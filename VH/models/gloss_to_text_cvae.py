import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gloss_to_text import PositionalEncoding


class LatentDomainSlotGlossToText(nn.Module):
    """
    CVAE-style latent Gloss-to-Text model.

    The prior path predicts a latent variable from the gloss encoder. During
    training, a posterior path also sees the target text and regularizes the
    prior through a KL term. During validation/test/generation, only the prior is
    used, so the model does not peek at the reference translation.
    """

    def __init__(
        self,
        gloss_vocab_size: int,
        text_vocab_size: int,
        gloss_pad_idx: int = 1,
        text_pad_idx: int = 1,
        slot_dim: int = 6,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        domain_prefix_len: int = 1,
        use_weather_slots: bool = True,
        use_conv_gate: bool = True,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tgt_len = max_tgt_len
        self.gloss_pad_idx = gloss_pad_idx
        self.text_pad_idx = text_pad_idx
        self.domain_prefix_len = domain_prefix_len
        self.use_weather_slots = use_weather_slots
        self.use_conv_gate = use_conv_gate
        self.latent_dim = latent_dim
        self.scale = math.sqrt(d_model)
        self._last_kl_loss: Optional[torch.Tensor] = None

        self.gloss_embedding = nn.Embedding(gloss_vocab_size, d_model, padding_idx=gloss_pad_idx)
        self.text_embedding = nn.Embedding(text_vocab_size, d_model, padding_idx=text_pad_idx)
        self.pos_src = PositionalEncoding(d_model, max_src_len + domain_prefix_len + 5, dropout)
        self.pos_tgt = PositionalEncoding(d_model, max_tgt_len + 4, dropout)

        self.domain_prefix = nn.Parameter(torch.randn(1, domain_prefix_len, d_model) * 0.02)
        if use_weather_slots:
            self.slot_proj = nn.Sequential(
                nn.Linear(slot_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.slot_proj = None

        if use_conv_gate:
            self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            self.gate = nn.Linear(d_model * 2, d_model)
            self.gate_norm = nn.LayerNorm(d_model)
            self.gate_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.prior = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.prior_mu = nn.Linear(d_model, latent_dim)
        self.prior_logvar = nn.Linear(d_model, latent_dim)

        self.posterior = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.posterior_mu = nn.Linear(d_model, latent_dim)
        self.posterior_logvar = nn.Linear(d_model, latent_dim)

        self.latent_to_memory = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.out_proj = nn.Linear(d_model, text_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if "embedding" in name or "domain_prefix" in name:
                continue
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    @staticmethod
    def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        valid = (~pad_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _apply_conv_gate(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        conv = F.gelu(conv)
        gate = torch.sigmoid(self.gate(torch.cat([x, conv], dim=-1)))
        mixed = gate * conv + (1.0 - gate) * x
        return self.gate_norm(x + self.gate_dropout(mixed))

    def _encode_source(
        self,
        gloss: torch.Tensor,
        gloss_lens: torch.Tensor,
        weather_slots: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, src_len = gloss.shape
        x = self.gloss_embedding(gloss) * self.scale
        x = self.pos_src(x)
        if self.use_conv_gate:
            x = self._apply_conv_gate(x)

        prefix_parts = [self.domain_prefix.expand(batch_size, -1, -1)]
        if self.slot_proj is not None and weather_slots is not None:
            prefix_parts.append(self.slot_proj(weather_slots).unsqueeze(1))
        prefix = torch.cat(prefix_parts, dim=1)

        src = torch.cat([prefix, x], dim=1)
        gloss_pad = self.make_padding_mask(gloss_lens, src_len)
        prefix_pad = torch.zeros(batch_size, prefix.size(1), dtype=torch.bool, device=gloss.device)
        src_pad = torch.cat([prefix_pad, gloss_pad], dim=1)
        return self.encoder(src, src_key_padding_mask=src_pad), src_pad

    def _prior_stats(self, src_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.prior(src_summary)
        return self.prior_mu(hidden), self.prior_logvar(hidden).clamp(min=-8.0, max=8.0)

    def _posterior_stats(
        self,
        tgt: torch.Tensor,
        tgt_lens: torch.Tensor,
        src_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tgt_emb = self.text_embedding(tgt) * self.scale
        tgt_pad = self.make_padding_mask(tgt_lens, tgt.size(1))
        tgt_summary = self._masked_mean(tgt_emb, tgt_pad)
        hidden = self.posterior(torch.cat([src_summary, tgt_summary], dim=-1))
        return self.posterior_mu(hidden), self.posterior_logvar(hidden).clamp(min=-8.0, max=8.0)

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def _kl_divergence(
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        prior_var = torch.exp(prior_logvar)
        posterior_var = torch.exp(posterior_logvar)
        kl = prior_logvar - posterior_logvar
        kl = kl + (posterior_var + (posterior_mu - prior_mu).pow(2)) / prior_var.clamp(min=1e-8)
        kl = 0.5 * (kl - 1.0)
        return kl.sum(dim=-1).mean()

    def _append_latent(
        self,
        memory: torch.Tensor,
        src_pad: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_token = self.latent_to_memory(z).unsqueeze(1)
        latent_pad = torch.zeros(memory.size(0), 1, dtype=torch.bool, device=memory.device)
        return torch.cat([latent_token, memory], dim=1), torch.cat([latent_pad, src_pad], dim=1)

    def encode(
        self,
        gloss: torch.Tensor,
        gloss_lens: torch.Tensor,
        weather_slots: Optional[torch.Tensor] = None,
        tgt: Optional[torch.Tensor] = None,
        tgt_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, src_pad = self._encode_source(gloss, gloss_lens, weather_slots)
        src_summary = self._masked_mean(memory, src_pad)
        prior_mu, prior_logvar = self._prior_stats(src_summary)

        if self.training and tgt is not None and tgt_lens is not None:
            posterior_mu, posterior_logvar = self._posterior_stats(tgt, tgt_lens, src_summary)
            z = self._sample(posterior_mu, posterior_logvar)
            self._last_kl_loss = self._kl_divergence(
                posterior_mu, posterior_logvar, prior_mu, prior_logvar
            )
        else:
            z = prior_mu
            self._last_kl_loss = None

        return self._append_latent(memory, src_pad, z)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_lens: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt_emb = self.text_embedding(tgt) * self.scale
        tgt_emb = self.pos_tgt(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        tgt_pad = self.make_padding_mask(tgt_lens, tgt.size(1)) if tgt_lens is not None else None
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(out)

    def forward(
        self,
        gloss: torch.Tensor,
        gloss_lens: torch.Tensor,
        tgt: torch.Tensor,
        tgt_lens: torch.Tensor,
        weather_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory, src_pad = self.encode(gloss, gloss_lens, weather_slots, tgt, tgt_lens)
        return self.decode(tgt, memory, tgt_lens, src_pad)

    def auxiliary_loss(self) -> Optional[torch.Tensor]:
        return self._last_kl_loss

    @torch.no_grad()
    def greedy_decode(
        self,
        gloss: torch.Tensor,
        gloss_lens: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        weather_slots: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        memory, src_pad = self.encode(gloss, gloss_lens, weather_slots)
        batch_size = gloss.size(0)
        max_len = max_len or self.max_tgt_len

        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=gloss.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=gloss.device)
        lengths = torch.ones(batch_size, dtype=torch.long, device=gloss.device)

        for _ in range(max_len):
            logits = self.decode(ys, memory, lengths, src_pad)
            next_token = logits[:, -1].argmax(dim=-1)
            finished |= next_token.eq(eos_idx)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            lengths = lengths + (~finished).long()
            if finished.all():
                break

        if was_training:
            self.train()
        return ys[:, 1:]
