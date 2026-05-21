import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class DomainSlotGlossToText(nn.Module):
    """
    Fast Gloss-to-Text SLT model for PHOENIX-2014T.

    The encoder is domain-aware through a learned weather-domain prefix and an
    optional weather-slot token extracted from gloss keywords. The Conv1D gate is
    a lightweight ADAT-inspired local/global fusion block.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tgt_len = max_tgt_len
        self.gloss_pad_idx = gloss_pad_idx
        self.text_pad_idx = text_pad_idx
        self.domain_prefix_len = domain_prefix_len
        self.use_weather_slots = use_weather_slots
        self.use_conv_gate = use_conv_gate
        self.scale = math.sqrt(d_model)

        self.gloss_embedding = nn.Embedding(gloss_vocab_size, d_model, padding_idx=gloss_pad_idx)
        self.text_embedding = nn.Embedding(text_vocab_size, d_model, padding_idx=text_pad_idx)
        self.pos_src = PositionalEncoding(d_model, max_src_len + domain_prefix_len + 4, dropout)
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
        self.out_proj = nn.Linear(d_model, text_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    def _apply_conv_gate(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        conv = F.gelu(conv)
        gate = torch.sigmoid(self.gate(torch.cat([x, conv], dim=-1)))
        mixed = gate * conv + (1.0 - gate) * x
        return self.gate_norm(x + self.gate_dropout(mixed))

    def encode(
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

        memory = self.encoder(src, src_key_padding_mask=src_pad)
        return memory, src_pad

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
        memory, src_pad = self.encode(gloss, gloss_lens, weather_slots)
        return self.decode(tgt, memory, tgt_lens, src_pad)

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
        return ys[:, 1:]
