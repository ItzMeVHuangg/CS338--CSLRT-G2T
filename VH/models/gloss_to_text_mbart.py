from typing import List, Optional

import torch
import torch.nn as nn


class PretrainedSeq2SeqGlossToText(nn.Module):
    """
    Gloss-to-Text wrapper around a pretrained HuggingFace seq2seq model.

    This path is meant for mBART/mT5-style experiments where the model owns its
    tokenizer and target vocabulary. It consumes the raw gloss/translation text
    strings kept in the existing GlossTextDataset batches, so it can coexist with
    the custom-vocab Transformer experiments.
    """

    def __init__(
        self,
        model_name: str,
        source_prefix: str = "translate gloss to German: ",
        src_lang: Optional[str] = "de_DE",
        tgt_lang: Optional[str] = "de_DE",
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        use_fast_tokenizer: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "PretrainedSeq2SeqGlossToText requires transformers. "
                "Install requirements_g2t_slt.txt or run: pip install transformers sentencepiece"
            ) from exc

        # mBART50 fast tokenizer conversion is brittle on some cluster
        # transformer/sentencepiece stacks, so default to the slow tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model_name = model_name
        self.source_prefix = source_prefix
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.num_beams = num_beams
        self.length_penalty = length_penalty

        if src_lang and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang
        self.forced_bos_token_id = self._resolve_forced_bos_token_id(tgt_lang)

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

        if freeze_encoder and hasattr(self.model, "get_encoder"):
            for param in self.model.get_encoder().parameters():
                param.requires_grad = False

    def _resolve_forced_bos_token_id(self, lang: Optional[str]) -> Optional[int]:
        if not lang:
            return None
        lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict) and lang in lang_code_to_id:
            return lang_code_to_id[lang]
        token_id = self.tokenizer.convert_tokens_to_ids(lang)
        if isinstance(token_id, int) and token_id != self.tokenizer.unk_token_id:
            return token_id
        return None

    def _source_texts(self, batch: dict) -> List[str]:
        return [self.source_prefix + text for text in batch["gloss_text"]]

    def _target_texts(self, batch: dict) -> List[str]:
        return list(batch["translation_text"])

    def _tokenize_sources(self, texts: List[str], device: torch.device) -> dict:
        if self.src_lang and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = self.src_lang
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_src_len,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in encoded.items()}

    def _tokenize_targets(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if self.tgt_lang and hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = self.tgt_lang
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_tgt_len,
                return_tensors="pt",
            )["input_ids"]
        labels = labels.to(device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels = labels.masked_fill(labels.eq(pad_id), -100)
        return labels

    def forward_loss(self, batch: dict, device: torch.device) -> torch.Tensor:
        model_inputs = self._tokenize_sources(self._source_texts(batch), device)
        labels = self._tokenize_targets(self._target_texts(batch), device)
        outputs = self.model(**model_inputs, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate_text(self, batch: dict, device: torch.device, max_len: Optional[int] = None) -> List[str]:
        model_inputs = self._tokenize_sources(self._source_texts(batch), device)
        generate_kwargs = {
            "max_length": max_len or self.max_tgt_len,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "early_stopping": True,
        }
        if self.forced_bos_token_id is not None:
            generate_kwargs["forced_bos_token_id"] = self.forced_bos_token_id
        generated = self.model.generate(**model_inputs, **generate_kwargs)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
