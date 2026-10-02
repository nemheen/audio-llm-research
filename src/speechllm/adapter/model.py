"""End-to-end speech LLM: Whisper encoder -> projector -> decoder-only LLM.

Design (see docs/REPORT.md, "Phase 3"):

* **Audio encoder** - Whisper's encoder, frozen. Qwen-Audio (Chu et al., 2023) also
  initialises from Whisper-large-v2 and shows the encoder already carries language,
  speaker and paralinguistic information, not just phonetics.
* **Projector** - Ultravox-style: stack ``k`` consecutive 20 ms frames (50 Hz ->
  50/k Hz) and map them with a 2-layer MLP into the LLM's embedding space. This is the
  only component trained in stage 1, so it fits on one consumer GPU.
* **LLM** - any HF causal LM with a chat template, frozen (optionally LoRA in stage 2).
* **Prompt** - audio embeddings are spliced into the chat prompt in place of an
  ``<AUDIO>`` marker, so no new tokens / embedding resizing are needed. Task and
  language are expressed as explicit tags (Qwen-Audio's hierarchical-tag idea) so one
  model can do ASR, language ID and instruction following without interference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

AUDIO_MARKER = "<AUDIO>"
WHISPER_FRAMES_PER_SECOND = 50  # encoder output rate (two stride-2 convs over 100 Hz mels)


@dataclass
class AudioLLMConfig:
    encoder_id: str = "openai/whisper-small"
    llm_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    stack_factor: int = 4  # 50 Hz -> 12.5 Hz audio tokens (Ultravox uses 8)
    projector_hidden: int = 2048
    freeze_llm: bool = True


class FrameStackProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, k: int):
        super().__init__()
        self.k = k
        self.norm = nn.LayerNorm(in_dim * k)
        self.fc1 = nn.Linear(in_dim * k, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)  # match the LLM's embedding scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        pad = (-t) % self.k
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        x = x.reshape(b, (t + pad) // self.k, d * self.k)
        return self.out_norm(self.fc2(F.gelu(self.fc1(self.norm(x)))))


class AudioLLM(nn.Module):
    def __init__(self, cfg: AudioLLMConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

        self.cfg = cfg
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.encoder_id)
        self.encoder = WhisperModel.from_pretrained(cfg.encoder_id, dtype=dtype).get_encoder()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_id)
        self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_id, dtype=dtype)
        self.projector = FrameStackProjector(
            self.encoder.config.d_model, self.llm.config.hidden_size, cfg.projector_hidden, cfg.stack_factor
        )

        self.encoder.requires_grad_(False).eval()
        if cfg.freeze_llm:
            self.llm.requires_grad_(False).eval()

    # ---------------------------------------------------------------- audio side
    def audio_features(self, wavs: list) -> tuple[torch.Tensor, list[int]]:
        feats = self.feature_extractor(wavs, sampling_rate=16_000, return_tensors="pt").input_features
        n_frames = [math.ceil(len(w) / 16_000 * WHISPER_FRAMES_PER_SECOND) for w in wavs]
        return feats, n_frames

    def encode_audio(self, feats: torch.Tensor, n_frames: list[int]) -> list[torch.Tensor]:
        """Returns one ``(n_audio_tokens, llm_dim)`` tensor per clip, padding frames dropped."""
        dev = next(self.projector.parameters()).device
        with torch.no_grad():
            h = self.encoder(feats.to(dev, dtype=self.encoder.dtype)).last_hidden_state
        h = self.projector(h.to(self.projector.fc1.weight.dtype))
        k = self.cfg.stack_factor
        return [h[i, : math.ceil(n / k)] for i, n in enumerate(n_frames)]

    # ---------------------------------------------------------------- text side
    def _embed(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        ids = ids.to(self.llm.device)
        return ids, self.llm.get_input_embeddings()(ids)

    def build_inputs(self, prompts: list[str], audio: list[torch.Tensor], targets: list[str] | None = None):
        """Splice audio embeddings into rendered chat prompts; right-pad into a batch.

        ``prompts`` are fully rendered chat-template strings containing exactly one
        ``AUDIO_MARKER``. If ``targets`` is given, they are appended (plus EOS) and
        are the only positions that receive loss.
        """
        embeds, labels = [], []
        eos = self.tokenizer.eos_token or ""
        for i, (prompt, a) in enumerate(zip(prompts, audio)):
            before, after = prompt.split(AUDIO_MARKER)
            ids_b, emb_b = self._embed(before)
            ids_a, emb_a = self._embed(after)
            parts, labs = [emb_b, a.to(emb_b.dtype), emb_a], [torch.full((len(ids_b) + len(a) + len(ids_a),), -100)]
            if targets is not None:
                ids_t, emb_t = self._embed(targets[i] + eos)
                parts.append(emb_t)
                labs.append(ids_t.cpu())
            embeds.append(torch.cat(parts))
            labels.append(torch.cat(labs))

        max_len = max(e.shape[0] for e in embeds)
        dim = embeds[0].shape[1]
        batch = torch.zeros(len(embeds), max_len, dim, dtype=embeds[0].dtype, device=embeds[0].device)
        mask = torch.zeros(len(embeds), max_len, dtype=torch.long, device=embeds[0].device)
        lab = torch.full((len(embeds), max_len), -100, dtype=torch.long)
        for i, (e, l) in enumerate(zip(embeds, labels)):
            batch[i, : len(e)] = e
            mask[i, : len(e)] = 1
            lab[i, : len(l)] = l
        return batch, mask, lab.to(batch.device)

    # ---------------------------------------------------------------- train / infer
    def forward(self, feats, n_frames, prompts, targets):
        audio = self.encode_audio(feats, n_frames)
        embeds, mask, labels = self.build_inputs(prompts, audio, targets)
        return self.llm(inputs_embeds=embeds, attention_mask=mask, labels=labels)

    @torch.no_grad()
    def generate(self, feats, n_frames, prompts, max_new_tokens: int = 128) -> list[str]:
        audio = self.encode_audio(feats, n_frames)
        outs = []
        for p, a in zip(prompts, audio):  # batch of 1: avoids left-padding bookkeeping
            embeds, mask, _ = self.build_inputs([p], [a])
            ids = self.llm.generate(
                inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            outs.append(self.tokenizer.decode(ids[0], skip_special_tokens=True).strip())
        return outs

    def trainable_state_dict(self) -> dict:
        return {f"projector.{k}": v for k, v in self.projector.state_dict().items()}
