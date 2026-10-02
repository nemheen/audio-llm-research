"""Cascade pipeline: Whisper ASR -> instruction-tuned LLM.

v0 (notebooks/00_cascade_v0_colab.ipynb) used ``pipeline("automatic-speech-recognition")``
with default decoding and fed the transcript to ``gemma-2b-it`` as a raw string. The
failure analysis (docs/REPORT.md) traced most errors to three causes, each fixed here:

1. Runaway decodes on non-English speech -> Whisper's own guard rails (Sec. 4.5 of the
   paper): temperature fallback on compression-ratio > 2.4 / avg-logprob < -1, plus a
   duration-aware ``max_new_tokens`` cap.
2. Language identity was discarded -> we expose Whisper's language-ID posterior and
   hand it to the LLM as evidence (in the spirit of Qwen-Audio's explicit language tags).
3. The LLM saw no chat template and sampled at T=0.7 -> it echoed the prompt
   scaffold. We apply the chat template, decode greedily and constrain the output to a
   closed label set returned as JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import torch

from .data import REASON_LABELS, SAMPLE_RATE
from .metrics import compression_ratio, is_degenerate

FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class ASRConfig:
    model_id: str = "openai/whisper-small"
    language: str | None = None  # None = let Whisper detect; e.g. "en", "mn" to force
    guard: bool = True  # temperature fallback + duration-aware length cap
    tokens_per_second: float = 12.0  # generous: Cyrillic costs ~3x more BPE tokens than English
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ASRResult:
    text: str
    lang: str
    lang_prob: float
    compression_ratio: float
    degenerate: bool
    lang_top3: list[tuple[str, float]] = field(default_factory=list)


class WhisperASR:
    def __init__(self, cfg: ASRConfig):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.cfg = cfg
        dtype = torch.float16 if cfg.device.startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(cfg.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id, dtype=dtype).to(cfg.device).eval()
        gc = self.model.generation_config
        self._lang_ids = torch.tensor(list(gc.lang_to_id.values()), device=cfg.device)
        self._lang_codes = [tok.strip("<|>") for tok in gc.lang_to_id]

    def _features(self, wav: np.ndarray) -> torch.Tensor:
        feats = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        return feats.to(self.cfg.device, dtype=self.model.dtype)

    @torch.no_grad()
    def language_posterior(self, feats: torch.Tensor) -> torch.Tensor:
        """Softmax over Whisper's language tokens at the first decoder step."""
        start = torch.full((feats.shape[0], 1), self.model.generation_config.decoder_start_token_id, device=feats.device)
        logits = self.model(input_features=feats, decoder_input_ids=start, use_cache=False).logits[:, -1]
        return logits[:, self._lang_ids].float().softmax(-1)

    @torch.no_grad()
    def transcribe(self, wav: np.ndarray) -> ASRResult:
        feats = self._features(wav)
        post = self.language_posterior(feats)[0]
        top = post.topk(3)
        top3 = [(self._lang_codes[i], float(p)) for p, i in zip(top.values, top.indices)]
        lang = self.cfg.language or top3[0][0]

        gen_kwargs: dict = {"language": lang, "task": "transcribe"}
        if self.cfg.guard:
            duration = len(wav) / SAMPLE_RATE
            gen_kwargs.update(
                temperature=FALLBACK_TEMPERATURES,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                do_sample=False,
                max_new_tokens=min(440, int(self.cfg.tokens_per_second * duration) + 16),
            )
        ids = self.model.generate(feats, **gen_kwargs)
        if not torch.is_tensor(ids):  # some transformers versions return a ModelOutput
            ids = ids["sequences"]
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return ASRResult(
            text=text,
            lang=lang,
            lang_prob=float(post[self._lang_codes.index(lang)]) if lang in self._lang_codes else float("nan"),
            compression_ratio=compression_ratio(text),
            degenerate=is_degenerate(text),
            lang_top3=top3,
        )


# --------------------------------------------------------------------------------------
# LLM reasoner
# --------------------------------------------------------------------------------------

LABEL_DEFINITIONS = {
    "different-language": "the speaker was asked to read a sentence that is not in the expected language",
    "grammar-or-spelling": "the prompt sentence contains a grammar or spelling mistake",
    "offensive-language": "the prompt sentence contains offensive or sensitive content",
    "difficult-pronunciation": "the prompt sentence contains words that are hard to pronounce (names, jargon)",
    "other": "none of the above",
}

SYSTEM_PROMPT = (
    "You audit a crowd-sourced speech corpus. Each item is a recording of a volunteer reading a prompt "
    "sentence aloud. Volunteers can report a prompt sentence for one of these reasons:\n"
    + "\n".join(f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items())
    + "\n\nYou receive the automatic transcript of the recording and the language-identification "
    "evidence from the speech recogniser. The transcript may itself contain recognition errors. "
    'Answer with a single JSON object: {"label": <one of the labels>, "explanation": <one short sentence>}.'
)


@dataclass
class ReasonerConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_lid: bool = True
    max_new_tokens: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LLMReasoner:
    def __init__(self, cfg: ReasonerConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        dtype = torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype).to(cfg.device).eval()

    def build_messages(self, asr: ASRResult) -> list[dict]:
        lines = [f"Transcript: {asr.text!r}"]
        if self.cfg.use_lid:
            lid = ", ".join(f"{code}={p:.2f}" for code, p in asr.lang_top3)
            lines.append(f"Detected spoken language (top-3 posterior): {lid}")
            if asr.degenerate:
                lines.append("Warning: the recogniser produced a repetitive, low-confidence transcript.")
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]

    @torch.no_grad()
    def __call__(self, asr: ASRResult) -> tuple[str, str]:
        prompt = self.tok.apply_chat_template(self.build_messages(asr), tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.cfg.device)
        out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        raw = self.tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        return parse_label(raw), raw


_JSON_OBJ = re.compile(r"\{.*?\}", flags=re.S)


def parse_label(raw: str) -> str:
    """Parse the JSON answer; fall back to keyword search; else ``other``."""
    m = _JSON_OBJ.search(raw or "")
    if m:
        try:
            label = str(json.loads(m.group(0)).get("label", "")).strip().lower()
            if label in REASON_LABELS:
                return label
        except json.JSONDecodeError:
            pass
    return label_from_freetext(raw)


def label_from_freetext(text: str) -> str:
    """Keyword matcher used to score free-text outputs (e.g. v0's Gemma generations)."""
    t = (text or "").lower()
    rules = [
        ("different-language", ("different-language", "different language", "foreign", "not english", "another language")),
        ("offensive-language", ("offensive",)),
        ("difficult-pronunciation", ("pronunc", "pronounce")),
        ("grammar-or-spelling", ("grammar", "spelling", "misspell", "typo")),
    ]
    for label, keys in rules:
        if any(k in t for k in keys):
            return label
    return "other"
