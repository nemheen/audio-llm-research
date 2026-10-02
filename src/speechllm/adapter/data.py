"""Multi-task samples for the audio adapter.

Every sample is (audio, task tags, instruction, target). Tags follow Qwen-Audio's
"hierarchical tags" idea in plain text so the frozen LLM can read them without new
vocabulary::

    <AUDIO>
    [task: transcribe] [lang: mn]
    Transcribe the speech.

Tasks
-----
transcribe   target = reference sentence (language tag given)
lid          target = language code (no language tag - that is the answer)
lid+asr      target = "lang: mn\\ntext: ..." (language unknown; model must infer both)
instruct     target = what the *text* LLM answers when given the reference transcript
             and the instruction. This is the DiVA / Ultravox distillation recipe: it
             needs no labelled instruction data and teaches the adapter to make audio
             "look like" text to the LLM. Targets come from scripts/make_distill_targets.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data import load_audio
from .model import AUDIO_MARKER

LANG_NAMES = {"en": "English", "mn": "Mongolian"}

INSTRUCTIONS = {
    "transcribe": "Transcribe the speech.",
    "lid": "Which language is spoken? Answer with the ISO 639-1 code.",
    "lid+asr": "Identify the spoken language, then transcribe the speech.",
}


@dataclass
class Sample:
    audio: str
    task: str
    lang: str
    target: str
    instruction: str | None = None


def render_user_turn(task: str, lang: str | None, instruction: str | None = None) -> str:
    tags = f"[task: {task}]" + (f" [lang: {lang}]" if lang and task in ("transcribe", "instruct") else "")
    return f"{AUDIO_MARKER}\n{tags}\n{instruction or INSTRUCTIONS[task]}"


def render_prompt(tokenizer, task: str, lang: str | None, instruction: str | None = None) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can listen to audio."},
        {"role": "user", "content": render_user_turn(task, lang, instruction)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def samples_from_manifest(df: pd.DataFrame, tasks: dict[str, float], seed: int = 0) -> list[Sample]:
    """Draw one task per clip according to ``tasks`` weights.

    ``df`` needs ``audio``, ``sentence``, ``locale`` and, for ``instruct``, columns
    ``instruction`` and ``distill_target``.
    """
    rng = random.Random(seed)
    names, weights = zip(*tasks.items())
    out = []
    for row in df.itertuples():
        task = rng.choices(names, weights)[0]
        if task == "instruct" and not isinstance(getattr(row, "distill_target", None), str):
            task = "transcribe"
        if task == "transcribe":
            out.append(Sample(row.audio, task, row.locale, row.sentence))
        elif task == "lid":
            out.append(Sample(row.audio, task, row.locale, row.locale))
        elif task == "lid+asr":
            out.append(Sample(row.audio, task, row.locale, f"lang: {row.locale}\ntext: {row.sentence}"))
        else:
            out.append(Sample(row.audio, task, row.locale, row.distill_target, row.instruction))
    return out


class AudioTaskDataset(Dataset):
    def __init__(self, samples: list[Sample], max_seconds: float = 30.0):
        self.samples = samples
        self.max_len = int(max_seconds * 16_000)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[Sample, torch.Tensor]:
        s = self.samples[i]
        return s, load_audio(s.audio)[: self.max_len]


class Collator:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        samples, wavs = zip(*batch)
        feats, n_frames = self.model.audio_features(list(wavs))
        prompts = [render_prompt(self.model.tokenizer, s.task, s.lang, s.instruction) for s in samples]
        return {"feats": feats, "n_frames": n_frames, "prompts": prompts, "targets": [s.target for s in samples], "samples": samples}
