"""Common Voice loading and evaluation-set construction.

Layout expected under ``root`` (an extracted Common Voice release or delta)::

    root/<locale>/clips/*.mp3
    root/<locale>/{validated,other,reported,clip_durations,...}.tsv
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 16_000

# The free-text "reason" field of reported.tsv, folded into a closed label set.
REASON_LABELS = ["different-language", "grammar-or-spelling", "offensive-language", "difficult-pronunciation", "other"]
_REASON_MAP = {
    "different-language": "different-language",
    "sc-different-language": "different-language",
    "grammar-or-spelling": "grammar-or-spelling",
    "offensive-language": "offensive-language",
    "difficult-pronounce": "difficult-pronunciation",
    "difficult-pronunciation": "difficult-pronunciation",
}


def canonical_reason(reason: str) -> str:
    return _REASON_MAP.get(str(reason).strip().lower(), "other")


def read_tsv(root: str | Path, locale: str, name: str) -> pd.DataFrame:
    # Common Voice sentences contain bare quotes, so disable quote handling.
    return pd.read_csv(Path(root) / locale / f"{name}.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def load_split(root: str | Path, locale: str, split: str = "validated") -> pd.DataFrame:
    """One row per clip with absolute ``audio`` path, ``sentence`` and ``duration_s``."""
    df = read_tsv(root, locale, split)
    dur = read_tsv(root, locale, "clip_durations").rename(columns={"clip": "path"})
    df = df.merge(dur, on="path", how="left")
    df["duration_s"] = pd.to_numeric(df["duration[ms]"], errors="coerce") / 1000.0
    df["audio"] = [str(Path(root) / locale / "clips" / p) for p in df["path"]]
    df["locale"] = locale
    return df.drop(columns=["duration[ms]"])


def build_reported_eval_set(root: str | Path, locale: str = "en") -> pd.DataFrame:
    """Clips whose *prompt sentence* was reported by the community, one row per clip.

    The v0 notebook inner-joined ``other`` x ``reported`` on ``sentence_id``. Because a
    sentence can be reported several times, that join duplicated clips (148 rows for
    107 clips) and over-weighted a handful of hallucinating ones. Here every clip
    appears once and carries the *set* of canonical reasons plus a primary label (the
    most frequent reason; ties broken by REASON_LABELS order).
    """
    clips = load_split(root, locale, "other")
    rep = read_tsv(root, locale, "reported")
    rep["label"] = rep["reason"].map(canonical_reason)

    def _agg(labels: pd.Series) -> pd.Series:
        counts = Counter(labels)
        primary = max(counts, key=lambda lab: (counts[lab], -REASON_LABELS.index(lab)))
        return pd.Series({"labels": sorted(counts), "label": primary, "raw_reasons": "|".join(sorted(set(labels)))})

    per_sentence = rep.groupby("sentence_id")["label"].apply(_agg).unstack()
    out = clips.merge(per_sentence, left_on="sentence_id", right_index=True, how="inner")
    return out.drop_duplicates("path").reset_index(drop=True)


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 waveform at ``sr``. libsndfile >= 1.1 reads mp3 directly."""
    import librosa
    import soundfile as sf

    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav.astype(np.float32)
