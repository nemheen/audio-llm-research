"""ASR and downstream-task metrics.

Conventions
-----------
* WER / CER are returned as *fractions* (0.25 == 25 %). The v0 report misread
  ``evaluate``'s WER of 1.39 as "1.39 %"; it was 139 %.
* Corpus-level WER is ``sum(S+D+I) / sum(N)`` over all utterances (what ``jiwer``
  computes for a list), not the mean of per-utterance WERs.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Sequence

import jiwer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .text import normalize


def _prep(refs: Sequence[str], hyps: Sequence[str], norm: bool) -> tuple[list[str], list[str]]:
    if norm:
        refs = [normalize(r) for r in refs]
        hyps = [normalize(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]  # jiwer rejects empty references
    return [p[0] for p in pairs], [p[1] for p in pairs]


def wer(refs: Sequence[str], hyps: Sequence[str], norm: bool = True) -> float:
    r, h = _prep(refs, hyps, norm)
    return float(jiwer.wer(r, h)) if r else float("nan")


def cer(refs: Sequence[str], hyps: Sequence[str], norm: bool = True) -> float:
    r, h = _prep(refs, hyps, norm)
    return float(jiwer.cer(r, h)) if r else float("nan")


def utterance_wer(ref: str, hyp: str, norm: bool = True) -> float:
    return wer([ref], [hyp], norm=norm)


def compression_ratio(text: str) -> float:
    """gzip compression ratio, the hallucination heuristic from Whisper (Sec. 4.5).

    Whisper re-decodes at a higher temperature when this exceeds 2.4, because
    repetitive loops ("the the the ...") compress extremely well.
    """
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw))


def is_degenerate(hyp: str, ref: str | None = None, cr_threshold: float = 2.4, len_ratio: float = 3.0) -> bool:
    """Flag repetition loops / runaway decodes.

    A hypothesis is degenerate if it is highly compressible *or* (when a reference is
    available) far longer than the reference.
    """
    if compression_ratio(hyp) > cr_threshold:
        return True
    if ref:
        n_ref, n_hyp = len(normalize(ref).split()), len(normalize(hyp).split())
        return n_hyp > len_ratio * n_ref + 5
    return False


def classification_report(gold: Iterable[str], pred: Iterable[str], labels: Sequence[str]) -> dict:
    gold, pred = list(gold), list(pred)
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=list(labels), average="macro", zero_division=0)),
        "per_class_f1": {
            lab: float(f)
            for lab, f in zip(labels, f1_score(gold, pred, labels=list(labels), average=None, zero_division=0))
        },
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-sample scores (small eval sets!)."""
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wer_bootstrap_ci(
    refs: Sequence[str], hyps: Sequence[str], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap CI for *corpus* WER: resample utterances, recompute sum(errors)/sum(words)."""
    r, h = _prep(refs, hyps, norm=True)
    if not r:
        return float("nan"), float("nan")
    errs, words = [], []
    for ref, hyp in zip(r, h):
        o = jiwer.process_words(ref, hyp)
        errs.append(o.substitutions + o.deletions + o.insertions)
        words.append(o.substitutions + o.deletions + o.hits)
    errs, words = np.asarray(errs), np.asarray(words)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    stats = errs[idx].sum(1) / words[idx].sum(1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
