"""Score a predictions table produced by any pipeline version.

Expected columns: ``sentence`` (reference), ``hyp`` (ASR output), and optionally
``label`` / ``labels`` (gold report reason, primary / set) and ``pred_label``.
"""

from __future__ import annotations

import pandas as pd

from . import metrics as M
from .data import REASON_LABELS


def score_asr(df: pd.DataFrame, lang_col: str | None = "is_english") -> dict:
    refs, hyps = df["sentence"].tolist(), df["hyp"].fillna("").tolist()
    degenerate = [M.is_degenerate(h, r) for r, h in zip(refs, hyps)]
    out = {
        "n_utts": len(df),
        "wer": M.wer(refs, hyps),
        "wer_ci95": M.wer_bootstrap_ci(refs, hyps),
        "wer_unnormalised": M.wer(refs, hyps, norm=False),
        "cer": M.cer(refs, hyps),
        "degenerate_rate": sum(degenerate) / max(len(df), 1),
        "wer_excl_degenerate": M.wer([r for r, d in zip(refs, degenerate) if not d], [h for h, d in zip(hyps, degenerate) if not d]),
    }
    if lang_col and lang_col in df:
        en = df[df[lang_col]]
        out["wer_english_prompts"] = M.wer(en["sentence"].tolist(), en["hyp"].fillna("").tolist())
        out["n_english_prompts"] = len(en)
    return out


def score_labels(df: pd.DataFrame) -> dict:
    rep = M.classification_report(df["label"], df["pred_label"], REASON_LABELS)
    # A clip may have been reported for several reasons; count a hit if any matches.
    rep["any_match_accuracy"] = float(
        sum(p in set(g) for p, g in zip(df["pred_label"], df["labels"])) / max(len(df), 1)
    )
    return rep


def looks_english(sentence: str) -> bool:
    """Heuristic text-language check for the *prompt* sentence (not the audio).

    Several "different-language" prompts in the English corpus are Swahili. We use a
    tiny stop-word test so the split is transparent and dependency-free.
    """
    words = set(sentence.lower().replace(",", " ").replace(".", " ").split())
    sw_markers = {"ya", "na", "kwa", "wa", "ni", "kuwa", "hii", "katika", "kwenye", "yenye", "haya", "mara"}
    if not sentence.isascii() and not all(ch in "‘’“”—–" for ch in sentence if not ch.isascii()):
        return False
    return len(words & sw_markers) == 0
