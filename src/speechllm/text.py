"""Text normalisation applied before WER/CER scoring.

Following the Whisper paper (Radford et al., 2022, Sec. 4.1 / App. C), WER is only
meaningful after both reference and hypothesis are normalised: otherwise casing and
punctuation differences are counted as errors. We keep the normaliser deliberately
small and script-agnostic (it works for Latin and Mongolian Cyrillic alike) instead of
pulling in Whisper's English-only spelling map.
"""

from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "`": "'"})
_DASHES = re.compile(r"[‐-―-]")
# Anything that is not a letter, digit, whitespace or apostrophe becomes a space.
_NON_WORD = re.compile(r"[^\w\s']", flags=re.UNICODE)
_EDGE_APOSTROPHE = re.compile(r"(^|\s)'+|'+(?=\s|$)")


def normalize(text: str | None) -> str:
    """Lower-case, strip punctuation and collapse whitespace.

    Initials such as ``З.Энхболд`` become two tokens (``з энхболд``), which matches how
    they are read aloud.
    """
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).lower()
    text = _DASHES.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    text = _EDGE_APOSTROPHE.sub(" ", text)
    return " ".join(text.split())
