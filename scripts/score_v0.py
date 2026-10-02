"""Re-score the v0 cascade (Whisper-small -> Gemma-2B-it) outputs.

v0 was run on Colab (notebooks/00_cascade_v0_colab.ipynb) and its raw outputs are
kept verbatim in results/v0_cascade/raw_outputs.csv. This script:

* reproduces the number printed by the notebook (WER 1.3936 on 148 rows), then
* re-scores on the de-duplicated 107 clips with proper normalisation, and
* scores the LLM stage as a classification task instead of ROUGE-L against a
  one-word label.

    python scripts/score_v0.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechllm.cascade import label_from_freetext  # noqa: E402
from speechllm.data import canonical_reason  # noqa: E402
from speechllm.evaluate import looks_english, score_asr, score_labels  # noqa: E402

OUT = Path("results/v0_cascade")


def main() -> None:
    raw = pd.read_csv(OUT / "raw_outputs.csv", index_col=0)
    raw["asr_transcript"] = raw["asr_transcript"].fillna("")

    as_reported = {
        "n_rows": len(raw),
        "n_unique_clips": int(raw["path"].nunique()),
        # exactly what `evaluate.load("wer")` computed in the notebook: no normalisation
        "wer_as_printed": float(jiwer.wer(raw["sentence_x"].tolist(), raw["asr_transcript"].tolist())),
    }

    # Collapse duplicated clips; keep every report reason as a gold label set.
    raw["label_c"] = raw["reason"].map(canonical_reason)
    grp = raw.groupby("path")
    df = grp.first().reset_index()
    df["labels"] = grp["label_c"].apply(lambda s: sorted(set(s))).values
    df["label"] = grp["label_c"].agg(lambda s: s.value_counts().index[0]).values
    df = df.rename(columns={"sentence_x": "sentence", "asr_transcript": "hyp", "llm_reasoning_output": "llm_raw"})
    df["is_english"] = df["sentence"].map(looks_english)
    df["pred_label"] = df["llm_raw"].map(label_from_freetext)

    metrics = {"as_reported": as_reported, "asr": score_asr(df), "reasoning": score_labels(df)}
    cols = ["path", "sentence", "hyp", "is_english", "label", "labels", "pred_label", "llm_raw", "duration[ms]", "gender", "accents"]
    df[cols].to_csv(OUT / "predictions.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
