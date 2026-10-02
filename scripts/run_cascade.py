"""Run the (fixed) cascade on a Common Voice evaluation set and score it.

Examples
--------
English reported-sentence set, full cascade (v1)::

    python scripts/run_cascade.py --config configs/cascade_v1_en.yaml

Mongolian zero-shot ASR only::

    python scripts/run_cascade.py --config configs/asr_mn_zeroshot.yaml --asr-model openai/whisper-large-v3-turbo

Any YAML key can be overridden on the command line (``--asr-model``, ``--language`` ...).
ASR outputs are cached per (model, language, guard) so LLM ablations reuse them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechllm.cascade import ASRConfig, ASRResult, LLMReasoner, ReasonerConfig, WhisperASR  # noqa: E402
from speechllm.data import build_reported_eval_set, load_audio, load_split  # noqa: E402
from speechllm.evaluate import looks_english, score_asr, score_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root")
    p.add_argument("--locale")
    p.add_argument("--eval-set", choices=["reported", "other", "validated", "test"])
    p.add_argument("--asr-model")
    p.add_argument("--language", help="force a Whisper language code; omit to auto-detect")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--llm-model", help="omit (and leave unset in YAML) for ASR-only runs")
    p.add_argument("--no-lid", action="store_true", help="ablation: hide language-ID evidence from the LLM")
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(Path(args.config).read_text())
    overrides = {
        "root": args.root, "locale": args.locale, "eval_set": args.eval_set, "asr_model": args.asr_model,
        "language": args.language, "llm_model": args.llm_model, "limit": args.limit, "out": args.out,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.no_guard:
        cfg["guard"] = False
    if args.no_lid:
        cfg["use_lid"] = False
    return cfg


def load_eval_set(cfg: dict) -> pd.DataFrame:
    if cfg["eval_set"] == "reported":
        df = build_reported_eval_set(cfg["root"], cfg["locale"])
    else:
        df = load_split(cfg["root"], cfg["locale"], cfg["eval_set"])
    df["is_english"] = df["sentence"].map(looks_english) if cfg["locale"] == "en" else False
    return df.head(cfg["limit"]) if cfg.get("limit") else df


def run_asr(df: pd.DataFrame, cfg: dict, cache: Path) -> pd.DataFrame:
    """Transcribe every clip, appending to a JSONL cache so interrupted runs resume."""
    done = set()
    if cache.exists():
        done = set(pd.read_json(cache, orient="records", lines=True)["path"])
    todo = df[~df["path"].isin(done)]
    if len(todo):
        print(f"[asr] {len(done)} cached, {len(todo)} to transcribe -> {cache}")
        asr = WhisperASR(ASRConfig(model_id=cfg["asr_model"], language=cfg.get("language"), guard=cfg.get("guard", True)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as f:
            for path, audio in tqdm(zip(todo["path"], todo["audio"]), total=len(todo), desc=f"ASR {cfg['asr_model']}"):
                t0 = time.time()
                rec = {"path": path, **asdict(asr.transcribe(load_audio(audio))), "asr_seconds": round(time.time() - t0, 2)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    out = pd.read_json(cache, orient="records", lines=True)
    return out[out["path"].isin(set(df["path"]))].drop_duplicates("path")


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval_set(cfg)
    tag = f"{cfg['asr_model'].split('/')[-1]}_{cfg.get('language') or 'auto'}_{'guard' if cfg.get('guard', True) else 'noguard'}"
    asr = run_asr(df, cfg, Path("results/.asr_cache") / cfg["locale"] / cfg["eval_set"] / f"{tag}.jsonl")
    df = df.merge(asr, on="path", how="left").rename(columns={"text": "hyp"})

    metrics: dict = {"config": cfg, "asr": score_asr(df)}
    if cfg["locale"] != "en":
        metrics["asr"].pop("wer_english_prompts", None)
    metrics["asr"]["lid_distribution"] = df["lang"].value_counts().to_dict()

    if cfg.get("llm_model") and "label" in df:
        reasoner = LLMReasoner(ReasonerConfig(model_id=cfg["llm_model"], use_lid=cfg.get("use_lid", True)))
        preds = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"LLM {cfg['llm_model']}"):
            res = ASRResult(row["hyp"], row["lang"], row["lang_prob"], row["compression_ratio"], row["degenerate"], [tuple(x) for x in row["lang_top3"]])
            preds.append(reasoner(res))
        df["pred_label"], df["llm_raw"] = zip(*preds)
        metrics["reasoning"] = score_labels(df)

    keep = [c for c in ["path", "sentence", "hyp", "lang", "lang_prob", "lang_top3", "degenerate", "is_english", "label", "labels", "pred_label", "llm_raw", "duration_s"] if c in df]
    df[keep].to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
