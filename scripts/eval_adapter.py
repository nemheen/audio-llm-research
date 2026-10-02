"""Evaluate a trained projector on a Common Voice split (ASR task).

    python scripts/eval_adapter.py --run checkpoints/adapter_mn --root <cv-root> --locale mn --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechllm.adapter import AudioLLM, AudioLLMConfig  # noqa: E402
from speechllm.adapter.data import render_prompt  # noqa: E402
from speechllm.data import load_audio, load_split  # noqa: E402
from speechllm.evaluate import score_asr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory written by train_adapter.py")
    ap.add_argument("--root", required=True)
    ap.add_argument("--locale", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "lid+asr"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioLLM(AudioLLMConfig(**cfg["model"]), dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    state = torch.load(run / "projector.pt", map_location=device)
    model.projector.load_state_dict({k.removeprefix("projector."): v for k, v in state.items()})
    model.projector.to(model.llm.dtype).eval()

    df = load_split(args.root, args.locale, args.split)
    if args.limit:
        df = df.head(args.limit)
    hyps = []
    for audio in tqdm(df["audio"], desc="adapter"):
        wav = load_audio(audio)
        feats, n = model.audio_features([wav])
        prompt = render_prompt(model.tokenizer, args.task, args.locale)
        out = model.generate(feats, n, [prompt])[0]
        hyps.append(out.split("text:", 1)[-1].strip() if args.task == "lid+asr" else out)
    df["hyp"] = hyps

    out = Path(args.out or run / f"eval_{args.locale}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    df[["path", "sentence", "hyp"]].to_csv(out / "predictions.csv", index=False)
    metrics = score_asr(df, lang_col=None)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
