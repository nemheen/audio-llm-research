"""Stage-1 training of the audio projector (encoder and LLM frozen).

    python scripts/train_adapter.py --config configs/adapter_mn.yaml
    python scripts/train_adapter.py --config configs/adapter_smoke.yaml   # CPU, ~minutes

Writes ``<out>/projector.pt``, ``<out>/train_log.jsonl`` and ``<out>/config.yaml``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechllm.adapter import AudioLLM, AudioLLMConfig  # noqa: E402
from speechllm.adapter.data import AudioTaskDataset, Collator, samples_from_manifest  # noqa: E402
from speechllm.data import load_split  # noqa: E402


def load_manifest(sources: list[dict]) -> pd.DataFrame:
    frames = []
    for src in sources:
        df = load_split(src["root"], src["locale"], src.get("split", "validated"))
        if src.get("max_clips"):
            df = df.sample(n=min(src["max_clips"], len(df)), random_state=0)
        if src.get("distill"):  # optional parquet/jsonl with path, instruction, distill_target
            df = df.merge(pd.read_json(src["distill"], lines=True), on="path", how="left")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def lr_at(step: int, cfg: dict) -> float:
    warm, total = cfg["warmup_steps"], cfg["max_steps"]
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    tcfg = cfg["train"]
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.manual_seed(tcfg.get("seed", 0))

    device = tcfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AudioLLM(AudioLLMConfig(**cfg["model"]), dtype=dtype).to(device)
    model.projector.float()  # keep trainable weights in fp32
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M")

    samples = samples_from_manifest(load_manifest(cfg["data"]["train"]), cfg["data"]["tasks"], seed=tcfg.get("seed", 0))
    loader = DataLoader(
        AudioTaskDataset(samples), batch_size=tcfg["batch_size"], shuffle=True,
        collate_fn=Collator(model), num_workers=tcfg.get("num_workers", 0), drop_last=True,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tcfg["lr"], weight_decay=0.01)

    step, accum, t0 = 0, tcfg.get("grad_accum", 1), time.time()
    log = (out / "train_log.jsonl").open("w")
    while step < tcfg["max_steps"]:
        for i, batch in enumerate(loader):
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                loss = model(batch["feats"], batch["n_frames"], batch["prompts"], batch["targets"]).loss / accum
            loss.backward()
            if (i + 1) % accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tcfg)
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            rec = {"step": step, "loss": loss.item() * accum, "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % tcfg.get("log_every", 10) == 0:
                print(rec)
            step += 1
            if step % tcfg.get("save_every", 1000) == 0 or step >= tcfg["max_steps"]:
                torch.save(model.trainable_state_dict(), out / "projector.pt")
            if step >= tcfg["max_steps"]:
                break
    log.close()
    print(f"done: {step} steps in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
