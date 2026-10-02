"""Build instruction-following targets by self-distillation from the text LLM.

For each clip we pick an instruction and record what the *frozen text LLM* answers
when it is given the reference transcript. Training the adapter on (audio,
instruction) -> that answer teaches it to make speech "look like" the transcript to
the LLM, without any human-written spoken-QA data (DiVA, Held et al. 2024; Ultravox).

    python scripts/make_distill_targets.py --root <cv-root> --locale mn --split train \
        --llm Qwen/Qwen2.5-1.5B-Instruct --out data/processed/distill_mn_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechllm.data import load_split  # noqa: E402

INSTRUCTIONS = {
    "en": [
        "Summarise what the speaker says in one sentence.",
        "What is the main topic of this utterance?",
        "Translate what the speaker says into Mongolian.",
        "Is the statement positive, negative or neutral? Explain briefly.",
        "List the named entities mentioned by the speaker.",
    ],
    "mn": [
        "Илтгэгчийн хэлсэн зүйлийг нэг өгүүлбэрээр хураангуйл.",  # summarise in one sentence
        "Энэ яриа ямар сэдвийн тухай вэ?",  # what is the topic
        "Илтгэгчийн хэлснийг англи хэл рүү орчуул.",  # translate to English
        "Дурдагдсан хүн, газар, байгууллагын нэрсийг жагсаа.",  # list named entities
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--locale", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

    df = load_split(args.root, args.locale, args.split)
    if args.limit:
        df = df.head(args.limit)
    rng = random.Random(0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for path, sentence in tqdm(zip(df["path"], df["sentence"]), total=len(df)):
            instruction = rng.choice(INSTRUCTIONS[args.locale])
            msgs = [{"role": "user", "content": f'Transcript: "{sentence}"\n\n{instruction}'}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = llm.generate(ids, max_new_tokens=128, do_sample=False)
            answer = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()
            f.write(json.dumps({"path": path, "instruction": instruction, "distill_target": answer}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
