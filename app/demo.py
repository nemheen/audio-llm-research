"""Gradio demo of the v1 cascade: speak or upload audio -> transcript, language ID, LLM answer.

    pip install -e ".[demo]"
    python app/demo.py --asr openai/whisper-small --llm Qwen/Qwen2.5-1.5B-Instruct

Unlike the evaluation pipeline, the demo lets the user type a free instruction that the
LLM applies to the transcript (e.g. "summarise", "translate to English", "is this a question?").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechllm.cascade import ASRConfig, LLMReasoner, ReasonerConfig, WhisperASR  # noqa: E402
from speechllm.data import load_audio  # noqa: E402


def main() -> None:
    import gradio as gr

    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    asr = WhisperASR(ASRConfig(model_id=args.asr))
    llm = LLMReasoner(ReasonerConfig(model_id=args.llm, max_new_tokens=256))

    def run(audio_path: str | None, language: str, instruction: str):
        if not audio_path:
            return "", "", "Record or upload audio first."
        asr.cfg.language = None if language == "auto" else language
        res = asr.transcribe(load_audio(audio_path))
        lid = ", ".join(f"{c} {p:.2f}" for c, p in res.lang_top3)
        msgs = [
            {"role": "system", "content": "You answer questions about a speech recording, given its automatic transcript and the detected language."},
            {"role": "user", "content": f"Detected language: {lid}\nTranscript: {res.text!r}\n\n{instruction}"},
        ]
        prompt = llm.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = llm.tok(prompt, return_tensors="pt").to(llm.cfg.device)
        with torch.no_grad():
            out = llm.model.generate(**ids, max_new_tokens=256, do_sample=False)
        answer = llm.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
        return res.text, lid + (" (repetitive decode!)" if res.degenerate else ""), answer

    gr.Interface(
        fn=run,
        inputs=[
            gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio"),
            gr.Radio(["auto", "en", "mn"], value="auto", label="Language"),
            gr.Textbox(value="Summarise what the speaker says in one sentence.", label="Instruction"),
        ],
        outputs=[gr.Textbox(label="Transcript"), gr.Textbox(label="Language ID (top-3)"), gr.Textbox(label="LLM answer", lines=6)],
        title="Speech LLM cascade (Whisper → Qwen2.5)",
        flagging_mode="never",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
