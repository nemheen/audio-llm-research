# References and what we took from each

**Whisper**: Radford, Kim, Xu, Brockman, McLeavey, Sutskever. *Robust Speech Recognition via
Large-Scale Weak Supervision.* arXiv:2212.04356, 2022.
- Sec. 4.5 decoding heuristics: temperature fallback (0.0 → 1.0 in 0.2 steps) when the gzip
  compression ratio exceeds 2.4 or the average log-probability falls below −1. v0's main
  failure mode (repetition loops on Swahili/Welsh mis-identified audio) is exactly what
  these heuristics were designed for. Used in `cascade.WhisperASR` and `metrics.is_degenerate`.
- Text normalisation before WER (App. C). Used in `text.normalize`.
- Language-ID head: the first decoder step is a distribution over language tokens. We
  expose it as a calibrated feature for the LLM rather than discarding it.

**Qwen-Audio**: Chu, Xu, Zhou, Yang, Zhang, Yan, Zhou, Zhou. *Qwen-Audio: Advancing Universal
Audio Understanding via Unified Large-Scale Audio-Language Models.* arXiv:2311.07919, 2023.
- Whisper-encoder initialisation for the audio side of an audio-LLM.
- *Hierarchical tags* (task, language, timestamps …) as the decoder condition, which reduces
  interference when one model is co-trained on many tasks with differently-shaped labels.
  We use a plain-text version (`[task: transcribe] [lang: mn]`) so a frozen LLM can read it
  without new vocabulary.

**DiVA**: Held, Li, Ryan, Shi, Zhang, Yang. *Distilling an End-to-End Voice Assistant Without
Instruction Training Data.* arXiv:2410.02678, 2024.
- Self-distillation: the speech model is trained to reproduce the frozen text LLM's response to
  the transcript. This removes the need for spoken instruction data, which does not exist
  for Mongolian. Implemented in `scripts/make_distill_targets.py` and the `instruct` task.

**Ultravox**: fixie-ai, github.com/fixie-ai/ultravox.
- Freeze both the audio encoder and the LLM and train only the projector in stage 1.
- Frame-stacking projector that reduces the 50 Hz encoder output to a few audio tokens per
  second before a small MLP.

**Common Voice**: Ardila et al. *Common Voice: A Massively-Multilingual Speech Corpus.* LREC 2020.
Release 22.0 (delta 2025-06-20) for `en` and `mn`.
