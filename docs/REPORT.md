# Speech-aware LLMs for English and Mongolian: technical report

## 1. Problem

We want an LLM that works on **speech** directly: it should transcribe, know which
language is being spoken, and answer instructions about the audio ("what is this about?",
"is this the expected language?"). Mongolian is the target language. It is poorly served by
general-purpose ASR, and spoken instruction data for it essentially doesn't exist.

There are two ways to build this:

* **Cascade**: ASR → text → LLM. Simple and modular, but everything the ASR drops (language
  identity, confidence, prosody) is invisible to the LLM, and ASR errors propagate.
* **End-to-end adapter**: feed audio-encoder states to the LLM through a learned projector
  (Qwen-Audio, Ultravox, DiVA). The LLM sees richer evidence, but this needs training data
  and a GPU.

We started with the cascade to get a baseline and a test bed quickly (v0). We then analysed
its failures (§3), fixed what could be fixed inside the cascade (v1, §4), and designed the
adapter around what the cascade could not fix (Phase 3, §6).

## 2. Data and task

**Evaluation set: reported sentences (English).** Common Voice volunteers can report a
*prompt sentence* as wrong-language, misspelled, offensive or hard to pronounce. We took all
clips in the CV-22.0 delta (`en/other.tsv`) whose sentence appears in `reported.tsv`. This
set is deliberately adversarial for ASR: 38 of the 107 prompts are **Swahili** sentences
that leaked into the English sentence pool, read by (mostly) non-Swahili speakers.

The downstream task is predicting why the sentence was reported. This gives a closed label
set we can score, unlike open-ended "reasoning":

| label | clips (primary) |
|---|---|
| different-language | 60 |
| grammar-or-spelling | 28 |
| offensive-language | 9 |
| difficult-pronunciation | 6 |
| other (free-text reasons) | 4 |

**Mongolian.** The CV-22.0 delta has 27 `mn` clips (3.3 min, `other.tsv`, not yet
validated). It is too small to train on but enough for a first zero-shot read. Training
(Phase 3) uses the full CV `mn` release.

## 3. v0: cascade baseline and what went wrong

v0 ([notebook](../notebooks/00_cascade_v0_colab.ipynb)): `whisper-small` through the HF
`pipeline` with default decoding, then `gemma-2b-it` (4-bit) given the transcript and a
"topic / reason / 3 facts" instruction, sampled at T=0.7. It was scored with WER and ROUGE-L
against the report reason.

The v0 write-up claimed *"WER 1.39 %, excellent ASR; ROUGE-L 0.0051"*. Re-scoring the archived
outputs (`scripts/score_v0.py`) shows that summary was wrong:

| | as written in v0 | re-scored |
|---|---|---|
| rows / unique clips | 148 | **107** (the `other ⋈ reported` join duplicated clips reported more than once) |
| WER | "1.39 %" | **{{v0_wer}}** on 107 clips (the 1.3936 printed by `evaluate` is a fraction, i.e. 139 %) |
| WER, English prompts only | | {{v0_wer_en}} |
| WER excluding degenerate decodes | | {{v0_wer_nodeg}} |
| clips with degenerate (looping) decodes | | {{v0_deg}} |
| LLM stage | ROUGE-L 0.0051 / 0.0287 (the report and the cell output disagree) | accuracy **{{v0_acc}}**, macro-F1 {{v0_f1}} |

### Failure analysis

1. **Hallucination loops dominate the WER.** A handful of clips produce 148-token loops:
   `ḍ ḍ ḍ …` for Swahili speech, and `Mae'r ffordd o'r ffordd …` (Welsh) for an English
   sentence full of brand names. One such clip scores a WER of 21 on its own. These are exactly the
   failures that Whisper's paper (§4.5) handles with temperature fallback on compression
   ratio > 2.4. The HF pipeline defaults don't enable it.
2. **Language identity was thrown away.** For the most frequent label, "different-language",
   the strongest evidence is Whisper's own language-ID head. The pipeline used it internally
   and then discarded it, so the LLM had to guess the language from a garbled transcript.
3. **The LLM was never really prompted.** `gemma-2b-it` was called on a raw string with no
   chat template, so it continued the instruction scaffold
   (`Additional Information: <contextual details…>`) instead of answering. Sampling at T=0.7
   made runs non-reproducible.
4. **The metric couldn't measure the task.** ROUGE-L between a paragraph and a one-word
   label is near zero whatever the quality. We replaced it with accuracy / macro-F1 over the
   closed label set, plus any-match accuracy for clips with several reasons.
5. **Evaluation hygiene.** Duplicated rows over-weighted the worst clips (the Welsh-loop
   clip appears 5 times). There were no confidence intervals on a set of about 100 utterances.

## 4. v1: fixing the cascade

Every change targets one failure above (`src/speechllm/cascade.py`):

| Change | Fixes | Inspired by |
|---|---|---|
| Temperature fallback (0 → 1.0), compression-ratio 2.4 / log-prob −1 thresholds, duration-aware `max_new_tokens` | 1 | Whisper §4.5 |
| Expose the language-ID posterior (top-3) and a "repetitive decode" flag to the LLM | 2 | Qwen-Audio's explicit language tags |
| Chat template, greedy decoding, closed label set returned as JSON | 3 | — |
| Accuracy / macro-F1 / any-match; bootstrap CIs; one row per clip | 4, 5 | — |
| LLM: `Qwen2.5-1.5B-Instruct` (ungated, chat-tuned, runs on CPU) | 3 | — |

The ASR model is unchanged (`whisper-small`), so the ASR deltas come from decoding alone.

### Results: ASR (English reported set, 107 clips, normalised WER)

{{asr_table}}

### Results: report-reason classification (107 clips)

{{cls_table}}

### Reading the results

{{v1_discussion}}

## 5. Mongolian: zero-shot ASR

{{mn_table}}

{{mn_discussion}}

## 6. Phase 3: end-to-end audio adapter

The cascade analysis points to what an end-to-end model should keep: **language identity**
and **confidence** live in the encoder states, and the transcript bottleneck throws them
away. For Mongolian, the ASR front-end is also the weak link (§5), so improving it inside the
cascade would mean fine-tuning Whisper itself. An adapter lets the LLM's own Mongolian
knowledge help.

### Architecture (`src/speechllm/adapter/model.py`)

```
audio 16 kHz ─► Whisper encoder (frozen) ─► 50 Hz states ─► stack k=4 frames ─► LN ─► MLP ─► LN ─► 12.5 Hz "audio tokens"
                                                                                           │
chat prompt:  <system> … <user> [AUDIO TOKENS] [task: transcribe] [lang: mn] Transcribe … ─►  frozen LLM (Qwen2.5) ─► text
```

* **Encoder: Whisper.** As in Qwen-Audio, which initialises from Whisper-large-v2. The encoder
  has seen Mongolian during Whisper pre-training, even if the decoder is weak on it.
* **Projector: frame stacking + 2-layer MLP**, the only trainable part in stage 1 (about 10M
  parameters for whisper-small → Qwen2.5-1.5B). This follows the Ultravox recipe: freeze
  both ends, train the bridge.
* **Audio is spliced into the chat template** at an `<AUDIO>` marker as `inputs_embeds`,
  so no new tokens or embedding resizing are needed. Padding frames from Whisper's fixed 30 s
  window are dropped (one audio token per 80 ms of real audio).
* **Task tags** (`[task: …] [lang: …]`) are Qwen-Audio's hierarchical-tag idea in plain
  text. One model can do ASR, language ID, and joint language-ID + ASR without one task's
  label format leaking into another. The language tag is withheld when the language *is*
  the answer.

### Training data recipe (`src/speechllm/adapter/data.py`, `configs/adapter_mn.yaml`)

| task | target | weight |
|---|---|---|
| transcribe | reference sentence | 0.60 |
| lid + asr | `lang: mn\ntext: …` | 0.15 |
| lid | language code | 0.05 |
| instruct | the text LLM's own answer to (transcript, instruction) | 0.20 |

The **instruct** task is DiVA's self-distillation (`scripts/make_distill_targets.py`). We ask the
frozen text LLM to summarise, translate, find the topic or list entities *from the reference
transcript*, and train the adapter to produce the same answer *from audio*. No human-written
Mongolian spoken-QA data is needed, which is what makes instruction following on Mongolian
audio reachable at all.

### Status

* Implemented and unit-tested (`tests/test_adapter_shapes.py`).
* **CPU smoke test** (`configs/adapter_smoke.yaml`): whisper-tiny + Qwen2.5-0.5B, projector
  only, 27 Mongolian clips, 60 steps. {{smoke}} This shows the gradient path through the
  spliced embeddings works. With 27 clips it is memorisation, **not** a quality result.
* **Not yet run**: the full stage-1 training on CV-mn + CV-en (needs a single 24 GB GPU; about
  10k steps at batch 32) and evaluation on CV-mn `test.tsv` against the Whisper zero-shot
  numbers in §5.

## 7. Limitations

* The English evaluation set is small (107 clips) and unusual by construction, which is why
  bootstrap CIs are reported. The Mongolian set (27 clips) is unvalidated and only indicative.
* Four of the five labels describe the **text** of the prompt, not the audio. An audio
  system can only infer them through a transcript, and a perfect transcript would make the
  task text classification. They are kept as a test of the reasoning stage, not as a claim
  about what is audible.
* `looks_english` (used for the English/Swahili split) is a stop-word heuristic. It was
  checked by eye on all 38 Swahili prompts.
* v1 was run on CPU with float32. Numbers may move slightly on GPU in half precision.

## 8. Next steps

1. Run Phase 3 stage 1 on GPU; report CV-mn `test` WER/CER against Whisper-small and
   Whisper-large-v3-turbo zero-shot.
2. Stage 2: LoRA on the LLM with the same task mix, and compare against fine-tuning Whisper
   itself on CV-mn (the strongest cascade baseline for Mongolian ASR).
3. Speaker-aware tasks: Common Voice `client_id` gives speaker identity for free. Concatenating
   clips from two speakers yields "how many speakers?" / "who said X?" training data,
   following Qwen-Audio's multi-task framing.
4. A proper spoken-instruction eval for Mongolian: a small human-written set of questions about
   CV-mn test clips, scored by exact match or an LLM judge.
