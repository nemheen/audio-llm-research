# Speech-aware LLMs for English and Mongolian

Getting an LLM to work directly on speech: transcribe it, identify the language, and
follow instructions about it. The target is Mongolian, a low-resource language that
general-purpose ASR handles poorly.

The project ran in three phases:

| Phase | What | Status |
|---|---|---|
| **v0: cascade baseline** | Whisper-small → Gemma-2B-it on Common Voice clips whose prompt sentence was reported by the community | done (Colab), re-scored |
| **v1: fixed cascade** | failure analysis of v0 → Whisper guard rails, language-ID evidence, chat-templated LLM with a closed label set | done (CPU), results below |
| **Phase 3: end-to-end audio adapter** | Whisper encoder → frame-stacking projector → frozen LLM, multi-task with Qwen-Audio-style task tags and DiVA/Ultravox-style distillation; English + Mongolian | implemented and smoke-tested; full training pending a GPU |

The full write-up, with the failure analysis, design decisions and paper notes, is in
**[docs/REPORT.md](docs/REPORT.md)**. The figures are in
[notebooks/01_results_analysis.ipynb](notebooks/01_results_analysis.ipynb).

## Headline results

RESULTS_TABLE

## Repository layout

```
src/speechllm/
  text.py          normalisation for WER/CER (Latin + Cyrillic)
  metrics.py       WER/CER, bootstrap CIs, Whisper compression-ratio loop detector, classification
  data.py          Common Voice loading, de-duplicated "reported" eval set, label canonicalisation
  cascade.py       WhisperASR (LID posterior, temperature fallback) + LLMReasoner (JSON labels)
  evaluate.py      one scoring path for every pipeline version
  adapter/         end-to-end model (encoder → projector → LLM) and multi-task data
scripts/
  score_v0.py              re-score the archived v0 outputs
  run_cascade.py           run + score any cascade config
  train_adapter.py         stage-1 projector training
  eval_adapter.py          ASR eval of a trained adapter
  make_distill_targets.py  instruction targets by self-distillation from the text LLM
  list_reported_clips.py   selective extraction of the 107 English clips
configs/           every run in results/ has a config here
results/           metrics.json + predictions.csv per run (small, committed)
notebooks/
  00_cascade_v0_colab.ipynb    original v0 run (with erratum)
  01_results_analysis.ipynb    figures and tables for the report
app/demo.py        Gradio demo (cascade, free-form instruction)
tests/             CPU-only unit tests
```

## Reproducing

```bash
pip install -e ".[analysis,dev]"
# data: see data/README.md (Common Voice 22.0 delta, en + mn)

python scripts/score_v0.py                                      # v0 re-scored
python scripts/run_cascade.py --config configs/asr_en_v0_reproduce.yaml
python scripts/run_cascade.py --config configs/asr_en_guarded.yaml
python scripts/run_cascade.py --config configs/asr_en_force_en.yaml
python scripts/run_cascade.py --config configs/cascade_v1_en.yaml
python scripts/run_cascade.py --config configs/cascade_v1_en.yaml --no-lid --out results/ablations/v1_no_lid
python scripts/run_cascade.py --config configs/asr_mn_zeroshot.yaml
python scripts/train_adapter.py --config configs/adapter_smoke.yaml  # CPU, ~30 min
pytest
```

The v1 numbers above were produced on CPU (no GPU), with greedy decoding and fixed seeds.
ASR outputs are cached under `results/.asr_cache/`, so LLM ablations don't re-run Whisper.

## References

See [docs/references.md](docs/references.md). Main influences: Whisper (Radford et al.,
2022) for the decoding guard rails and normalised WER; Qwen-Audio (Chu et al., 2023) for
the encoder choice and hierarchical task tags; Ultravox and DiVA (Held et al., 2024) for
the projector design and the distillation recipe.
