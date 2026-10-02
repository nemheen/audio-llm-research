# Data

Nothing under `data/raw/` or `data/processed/` is committed. Everything is rebuilt from
public Mozilla Common Voice releases ([commonvoice.mozilla.org/datasets](https://commonvoice.mozilla.org/en/datasets), CC0).

## What is used where

| Use | Release | Locale | File(s) | Size |
|---|---|---|---|---|
| English cascade eval (v0, v1) | CV 22.0 **delta** (2025-06-20) | `en` | `other.tsv` ⋈ `reported.tsv`, `clip_durations.tsv`, 107 clips | 1.1 GB archive |
| Mongolian zero-shot ASR | CV 22.0 **delta** (2025-06-20) | `mn` | `other.tsv` (27 clips, 3.3 min) | 1.6 MB archive |
| Adapter training (Phase 3) | CV 22.0 **full** | `mn`, `en` | `train.tsv`, `dev.tsv`, `test.tsv` | see Mozilla site |

The delta release has no `train/dev/test` split and only 27 Mongolian clips, all from
`other.tsv`, so their transcripts have **not yet been validated** by the community. That is
fine for a first zero-shot read, but it is not a benchmark. Phase 3 numbers must come from
the full release's `test.tsv`.

## Layout

```
data/raw/
  cv-corpus-22.0-delta-2025-06-20/
    en/{clips/, other.tsv, reported.tsv, clip_durations.tsv}
    mn/{clips/, other.tsv, validated_sentences.tsv, ...}
  cv-corpus-22.0-2025-06-20/          # full release, Phase 3 only
    mn/{clips/, train.tsv, dev.tsv, test.tsv, validated.tsv, clip_durations.tsv}
data/processed/
  distill_{mn,en}_train.jsonl         # scripts/make_distill_targets.py
```

## Extracting

The English delta is large. Only the 107 reported clips are needed for the cascade
evaluation, so extract those selectively:

```bash
mkdir -p data/raw
# Mongolian delta: small, extract everything
tar -xzf cv-corpus-22.0-delta-2025-06-20-mn.tar.gz -C data/raw

# English delta: TSVs first, then only the clips whose sentence was reported
P=cv-corpus-22.0-delta-2025-06-20/en
tar -xzf cv-corpus-22.0-delta-2025-06-20-en.tar.gz -C data/raw $P/other.tsv $P/reported.tsv $P/clip_durations.tsv
python scripts/list_reported_clips.py data/raw > /tmp/clips.txt
tar -xzf cv-corpus-22.0-delta-2025-06-20-en.tar.gz -C data/raw -T /tmp/clips.txt
```

On Windows with Git-Bash's `tar`, add `--force-local` (otherwise `C:` is read as a remote
host) and make sure the list file has LF line endings.

## Known properties of the reported-sentence set

* `reported.tsv` is about the **prompt sentence**, not the recording. A "different-language"
  report means the *text* was not English. 38 of the 107 prompts are Swahili.
* One sentence can be reported several times with different reasons, so a clip carries a
  *set* of labels (`labels`) plus a primary label (`label`, the most frequent one).
* Reasons are free text. They are folded into 5 classes by `speechllm.data.canonical_reason`.
