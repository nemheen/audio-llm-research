"""Generate notebooks/01_results_analysis.ipynb from the files in results/.

    python scripts/build_analysis_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_results_analysis.ipynb
"""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Results and error analysis

All numbers are computed from `results/*/predictions.csv` with `speechllm.evaluate`, which is the
same code path the scripts use. Narrative and interpretation: [`docs/REPORT.md`](../docs/REPORT.md)."""),
    code("""import ast, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from speechllm.metrics import wer, utterance_wer, is_degenerate
from speechllm.data import REASON_LABELS

R = ROOT / "results"
FIG = R / "figures"; FIG.mkdir(exist_ok=True)

# Reference categorical slots (blue, orange, aqua) + recessive ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

def load(run):
    p = R / run
    if not (p / "metrics.json").exists():
        return None, None
    return json.loads((p / "metrics.json").read_text()), pd.read_csv(p / "predictions.csv")"""),
    md("## 1. ASR: from v0 to guarded decoding (English reported-sentence set, 107 clips)"),
    code("""RUNS = {
    "v0 (Colab, as run)": "v0_cascade",
    "v0 decoding, re-run": "ablations/asr_v0_decoding",
    "+ guard rails": "ablations/asr_guarded",
    "+ guard, forced en": "ablations/asr_guarded_force_en",
}
rows, preds = [], {}
for name, run in RUNS.items():
    m, p = load(run)
    if m is None:
        continue
    a = m["asr"]; preds[name] = p
    sw = p[~p["is_english"]]
    rows.append({
        "run": name, "WER": a["wer"], "95% CI": f"{a['wer_ci95'][0]:.2f}-{a['wer_ci95'][1]:.2f}",
        "WER en prompts (n=69)": a["wer_english_prompts"],
        "WER sw prompts (n=38)": wer(sw["sentence"].tolist(), sw["hyp"].fillna("").tolist()),
        "CER": a["cer"], "degenerate %": 100 * a["degenerate_rate"],
    })
asr_table = pd.DataFrame(rows).set_index("run")
asr_table.style.format(precision=3)"""),
    md("""The **as-printed** v0 number was `1.3936`, computed on 148 rows with duplicated clips and
no normalisation. It is a *fraction*, so it means 139 % WER. The table above is on the 107 unique clips."""),
    code("""m0, _ = load("v0_cascade")
m0["as_reported"]"""),
    md("### Where the errors are: per-utterance WER, English vs Swahili prompts"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
order = [k for k in ["v0 decoding, re-run", "+ guard rails", "+ guard, forced en"] if k in preds] or list(preds)
colors = [C1, C2, C3]
for ax, (subset, title) in zip(axes, [(True, "English prompts"), (False, "Swahili prompts")]):
    for i, name in enumerate(order):
        p = preds[name][preds[name]["is_english"] == subset]
        u = np.array([utterance_wer(r, h if isinstance(h, str) else "") for r, h in zip(p["sentence"], p["hyp"])])
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(u))
        ax.scatter(np.full(len(u), i) + jitter, np.clip(u, 0, 5), s=14, color=colors[i], alpha=0.75, linewidths=0)
        ax.hlines(np.median(u), i - 0.3, i + 0.3, color=INK, linewidth=2)
    ax.set_xticks(range(len(order)), order, rotation=0, fontsize=8.5)
    ax.set_title(f"{title} (n={len(p)})", fontsize=10, color=INK, loc="left")
axes[0].set_ylabel("utterance WER (clipped at 5)")
fig.suptitle("Per-utterance WER; black bar = median", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(FIG / "asr_per_utterance_wer.png", bbox_inches="tight")"""),
    md("### Whisper's language ID already separates the two populations"),
    code("""_, g = load("ablations/asr_guarded")
if g is not None:
    g["p_en"] = [dict(map(tuple, ast.literal_eval(t))).get("en", 0.0) for t in g["lang_top3"]]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for k, (subset, lab, col) in enumerate([(True, "English prompt", C1), (False, "Swahili prompt", C2)]):
        v = g.loc[g["is_english"] == subset, "p_en"]
        ax.scatter(v, np.full(len(v), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), s=16, color=col, alpha=0.8, linewidths=0)
    ax.set_yticks([0, 1], ["English prompt", "Swahili prompt"]); ax.set_xlabel("Whisper P(language = en)")
    ax.set_xlim(-0.02, 1.02); ax.grid(axis="y", visible=False)
    ax.set_title("Language-ID posterior per clip", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "lid_posterior.png", bbox_inches="tight")
    display(pd.crosstab(g["is_english"], g["lang"]))"""),
    md("## 2. Reasoning stage: report-reason classification"),
    code("""RRUNS = {"v0 Gemma-2B (free text, keyword-scored)": "v0_cascade",
         "v1 Qwen2.5-1.5B, no LID": "ablations/v1_no_lid",
         "v1 Qwen2.5-1.5B + LID": "v1_cascade_en"}
rows = []
for name, run in RRUNS.items():
    m, p = load(run)
    if m is None or "reasoning" not in m:
        continue
    r = m["reasoning"]
    rows.append({"run": name, "accuracy": r["accuracy"], "any-match acc.": r["any_match_accuracy"], "macro-F1": r["macro_f1"],
                 **{f"F1 {k}": v for k, v in r["per_class_f1"].items()}})
# Trivial references
_, p0 = load("v0_cascade")
maj = (p0["label"] == "different-language").mean()
_, g = load("ablations/asr_guarded")
if g is not None:
    rule = np.where(g["lang"] != "en", "different-language", "grammar-or-spelling")
    rows.append({"run": "rule: LID!=en -> different-language else grammar", "accuracy": (rule == g["label"]).mean()})
rows.append({"run": "majority class", "accuracy": maj})
pd.DataFrame(rows).set_index("run").style.format(precision=3, na_rep="-")"""),
    code("""m, p = load("v1_cascade_en")
if p is not None:
    cm = pd.crosstab(p["label"], p["pred_label"]).reindex(index=REASON_LABELS, columns=REASON_LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(cm.values, cmap="Blues"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm.values):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.values.max() / 2 else INK, fontsize=9)
    short = [l.split("-")[0] for l in REASON_LABELS]
    ax.set_xticks(range(5), short, rotation=30); ax.set_yticks(range(5), short)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold (primary reason)")
    ax.set_title("v1 confusion matrix", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "v1_confusion.png", bbox_inches="tight")"""),
    md("""Labels other than *different-language* describe the **text** of the prompt
(spelling, offensiveness). Audio carries little evidence for them, and the transcript is a
noisy copy of a text we could simply read. They are kept as a stress test of the reasoning
stage, but the realistic ceiling for an audio-only system is set by the language-ID part."""),
    md("## 3. Mongolian zero-shot ASR (27 unvalidated clips, CV-22 delta)"),
    code("""rows = []
for run in sorted((R / "mn_zeroshot").glob("*")):
    m, p = load(f"mn_zeroshot/{run.name}")
    if m is None:
        continue
    rows.append({"model": run.name, "WER": m["asr"]["wer"], "CER": m["asr"]["cer"], "degenerate %": 100 * m["asr"]["degenerate_rate"],
                 "WER 95% CI": f"{m['asr']['wer_ci95'][0]:.2f}-{m['asr']['wer_ci95'][1]:.2f}"})
display(pd.DataFrame(rows).set_index("model").style.format(precision=3))
_, p = load("mn_zeroshot/whisper-small")
if p is not None:
    pd.set_option("display.max_colwidth", 90)
    display(p[["sentence", "hyp"]].head(6))"""),
    md("## 4. Phase 3 adapter: CPU smoke test\n\nWhisper-tiny encoder + Qwen2.5-0.5B, projector only, 27 Mongolian clips. This checks that the plumbing works (the loss goes down). It is **not** a quality result."),
    code("""log = ROOT / "checkpoints/adapter_smoke/train_log.jsonl"
src = log if log.exists() else R / "adapter_smoke/train_log.jsonl"
if src.exists():
    L = pd.read_json(src, lines=True)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(L["step"], L["loss"], color=C1, linewidth=1, alpha=0.4)
    ax.plot(L["step"], L["loss"].rolling(5, min_periods=1).mean(), color=C1, linewidth=2)
    ax.set_xlabel("step"); ax.set_ylabel("train loss (target tokens)")
    ax.set_title("Projector-only training, smoke test", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "adapter_smoke_loss.png", bbox_inches="tight")
    print(L.iloc[[0, -1]])"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/01_results_analysis.ipynb")
print("wrote notebooks/01_results_analysis.ipynb")
