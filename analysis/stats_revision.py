"""
B1: Bootstrap 95% CIs for F1 (and accuracy) per config × dataset
B2: Rank-biserial r (effect size) for key Wilcoxon pairwise comparisons
B3: λ-similarity table per config × dataset

Outputs:
  - Console: formatted tables
  - analysis/bootstrap_ci.tex  — LaTeX table for B1+B2
  - analysis/lambda_table.tex  — LaTeX table for B3
"""

from __future__ import annotations
import json
import glob
from pathlib import Path
import numpy as np
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parents[1] / "results" / "final5"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)

DATASETS = ["mushroom", "heart", "monk_1", "monk_2", "monk_3", "breast_cancer"]
DATASET_LABELS = {
    "mushroom": "Mushroom",
    "heart": "Heart",
    "monk_1": "MONK-1",
    "monk_2": "MONK-2",
    "monk_3": "MONK-3",
    "breast_cancer": "Breast Cancer",
}
METHODS = ["lm_residual", "ste", "ste_residual", "proximal", "proximal_residual"]
METHOD_LABELS = {
    "lm_residual":       "LM-Res",
    "ste":               "STE-Flat",
    "ste_residual":      "STE-Res",
    "proximal":          "Prox-Flat",
    "proximal_residual": "Prox-Res",
}

N_BOOT = 10_000
RNG = np.random.default_rng(42)

# ── Helpers ────────────────────────────────────────────────────────────────────

def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI."""
    boots = [stat(RNG.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return stat(x), lo, hi

def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation from Wilcoxon signed-rank test."""
    diffs = x - y
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    r_plus = np.sum(ranks[diffs > 0])
    r_minus = np.sum(ranks[diffs < 0])
    n = len(diffs)
    w = max(r_plus, r_minus)
    # r = (R+ - R-) / (R+ + R-)
    r = (r_plus - r_minus) / (n * (n + 1) / 2)
    return float(r)

def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    diffs = x - y
    if np.all(diffs == 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(x, y)
    except ValueError:
        p = 1.0
    return float(p)

# ── Load data ─────────────────────────────────────────────────────────────────

def load_all() -> dict:
    """Returns data[dataset][method] = list of trial dicts."""
    data = {}
    for ds in DATASETS:
        pattern = str(RESULTS_DIR / f"{ds}_*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No final5 JSON for dataset '{ds}' in {RESULTS_DIR}")
        with open(files[-1]) as f:
            d = json.load(f)
        data[ds] = {m: d[m] for m in METHODS}
    return data

def extract(trials: list, key: str) -> np.ndarray:
    return np.array([t[key] for t in trials], dtype=float)

# ── B1 + B2: Bootstrap CI + rank-biserial r ───────────────────────────────────

def compute_b1_b2(data: dict):
    """
    For each dataset × method: compute bootstrap CI for F1 and accuracy.
    For key comparisons (STE-Flat vs STE-Res, best vs rest): compute r.
    """
    # --- per-cell stats
    ci = {}  # ci[ds][method] = {'f1': (mean, lo, hi), 'acc': ..., 'lam': ...}
    for ds in DATASETS:
        ci[ds] = {}
        for m in METHODS:
            trials = data[ds][m]
            f1_arr  = extract(trials, "f1")
            acc_arr = extract(trials, "accuracy")
            lam_arr = extract(trials, "lambda_similarity")
            ci[ds][m] = {
                "f1":  bootstrap_ci(f1_arr),
                "acc": bootstrap_ci(acc_arr),
                "lam": bootstrap_ci(lam_arr),
                "cryst": int(np.sum([t["is_crystallized"] for t in trials])),
            }

    # --- pairwise effect sizes (key comparisons)
    KEY_PAIRS = [
        ("ste", "ste_residual"),        # STE-Flat vs STE-Res
        ("proximal", "proximal_residual"),
        ("lm_residual", "ste_residual"), # LM-Res vs STE-Res (best two)
    ]
    effects = {}  # effects[ds][(m1,m2)] = {'r': float, 'p': float}
    for ds in DATASETS:
        effects[ds] = {}
        for m1, m2 in KEY_PAIRS:
            f1_m1 = extract(data[ds][m1], "f1")
            f1_m2 = extract(data[ds][m2], "f1")
            r = rank_biserial_r(f1_m2, f1_m1)   # positive = m2 better
            p = wilcoxon_p(f1_m2, f1_m1)
            effects[ds][(m1, m2)] = {"r": r, "p": p}

    return ci, effects

# ── B3: λ-similarity table ────────────────────────────────────────────────────

def compute_b3(data: dict) -> dict:
    """Per config × dataset: mean λ ± CI, fraction needing rewriting algebra."""
    lam = {}
    for ds in DATASETS:
        lam[ds] = {}
        for m in METHODS:
            trials = data[ds][m]
            lam_arr = extract(trials, "lambda_similarity")
            cryst   = np.array([t["is_crystallized"] for t in trials], dtype=bool)
            mean, lo, hi = bootstrap_ci(lam_arr)
            # fraction where formula is exact (crystallized AND lambda ~ 1)
            exact = int(np.sum(cryst & (lam_arr > 0.99)))
            lam[ds][m] = {
                "mean": mean, "lo": lo, "hi": hi,
                "std":  float(np.std(lam_arr, ddof=1)),
                "cryst": int(np.sum(cryst)),
                "exact": exact,
            }
    return lam

# ── Print console tables ───────────────────────────────────────────────────────

def print_b1(ci: dict):
    print("\n" + "="*90)
    print("B1: Bootstrap 95% CI for F1 — mean [lo, hi]")
    print("="*90)
    hdr = f"{'Dataset':<14}" + "".join(f"{METHOD_LABELS[m]:>18}" for m in METHODS)
    print(hdr)
    print("-"*90)
    for ds in DATASETS:
        row = f"{DATASET_LABELS[ds]:<14}"
        for m in METHODS:
            mn, lo, hi = ci[ds][m]["f1"]
            row += f"  {mn:.3f} [{lo:.3f},{hi:.3f}]"
        print(row)

def print_b1_acc(ci: dict):
    print("\n" + "="*90)
    print("B1: Bootstrap 95% CI for Accuracy — mean [lo, hi]")
    print("="*90)
    hdr = f"{'Dataset':<14}" + "".join(f"{METHOD_LABELS[m]:>18}" for m in METHODS)
    print(hdr)
    print("-"*90)
    for ds in DATASETS:
        row = f"{DATASET_LABELS[ds]:<14}"
        for m in METHODS:
            mn, lo, hi = ci[ds][m]["acc"]
            row += f"  {mn:.3f} [{lo:.3f},{hi:.3f}]"
        print(row)

def print_b2(effects: dict):
    print("\n" + "="*80)
    print("B2: Rank-biserial r (effect size) for key pairwise F1 comparisons")
    print("    Positive r = second method better than first")
    print("="*80)
    pairs = [("ste","ste_residual"), ("proximal","proximal_residual"), ("lm_residual","ste_residual")]
    pair_labels = ["STE-Flat→STE-Res", "Prox-Flat→Prox-Res", "LM-Res→STE-Res"]
    hdr = f"{'Dataset':<14}" + "".join(f"{pl:>22}" for pl in pair_labels)
    print(hdr)
    print("-"*80)
    for ds in DATASETS:
        row = f"{DATASET_LABELS[ds]:<14}"
        for pair in pairs:
            e = effects[ds][pair]
            sig = "*" if e["p"] < 0.05 else " "
            row += f"  r={e['r']:+.3f}  p={e['p']:.3f}{sig}  "
        print(row)
    print("\n  * p < 0.05 (Wilcoxon signed-rank)")

def print_b3(lam: dict):
    print("\n" + "="*90)
    print("B3: λ-similarity — mean ± std  [95% CI]  | cryst/10 | exact (λ>0.99)/10")
    print("="*90)
    for m in METHODS:
        print(f"\n  {METHOD_LABELS[m]}")
        print(f"  {'Dataset':<16} {'mean±std':>12}  {'95% CI':>18}  cryst  exact")
        print(f"  {'-'*65}")
        for ds in DATASETS:
            v = lam[ds][m]
            print(f"  {DATASET_LABELS[ds]:<16} {v['mean']:.4f}±{v['std']:.4f}"
                  f"  [{v['lo']:.4f}, {v['hi']:.4f}]"
                  f"    {v['cryst']}/10   {v['exact']}/10")

# ── LaTeX output ───────────────────────────────────────────────────────────────

def latex_b1b2(ci: dict, effects: dict) -> str:
    lines = []
    lines.append(r"% === Table: Bootstrap 95% CI for F1 (B1) + rank-biserial r (B2) ===")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Bootstrap 95\% confidence intervals for mean $F_1$ score")
    lines.append(r"         (10 trials per configuration) and rank-biserial $r$ effect sizes")
    lines.append(r"         for key pairwise comparisons (Wilcoxon signed-rank).}")
    lines.append(r"\label{tab:bootstrap}")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l ccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & LM-Res & STE-Flat & STE-Res & Prox-Flat & Prox-Res \\")
    lines.append(r"        & $\bar{F}_1$ [95\%~CI] & $\bar{F}_1$ [95\%~CI]"
                 r" & $\bar{F}_1$ [95\%~CI] & $\bar{F}_1$ [95\%~CI] & $\bar{F}_1$ [95\%~CI] \\")
    lines.append(r"\midrule")
    for ds in DATASETS:
        cells = []
        for m in METHODS:
            mn, lo, hi = ci[ds][m]["f1"]
            cells.append(f"{mn:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"{DATASET_LABELS[ds]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Effect sizes: rank-biserial $r$"
                 r" (positive = second method better; ${}^*p<0.05$)}} \\")
    lines.append(r"& \multicolumn{2}{c}{STE-Flat $\to$ STE-Res}"
                 r" & \multicolumn{2}{c}{Prox-Flat $\to$ Prox-Res}"
                 r" & LM-Res $\to$ STE-Res \\")
    pairs = [("ste","ste_residual"), ("proximal","proximal_residual"), ("lm_residual","ste_residual")]
    for ds in DATASETS:
        cells = []
        for i, pair in enumerate(pairs):
            e = effects[ds][pair]
            sig = r"$^*$" if e["p"] < 0.05 else ""
            cells.append(f"$r={e['r']:+.3f}${sig}")
        # fit 5 columns: empty, col1 spans 2, col2 spans 2, col3
        lines.append(f"{DATASET_LABELS[ds]} & \\multicolumn{{2}}{{c}}{{{cells[0]}}}"
                     f" & \\multicolumn{{2}}{{c}}{{{cells[1]}}}"
                     f" & {cells[2]} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def latex_b3(lam: dict) -> str:
    lines = []
    lines.append(r"% === Table: λ-similarity by configuration × dataset (B3) ===")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{$\lambda$-similarity (mean\,$\pm$\,std) and crystallization"
                 r" rates across all configurations and datasets ($n=10$ trials each)."
                 r" ``Exact'' counts trials where $\lambda > 0.99$,"
                 r" i.e.\ the extracted formula satisfies Proposition~1 without approximation.}")
    lines.append(r"\label{tab:lambda}")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{l ccc ccc ccc ccc ccc}")
    lines.append(r"\toprule")
    # Header: 5 methods × (λ, cryst, exact)
    lines.append(r"& \multicolumn{3}{c}{LM-Res}"
                 r" & \multicolumn{3}{c}{STE-Flat}"
                 r" & \multicolumn{3}{c}{STE-Res}"
                 r" & \multicolumn{3}{c}{Prox-Flat}"
                 r" & \multicolumn{3}{c}{Prox-Res} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}"
                 r"\cmidrule(lr){11-13}\cmidrule(lr){14-16}")
    lines.append(r"Dataset"
                 + r" & $\lambda$ & C & E" * 5 + r" \\")
    lines.append(r"\midrule")
    for ds in DATASETS:
        cells = []
        for m in METHODS:
            v = lam[ds][m]
            cells.append(f"{v['mean']:.3f}$\\pm${v['std']:.3f} & {v['cryst']}/10 & {v['exact']}/10")
        lines.append(f"{DATASET_LABELS[ds]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{16}{l}{\small C\,=\,crystallized/10;\;"
                 r"E\,=\,exact ($\lambda>0.99$)/10} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading final5 data...")
    data = load_all()
    print(f"  Loaded {len(DATASETS)} datasets × {len(METHODS)} configs × 10 trials\n")

    ci, effects = compute_b1_b2(data)
    lam = compute_b3(data)

    print_b1(ci)
    print_b1_acc(ci)
    print_b2(effects)
    print_b3(lam)

    tex_b1b2 = latex_b1b2(ci, effects)
    tex_b3   = latex_b3(lam)

    out1 = OUT_DIR / "bootstrap_ci.tex"
    out2 = OUT_DIR / "lambda_table.tex"
    out1.write_text(tex_b1b2)
    out2.write_text(tex_b3)

    print(f"\n  LaTeX tables written to:")
    print(f"    {out1}")
    print(f"    {out2}")

if __name__ == "__main__":
    main()
