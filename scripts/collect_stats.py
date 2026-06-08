#!/usr/bin/env python3
"""
collect_stats.py — headless execution of all three benchmark suites.

Mirrors the three Jupyter notebooks without interactive display.
Saves per-trial CSV + learning-curve PNGs to results/{dataset}/.

Usage:
    python scripts/collect_stats.py            # 10 trials per method
    python scripts/collect_stats.py --quick    # 3 trials, 4000 iters (fast)
    python scripts/collect_stats.py --dataset heart
"""

import sys, os, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from luknn.benchmark.datasets import load_truth_table, load_mushroom, load_heart_disease
from luknn.layers.lukasiewicz_linear import LukasiewiczNet
from luknn.optimizers import LMOptimizer, STEOptimizer, ProximalOptimizer
from luknn.benchmark.metrics import (
    compute_accuracy, compute_f1, compute_lambda_similarity, compute_delta_n,
)
from luknn.extraction.extractor import extract_formula

try:
    from sklearn.metrics import confusion_matrix as _cm
    def _sens_spec(pred_np, y_np):
        p = (pred_np >= 0.5).astype(int)
        t = y_np.round().astype(int)
        unique = set(t.tolist())
        if not unique.issubset({0, 1}):
            return float('nan'), float('nan')
        try:
            tn, fp, fn, tp = _cm(t, p, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return round(float(sens), 4), round(float(spec), 4)
        except Exception:
            return float('nan'), float('nan')
except ImportError:
    def _sens_spec(pred_np, y_np):
        return float('nan'), float('nan')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
parser.add_argument('--dataset', choices=['all', 'truth', 'mushroom', 'heart'],
                    default='all')
args = parser.parse_args()

QUICK    = args.quick
N_TRIALS = 3  if QUICK else 10
BASE_SEED = 42
RES_BASE  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

MAX_ITERS = {
    # (method, dataset): max_iterations
    ('LM',      'truth'):    400,
    ('STE',     'truth'):    4000 if QUICK else 8000,
    ('Proximal','truth'):    4000 if QUICK else 8000,
    ('LM',      'mushroom'): 400,
    ('STE',     'mushroom'): 5000 if QUICK else 15000,
    ('Proximal','mushroom'): 5000 if QUICK else 12000,
    ('LM',      'heart'):    400,
    ('STE',     'heart'):    5000 if QUICK else 10000,
    ('Proximal','heart'):    5000 if QUICK else 10000,
}
BATCH_SIZE = {'truth': 0, 'mushroom': 512, 'heart': 0}
TOL        = {'truth': 2e-3, 'mushroom': 2e-3, 'heart': 5e-2}
ARCH       = {'truth': [4, 4], 'mushroom': [16, 8], 'heart': [6, 4]}
OPTIMIZER_MODE = {'LM': 'continuous', 'STE': 'ste', 'Proximal': 'clamp'}

# ── Optimizer factory ─────────────────────────────────────────────────────────
def make_opt(method, model, ds_key):
    bs = BATCH_SIZE[ds_key]
    if method == 'LM':
        return LMOptimizer(model, mu_init=0.01, patience=50,
                           crystallize_n=2, prune=True, batch_size=bs)
    elif method == 'STE':
        return STEOptimizer(model, lr=0.005, clip_grad=1.0)
    else:
        return ProximalOptimizer(model, lr=0.008, lambda_sparse=0.002,
                                 lambda_attract=0.08, prox_threshold=3e-4,
                                 phase1_fraction=0.65)

# ── Single trial ──────────────────────────────────────────────────────────────
def run_trial(method, ds, ds_key, trial, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    mode  = OPTIMIZER_MODE[method]
    model = LukasiewiczNet(ds.n_features, ARCH[ds_key], mode=mode)
    opt   = make_opt(method, model, ds_key)
    mi    = MAX_ITERS[(method, ds_key)]
    tol   = TOL[ds_key]

    t0 = time.perf_counter()
    result = opt.train(ds.X_train, ds.y_train,
                       tol_mse=tol, max_iter=mi, verbose=False)
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        pred = model(ds.X_test)

    pred_np = pred.numpy()
    y_np    = ds.y_test.numpy()

    acc  = compute_accuracy(pred, ds.y_test)
    f1   = compute_f1(pred, ds.y_test)
    lam  = compute_lambda_similarity(model, ds.X_test, ds.y_test)
    dn   = compute_delta_n(model)
    crys = dn < 1e-3
    sens, spec = _sens_spec(pred_np, y_np)

    # Active input features (layer 0)
    W0 = model.weight_matrix_repr()[0][0]
    active_mask = (W0.abs() > 0.5).any(dim=0).cpu().numpy()
    fn = (ds.feature_names or [f'x{i}' for i in range(ds.n_features)])
    fn = fn[:ds.n_features]
    active_names = [fn[i] for i in range(len(fn)) if i < len(active_mask) and active_mask[i]]

    formula_str = None
    if crys:
        try:
            er = extract_formula(model, fn, n_values=3)
            formula_str = er.formula
        except Exception as e:
            formula_str = f'[err:{e}]'

    return {
        'method': method, 'trial': trial, 'seed': seed,
        'mse': round(result.final_mse, 6),
        'accuracy': round(acc, 4),
        'f1': round(f1, 4) if f1 == f1 else float('nan'),
        'sensitivity': sens, 'specificity': spec,
        'lambda': round(lam, 4),
        'delta_n': round(dn, 6),
        'crystallized': crys,
        'converged': result.converged,
        'iterations': result.iterations,
        'active_features': len(active_names),
        'active_feature_names': '|'.join(active_names[:20]),
        'time_s': round(elapsed, 2),
        'extracted_formula': formula_str,
        'mse_history': result.mse_history,
    }

# ── Output helpers ────────────────────────────────────────────────────────────
def separator(title, width=70):
    print(f'\n{"═"*width}\n  {title}\n{"═"*width}')

def save_csv(rows, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'mse_history'}
                        for r in rows])
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    print(f'  → {path}')
    return df

def save_curves(records, out_dir, fname, tol):
    os.makedirs(out_dir, exist_ok=True)
    colors = {'LM': 'steelblue', 'STE': 'darkorange', 'Proximal': 'seagreen'}
    methods = ['LM', 'STE', 'Proximal']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, method in zip(axes, methods):
        for r in records:
            if r['method'] != method:
                continue
            hist = r.get('mse_history', [])
            if not hist:
                continue
            ls = '-' if r['crystallized'] else '--'
            ax.semilogy(hist, color=colors[method], alpha=0.55,
                        linewidth=0.7, linestyle=ls)
        ax.axhline(tol, color='red', linestyle=':', linewidth=1.2,
                   label=f'tol={tol}')
        ax.set_title(method)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Truth Tables
# ══════════════════════════════════════════════════════════════════════════════
if args.dataset in ('all', 'truth'):
    separator('EXPERIMENT 1 — Truth Table Reconstruction (f1, f2, 3-valued)')
    FORMULAS = ['f1', 'f2']
    OUT      = os.path.join(RES_BASE, 'truth_tables')
    all_rows = []
    all_recs = []

    for formula in FORMULAS:
        for method in ['LM', 'STE', 'Proximal']:
            successes = 0
            t_start = time.time()
            for trial in range(N_TRIALS):
                seed = BASE_SEED + FORMULAS.index(formula) * 100 + trial * 1000
                ds = load_truth_table(formula=formula, n_values=3, seed=seed)
                rec = run_trial(method, ds, 'truth', trial, seed)
                rec['formula'] = formula
                all_recs.append(rec)
                all_rows.append({k: v for k, v in rec.items() if k != 'mse_history'})
                m = '✓' if rec['crystallized'] else '✗'
                print(f'  {formula}/{method} t{trial}: '
                      f'mse={rec["mse"]:.5f}  f1={rec["f1"]:.3f}  '
                      f'crys={m}  Δ={rec["delta_n"]:.5f}  {rec["time_s"]:.1f}s')
                if rec['crystallized']:
                    successes += 1
            print(f'  └─ {formula}/{method}: {successes}/{N_TRIALS} crystallized'
                  f'  ({time.time()-t_start:.1f}s)')

    df_tt = save_csv(all_rows, OUT, 'all_trials.csv')
    save_curves(all_recs, OUT, 'learning_curves.png', TOL['truth'])

    separator('Truth Table — Aggregate')
    agg = df_tt.groupby(['formula', 'method']).agg(
        mse_mean=('mse', 'mean'),    mse_std=('mse', 'std'),
        f1_mean=('f1', 'mean'),      f1_max=('f1', 'max'),
        lam_mean=('lambda', 'mean'),
        crys_rate=('crystallized', 'mean'),
        iter_mean=('iterations', 'mean'),
        time_mean=('time_s', 'mean'),
    ).round(4)
    print(agg.to_string())

    print('\n  Extracted formulas:')
    for r in all_recs:
        if r['crystallized'] and r['extracted_formula']:
            print(f"    [{r['formula']}/{r['method']} t{r['trial']}]"
                  f"  F1={r['f1']:.4f}  {r['extracted_formula']}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Mushroom
# ══════════════════════════════════════════════════════════════════════════════
if args.dataset in ('all', 'mushroom'):
    separator('EXPERIMENT 2 — UCI Mushroom (111 features)')
    OUT      = os.path.join(RES_BASE, 'mushroom')
    all_rows = []
    all_recs = []

    for method in ['LM', 'STE', 'Proximal']:
        successes = 0
        t_start = time.time()
        for trial in range(N_TRIALS):
            seed = BASE_SEED + trial * 1000
            ds   = load_mushroom(enrich=True, seed=seed)
            rec  = run_trial(method, ds, 'mushroom', trial, seed)
            all_recs.append(rec)
            all_rows.append({k: v for k, v in rec.items() if k != 'mse_history'})
            m = '✓' if rec['crystallized'] else '✗'
            print(f'  {method} t{trial}: '
                  f'mse={rec["mse"]:.4f}  f1={rec["f1"]:.3f}  '
                  f'acc={rec["accuracy"]:.3f}  crys={m}  '
                  f'active={rec["active_features"]}  {rec["time_s"]:.1f}s')
            if rec['crystallized']:
                successes += 1
        print(f'  └─ {method}: {successes}/{N_TRIALS} crystallized'
              f'  ({time.time()-t_start:.1f}s)')

    df_m = save_csv(all_rows, OUT, 'all_trials.csv')
    save_curves(all_recs, OUT, 'learning_curves.png', TOL['mushroom'])

    separator('Mushroom — Aggregate')
    agg = df_m.groupby('method').agg(
        mse_mean=('mse', 'mean'),     mse_std=('mse', 'std'),
        acc_mean=('accuracy', 'mean'),
        f1_mean=('f1', 'mean'),       f1_max=('f1', 'max'),
        lam_mean=('lambda', 'mean'),
        crys_rate=('crystallized', 'mean'),
        active_mean=('active_features', 'mean'),
        time_mean=('time_s', 'mean'),
    ).round(4)
    print(agg.to_string())

    best_all = max(all_recs, key=lambda r: r['f1'] if r['f1'] == r['f1'] else -1)
    print(f'\n  Overall best: {best_all["method"]} t{best_all["trial"]}'
          f'  F1={best_all["f1"]:.4f}  active={best_all["active_features"]}')
    if best_all['extracted_formula']:
        print(f'  Formula: {best_all["extracted_formula"]}')


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Heart Disease
# ══════════════════════════════════════════════════════════════════════════════
if args.dataset in ('all', 'heart'):
    separator('EXPERIMENT 3 — UCI Heart Disease Cleveland (22 features, 303 samples)')
    OUT      = os.path.join(RES_BASE, 'heart_disease')
    all_rows = []
    all_recs = []

    for method in ['LM', 'STE', 'Proximal']:
        successes = 0
        t_start = time.time()
        for trial in range(N_TRIALS):
            seed = BASE_SEED + trial * 1000
            ds   = load_heart_disease(seed=seed)
            rec  = run_trial(method, ds, 'heart', trial, seed)
            all_recs.append(rec)
            all_rows.append({k: v for k, v in rec.items() if k != 'mse_history'})
            m = '✓' if rec['crystallized'] else '✗'
            print(f'  {method} t{trial}: '
                  f'mse={rec["mse"]:.4f}  f1={rec["f1"]:.3f}  '
                  f'sens={rec["sensitivity"]:.3f}  spec={rec["specificity"]:.3f}  '
                  f'crys={m}  Δ={rec["delta_n"]:.4f}  {rec["time_s"]:.1f}s')
            if rec['crystallized']:
                successes += 1
        print(f'  └─ {method}: {successes}/{N_TRIALS} crystallized'
              f'  ({time.time()-t_start:.1f}s)')

    df_h = save_csv(all_rows, OUT, 'all_trials.csv')
    save_curves(all_recs, OUT, 'learning_curves.png', TOL['heart'])

    separator('Heart Disease — Aggregate')
    agg = df_h.groupby('method').agg(
        mse_mean=('mse', 'mean'),          mse_std=('mse', 'std'),
        acc_mean=('accuracy', 'mean'),     acc_std=('accuracy', 'std'),
        f1_mean=('f1', 'mean'),            f1_max=('f1', 'max'),
        sens_mean=('sensitivity', 'mean'), sens_max=('sensitivity', 'max'),
        spec_mean=('specificity', 'mean'),
        lam_mean=('lambda', 'mean'),
        crys_rate=('crystallized', 'mean'),
        active_mean=('active_features', 'mean'),
        time_mean=('time_s', 'mean'),
    ).round(4)
    print(agg.to_string())

    # Per-method best
    separator('Heart Disease — Best trial per method')
    for method in ['LM', 'STE', 'Proximal']:
        sub = [r for r in all_recs if r['method'] == method]
        if not sub:
            continue
        best = max(sub, key=lambda r: r['f1'] if r['f1'] == r['f1'] else -1)
        print(f'  {method}  trial={best["trial"]}  '
              f'F1={best["f1"]:.4f}  acc={best["accuracy"]:.4f}  '
              f'sens={best["sensitivity"]:.3f}  spec={best["specificity"]:.3f}  '
              f'crys={best["crystallized"]}  active={best["active_features"]}')
        if best['extracted_formula']:
            print(f'    features: {best["active_feature_names"]}')
            print(f'    formula:  {best["extracted_formula"]}')

    # Feature selection frequency
    separator('Heart Disease — Feature selection frequency (crystallized trials)')
    for method in ['LM', 'STE', 'Proximal']:
        crys = [r for r in all_recs if r['method'] == method and r['crystallized']]
        if not crys:
            print(f'  {method}: 0 crystallized trials')
            continue
        cnt = Counter()
        for r in crys:
            for f in r['active_feature_names'].split('|'):
                if f:
                    cnt[f] += 1
        top = cnt.most_common(10)
        print(f'  {method} ({len(crys)} crys trials): '
              + '  '.join(f'{f}={n}' for f, n in top))


# ── Done ──────────────────────────────────────────────────────────────────────
separator('ALL RUNS COMPLETE')
run_mode = 'quick (3 trials)' if QUICK else 'full (10 trials)'
print(f'  Mode:    {run_mode}')
print(f'  Results: {RES_BASE}')
print()
