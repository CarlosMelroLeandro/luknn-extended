"""
Dataset wrappers for benchmarking.

Supported datasets
------------------
truth_table     Synthetic (n+1)-valued truth sub-tables for Ł formulas.
mushroom        UCI Mushroom (8124 samples, 22 nominal attributes → 111 binary).
heart_disease   UCI Heart Disease — Cleveland subset (303 samples, 22 features).
babi            Facebook bAbI Task-11 (path reasoning) converted to binary features.
monk            UCI MONK problems 1/2/3 (122–556 samples, 6 nominal attrs → 17 binary).
breast_cancer   UCI Breast Cancer Ljubljana (286 samples, 9 nominal attrs → binary).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from ..logic.connectives import truth_subtable, evaluate_formula


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    X_train: Tensor
    y_train: Tensor
    X_test: Tensor
    y_test: Tensor
    n_features: int
    name: str
    feature_names: list[str] | None = None
    X_val: Tensor | None = None
    y_val: Tensor | None = None


# ---------------------------------------------------------------------------
# Class-weighting utility
# ---------------------------------------------------------------------------

def compute_class_weight(y: Tensor) -> Tensor:
    """
    Compute per-sample class weights for binary classification.

    Uses the sklearn 'balanced' formula: w_c = n / (2 * n_c).
    Returns a float32 tensor of shape (N,) with one weight per sample.
    The minority class receives a weight > 1; majority class < 1.
    """
    n = len(y)
    n1 = y.sum().item()
    n0 = n - n1
    w1 = n / (2.0 * n1) if n1 > 0 else 1.0
    w0 = n / (2.0 * n0) if n0 > 0 else 1.0
    weights = torch.where(y > 0.5,
                          torch.tensor(w1, dtype=torch.float32),
                          torch.tensor(w0, dtype=torch.float32))
    return weights


# ---------------------------------------------------------------------------
# Formula registry
# ---------------------------------------------------------------------------

def _f1(x1, x2, x3, x4, x5, x6):
    from ..logic.connectives import tnorm, residuum
    return tnorm(x1, residuum(x3, x6))

def _f2(x1, x2, x3, x4, x5, x6):
    from ..logic.connectives import tnorm, residuum
    return tnorm(residuum(x4, x6), residuum(x6, x2))

def _f6(x1, x2, x3, x4, x5, x6):
    from ..logic.connectives import tnorm, residuum
    return tnorm(
        tnorm(
            tnorm(residuum(tnorm(x4, x5), x6), residuum(tnorm(x1, x5), x2)),
            residuum(tnorm(x1, x2), x3),
        ),
        residuum(x6, x4),
    )

FORMULA_REGISTRY: dict[str, tuple[Callable, int]] = {
    "f1": (_f1, 6),
    "f2": (_f2, 6),
    "f6": (_f6, 6),
}


def _random_formula(n_vars: int):
    """Generate a random conjunctive-disjunctive formula (3 to 5 connectives)."""
    from ..logic.connectives import tnorm, disjunction, negation

    def fn(*xs):
        ops = [tnorm, disjunction]
        rng = np.random.default_rng(seed=n_vars)
        result = xs[0]
        for i in range(1, n_vars):
            op = ops[rng.integers(0, 2)]
            x = xs[i] if rng.random() > 0.3 else negation(xs[i])
            result = op(result, x)
        return result
    return fn


# ---------------------------------------------------------------------------
# Truth table dataset
# ---------------------------------------------------------------------------

def load_truth_table(
    formula: str = "f1",
    n_values: int = 3,
    n_vars: int = 6,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Dataset:
    if formula == "random":
        fn = _random_formula(n_vars)
    elif formula in FORMULA_REGISTRY:
        fn, n_vars_req = FORMULA_REGISTRY[formula]
        n_vars = n_vars_req
    else:
        raise ValueError(f"Unknown formula {formula!r}. Available: {list(FORMULA_REGISTRY)}")

    X = truth_subtable(n_vars, n_values)
    y = evaluate_formula(fn, n_vars, n_values)

    N = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_test = max(1, int(N * test_fraction))
    train_idx = torch.from_numpy(idx[n_test:])
    test_idx = torch.from_numpy(idx[:n_test])

    return Dataset(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        n_features=n_vars,
        name=f"truth_table_{formula}_{n_values}v",
        feature_names=[f"x{i+1}" for i in range(n_vars)],
    )


# ---------------------------------------------------------------------------
# Mushroom dataset (UCI)
# ---------------------------------------------------------------------------

_MUSHROOM_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "mushroom/agaricus-lepiota.data"
)
_MUSHROOM_COLS = [
    "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
    "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
    "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring",
    "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", "veil_color",
    "ring_number", "ring_type", "spore_print_color", "population", "habitat",
]


def load_mushroom(
    data_dir: str | Path = "data/real/mushroom",
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    enrich: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    Load + binarize mushroom dataset.  Downloads if not cached.
    enrich=True adds synthetic half-valued negative cases (paper §5).
    """
    import pandas as pd
    from sklearn.preprocessing import LabelBinarizer

    data_dir = Path(data_dir)
    raw_path = data_dir / "agaricus-lepiota.data"

    if not raw_path.exists():
        import urllib.request
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MUSHROOM_URL, raw_path)

    df = pd.read_csv(raw_path, header=None, names=_MUSHROOM_COLS)
    y = (df["class"] == "e").astype(float).values
    feat_cols = [c for c in df.columns if c != "class"]
    df_feat = df[feat_cols].replace("?", np.nan).fillna(
        df[feat_cols].mode().iloc[0]
    )
    parts = [LabelBinarizer().fit_transform(df_feat[c]) for c in feat_cols]
    X = np.hstack(parts).astype(float)

    if enrich:
        pos_mask = y == 1
        X = np.vstack([X, X[pos_mask] * 0.5])
        y = np.concatenate([y, np.zeros(pos_mask.sum())])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    train_all = idx[n_test:]
    test_idx  = idx[:n_test]

    if val_fraction > 0:
        n_val    = max(1, int(len(train_all) * val_fraction))
        val_idx  = train_all[:n_val]
        tr_idx   = train_all[n_val:]
    else:
        val_idx  = None
        tr_idx   = train_all

    return Dataset(
        X_train=torch.tensor(X[tr_idx],   dtype=torch.float32),
        y_train=torch.tensor(y[tr_idx],   dtype=torch.float32),
        X_test =torch.tensor(X[test_idx], dtype=torch.float32),
        y_test =torch.tensor(y[test_idx], dtype=torch.float32),
        n_features=X.shape[1],
        name="mushroom",
        X_val=torch.tensor(X[val_idx], dtype=torch.float32) if val_idx is not None else None,
        y_val=torch.tensor(y[val_idx], dtype=torch.float32) if val_idx is not None else None,
    )


def load_mushroom_grouped(
    data_dir: str | Path = "data/real/mushroom",
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    enrich: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    Mushroom dataset with reduced fan-in: 22 features instead of 111.

    Instead of one-hot encoding each categorical attribute (→ 111 binary
    columns), each of the 22 attributes is encoded as a single integer label
    normalised to [0, 1].  This keeps the input in the Łukasiewicz truth-value
    range while reducing the fan-in from 111 to 22, making each surviving
    weight ~5× larger and more likely to survive crystallisation to ±1.

    All other options (enrich, splits, seed) are identical to load_mushroom().
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    data_dir = Path(data_dir)
    raw_path = data_dir / "agaricus-lepiota.data"

    if not raw_path.exists():
        import urllib.request
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MUSHROOM_URL, raw_path)

    df = pd.read_csv(raw_path, header=None, names=_MUSHROOM_COLS)
    y = (df["class"] == "e").astype(float).values
    feat_cols = [c for c in df.columns if c != "class"]
    df_feat = df[feat_cols].replace("?", np.nan).fillna(
        df[feat_cols].mode().iloc[0]
    )

    parts = []
    for c in feat_cols:
        le = LabelEncoder()
        enc = le.fit_transform(df_feat[c]).astype(float)
        n_cats = len(le.classes_)
        parts.append((enc / max(n_cats - 1, 1)).reshape(-1, 1))
    X = np.hstack(parts)  # shape (n_samples, 22)

    if enrich:
        pos_mask = y == 1
        X = np.vstack([X, X[pos_mask] * 0.5])
        y = np.concatenate([y, np.zeros(pos_mask.sum())])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    train_all = idx[n_test:]
    test_idx  = idx[:n_test]

    if val_fraction > 0:
        n_val   = max(1, int(len(train_all) * val_fraction))
        val_idx = train_all[:n_val]
        tr_idx  = train_all[n_val:]
    else:
        val_idx = None
        tr_idx  = train_all

    return Dataset(
        X_train=torch.tensor(X[tr_idx],   dtype=torch.float32),
        y_train=torch.tensor(y[tr_idx],   dtype=torch.float32),
        X_test =torch.tensor(X[test_idx], dtype=torch.float32),
        y_test =torch.tensor(y[test_idx], dtype=torch.float32),
        n_features=X.shape[1],
        name="mushroom_grouped",
        X_val=torch.tensor(X[val_idx], dtype=torch.float32) if val_idx is not None else None,
        y_val=torch.tensor(y[val_idx], dtype=torch.float32) if val_idx is not None else None,
    )


# ---------------------------------------------------------------------------
# bAbI Task-11 (path reasoning)
# ---------------------------------------------------------------------------

def load_babi_task11(
    data_dir: str | Path = "data/real/babi",
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Dataset:
    """
    Load bAbI Task-11 (basic coreference) and encode as binary feature vectors.

    Data format (each story):
        1  Mary moved to the bathroom.
        2  Sandra journeyed to the bedroom.
        ...
        N  Where is Mary?  <answer> <support>

    Feature encoding: bag-of-words over entity × location pairs (binary).
    Download from: https://research.fb.com/downloads/babi/

    Raises FileNotFoundError if data_dir does not exist.
    """
    data_dir = Path(data_dir)
    train_file = data_dir / "qa11_basic-coreference_train.txt"
    test_file = data_dir / "qa11_basic-coreference_test.txt"

    if not train_file.exists():
        raise FileNotFoundError(
            f"bAbI Task-11 data not found at {train_file}.\n"
            "Download from https://research.fb.com/downloads/babi/ and place "
            f"the .txt files in {data_dir}/"
        )

    def parse_file(path: Path):
        stories, labels = [], []
        story: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    story = []
                    continue
                idx, rest = line.split(" ", 1)
                if "\t" in rest:
                    # Question line
                    question, answer, *_ = rest.split("\t")
                    stories.append((" ".join(story), question))
                    labels.append(answer.strip().lower())
                else:
                    story.append(rest)
        return stories, labels

    train_stories, train_labels = parse_file(train_file)
    test_stories, test_labels = parse_file(test_file)
    all_stories = train_stories + test_stories
    all_labels = train_labels + test_labels

    # Build binary feature matrix: unique (word) presence per story+question
    from sklearn.feature_extraction.text import CountVectorizer
    texts = [s + " " + q for s, q in all_stories]
    vec = CountVectorizer(binary=True, min_df=2)
    X = vec.fit_transform(texts).toarray().astype(float)

    # Binary label: correct answer = first label in sorted unique labels?
    unique_labels = sorted(set(all_labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in all_labels], dtype=float)
    # Binarize: label 0 vs rest
    y = (y == 0).astype(float)

    n_train = len(train_stories)
    X_tr = torch.tensor(X[:n_train], dtype=torch.float32)
    y_tr = torch.tensor(y[:n_train], dtype=torch.float32)
    X_te = torch.tensor(X[n_train:], dtype=torch.float32)
    y_te = torch.tensor(y[n_train:], dtype=torch.float32)

    return Dataset(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        n_features=X.shape[1],
        name="babi_task11",
        feature_names=vec.get_feature_names_out().tolist(),
    )


# ---------------------------------------------------------------------------
# Heart Disease dataset (UCI)
# ---------------------------------------------------------------------------

_HEART_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

# Column metadata for the 13 input features of processed.cleveland.data
# (col 13 is the target)
_HEART_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num",
]
# Continuous features: normalised to [0,1] via MinMax over training set
_CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Binary features: already 0/1
_BINARY = ["sex", "fbs", "exang"]
# Categorical features: one-hot encoded
# cp: 1-4 (4 types of chest pain)
# restecg: 0-2 (resting ECG result)
# slope: 1-3 (slope of ST segment)
# thal: 3=normal, 6=fixed defect, 7=reversible defect
_CATEGORICAL = ["cp", "restecg", "slope", "thal"]
# Ordinal: ca (0–3 major vessels) — normalised to [0,1] like continuous
_ORDINAL = ["ca"]   # max=3, so ca/3 ∈ {0, 1/3, 2/3, 1}


def load_heart_disease(
    data_dir: str | Path = "data/real/heart",
    subset: str = "cleveland",
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> Dataset:
    """
    Load and preprocess the UCI Heart Disease dataset.

    subset: 'cleveland' (303 rows, standard benchmark)
            'all'       (cleveland + hungarian + va + switzerland, ~920 rows)

    Feature engineering (→ 22 binary/continuous features in [0,1]):
      • Continuous (age, trestbps, chol, thalach, oldpeak, ca): MinMax [0,1]
      • Binary (sex, fbs, exang): kept as-is
      • Categorical (cp, restecg, slope, thal): one-hot encoded

    Target: binarised — 0 = no heart disease, 1 = disease (num ≥ 1).
    Missing values (cols ca and thal in Cleveland): filled with column median.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

    data_dir = Path(data_dir)
    zip_path = data_dir / "heart_disease.zip"

    # Download if missing
    if not (data_dir / "processed.cleveland.data").exists():
        import urllib.request, zipfile
        data_dir.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            urllib.request.urlretrieve(_HEART_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)

    # Choose which files to load
    if subset == "all":
        files = [
            "processed.cleveland.data",
            "processed.hungarian.data",
            "processed.va.data",
            "processed.switzerland.data",
        ]
    else:
        files = ["processed.cleveland.data"]

    frames = []
    for fname in files:
        path = data_dir / fname
        if path.exists():
            frames.append(pd.read_csv(path, header=None,
                                      names=_HEART_COLS, na_values="?"))
    df = pd.concat(frames, ignore_index=True)

    # Binarise target: 0=healthy, 1=disease
    y = (df["num"] >= 1).astype(float).values
    df = df.drop(columns=["num"])

    # Impute missing values with column median (only ca and thal have missing)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Build feature matrix
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_fraction))
    train_all = idx[n_test:]
    test_idx  = idx[:n_test]

    if val_fraction > 0:
        n_val    = max(1, int(len(train_all) * val_fraction))
        val_idx  = train_all[:n_val]
        train_idx = train_all[n_val:]
    else:
        val_idx   = None
        train_idx = train_all

    # Fit scalers on train_proper only (never on val or test)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True

    parts = []
    feat_names: list[str] = []

    # Continuous + ordinal → MinMax
    for col in _CONTINUOUS + _ORDINAL:
        vals = df[[col]].values.astype(float)
        scaler = MinMaxScaler()
        scaler.fit(vals[train_mask])
        parts.append(scaler.transform(vals))
        feat_names.append(col)

    # Binary → as-is
    for col in _BINARY:
        parts.append(df[[col]].values.astype(float))
        feat_names.append(col)

    # Categorical → one-hot (fit on all data to cover all categories)
    for col in _CATEGORICAL:
        lb = LabelBinarizer()
        lb.fit(df[col].values)
        encoded = lb.transform(df[col].values).astype(float)
        if encoded.shape[1] == 1:   # binary attribute (shouldn't happen here)
            parts.append(encoded)
            feat_names.append(f"{col}=1")
        else:
            parts.append(encoded)
            feat_names.extend(f"{col}={c}" for c in lb.classes_)

    X = np.hstack(parts)

    return Dataset(
        X_train=torch.tensor(X[train_idx], dtype=torch.float32),
        y_train=torch.tensor(y[train_idx], dtype=torch.float32),
        X_test =torch.tensor(X[test_idx],  dtype=torch.float32),
        y_test =torch.tensor(y[test_idx],  dtype=torch.float32),
        n_features=X.shape[1],
        name=f"heart_disease_{subset}",
        feature_names=feat_names,
        X_val=torch.tensor(X[val_idx], dtype=torch.float32) if val_idx is not None else None,
        y_val=torch.tensor(y[val_idx], dtype=torch.float32) if val_idx is not None else None,
    )


# ---------------------------------------------------------------------------
# MONK problems (UCI)
# ---------------------------------------------------------------------------

# Attribute names and their cardinalities
_MONK_ATTR_NAMES   = ["head_shape", "body_shape", "is_smiling", "holding",
                      "jacket_color", "has_tie"]
_MONK_CARDINALITIES = [3, 3, 2, 3, 4, 2]  # number of distinct values per attr

# Ground-truth rules (for reference in notebooks)
MONK_RULES = {
    1: "(a1 == a2) OR (a5 == 1)",
    2: "exactly 2 of {a1=1, a2=1, a3=1, a4=1, a5=1, a6=1} are true",
    3: "(a5 == 3 AND a4 == 1) OR (a5 != 4 AND a2 != 3)  [~5% label noise]",
}

# Original UCI train-set sizes for each problem (test = full 432 domain)
_MONK_TRAIN_SIZES = {1: 124, 2: 169, 3: 122}


def _monk_label(problem: int, row: np.ndarray, rng=None) -> int:
    """Apply the ground-truth classification rule for a single MONK instance."""
    a1, a2, a3, a4, a5, a6 = row
    if problem == 1:
        return int((a1 == a2) or (a5 == 1))
    elif problem == 2:
        return int(sum(v == 1 for v in [a1, a2, a3, a4, a5, a6]) == 2)
    else:  # MONK-3 with ~5% label noise on training
        label = int((a5 == 3 and a4 == 1) or (a5 != 4 and a2 != 3))
        if rng is not None and rng.random() < 0.05:
            label = 1 - label
        return label


def _generate_monk(problem: int, seed: int) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """
    Generate the full MONK domain (432 samples = all attribute combinations)
    and split into train/test following the original UCI train-set sizes.

    The complete domain is always used as the test set (no hold-out noise),
    matching the UCI evaluation protocol.
    """
    from itertools import product

    # Full domain: all 3×3×2×3×4×2 = 432 combinations (1-indexed values)
    domain = np.array(list(product(
        range(1, 4), range(1, 4), range(1, 3),
        range(1, 4), range(1, 5), range(1, 3),
    )), dtype=int)  # shape (432, 6)

    # Test labels: clean (no noise)
    y_test = np.array([_monk_label(problem, row) for row in domain], dtype=float)

    # Train split: random subset of _MONK_TRAIN_SIZES[problem] from the domain
    rng = np.random.default_rng(seed)
    n_train = _MONK_TRAIN_SIZES[problem]
    train_idx = rng.choice(len(domain), size=n_train, replace=False)

    X_train = domain[train_idx].copy()
    # MONK-3 train labels get 5% noise
    noise_rng = np.random.default_rng(seed + 1) if problem == 3 else None
    y_train = np.array([
        _monk_label(problem, row,
                    rng=noise_rng if problem == 3 else None)
        for row in X_train
    ], dtype=float)

    return X_train, y_train, domain, y_test


def load_monk(
    problem: int = 1,
    data_dir: str | Path = "data/real/monk",
    seed: int = 42,
) -> Dataset:
    """
    Load a MONK benchmark problem (1, 2, or 3).

    Generates the full domain (432 samples) from the known rules.  Train set
    follows the original UCI sizes (MONK-1: 124, MONK-2: 169, MONK-3: 122).
    Test set is always the complete 432-sample domain (clean labels).
    Each of the 6 attributes is one-hot encoded → 17 binary features.

    The ground-truth logical rule for each problem is available in MONK_RULES.
    """
    if problem not in (1, 2, 3):
        raise ValueError("problem must be 1, 2, or 3")

    X_tr_raw, y_tr, X_te_raw, y_te = _generate_monk(problem, seed)

    def one_hot(X_raw: np.ndarray) -> np.ndarray:
        parts = []
        for col_idx, n_vals in enumerate(_MONK_CARDINALITIES):
            col = X_raw[:, col_idx]
            enc = np.zeros((len(col), n_vals), dtype=float)
            for row_i, v in enumerate(col):
                enc[row_i, v - 1] = 1.0
            parts.append(enc)
        return np.hstack(parts)

    X_tr = one_hot(X_tr_raw)
    X_te = one_hot(X_te_raw)

    feat_names: list[str] = []
    for attr, n_vals in zip(_MONK_ATTR_NAMES, _MONK_CARDINALITIES):
        feat_names.extend(f"{attr}={v+1}" for v in range(n_vals))

    return Dataset(
        X_train=torch.tensor(X_tr, dtype=torch.float32),
        y_train=torch.tensor(y_tr, dtype=torch.float32),
        X_test=torch.tensor(X_te, dtype=torch.float32),
        y_test=torch.tensor(y_te, dtype=torch.float32),
        n_features=X_tr.shape[1],
        name=f"monk_{problem}",
        feature_names=feat_names,
    )


def load_monk_grouped(
    problem: int = 1,
    seed: int = 42,
) -> Dataset:
    """
    MONK problem with reduced fan-in: 6 ordinal features instead of 17.

    Each of the 6 categorical attributes is encoded as a single value
    normalised to [0, 1]:  (raw_value - 1) / (cardinality - 1).
    This keeps inputs in the Łukasiewicz truth-value range while reducing
    fan-in from 17 to 6, making surviving weights ~3× larger and more
    likely to survive crystallisation.

    All other generation options (split, seed, noise) are identical to
    load_monk().
    """
    if problem not in (1, 2, 3):
        raise ValueError("problem must be 1, 2, or 3")

    X_tr_raw, y_tr, X_te_raw, y_te = _generate_monk(problem, seed)

    def ordinal(X_raw: np.ndarray) -> np.ndarray:
        parts = []
        for col_idx, n_vals in enumerate(_MONK_CARDINALITIES):
            col = (X_raw[:, col_idx] - 1).astype(float)
            col = col / max(n_vals - 1, 1)      # normalise to [0, 1]
            parts.append(col.reshape(-1, 1))
        return np.hstack(parts)

    X_tr = ordinal(X_tr_raw)
    X_te = ordinal(X_te_raw)

    return Dataset(
        X_train=torch.tensor(X_tr, dtype=torch.float32),
        y_train=torch.tensor(y_tr, dtype=torch.float32),
        X_test=torch.tensor(X_te, dtype=torch.float32),
        y_test=torch.tensor(y_te, dtype=torch.float32),
        n_features=X_tr.shape[1],
        name=f"monk_{problem}_grouped",
        feature_names=_MONK_ATTR_NAMES,
    )


# ---------------------------------------------------------------------------
# Breast Cancer Ljubljana (UCI)
# ---------------------------------------------------------------------------

_BREAST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer/breast-cancer.data"
)

_BREAST_COLS = [
    "class", "age", "menopause", "tumor_size", "inv_nodes",
    "node_caps", "deg_malig", "breast", "breast_quad", "irradiat",
]

# Ordinal attributes: encode as integer rank / (n_levels - 1) → [0, 1]
_BREAST_ORDINAL = {
    "age":        ["10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90-99"],
    "tumor_size": ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54"],
    "inv_nodes":  ["0-2","3-5","6-8","9-11","12-14","15-17","18-20","21-23","24-26","27-29","30-32","33-35","36-39"],
    "deg_malig":  [1, 2, 3],
}

# Nominal attributes: one-hot
_BREAST_NOMINAL = ["menopause", "node_caps", "breast", "breast_quad", "irradiat"]


def load_breast_cancer(
    data_dir: str | Path = "data/real/breast_cancer",
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> Dataset:
    """
    Load the UCI Breast Cancer (Ljubljana) dataset.

    286 samples, 9 mixed attributes, binary target:
      0 = no-recurrence-events, 1 = recurrence-events.

    Encoding:
      • Ordinal (age, tumor_size, inv_nodes, deg_malig): rank / (n−1) → [0,1]
      • Nominal (menopause, node_caps, breast, breast_quad, irradiat): one-hot
    Missing values (node_caps, breast_quad): replaced with mode.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelBinarizer

    data_dir = Path(data_dir)
    raw_path = data_dir / "breast-cancer.data"

    if not raw_path.exists():
        import urllib.request
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_BREAST_URL, raw_path)

    df = pd.read_csv(raw_path, header=None, names=_BREAST_COLS, na_values="?")
    y = (df["class"] == "recurrence-events").astype(float).values
    df = df.drop(columns=["class"])

    # Impute missing with mode (column-wise)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    parts: list[np.ndarray] = []
    feat_names: list[str] = []

    for col, levels in _BREAST_ORDINAL.items():
        levels_str = [str(l) for l in levels]
        rank_map = {v: i / (len(levels_str) - 1) for i, v in enumerate(levels_str)}
        vals = df[col].astype(str).map(rank_map).fillna(0.5).values.reshape(-1, 1)
        parts.append(vals)
        feat_names.append(col)

    for col in _BREAST_NOMINAL:
        lb = LabelBinarizer()
        enc = lb.fit_transform(df[col].astype(str).values).astype(float)
        if enc.shape[1] == 1:
            parts.append(enc)
            feat_names.append(f"{col}=1")
        else:
            parts.append(enc)
            feat_names.extend(f"{col}={c}" for c in lb.classes_)

    X = np.hstack(parts)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(len(y) * test_fraction))
    train_all = idx[n_test:]
    te_idx    = idx[:n_test]

    if val_fraction > 0:
        n_val    = max(1, int(len(train_all) * val_fraction))
        val_idx  = train_all[:n_val]
        tr_idx   = train_all[n_val:]
    else:
        val_idx  = None
        tr_idx   = train_all

    return Dataset(
        X_train=torch.tensor(X[tr_idx], dtype=torch.float32),
        y_train=torch.tensor(y[tr_idx], dtype=torch.float32),
        X_test =torch.tensor(X[te_idx], dtype=torch.float32),
        y_test =torch.tensor(y[te_idx], dtype=torch.float32),
        n_features=X.shape[1],
        name="breast_cancer",
        feature_names=feat_names,
        X_val=torch.tensor(X[val_idx], dtype=torch.float32) if val_idx is not None else None,
        y_val=torch.tensor(y[val_idx], dtype=torch.float32) if val_idx is not None else None,
    )


def load_bc_grouped(
    data_dir: str | Path = "data/real/breast_cancer",
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> Dataset:
    """
    Breast Cancer dataset with reduced fan-in: 9 features instead of 15.

    The standard loader one-hot encodes the 5 nominal attributes
    (menopause, node_caps, breast, breast_quad, irradiat) producing 11
    extra columns.  Here every attribute is represented as a single value
    normalised to [0, 1]:
      • Ordinal attrs: same rank encoding as load_breast_cancer().
      • Nominal attrs: LabelEncoder integer / (n_classes − 1).

    This reduces fan-in from 15 to 9 while keeping all inputs in the
    Łukasiewicz truth-value range.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    data_dir = Path(data_dir)
    raw_path = data_dir / "breast-cancer.data"

    if not raw_path.exists():
        import urllib.request
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_BREAST_URL, raw_path)

    df = pd.read_csv(raw_path, header=None, names=_BREAST_COLS, na_values="?")
    y = (df["class"] == "recurrence-events").astype(float).values
    df = df.drop(columns=["class"])

    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    parts: list[np.ndarray] = []

    # Ordinal attributes: same encoding as load_breast_cancer
    for col, levels in _BREAST_ORDINAL.items():
        levels_str = [str(l) for l in levels]
        rank_map = {v: i / (len(levels_str) - 1) for i, v in enumerate(levels_str)}
        vals = df[col].astype(str).map(rank_map).fillna(0.5).values.reshape(-1, 1)
        parts.append(vals)

    # Nominal attributes: ordinal (LabelEncoder) normalised to [0, 1]
    for col in _BREAST_NOMINAL:
        le = LabelEncoder()
        enc = le.fit_transform(df[col].astype(str).values).astype(float)
        n_cls = len(le.classes_)
        parts.append((enc / max(n_cls - 1, 1)).reshape(-1, 1))

    X = np.hstack(parts)   # shape (286, 9)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(len(y) * test_fraction))
    train_all = idx[n_test:]
    te_idx    = idx[:n_test]

    if val_fraction > 0:
        n_val   = max(1, int(len(train_all) * val_fraction))
        val_idx = train_all[:n_val]
        tr_idx  = train_all[n_val:]
    else:
        val_idx = None
        tr_idx  = train_all

    return Dataset(
        X_train=torch.tensor(X[tr_idx], dtype=torch.float32),
        y_train=torch.tensor(y[tr_idx], dtype=torch.float32),
        X_test =torch.tensor(X[te_idx], dtype=torch.float32),
        y_test =torch.tensor(y[te_idx], dtype=torch.float32),
        n_features=X.shape[1],
        name="breast_cancer_grouped",
        feature_names=list(_BREAST_ORDINAL.keys()) + _BREAST_NOMINAL,
        X_val=torch.tensor(X[val_idx], dtype=torch.float32) if val_idx is not None else None,
        y_val=torch.tensor(y[val_idx], dtype=torch.float32) if val_idx is not None else None,
    )


# ---------------------------------------------------------------------------
# Spambase (UCI)
# ---------------------------------------------------------------------------

_SPAMBASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "spambase/spambase.data"
)

# 57 feature names (from spambase.names)
_SPAMBASE_FEATURES = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you",
    "word_freq_credit", "word_freq_your", "word_freq_font", "word_freq_000",
    "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george",
    "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet",
    "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
    "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm",
    "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original",
    "word_freq_project", "word_freq_re", "word_freq_edu", "word_freq_table",
    "word_freq_conference", "char_freq_semicolon", "char_freq_lparen",
    "char_freq_lbracket", "char_freq_exclaim", "char_freq_dollar",
    "char_freq_hash", "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total",
]


def load_spambase(
    data_dir: str | Path = "data/real/spambase",
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> Dataset:
    """
    Load and preprocess the UCI Spambase dataset.

    4601 samples, 57 continuous features (word/char frequencies + capital stats).
    Target: 1 = spam, 0 = not spam.
    Features normalised to [0, 1] via MinMaxScaler (fit on train only).
    No missing values.  Downloads automatically if not cached.
    """
    from sklearn.preprocessing import MinMaxScaler

    data_dir = Path(data_dir)
    raw_path = data_dir / "spambase.data"

    if not raw_path.exists():
        import urllib.request
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_SPAMBASE_URL, raw_path)

    data = np.loadtxt(raw_path, delimiter=",")
    X_raw = data[:, :57]
    y = data[:, 57].astype(float)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(len(y) * test_fraction))
    train_all = idx[n_test:]
    te_idx    = idx[:n_test]

    if val_fraction > 0:
        n_val   = max(1, int(len(train_all) * val_fraction))
        val_idx = train_all[:n_val]
        tr_idx  = train_all[n_val:]
    else:
        val_idx = None
        tr_idx  = train_all

    scaler = MinMaxScaler()
    scaler.fit(X_raw[tr_idx])
    X = scaler.transform(X_raw).astype(float)

    return Dataset(
        X_train=torch.tensor(X[tr_idx], dtype=torch.float32),
        y_train=torch.tensor(y[tr_idx], dtype=torch.float32),
        X_test =torch.tensor(X[te_idx], dtype=torch.float32),
        y_test =torch.tensor(y[te_idx], dtype=torch.float32),
        n_features=X.shape[1],
        name="spambase",
        feature_names=_SPAMBASE_FEATURES,
        X_val=torch.tensor(X[val_idx], dtype=torch.float32) if val_idx is not None else None,
        y_val=torch.tensor(y[val_idx], dtype=torch.float32) if val_idx is not None else None,
    )


# ---------------------------------------------------------------------------
# Musk v2 (UCI)
# ---------------------------------------------------------------------------

_MUSK_URL = "https://archive.ics.uci.edu/static/public/75/data.csv"


def load_musk(
    data_dir: str | Path = "data/real/musk",
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> Dataset:
    """
    Load and preprocess the UCI Musk (version 2) dataset.

    6598 samples, 166 numeric features (conformational descriptors).
    Target: 1 = musk molecule, 0 = non-musk.
    First two columns (molecule_name, conformation_name) are dropped.
    Features normalised to [0, 1] via MinMaxScaler (fit on train only).
    No missing values.  Downloads automatically if not cached.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    data_dir = Path(data_dir)
    raw_path = data_dir / "musk_v2.csv"

    if not raw_path.exists():
        import urllib.request
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MUSK_URL, raw_path)

    df = pd.read_csv(raw_path)
    # Columns: molecule_name, conformation_name, f1..f166, class
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X_raw = df[feat_cols].values.astype(float)
    # class column may have trailing dots ("1." / "0.") — coerce via float
    y = pd.to_numeric(df["class"], errors="coerce").fillna(0).values.astype(float)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(len(y) * test_fraction))
    train_all = idx[n_test:]
    te_idx    = idx[:n_test]

    if val_fraction > 0:
        n_val   = max(1, int(len(train_all) * val_fraction))
        val_idx = train_all[:n_val]
        tr_idx  = train_all[n_val:]
    else:
        val_idx = None
        tr_idx  = train_all

    scaler = MinMaxScaler()
    scaler.fit(X_raw[tr_idx])
    X = scaler.transform(X_raw).astype(float)

    return Dataset(
        X_train=torch.tensor(X[tr_idx], dtype=torch.float32),
        y_train=torch.tensor(y[tr_idx], dtype=torch.float32),
        X_test =torch.tensor(X[te_idx], dtype=torch.float32),
        y_test =torch.tensor(y[te_idx], dtype=torch.float32),
        n_features=X.shape[1],
        name="musk",
        feature_names=[f"f{i+1}" for i in range(166)],
        X_val=torch.tensor(X[val_idx], dtype=torch.float32) if val_idx is not None else None,
        y_val=torch.tensor(y[val_idx], dtype=torch.float32) if val_idx is not None else None,
    )


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_dataset(config) -> Dataset:
    """Dispatch to the right loader based on config.dataset_type."""
    dtype        = getattr(config, "dataset_type", "truth_table")
    seed         = getattr(config, "seed", 42)
    val_fraction = getattr(config, "val_fraction", 0.0)
    if dtype == "truth_table":
        return load_truth_table(
            formula=getattr(config, "formula", "f1"),
            n_values=getattr(config, "n_values", 3),
            n_vars=getattr(config, "n_vars", 6),
            seed=seed,
        )
    elif dtype == "mushroom":
        return load_mushroom(seed=seed, val_fraction=val_fraction)
    elif dtype == "heart_disease":
        return load_heart_disease(
            subset=getattr(config, "heart_subset", "cleveland"),
            seed=seed, val_fraction=val_fraction,
        )
    elif dtype == "babi":
        return load_babi_task11(seed=seed)
    elif dtype == "monk":
        return load_monk(
            problem=getattr(config, "monk_problem", 1),
            seed=seed,
        )
    elif dtype == "breast_cancer":
        return load_breast_cancer(seed=seed, val_fraction=val_fraction)
    elif dtype == "spambase":
        return load_spambase(seed=seed, val_fraction=val_fraction)
    elif dtype == "musk":
        return load_musk(seed=seed, val_fraction=val_fraction)
    else:
        raise ValueError(f"Unknown dataset type: {dtype!r}")
