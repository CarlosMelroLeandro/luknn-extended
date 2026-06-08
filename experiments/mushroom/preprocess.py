"""
Mushroom dataset preprocessing.

Replicates §5 of Leandro (ALT 2009):
  1.  Download from UCI (or use local copy in data/real/mushroom/).
  2.  One-hot encode all nominal attributes → 111 binary features.
  3.  Enrich: for each positive case add a "half-valued" negative case
      (multiply all attributes by 0.5).

The enriched dataset is saved to data/real/mushroom/mushroom_enriched.csv.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

MUSHROOM_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
)

COLUMNS = [
    "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
    "gill_attachment", "gill_spacing", "gill_size", "gill_color",
    "stalk_shape", "stalk_root", "stalk_surface_above_ring",
    "stalk_surface_below_ring", "stalk_color_above_ring", "stalk_color_below_ring",
    "veil_type", "veil_color", "ring_number", "ring_type", "spore_print_color",
    "population", "habitat",
]

DATA_DIR = Path(__file__).parents[2] / "data" / "real" / "mushroom"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    src = path or DATA_DIR / "agaricus-lepiota.data"
    if src.exists():
        return pd.read_csv(src, header=None, names=COLUMNS)
    print(f"Downloading mushroom dataset to {DATA_DIR}…")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MUSHROOM_URL, header=None, names=COLUMNS)
    df.to_csv(DATA_DIR / "agaricus-lepiota.data", index=False, header=False)
    return df


def binarize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode all attributes → (X, y) with y ∈ {0, 1}."""
    y = (df["class"] == "e").astype(float).values   # edible=1, poisonous=0
    feature_cols = [c for c in df.columns if c != "class"]
    # Replace missing '?' with most-frequent
    df_feat = df[feature_cols].replace("?", np.nan).fillna(
        df[feature_cols].mode().iloc[0]
    )
    parts = []
    for col in feature_cols:
        lb = LabelBinarizer()
        parts.append(lb.fit_transform(df_feat[col]))
    X = np.hstack(parts).astype(float)
    return X, y


def enrich(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Add synthetic negative cases by multiplying each positive sample by 0.5.
    This speeds up convergence and reduces un-representable configurations.
    """
    pos_mask = y == 1
    X_half = X[pos_mask] * 0.5
    y_half = np.zeros(pos_mask.sum())
    X_out = np.vstack([X, X_half])
    y_out = np.concatenate([y, y_half])
    return X_out, y_out


def prepare(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    df = load_raw(path)
    X, y = binarize(df)
    X, y = enrich(X, y)
    out_path = DATA_DIR / "mushroom_enriched.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).assign(label=y).to_csv(out_path, index=False)
    print(f"Saved enriched dataset ({len(y)} rows, {X.shape[1]} features) → {out_path}")
    return X, y


if __name__ == "__main__":
    prepare()
