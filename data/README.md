# Datasets

## Included in the repository

| Dataset | Path | Source | Size |
|---------|------|--------|------|
| Heart Disease (Cleveland) | `data/real/heart/processed.cleveland.data` | UCI / Detrano et al. (1989) | 303 instances, 13 features |
| Breast Cancer (Ljubljana) | `data/real/breast_cancer/breast-cancer.data` | UCI | 286 instances, 9 features |

MONK datasets are fetched programmatically via `sklearn.datasets` — no manual download needed.

## Must be downloaded separately

### Mushroom (Agaricus Lepiota)

The Mushroom dataset (~366 KB) is excluded from the repository.
Download with:

```bash
python scripts/download_mushroom.py
```

Or manually:
```bash
curl -o data/real/mushroom/agaricus-lepiota.data \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
```

**Attribution:** Schlimmer, J. (1987). *Mushroom Records from The Audubon Society Field Guide to North American Mushrooms.* UCI Machine Learning Repository.

## Preprocessing

All preprocessing is handled inside `src/luknn/benchmark/datasets.py`:

- **Heart Disease:** 5 continuous features → min-max scaled to [0,1]; 4 categorical features → one-hot encoded; binary/ordinal features kept or normalized. Output: 22 features.
- **Mushroom:** Categorical features → one-hot encoded. Output: 111 binary features.
- **MONK-1/2/3:** Nominal attributes → one-hot binary vectors (17 features). MONK-3 includes 5% attribute noise.
- **Breast Cancer:** Missing values dropped; categorical features → one-hot. Output: 20 features.

## UCI Terms of Use

These datasets are publicly available from the UCI Machine Learning Repository for research purposes. Please cite the original sources when using them in publications. See each dataset's documentation in the subdirectories for full attribution.
