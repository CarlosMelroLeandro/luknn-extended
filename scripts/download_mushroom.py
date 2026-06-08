"""Download the Mushroom dataset from UCI."""
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
DEST = Path(__file__).parents[1] / "data/real/mushroom/agaricus-lepiota.data"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already present: {DEST}")
        return
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    size_kb = DEST.stat().st_size // 1024
    print(f"Saved to {DEST} ({size_kb} KB)")

if __name__ == "__main__":
    main()
