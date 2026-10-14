"""
Dataset Download Utility
========================
Downloads the Kaggle Medicinal Flower Dataset using the `kagglehub` library.
Original dataset source: https://www.kaggle.com/datasets/jannatulferdaues/medicinal-flower-dataset

Note: Requires `kagglehub` package (`pip install kagglehub`).
"""

import sys

def download():
    try:
        import kagglehub
    except ImportError:
        print("[ERROR] 'kagglehub' package is required to download from Kaggle.")
        print("Install it using: pip install kagglehub")
        sys.exit(1)

    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("jannatulferdaues/medicinal-flower-dataset")
    print(f"[OK] Dataset successfully downloaded to:\n{path}")
    return path

if __name__ == "__main__":
    download()
