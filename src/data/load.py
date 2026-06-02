import shutil
from pathlib import Path

import pandas as pd

from src.config import KAGGLE_DATASET, RAW_DIR, RAW_FILENAME


def download_raw():
    """Grab the dataset off Kaggle and drop a copy in data/raw/."""
    import kagglehub

    cache = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / RAW_FILENAME
    shutil.copy2(cache / RAW_FILENAME, dest)
    return dest


def load_raw():
    path = RAW_DIR / RAW_FILENAME
    if not path.exists():
        path = download_raw()
    return pd.read_csv(path)
