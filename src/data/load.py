"""Load the raw Kaggle dataset from data/raw/."""
from pathlib import Path
import pandas as pd

from src.config import RAW_DIR, RAW_FILENAME


def load_raw(filename: str = RAW_FILENAME) -> pd.DataFrame:
    """Read the raw CSV from data/raw/.

    Raises FileNotFoundError with a clear message if the team has not yet
    placed the Kaggle download in the expected location.
    """
    path: Path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it from Kaggle and "
            f"place it there, or update RAW_FILENAME in src/config.py."
        )
    return pd.read_csv(path)
