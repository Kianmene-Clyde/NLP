import pandas as pd


def make_dataset(filename: str, expect_labels: bool | None = None) -> pd.DataFrame:
    df = pd.read_csv(filename)
    if "video_name" not in df.columns:
        raise ValueError("Le CSV doit contenir la colonne 'video_name'.")
    if expect_labels is True and "is_comic" not in df.columns:
        raise ValueError("Le CSV d'entraînement doit contenir 'is_comic' (0/1).")
    return df
