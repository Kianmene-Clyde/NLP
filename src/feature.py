from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def get_vectorizer() -> CountVectorizer:
    try:
        fr = set(stopwords.words("french"))
    except Exception:
        fr = None
    return CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=fr,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
    )


def make_features(df):
    X_text = df["video_name"].astype(str).fillna("")
    y = df["is_comic"] if "is_comic" in df.columns else None
    return X_text, y
