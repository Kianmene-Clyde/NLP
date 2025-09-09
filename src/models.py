from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from feature import get_vectorizer

RANDOM_STATE = 42


def make_model() -> Pipeline:
    return Pipeline([
        ("vect", get_vectorizer()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
