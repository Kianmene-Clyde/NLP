import click
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data import make_dataset
from feature import make_features
from models import make_model


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input_filename", default="data/train.csv", help="CSV train (video_name,is_comic)")
@click.option("--model_dump_filename", default="models/model.json", help="Chemin du modèle dumpé")
@click.option("--holdout_split", default=0.0, type=float, help="Split optionnel (ex: 0.2)")
@click.option("--holdout_output", default="data/test.csv", help="Où écrire le holdout si split > 0")
def train(input_filename, model_dump_filename, holdout_split, holdout_output):
    df = make_dataset(input_filename, expect_labels=True)
    X_text, y = make_features(df)

    if holdout_split and 0.0 < holdout_split < 1.0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_text, y, test_size=holdout_split, random_state=42, stratify=y
        )
        pd.DataFrame({"video_name": X_te, "is_comic": y_te}).to_csv(holdout_output, index=False)
        click.echo(f"[i] Holdout écrit: {holdout_output}")
        X_text, y = X_tr, y_tr

    model = make_model()
    model.fit(X_text, y)
    joblib.dump(model, model_dump_filename)
    click.echo(f"Modèle entraîné → {model_dump_filename}")


@cli.command()
@click.option("--input_filename", default="data/test.csv", help="CSV test (≥ video_name)")
@click.option("--model_dump_filename", default="models/model.json", help="Modèle dumpé")
@click.option("--output_filename", default="data/prediction.csv", help="CSV de sortie")
def predict(input_filename, model_dump_filename, output_filename):
    model = joblib.load(model_dump_filename)
    df = make_dataset(input_filename, expect_labels=False)
    if "video_name" not in df.columns:
        raise ValueError("Le CSV de prédiction doit contenir 'video_name'.")
    X_text = df["video_name"].astype(str).fillna("")
    y_pred = model.predict(X_text)

    out = pd.DataFrame({"video_name": X_text, "is_comic": y_pred})
    if hasattr(model, "predict_proba"):
        try:
            out["proba_is_comic"] = model.predict_proba(X_text)[:, 1]
        except Exception:
            pass
    out.to_csv(output_filename, index=False)
    click.echo(f"[✓] Prédictions écrites → {output_filename}")


@cli.command()
@click.option("--input_filename", default="data/raw/train.csv", help="CSV train (video_name,is_comic)")
def evaluate(input_filename):
    df = make_dataset(input_filename, expect_labels=True)
    X_text, y = make_features(df)
    model = make_model()

    # CV 5-fold (accuracy)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_text, y, cv=cv, scoring="accuracy")
    click.echo(f"[i] CV accuracy (5-fold): mean={scores.mean():.4f} ± {scores.std():.4f}")

    # Holdout 20% (métriques détaillées)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    click.echo(f"Accuracy (holdout 20%): {accuracy_score(y_te, y_pred):.4f}")
    click.echo(f"Confusion matrix:\n{confusion_matrix(y_te, y_pred)}")
    click.echo("\n" + classification_report(y_te, y_pred, digits=4))


if __name__ == "__main__":
    cli()
