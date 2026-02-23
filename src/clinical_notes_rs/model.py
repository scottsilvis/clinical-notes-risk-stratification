from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# load_structured_data takes the variable processed_dir and uses it to locate the patients.csv and 
# outcomes.csv files. These are loaded using the read_csv function in the pandas library. The two 
# dataframes are merged on the key patient_id. The type of join in this case is an inner join. This 
# is done to make sure all datapoints are properly aligned. The dataset is then split into X (age, 
# comorbidity_count, prior_admits, los_days) and y (readmit_30d).A tuple of X,y is then returned by 
# the function. 


def load_structured_data(processed_dir: Path):
    patients = pandas.read_csv(processed_dir / "patients.csv")
    outcomes = pandas.read_csv(processed_dir / "outcomes.csv")
    notes = pandas.read_csv(processed_dir / "notes.csv")

    df = patients.merge(outcomes, on="patient_id", how="inner")
    df = df.merge(notes, on="patient_id", how="left")

    if df["note_text"].isna().any():
        raise ValueError("Missing note_text values after join.")
    if df["readmit_30d"].isna().any():
        raise ValueError("Missing readmit_30d values after join.")

    X = df[["age", "comorbidity_count", "prior_admits", "los_days"]]
    y = df["readmit_30d"]
    z = df["note_text"]

    return X, y, z


# The function run_baseline takes the variable processed_dir and uses it to when it calls the 
# function load_structured_data. The function run_baseline also accepts the variable seed, which it 
# uses to instill a random state for reproducibility. The function accepts the tuple returned by 
# load_structured_data and assigns it to X and y. The variables X and y are then divided into a 
# testing and training dataset, where 25% of the data is stored as X_test or y_test. The remaining 
# 75% of the data is stored as X_train and y_train. Because the outcome is imbalanced, 
# stratification preserves the class distribution so the test set is representative and evaluation 
# is stable. A logistic regression model is then fit to the training data. The model is then used 
# to predict probabilities on the test set, and the ROC-AUC score is calculated and printed. A ROC 
# curve is then plotted using the true labels and predicted probabilities. The figure is saved to 
# the out_dir directory as baseline_roc.png.


def run_baseline(processed_dir: Path, out_dir: Path, seed: int = 7) -> None:
    X, y, _ = load_structured_data(processed_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"Baseline ROC-AUC: {auc:.3f}")

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"Baseline Logistic Regression (AUC = {auc:.3f})")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "baseline_roc.png", dpi=150, bbox_inches="tight")
    plt.close()

# The function run_text_baseline takes the variable processed_dir and uses it when it calls the 
# function load_structured_data. The function run__text_baseline also accepts the variable seed, 
# which it uses to instill a random state for reproducibility. The function accepts the tuple 
# returned by load_structured_data and assigns it to y and z. The first variablein the tuple is X, 
# which is purposfully dropped during assignment. The variables z and y are then divided into a 
# testing and training dataset, where 25% of the data is stored as z_test or y_test. The remaining 
# 75% of the data is stored as z_train and y_train. Because the outcome may be imbalanced, 
# stratification preserves the class distribution so the test set is representative and evaluation 
# is stable. The package scikit-learn is used to create a pipeline that consists of two steps: a 
# TfidfVectorizer and a LogisticRegression. The vectorizer is configured to consider both unigrams 
# and bigrams, convert all text to lowercase, and removes English stop words. TF-IDF assigns each 
# word or phrase a unique index and represents each document as a sparse vector of TF-IDF weights 
# rather than raw counts. These vectors are then used as inputs to the classifier. A logistic regression 
# model is then fit to this data along with the outcome data. The model is then used 
# to predict probabilities on the test set, and the ROC-AUC score is calculated and printed. A ROC 
# curve is then plotted using the true labels and predicted probabilities. The figure is saved to 
# the out_dir directory as baseline_roc.png.


def run_text_baseline(processed_dir: Path, out_dir: Path, seed: int = 7) -> None:
    _, y, z = load_structured_data(processed_dir)

    z_train, z_test, y_train, y_test = train_test_split(
        z,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    clf = Pipeline(
        steps=[
            (
                "tfidf", 
                TfidfVectorizer(
                    ngram_range=(1, 2),   # unigrams + bigrams
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )

    clf.fit(z_train, y_train)

    y_proba = clf.predict_proba(z_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"[model_b: text-only TF-IDF+LR] ROC-AUC: {auc:.3f}")

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve — model_b (text-only TF-IDF + logistic regression) (AUC = {auc:.3f})")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "roc_text_baseline.png", dpi=200, bbox_inches="tight")
    plt.close()

    return auc


# The main function is the entry point of the script. It uses argparse to parse command line 
# arguments. If the --baseline flag is provided, it calls the run_baseline function with the 
# appropriate directories. If the --text-baseline flag is provided, it calls the run_text_baseline function with the 
# appropriate directories. If neither flag is provided, it prints the help message.

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run structured-only baseline model")
    parser.add_argument("--text-baseline", action="store_true", help="Run text-only baseline model")
    args = parser.parse_args()

    root = Path.cwd()
    processed = root / "data" / "processed"
    out_dir = root / "reports" / "figures"

    if args.baseline:
        run_baseline(processed, out_dir)
    elif args.text_baseline:
        run_text_baseline(processed, out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
