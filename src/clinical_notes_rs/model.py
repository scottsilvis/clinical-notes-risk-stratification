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
from sklearn.compose import ColumnTransformer

# load_joined_clinical_data takes the variable processed_dir and uses it to locate the 
# patients.csv, notes.csv, and outcomes.csv files. These are loaded using the read_csv function in 
# the pandas library. because of the increasing complexity of this file, I have added sanity checks 
# to look for duplicate patient_ids. Before this, I remap the sex column from M/F to 1/0 to avoid 
# complicatoins in the models. The three dataframes are merged on the key patient_id. The type of 
# join in this case is an inner join for merging patients and outcomes, and a left join for merging 
# notes. This is because we want to keep all patients that have outcomes data, even if they are 
# missing notes. I then added sanity checks to ensure the required columns are present in each of 
# the three dataframes. If any of the sanity checks fail, a ValueError is raised with an 
# appropriate message. If all checks pass, the merged dataframe is returned.


def load_joined_clinical_data(processed_dir: Path) -> pandas.DataFrame:
    patients = pandas.read_csv(processed_dir / "patients.csv")
    outcomes = pandas.read_csv(processed_dir / "outcomes.csv")
    notes = pandas.read_csv(processed_dir / "notes.csv")

    sex_map = {"M": 1, "F": 0}
    patients["sex"] = patients["sex"].map(sex_map)

    if patients["patient_id"].duplicated().any():
        raise ValueError("patients.csv contains duplicate patient_id values.")
    if outcomes["patient_id"].duplicated().any():
        raise ValueError("outcomes.csv contains duplicate patient_id values.")
    if notes["patient_id"].duplicated().any():
        raise ValueError("notes.csv contains duplicate patient_id values.")

    df = patients.merge(outcomes, on="patient_id", how="inner")
    df = df.merge(notes, on="patient_id", how="left")

    if df.empty:
        raise ValueError("Joined dataframe is empty after merges.")
    if df["patient_id"].duplicated().any():
        raise ValueError("Joined dataframe has duplicate patient_id (merge may have created duplicates).")
    if df["note_text"].isna().any():
        raise ValueError("Missing note_text values after join.")
    if df["readmit_30d"].isna().any():
        raise ValueError("Missing readmit_30d values after join.")

    required_patients = {"patient_id", "age", "sex", "comorbidity_count", "prior_admits", "los_days"}
    required_outcomes = {"patient_id", "readmit_30d"}
    required_notes = {"patient_id", "note_text"}

    missing = required_patients - set(patients.columns)
    if missing:
        raise ValueError(f"patients.csv missing columns: {sorted(missing)}")

    missing = required_outcomes - set(outcomes.columns)
    if missing: 
        raise ValueError(f"outcomes.csv missing columns: {sorted(missing)}")

    missing = required_notes - set(notes.columns)
    if missing:
        raise ValueError(f"notes.csv missing columns: {sorted(missing)}")

    return df

# The function make_train_test_split takes a dataframe and splits it into a training and testing 
# set. The function accepts the following parameters: df (the input dataframe), target_col (the 
# name of the target column to stratify on) which defaults to "readmit_30d", test_size (the 
# percentage of the data to include in the test set) which defaults to 0.25, and seed (the random 
# state for reproducibility) which defaults to 7. The function uses the train_test_split function 
# from scikit-learn to perform the split, stratifying on the target column to preserve class 
# distribution. The function returns the training and testing dataframes.

def make_train_test_split(
    df: pandas.DataFrame,
    *,
    target_col: str = "readmit_30d",
    test_size: float = 0.25,
    seed: int = 7,
):
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
    )

    return train_df, test_df


# The function run_baseline takes the variable processed_dir and uses it to when it calls the 
# function load_joined_clinical_data. The function run_baseline also accepts the variable seed, 
# which it uses when calling the function make_train_test_split. The function accepts the datafame 
# returned by load_joined_clinical_data and immedately passes it to make_train_test_split. the 
# output of this is train_df and test_df. This is this split into X_train, y_train, X_test, and 
# y_test. A logistic regression model is then fit to the training data. The model is then used 
# to predict probabilities on the test set, and the ROC-AUC score is calculated and printed. A ROC 
# curve is then plotted using the true labels and predicted probabilities. The figure is saved to 
# the out_dir directory as baseline_roc.png.


def run_baseline(processed_dir: Path, out_dir: Path, seed: int = 7) -> None:
    df = load_joined_clinical_data(processed_dir)
    train_df, test_df = make_train_test_split(df, seed=seed)

    X_train = train_df[["age", "sex" , "comorbidity_count", "prior_admits", "los_days"]]
    y_train = train_df["readmit_30d"]

    X_test = test_df[["age", "sex", "comorbidity_count", "prior_admits", "los_days"]]
    y_test = test_df["readmit_30d"]

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
# function load_joined_clinical_data. The function run_text_baseline also accepts the variable 
# seed, which it uses when calling the function make_train_test_split. run_text_baseline accepts 
# the datafame returned by load_joined_clinical_data and immedately passes it to 
# make_train_test_split. The output of this is train_df and test_df. This is this split into 
# z_train, y_train, z_test, and y_test. The package scikit-learn is used to create a pipeline that 
# consists of two steps: a TfidfVectorizer and a LogisticRegression. The vectorizer is configured 
# to consider both unigrams and bigrams, convert all text to lowercase, and removes English stop 
# words. TF-IDF assigns each word or phrase a unique index and represents each document as a sparse 
# vector of TF-IDF weights rather than raw counts. These vectors are then used as inputs to the 
# classifier. A logistic regression model is then fit to this data along with the outcome data. The 
# model is then used to predict probabilities on the test set, and the ROC-AUC score is calculated 
# and printed. A ROC curve is then plotted using the true labels and predicted probabilities. The 
# figure is saved to the out_dir directory as roc_text_baseline.png.


def run_text_baseline(processed_dir: Path, out_dir: Path, seed: int = 7) -> None:
    df = load_joined_clinical_data(processed_dir)
    train_df, test_df = make_train_test_split(df, seed=seed)

    z_train = train_df["note_text"]
    y_train = train_df["readmit_30d"]

    z_test = test_df["note_text"]
    y_test = test_df["readmit_30d"]

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

# The function run_combined_model takes the variable processed_dir and uses it when it calls the 
# function load_joined_clinical_data. This function also accepts the variable seed, which it uses 
# when calling the function make_train_test_split. run_combined_model accepts the datafame 
# returned by load_joined_clinical_data and immedately passes it to make_train_test_split. The 
# output of this is train_df and test_df. This is this split into numeric and text features, which
# are used to make the X_train and X_test variables. The preprocessor is defined as a 
# ColumnTransformer that applies a TfidfVectorizer to the text feature (note_text) and passes 
# through the numeric features. The TfidfVectorizer is configured to consider both unigrams and 
# bigrams, convert all text to lowercase, and removes English stop words. The package scikit-learn 
# is used to create a pipeline that the above-mentioned preprocessing steps and a Logistic 
# Regression. The model is then used to predict probabilities on the test set, and the ROC-AUC 
# score is calculated and printed. A ROC curve is then plotted using the true labels and predicted 
# probabilities. The figure is saved to the out_dir directory as roc_combined.png.


def run_combined_model(processed_dir: Path, out_dir: Path, seed: int = 7) -> None:
    df = load_joined_clinical_data(processed_dir)
    train_df, test_df = make_train_test_split(df, seed=seed)

    y_train = train_df["readmit_30d"]
    y_test = test_df["readmit_30d"]

    numeric_features = ["age", "sex", "comorbidity_count", "prior_admits", "los_days"]
    text_feature = "note_text"

    X_train = train_df[numeric_features + [text_feature]]
    X_test = test_df[numeric_features + [text_feature]]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    stop_words="english",
                ),
                text_feature,
            ),
            (
                "numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    clf = Pipeline(
        steps=[
            (
                "preprocessor",preprocessor),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"[model_c: combined TF-IDF+LR] ROC-AUC: {auc:.3f}")

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve — model_c (combined TF-IDF + logistic regression) (AUC = {auc:.3f})")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "roc_combined.png", dpi=200, bbox_inches="tight")
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
    parser.add_argument("--combined", action="store_true", help="Run combined structured + text model (Model C).")
    args = parser.parse_args()

    root = Path.cwd()
    processed = root / "data" / "processed"
    out_dir = root / "reports" / "figures"

    if args.baseline:
        run_baseline(processed, out_dir)
    elif args.text_baseline:
        run_text_baseline(processed, out_dir)
    elif args.combined:
        run_combined_model(processed, out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
