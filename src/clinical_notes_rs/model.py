from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split

# load_structured_data takes the variable processed_dir and uses it to locate the patients.csv and outcomes.csv files. 
# These are loaded using the read_csv function in the pandas library. 
# The two dataframes are merged on the key patient_id. The type of join in this case is an inner join. 
# This is done to make sure all datapoints are properly aligned. 
# The dataset is then split into X (age, comorbidity_count, prior_admits, los_days) and y (readmit_30d).
# A tuple of X,y is then returned by the function. 


def load_structured_data(processed_dir: Path):
    patients = pandas.read_csv(processed_dir / "patients.csv")
    outcomes = pandas.read_csv(processed_dir / "outcomes.csv")

    df = patients.merge(outcomes, on="patient_id", how="inner")

    X = df[["age", "comorbidity_count", "prior_admits", "los_days"]]
    y = df["readmit_30d"]

    return X, y


# run_baseline takes the variable processed_dir and uses it to when it calls the function load_structured_data.
# run_baseline also accepts the variable seed, which it uses to instill a random state for reproducibility. 
# The function accepts the tuple returned by load_structured_data and assigns it to X and y. 
# X and y are then divided into a testing and training dataset, where 25% of the data is stored as X_test or y_test. 
# The remaining 75% of the data is stored as X_train and y_train. 
# Because the outcome is imbalanced, stratification preserves the class distribution so the test set is 
# representative and evaluation is stable.
# A logistic regression model is then fit to the training data.
# The model is then used to predict probabilities on the test set, and the ROC-AUC score is calculated and printed. 
# A ROC curve is then plotted using the true labels and predicted probabilities. 
# The figure is saved to the out_dir directory as baseline_roc.png.


def run_baseline(processed_dir: Path, out_dir: Path, seed: int = 7) -> None:
    X, y = load_structured_data(processed_dir)

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

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"Baseline Logistic Regression (AUC = {auc:.3f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "baseline_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# The main function is the entry point of the script. It uses argparse to parse command line arguments.
# If the --baseline flag is provided, it calls the run_baseline function with the appropriate directories. 
# If the flag is not provided, it prints the help message.

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run structured-only baseline model")
    args = parser.parse_args()

    root = Path.cwd()
    processed = root / "data" / "processed"
    out_dir = root / "reports" / "figures"

    if args.baseline:
        run_baseline(processed, out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
