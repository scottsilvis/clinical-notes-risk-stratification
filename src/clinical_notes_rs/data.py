from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    processed: Path

    @staticmethod
    def from_cwd() -> "Paths":
        # assumes you run commands from repo root
        root = Path.cwd()
        processed = root / "data" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        return Paths(repo_root=root, processed=processed)


def _make_note(rng: np.random.Generator, risk: float) -> str:
    """Generate a synthetic discharge-style note with risk-correlated phrases."""
    base = [
        "Discharge summary:",
        "Patient tolerated procedures.",
        "Follow-up with primary care.",
    ]

    low_risk_phrases = [
        "stable condition",
        "no complications",
        "medication adherence",
        "supportive family",
        "denies shortness of breath",
    ]
    high_risk_phrases = [
        "poor adherence",
        "limited social support",
        "frequent ED visits",
        "missed follow-up",
        "uncontrolled symptoms",
        "transportation barrier",
    ]

    # Mix phrases based on risk
    n = rng.integers(4, 9)
    phrases = []
    for _ in range(n):
        if rng.random() < risk:
            phrases.append(rng.choice(high_risk_phrases))
        else:
            phrases.append(rng.choice(low_risk_phrases))

    # Add a little noise / clinician-ish filler
    filler = [
        "Plan discussed with patient.",
        "Return precautions reviewed.",
        "Labs reviewed.",
        "Care coordination notified.",
    ]
    if rng.random() < 0.6:
        phrases.append(rng.choice(filler))

    return " ".join(base + phrases)


def generate_synthetic(n_patients: int = 1500, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    patient_id = np.arange(1, n_patients + 1)

    age = rng.normal(58, 16, size=n_patients).clip(18, 90).round().astype(int)
    sex = rng.choice(["F", "M"], size=n_patients, p=[0.52, 0.48])
    comorbidity_count = rng.poisson(lam=2.2, size=n_patients).clip(0, 10)
    prior_admits = rng.poisson(lam=0.8, size=n_patients).clip(0, 8)
    los_days = rng.gamma(shape=2.0, scale=2.0, size=n_patients).clip(1, 21).round(1)

    # risk score drives both note content and outcome probability
    risk_score = (
        0.03 * (age - 50)
        + 0.35 * comorbidity_count
        + 0.45 * prior_admits
        + 0.10 * (los_days - 3)
        + rng.normal(0, 0.8, size=n_patients)
    )

    # Convert to probability via logistic
    p_readmit = 1 / (1 + np.exp(-(risk_score - 1.2)))
    readmit_30d = (rng.random(n_patients) < p_readmit).astype(int)

    patients = pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "comorbidity_count": comorbidity_count,
            "prior_admits": prior_admits,
            "los_days": los_days,
        }
    )

    notes = pd.DataFrame(
        {
            "patient_id": patient_id,
            "note_text": [_make_note(rng, float(p)) for p in p_readmit],
        }
    )

    outcomes = pd.DataFrame(
        {
            "patient_id": patient_id,
            "readmit_30d": readmit_30d,
        }
    )

    return patients, notes, outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate synthetic processed CSVs")
    parser.add_argument("--n_patients", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    paths = Paths.from_cwd()

    if args.generate:
        patients, notes, outcomes = generate_synthetic(n_patients=args.n_patients, seed=args.seed)
        patients.to_csv(paths.processed / "patients.csv", index=False)
        notes.to_csv(paths.processed / "notes.csv", index=False)
        outcomes.to_csv(paths.processed / "outcomes.csv", index=False)
        print("Wrote:")
        print(" - data/processed/patients.csv")
        print(" - data/processed/notes.csv")
        print(" - data/processed/outcomes.csv")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
