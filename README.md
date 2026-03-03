# Clinical Notes Risk Stratification

Predict a clinically relevant risk signal (default: 30-day readmission) using a combination of:
- Structured patient data (demographics / utilization / comorbidity proxy)
- Unstructured clinical notes (NLP features via TF-IDF; optional LLM-based extraction)

This repository is designed as a quick demonstration of:
- Working with heterogeneous healthcare-style data
- Building reproducible analytical workflows in Python
- Applying baseline + ensemble modeling approaches
- Producing interpretable insights for clinical/operational stakeholders
- Framing lightweight MLOps and deployment considerations

> Note: The default dataset used here is synthetic (generated locally) to avoid PHI and access constraints while preserving realistic patterns and noise.

---

## Project Goals

1. Create a realistic, privacy-safe clinical analytics dataset (structured + notes)
2. Build a baseline model using structured data only
3. Add NLP-derived features from clinical notes and quantify incremental predictive lift beyond structured data
4. Provide interpretable model outputs (coefficients, top n-grams) suitable for clinical and operational stakeholders
5. Package the workflow as clean, reusable code with clear documentation

---

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── data/
│   ├── raw/                       # gitignored
│   └── processed/                 # gitignored
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_nlp_features.ipynb
    └── 03_modeling.ipynb
├── src/
│   └── clinical_notes_rs/
│       ├── __init__.py
│       ├── config.py              # config constants (paths, seeds, etc.)
│       ├── data.py                # data generation/loading
│       ├── features.py            # feature engineering (structured + NLP)
│       ├── model.py               # training utilities
│       └── eval.py                # evaluation + reporting
├── reports/
│   └── figures/
├── docs/
│   └── one_pager.md               # stakeholder-facing summary
```
---

## Reproducibility

All data used in this project is generated locally.

To reproduce results:

```bash
python -m clinical_notes_rs.data          # generate synthetic data
python -m clinical_notes_rs.model --baseline
python -m clinical_notes_rs.model --text-baseline
python -m clinical_notes_rs.model --combined
```

---

## Results at a Glance

All models were evaluated on the same stratified train/test split (75/25) using ROC-AUC.

| Model | Features | ROC-AUC |
|------|---------|--------|
| Model A | Structured data only | 0.670 |
| Model B | Clinical notes only (TF-IDF) | 0.679 |
| Model C | Structured + clinical notes | 0.687 |

**Conclusion:**  
Unstructured clinical notes encode outcome-relevant information and provide incremental predictive value beyond structured patient data.

---

## Future Extensions

- Contextual text embeddings (e.g., clinical BERT models)
- Temporal modeling of note sequences
- Model packaging for API-based decision support
- Monitoring model performance drift in production

---

## Healthcare Context & Limitations

This project uses synthetic data to mirror common challenges in healthcare analytics, including noisy documentation, heterogeneous data sources, and imperfect proxies for patient complexity.

While performance metrics are illustrative, the primary objective is to demonstrate:
- Sound analytical design
- Safe handling of unstructured clinical text
- Interpretable modeling approaches appropriate for healthcare settings