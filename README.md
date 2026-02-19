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
3. Add NLP-derived features from clinical notes and quantify lift
4. Provide interpretability (coefficients / top n-grams; optional SHAP)
5. Package the workflow as clean, reusable code with clear documentation

---

## Repository Structure

```text
clinical-notes-risk-stratification/
  README.md
  .gitignore
  data/
    raw/                       # gitignored
    processed/                 # gitignored
  notebooks/
    01_eda.ipynb
    02_nlp_features.ipynb
    03_modeling.ipynb
  src/
    clinical_notes_rs/
      __init__.py
      config.py                # config constants (paths, seeds, etc.)
      data.py                  # data generation/loading
      features.py              # feature engineering (structured + NLP)
      model.py                 # training utilities
      eval.py                  # evaluation + reporting
  reports/
    figures/
  docs/
    one_pager.md               # stakeholder-facing summary
```
