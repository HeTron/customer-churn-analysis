# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Churn Edge** — multi-page Streamlit app that predicts customer churn on the IBM Telco dataset, translates predictions to revenue-at-risk, and optimizes retention campaign sizing. See `README.md` for the public-facing overview.

This was rebuilt 2026-05-14 from a single-file LogisticRegression demo (`churn_app.py`, deleted) to a portfolio-grade app modeled on the sibling project `~/Projects/aibc/market-edge/`. If patterns are unclear, mirror Market Edge.

## Commands

Use the project venv at `.venv` (Python 3.11, inherits from miniconda).

```bash
source .venv/bin/activate
pip install -r requirements.txt          # first time / after dep changes
streamlit run churn_app.py                # local dev — opens on :8501
pytest tests/ -q                          # full test suite (32 tests)
pytest tests/test_features.py::TestBuildFeatureMatrix::test_no_nan_in_X -q   # single test
```

Smoke test the pipeline end-to-end without launching Streamlit:

```bash
python -c "from src.data import load_clean; from src.features import build_feature_matrix; df = load_clean(); X, y = build_feature_matrix(df); print(X.shape, round(y.mean(), 4))"
# Expected: (7032, 41) 0.2658
```

## Architecture

### Module purity rule (load-bearing)

`src/features.py` and `src/business.py` are **pure** — no Streamlit imports, no I/O, no caching decorators. They must stay importable from plain Python. Tests rely on this.

`src/data.py`, `src/model.py`, `src/explain.py` may import Streamlit because they own caching (`@st.cache_data`, `@st.cache_resource`). Their pure helpers should still be Streamlit-free; only the cached convenience wrappers (`load_clean`, `train_model`, `explain_churn`) depend on Streamlit.

If a future change wants to call a function from a script or notebook, it goes in the pure modules.

### Feature engineering round-trip

`build_feature_matrix(df)` returns `(X, y)` after one-hot encoding (`pd.get_dummies(drop_first=True)`). Column count = 41. Single-customer prediction on the Predict page goes through `prepare_single_customer(customer_dict, feature_cols)` which reindexes the encoded customer row to match the training column space exactly (missing dummies zero-filled, extras dropped). Breaking this contract silently misaligns features at predict time — tests guard the round-trip.

### Multi-page Streamlit layout

`churn_app.py` is the landing page (Streamlit Cloud's expected main file — kept this name for deploy compatibility; conceptually it's the same role as `streamlit_app.py` in Market Edge). Pages auto-discovered from `pages/` in filename order:

1. `1_📊_Cohort_Analysis.py` — filtered EDA with Plotly
2. `2_🤖_Predict.py` — single-customer scoring + Claude retention recommendation
3. `3_💰_Campaign_Optimizer.py` — batch scoring + ROI vs campaign-size curve (the differentiator)

All pages use `template="plotly_dark"` and the primary color `#00D4AA` (defined in `.streamlit/config.toml`).

### Cross-validation honesty caveat

`pages/3_Campaign_Optimizer.py` trains on the full dataset and scores the same rows — that's **in-sample** by design (we want a probability per customer for ranking, not a generalization estimate). The CV metrics shown on the Predict page come from `cross_validated_metrics()` and are the honest ones. `cross_validated_metrics()` also returns `out_of_fold_probas` — if a future change wants ranking to be honest too, swap the in-sample probas in the Optimizer page for OOF probas.

### Claude integration

`src/explain.py` uses `model = "claude-sonnet-4-6"`. When `ANTHROPIC_API_KEY` is missing, both helpers return a markdown fallback string — the rest of the app keeps working. Don't add hard failures around the API call.

## Deployment

Live URL: `https://hetron-customer-churn-analysis.streamlit.app` (Streamlit Cloud, repo `HeTron/customer-churn-analysis`).

Streamlit Cloud's main file is locked to `churn_app.py` on existing apps (the setting was removed from the dashboard). The landing page lives at that filename for that reason. Add `ANTHROPIC_API_KEY` under Settings → Secrets in the dashboard.

## Things not to touch

- `churn_data.csv` — the raw IBM Telco dataset, used by tests and the app.
- `churn_eda_model.ipynb` — kept as a portfolio artifact showing the original EDA process.
- `images/` — README screenshots. Tracked, not gitignored.
- `generate_data.py` — kept for reproducibility (re-fetches the public IBM Telco CSV).

## Gotchas

- The venv's `.venv/bin/python` resolves `sys.prefix` to `/Users/jocksolo/miniconda3` — it inherits site-packages. New deps (`lightgbm`, `plotly`, `anthropic`) are installed and importable; pandas warnings about old `numexpr`/`bottleneck` versions in miniconda's site-packages are cosmetic and don't affect Streamlit Cloud (which builds its own env).
- The 11 customers with blank `TotalCharges` are dropped during cleaning — final row count is 7,032, not 7,043. Tests assume this.
- `class_weight="balanced"` is set on LGBMClassifier because the dataset is ~26.6% churn. Removing it will silently degrade recall on the minority class.
