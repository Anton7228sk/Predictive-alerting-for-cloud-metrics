# Predictive Alerting for Cloud Metrics

A machine learning system that predicts cloud service incidents **before they occur**, using historical metrics. Targets ≥80% incident recall with a controlled false positive rate and ~10 minute lead time.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f7931e)

---

## Problem Formulation

Binary classification with a sliding-window structure:

> Given the previous **W** time steps of cloud metrics, predict whether an incident will occur within the next **H** time steps.

- **W = 60 minutes** — lookback window
- **H = 15 minutes** — prediction horizon

Each sample is a feature vector derived from the W-step history at time *t*, labeled `1` if any incident starts in *(t, t+H]*. The window slides one step at a time, generating one labeled sample per timestamp with no lookahead leakage.

---

## Model

**IsolationForest + XGBoost ensemble** (`score = 0.3 × iso + 0.7 × xgb`)

| Model | Role | Strength | Weakness |
|-------|------|----------|----------|
| IsolationForest | Anomaly detector trained on normal data only | Catches novel failure modes without labels | High FPR on routine spikes |
| XGBoost | Supervised classifier with `scale_pos_weight` | High precision on known incident patterns | Cannot generalize beyond training distribution |
| Ensemble | Weighted combination | Better recall-precision tradeoff than either alone | — |

Features: rolling statistics, lag values, rate-of-change, percentile features (IQR, q95), and cyclical time encoding — all derived from the W-step window.

---

## Results

Evaluated on an 18-day held-out test set (80/20 temporal split, 8 incidents, 90-day synthetic dataset):

| Threshold | Recall | FPR    | Mean Lead Time |
|-----------|--------|--------|----------------|
| 0.40      | 1.000  | 0.0062 | 5.5 min        |
| 0.50      | 1.000  | 0.0005 | 5.2 min        |
| 0.60      | 1.000  | 0.0000 | 4.8 min        |
| **0.65**  | **1.000** | **0.0000** | **2.0 min** |
| 0.70      | 1.000  | 0.0000 | —              |
| 0.80      | 0.750  | 0.0000 | —              |

**ROC-AUC: 0.9499**

Recall is measured at incident-interval level: an incident counts as detected only if at least one alert fires *before* its start. Lead time at threshold 0.70+ is reported as — because alerts fire at incident onset rather than in the pre-incident window.

To reproduce: `python -c "from src.data.generator import generate_dataset; from src.training.pipeline import TrainingPipeline; import yaml; df = generate_dataset(n_days=90, seed=42); config = yaml.safe_load(open('config/config.yaml')); print(TrainingPipeline(config).run(df)['metrics']['at_default_threshold'])"`

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── generator.py        synthetic dataset with incident injection
│   │   └── preprocessing.py    sliding-window feature engineering
│   ├── models/
│   │   ├── base.py
│   │   ├── isolation_forest.py
│   │   ├── xgboost_model.py
│   │   └── ensemble.py
│   ├── training/
│   │   └── pipeline.py
│   └── evaluation/
│       └── metrics.py          incident-level recall, FPR, lead time
├── notebooks/
│   └── exploration.ipynb       full walkthrough with analysis
├── tests/
├── config/config.yaml
├── requirements.txt
└── requirements-dev.txt
```

---

## Quick Start

```bash
git clone <repo-url>
cd predictive-alerting-cloud-metrics
pip install -e ".[dev]"
```

**Train and evaluate:**

```python
import yaml
from src.data.generator import generate_dataset
from src.training.pipeline import TrainingPipeline

df = generate_dataset(n_days=90, freq_minutes=1, seed=42)

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

result = TrainingPipeline(config).run(df)
print(result["metrics"]["at_optimal_threshold"])
```

**Run tests:**

```bash
pytest tests/ -v
```

**Explore in Jupyter** (full analysis with plots and design rationale):

```bash
jupyter notebook notebooks/exploration.ipynb
```
