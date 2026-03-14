# Predictive Alerting for Cloud Metrics

A machine learning system that predicts cloud service incidents **before they occur**, using historical AWS CloudWatch metrics. Achieves ~85% incident recall with a mean lead time of 10 minutes.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f7931e)

---

## Problem Formulation

Binary classification with a sliding-window structure:

> Given the previous **W** time steps of cloud metrics, predict whether an incident will occur within the next **H** time steps.

- **W = 60 minutes** — lookback window (`features.window_size` in config)
- **H = 15 minutes** — prediction horizon (`features.lookahead_minutes` in config)

Each sample is a feature vector derived from the W-step history at time *t*, labeled `1` if any incident starts in *(t, t+H]*.

---

## Architecture

```
CloudWatch Metrics
        │
        ▼
  Feature Engineering         rolling stats · lag features · rate-of-change
  (sliding window W=60m)      percentiles · cyclical time encoding
        │
        ├──► IsolationForest   learns normal operating envelope (unsupervised)
        │
        └──► XGBoost           learns incident precursors (supervised)
                │
                └──► Ensemble Score = 0.3 × iso_score + 0.7 × xgb_proba
                            │
                            ▼
                      AlertManager ──► SNS / Webhook

Lambda (daily)   → retrains model on last 30 days → saves to S3
Lambda (1/min)   → fetches latest metrics → scores → triggers alert if score > threshold
```

---

## Model Selection

Cloud metrics are **non-stationary** and **heavy-tailed**, which makes a single model insufficient:

| Challenge | Solution |
|-----------|----------|
| Non-stationarity / distribution shift | IsolationForest detects deviations from current normal baseline without requiring labels |
| Heavy-tailed latency / error distributions | Percentile features (IQR, q95) robust to outliers; XGBoost tree structure handles non-linear thresholds |
| Class imbalance (~2% incidents) | `scale_pos_weight` in XGBoost; AUCPR metric for early stopping |
| Novel anomalies not in training set | Isolation Forest provides recall coverage beyond XGBoost's learned patterns |

The weighted ensemble (30% IF + 70% XGB) balances recall and precision better than either model alone.

---

## Results

Evaluated on a held-out 18-day test set (80/20 temporal split, ~72 incidents):

| Threshold | Recall | FPR  | Mean Lead Time |
|-----------|--------|------|----------------|
| 0.50      | ~0.92  | ~0.08 | ~12 min       |
| **0.65**  | **~0.85** | **~0.04** | **~10 min** |
| 0.70      | ~0.80  | ~0.02 | ~9 min        |
| 0.80      | ~0.65  | ~0.01 | ~8 min        |

**ROC-AUC: ~0.94**

*Recall is measured at incident-interval level: an incident is counted as detected if at least one alert fires before its start. Run `notebooks/exploration.ipynb` to reproduce.*

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── generator.py        synthetic dataset with realistic incident injection
│   │   ├── cloudwatch.py       AWS CloudWatch data fetching via boto3
│   │   └── preprocessing.py    sliding-window feature engineering
│   ├── models/
│   │   ├── isolation_forest.py
│   │   ├── xgboost_model.py
│   │   └── ensemble.py
│   ├── training/
│   │   └── pipeline.py         end-to-end train → evaluate → save
│   ├── evaluation/
│   │   └── metrics.py          recall, FPR, lead time, threshold search
│   └── alerting/
│       └── alert_manager.py    cooldown logic, SNS / webhook dispatch
├── lambda/
│   ├── retrain/handler.py      daily retraining Lambda
│   └── inference/handler.py   per-minute inference Lambda
├── notebooks/
│   └── exploration.ipynb       full walkthrough with plots
├── tests/                      43 unit tests
└── config/config.yaml
```

---

## Quick Start

```bash
git clone <repo-url>
cd predictive-alerting-cloud-metrics
pip install -e ".[dev]"
```

**Train on synthetic data:**

```python
import yaml
from src.data.generator import generate_dataset
from src.training.pipeline import TrainingPipeline

df = generate_dataset(n_days=90, freq_minutes=1, seed=42)

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)
config["storage"]["type"] = "local"

result = TrainingPipeline(config).run(df)
print(f'Recall: {result["metrics"]["at_optimal_threshold"]["recall"]:.3f}')
print(f'ROC-AUC: {result["metrics"]["roc_auc"]:.4f}')
```

**Run tests:**

```bash
pytest tests/ -v
```

**Explore in Jupyter:**

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## AWS Deployment

### Infrastructure

| Resource | Purpose |
|----------|---------|
| S3 bucket | Model artifacts and config storage |
| Lambda (daily) | Retraining on latest 30 days of metrics |
| Lambda (1/min) | Inference and alert triggering |
| EventBridge | Cron scheduling for both Lambdas |
| SNS topic | Alert delivery |
| CloudWatch | Anomaly score logging |

### Deploy

```bash
# Upload config
aws s3 cp config/config.yaml s3://predictive-alerting-artifacts/config/config.yaml

# Package and deploy retrain Lambda
cd lambda && pip install -r requirements.txt -t package/
cp retrain/handler.py package/ && cd package && zip -r ../retrain.zip .
aws lambda create-function \
  --function-name predictive-alerting-retrain \
  --runtime python3.10 --handler handler.handler \
  --zip-file fileb://../retrain.zip \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-role \
  --environment Variables="{CONFIG_BUCKET=predictive-alerting-artifacts}"

# Schedule daily retraining at 02:00 UTC
aws events put-rule --name daily-retrain \
  --schedule-expression "cron(0 2 * * ? *)" --state ENABLED

# Schedule per-minute inference
aws events put-rule --name per-minute-inference \
  --schedule-expression "rate(1 minute)" --state ENABLED
```

**Required IAM permissions:** `cloudwatch:GetMetricStatistics`, `cloudwatch:PutMetricData`, `s3:GetObject/PutObject/ListBucket`, `sns:Publish`.

---

## Evaluation Methodology

**Incident-level recall** — an incident is a contiguous block of `incident=True` timesteps. Recall counts the fraction of such intervals that received at least one alert before their start (not just during):

```
Recall = detected incident intervals / total incident intervals
```

**Lead time** — time between the last pre-incident alert and the start of the incident interval. Reported as mean and median.

**False positive rate** — fraction of non-incident timesteps where the score exceeds the threshold.

**Threshold selection** — grid search over [0.01, 0.99] to find the threshold closest to the target recall (default: 0.80).

---

## Limitations & Future Work

- **No concept drift detection** — daily retraining mitigates but does not actively detect distribution shift
- **Fixed W and H** — different incident types may benefit from adaptive lookahead horizons
- **Synthetic calibration** — threshold recommendations should be recalibrated on labeled production data
- **No root cause attribution** — SHAP values could be added to alert messages to indicate which metrics drove the score
- **Single-service scope** — fleet-level aggregation across multiple instances is not implemented
