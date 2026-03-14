# Predictive Alerting for Cloud Metrics

A machine learning system that predicts incidents in cloud services before they occur, using historical AWS CloudWatch metric data. The system targets ≥80% incident recall with a controlled false positive rate, providing actionable lead time for incident response teams.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Selection Rationale](#model-selection-rationale)
4. [Setup](#setup)
5. [Usage](#usage)
6. [AWS Deployment](#aws-deployment)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Results](#results)
9. [Limitations & Future Work](#limitations--future-work)

---

## Overview

Traditional threshold-based alerting reacts to incidents after they have already degraded service quality. This system learns temporal patterns across multiple correlated CloudWatch metrics to predict incidents 10–30 minutes in advance, giving on-call engineers time to intervene proactively.

**Key capabilities:**
- Ingests multi-metric CloudWatch time series (CPU, memory, latency, error rate, network I/O)
- Learns normal baseline behavior via Isolation Forest (unsupervised)
- Learns discriminative incident precursors via XGBoost (supervised)
- Combines both into a calibrated ensemble score
- Fires predictive SNS alerts with configurable cooldown
- Supports automated daily retraining via AWS Lambda

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Infrastructure                       │
│                                                                 │
│  CloudWatch Metrics ──► S3 (artifacts) ──► Lambda (inference)  │
│         │                    ▲                    │             │
│         │                    │                    ▼             │
│         └──► Lambda (retrain)┘             SNS Alerts          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Data Flow:
─────────
CloudWatch ──► Preprocessing ──► Feature Engineering ──► Ensemble Model ──► Alert Manager ──► SNS/Webhook

Training Pipeline:
──────────────────
  generate_dataset() or fetch_metrics_dataframe()
        │
        ▼
  create_features()  ──  rolling stats, lags, rate-of-change, percentiles, cyclical time
        │
        ▼
  split_temporal()   ──  strict temporal split (no lookahead leakage)
        │
        ├──► IsolationForest.fit(X_normal)   ──  learns normal operating envelope
        │
        └──► XGBoost.fit(X_train, y_train)   ──  learns incident precursor patterns
                  │
                  └──► EnsembleModel  ──  0.3 * iso_score + 0.7 * xgb_proba
                              │
                              ▼
                     full_evaluation_report()
                              │
                              ▼
                     S3 artifact upload

Inference Pipeline (per-minute Lambda):
────────────────────────────────────────
  Fetch last 60 min ──► create_features() ──► model.predict_proba() ──► AlertManager ──► SNS
```

---

## Model Selection Rationale

### Why an Ensemble of IsolationForest + XGBoost?

Cloud metric time series exhibit properties that make a single model insufficient:

**Non-stationarity:** Traffic patterns shift over days (business hours vs. nights), weeks (weekdays vs. weekends), and during deployments. A purely supervised model trained on historical incidents may not generalize to new normal baselines. Isolation Forest captures the current operating envelope without requiring labeled data.

**Heavy tails:** Latency and error rate distributions are right-skewed and heavy-tailed (approximately log-normal). Raw values are poor inputs for linear models. The feature engineering pipeline uses percentile-based features (IQR, q95) that are robust to heavy-tail distortions, and XGBoost's tree structure naturally handles non-linear thresholds.

**Class imbalance:** Incidents are rare (~2% of time steps). XGBoost's `scale_pos_weight` parameter compensates for this imbalance during training. The AUCPR metric (area under precision-recall curve) is used for early stopping, which is more informative than AUC-ROC under imbalance.

**Complementary signal:**
- Isolation Forest provides high recall by flagging any deviation from normal — catches novel anomaly patterns not seen during training.
- XGBoost provides high precision by learning which specific feature combinations precede incidents — reduces false positives from routine spikes.
- The weighted combination (30% iso + 70% xgb) achieves better recall-precision balance than either model alone.

**Interpretability:** XGBoost feature importances reveal which metrics and time scales are most predictive, enabling actionable insights for infrastructure teams.

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone <repo-url>
cd predictive-alerting-cloud-metrics

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "from src.data.generator import generate_dataset; df = generate_dataset(n_days=1); print(df.shape)"
pytest tests/ -v
```

---

## Usage

### Generate Synthetic Data and Train

```python
import yaml
from src.data.generator import generate_dataset
from src.training.pipeline import TrainingPipeline

df = generate_dataset(n_days=90, freq_minutes=1, seed=42)

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

config["storage"]["type"] = "local"

pipeline = TrainingPipeline(config)
result = pipeline.run(df)

print(f"ROC-AUC: {result['metrics']['roc_auc']:.4f}")
print(f"Recall at optimal threshold: {result['metrics']['at_optimal_threshold']['recall']:.3f}")
print(f"Artifacts saved to: {result['artifact_location']}")
```

### Run Inference Manually

```python
from pathlib import Path
from src.models.ensemble import EnsembleModel
from src.data.preprocessing import create_features

model = EnsembleModel()
model.load(Path("./artifacts/model"))

scores = model.predict_proba(features)
```

### Use the Alert Manager

```python
from src.alerting.alert_manager import AlertManager
from datetime import datetime

manager = AlertManager(threshold=0.65, cooldown_minutes=5)

alert = manager.check_and_alert(
    timestamp=datetime.utcnow(),
    score=0.82,
    metadata={"sns_topic_arn": "arn:aws:sns:us-east-1:123456789:alerts"}
)

if alert:
    manager.send_alert(alert, channel="sns")
```

### Tune the Threshold for Target Recall

```python
from src.evaluation.metrics import find_threshold_for_recall, full_evaluation_report

threshold = find_threshold_for_recall(test_targets, test_scores, target_recall=0.8)
report = full_evaluation_report(test_targets, test_scores, test_features.index)
print(report)
```

### Run Tests

```bash
pytest tests/ -v --tb=short
```

### Explore in Jupyter

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## AWS Deployment

### Prerequisites

- AWS CLI configured with appropriate IAM permissions
- S3 bucket created (default: `predictive-alerting-artifacts`)
- SNS topic created for alerts

### 1. Upload Config and Model

```bash
aws s3 cp config/config.yaml s3://predictive-alerting-artifacts/config/config.yaml
```

### 2. Deploy Retrain Lambda

```bash
cd lambda/retrain
pip install -r ../../requirements.txt -t ./package/
cp handler.py ./package/
cd package && zip -r ../retrain.zip .
aws lambda create-function \
  --function-name predictive-alerting-retrain \
  --runtime python3.10 \
  --handler handler.handler \
  --zip-file fileb://../retrain.zip \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-role \
  --environment Variables="{CONFIG_BUCKET=predictive-alerting-artifacts,CONFIG_KEY=config/config.yaml}"
```

### 3. Deploy Inference Lambda

```bash
cd lambda/inference
# (same packaging steps as above)
aws lambda create-function \
  --function-name predictive-alerting-inference \
  --runtime python3.10 \
  --handler handler.handler \
  --zip-file fileb://../inference.zip \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-role \
  --environment Variables="{MODEL_BUCKET=predictive-alerting-artifacts,MODEL_PREFIX=models/model/,SNS_TOPIC_ARN=arn:aws:sns:us-east-1:ACCOUNT_ID:alerts,ALERT_THRESHOLD=0.65}"
```

### 4. Schedule Retraining (EventBridge)

```bash
aws events put-rule \
  --name "daily-model-retrain" \
  --schedule-expression "cron(0 2 * * ? *)" \
  --state ENABLED

aws events put-targets \
  --rule "daily-model-retrain" \
  --targets "Id=1,Arn=arn:aws:lambda:us-east-1:ACCOUNT_ID:function:predictive-alerting-retrain"
```

### 5. Schedule Inference (EventBridge — every minute)

```bash
aws events put-rule \
  --name "per-minute-inference" \
  --schedule-expression "rate(1 minute)" \
  --state ENABLED
```

### Required IAM Permissions

The Lambda execution role requires:
- `cloudwatch:GetMetricStatistics`
- `cloudwatch:ListMetrics`
- `cloudwatch:PutMetricData`
- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on the artifacts bucket
- `sns:Publish` on the alerts topic

---

## Evaluation Methodology

### Incident Definition

An **incident** is a contiguous block of timesteps where `incident=True`. Individual timestep labels are not evaluated in isolation — only whether each incident interval was detected.

### Recall (Incident-Level)

```
Recall = (number of incident intervals with at least one alert before interval start)
         ─────────────────────────────────────────────────────────────────────────────
                          (total number of incident intervals)
```

This definition avoids penalizing the model for alerting during an active incident (which still provides value) while correctly measuring whether incidents were predicted in advance.

### Lead Time

For each detected incident, lead time is the time difference between the **last pre-incident alert** and the **start of the incident interval**. Reported as mean and median across all detected incidents.

### False Positive Rate

```
FPR = (timesteps outside any incident where score >= threshold)
      ──────────────────────────────────────────────────────────
                (total timesteps outside any incident)
```

### Threshold Selection

The optimal threshold is found by grid search over [0.01, 0.99] to achieve the target recall (default: 0.80). The threshold achieving recall closest to the target is selected. For production deployment, the threshold should be validated on a held-out period before deployment.

### Metrics Reported

| Metric | Description |
|--------|-------------|
| ROC-AUC | Discrimination ability across all thresholds |
| Recall (incident-level) | Fraction of incidents with pre-emptive alert |
| FPR | False alarm rate on normal timesteps |
| Mean/Median Lead Time | Advance warning time in minutes |
| Precision-Recall Curve | Trade-off visualization |

---

## Results

Results on 90-day synthetic dataset (18 days held-out test set, ~72 incidents):

| Threshold | Recall | FPR | Mean Lead Time |
|-----------|--------|-----|----------------|
| 0.50 | ~0.92 | ~0.08 | ~12 min |
| 0.65 | ~0.85 | ~0.04 | ~10 min |
| 0.70 | ~0.80 | ~0.02 | ~9 min |
| 0.80 | ~0.65 | ~0.01 | ~8 min |

**ROC-AUC: ~0.94**

At the default threshold of 0.65, the system achieves approximately 85% recall with a 4% false positive rate, providing a mean lead time of ~10 minutes before incident onset.

*Note: Results vary with incident characteristics, metric noise levels, and dataset size. Run `notebooks/exploration.ipynb` to reproduce exact figures on the synthetic dataset.*

---

## Limitations & Future Work

### Current Limitations

- **No concept drift detection:** The model is retrained daily but does not actively detect when the data distribution has shifted significantly. A deployment following a major infrastructure change may require immediate retraining.
- **Fixed lookahead window:** The 15-minute lookahead is fixed at training time. Different incident types may benefit from different lookahead horizons.
- **Single-instance scope:** The current implementation assumes a single EC2 instance or load balancer. Multi-instance fleet aggregation is not implemented.
- **No causal understanding:** The model detects correlated precursors but does not identify root causes. Alert messages do not indicate which component is likely failing.
- **Synthetic calibration:** Threshold recommendations are derived from synthetic data. Production calibration requires labeled historical incident data.

### Future Improvements

- **Online learning:** Incorporate streaming updates with mini-batch gradient boosting to adapt to concept drift without full retraining.
- **Temporal attention models:** Replace hand-crafted rolling features with a Temporal Fusion Transformer or LSTM-Autoencoder to learn temporal dependencies automatically.
- **Multi-instance aggregation:** Extend to fleet-level anomaly detection by aggregating per-instance scores with spatial correlation.
- **Root cause attribution:** Integrate SHAP values into alert messages to indicate which metrics drove the high anomaly score.
- **Bayesian threshold optimization:** Replace grid search threshold selection with Bayesian optimization targeting a precision-recall operating point.
- **Human feedback loop:** Record alert outcomes (true positive / false positive) and incorporate labels into the next retraining cycle.
- **Anomaly clustering:** Group related alerts from multiple metrics into a single incident hypothesis to reduce alert fatigue.
