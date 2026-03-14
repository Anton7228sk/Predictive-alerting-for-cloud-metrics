import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "predictive-alerting-artifacts")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/model/")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
ALERT_THRESHOLD = float(os.environ.get("ALERT_THRESHOLD", "0.65"))
LOOKBACK_MINUTES = int(os.environ.get("LOOKBACK_MINUTES", "60"))

_model_cache: dict[str, Any] = {}


def _load_model_from_s3(bucket: str, prefix: str) -> "EnsembleModel":
    import boto3

    from src.models.ensemble import EnsembleModel

    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = response.get("Contents", [])

    if not objects:
        raise FileNotFoundError(f"No model artifacts found at s3://{bucket}/{prefix}")

    tmpdir = Path(tempfile.mkdtemp())

    for obj in objects:
        key = obj["Key"]
        relative = key[len(prefix):]
        local_path = tmpdir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))

    model = EnsembleModel()
    model.load(tmpdir)
    return model


def _get_model() -> "EnsembleModel":
    cache_key = f"{MODEL_BUCKET}/{MODEL_PREFIX}"

    if cache_key not in _model_cache:
        logger.info("Loading model from s3://%s/%s", MODEL_BUCKET, MODEL_PREFIX)
        _model_cache[cache_key] = _load_model_from_s3(MODEL_BUCKET, MODEL_PREFIX)
        logger.info("Model loaded and cached")

    return _model_cache[cache_key]


def _fetch_recent_metrics(lookback_minutes: int) -> "pd.DataFrame":
    import yaml
    import boto3

    from src.data.cloudwatch import fetch_metrics_dataframe

    s3 = boto3.client("s3")
    config_bucket = os.environ.get("CONFIG_BUCKET", MODEL_BUCKET)
    config_key = os.environ.get("CONFIG_KEY", "config/config.yaml")

    try:
        response = s3.get_object(Bucket=config_bucket, Key=config_key)
        config = yaml.safe_load(response["Body"].read())
        metric_configs = config["data"]["metrics"]
    except Exception as exc:
        logger.warning("Could not load config from S3 (%s), using defaults", exc)
        metric_configs = [
            {"namespace": "AWS/EC2", "metric_name": "CPUUtilization", "stat": "Average"},
        ]

    return fetch_metrics_dataframe(
        metric_configs, lookback_days=lookback_minutes / (60 * 24)
    )


def _log_to_cloudwatch(score: float, threshold: float, alert_triggered: bool) -> None:
    try:
        import boto3

        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace="PredictiveAlerting",
            MetricData=[
                {
                    "MetricName": "AnomalyScore",
                    "Value": score,
                    "Unit": "None",
                    "Timestamp": datetime.utcnow(),
                },
                {
                    "MetricName": "AlertTriggered",
                    "Value": 1.0 if alert_triggered else 0.0,
                    "Unit": "Count",
                    "Timestamp": datetime.utcnow(),
                },
            ],
        )
    except Exception as exc:
        logger.warning("Failed to log metrics to CloudWatch: %s", exc)


def _trigger_sns_alert(score: float, timestamp: datetime) -> None:
    try:
        import boto3

        sns = boto3.client("sns")
        message = {
            "alert_type": "PredictiveAlert",
            "timestamp": timestamp.isoformat(),
            "score": score,
            "threshold": ALERT_THRESHOLD,
            "message": f"Anomaly score {score:.3f} exceeded threshold {ALERT_THRESHOLD}. Incident may be imminent.",
        }
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject="Predictive Alert: High Anomaly Score Detected",
            Message=json.dumps(message, indent=2),
        )
        logger.info("SNS alert sent with score %.3f", score)
    except Exception as exc:
        logger.error("Failed to send SNS alert: %s", exc)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    import pandas as pd

    from src.data.preprocessing import create_features

    logger.info("Inference Lambda triggered at %s", datetime.utcnow().isoformat())

    try:
        model = _get_model()
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}

    try:
        df = _fetch_recent_metrics(LOOKBACK_MINUTES)
        if df.empty:
            logger.warning("No recent metrics available")
            return {"statusCode": 200, "body": json.dumps({"warning": "No data"})}
    except Exception as exc:
        logger.error("Failed to fetch metrics: %s", exc)
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}

    try:
        features = create_features(df)
        if features.empty:
            logger.warning("Feature creation produced empty result")
            return {"statusCode": 200, "body": json.dumps({"warning": "Empty features"})}

        scores = model.predict_proba(features)
        latest_score = float(scores[-1])
        latest_timestamp = features.index[-1].to_pydatetime()
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}

    alert_triggered = latest_score >= ALERT_THRESHOLD

    if alert_triggered and SNS_TOPIC_ARN:
        _trigger_sns_alert(latest_score, latest_timestamp)

    _log_to_cloudwatch(latest_score, ALERT_THRESHOLD, alert_triggered)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "timestamp": latest_timestamp.isoformat(),
                "score": latest_score,
                "threshold": ALERT_THRESHOLD,
                "alert_triggered": alert_triggered,
            }
        ),
    }
