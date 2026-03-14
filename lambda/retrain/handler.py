import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import yaml

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET", "")
CONFIG_KEY = os.environ.get("CONFIG_KEY", "config/config.yaml")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "30"))


def _load_config() -> dict[str, Any]:
    import boto3

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=CONFIG_BUCKET, Key=CONFIG_KEY)
    return yaml.safe_load(response["Body"].read())


def _fetch_cloudwatch_data(config: dict[str, Any]) -> "pd.DataFrame":
    import pandas as pd

    from src.data.cloudwatch import fetch_metrics_dataframe

    metric_configs = config["data"]["metrics"]
    lookback = config["data"].get("lookback_days", LOOKBACK_DAYS)
    return fetch_metrics_dataframe(metric_configs, lookback_days=lookback)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    import pandas as pd

    from src.training.pipeline import TrainingPipeline

    logger.info("Retrain Lambda triggered at %s", datetime.utcnow().isoformat())

    try:
        config = _load_config()
        logger.info("Config loaded from s3://%s/%s", CONFIG_BUCKET, CONFIG_KEY)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        raise

    try:
        df = _fetch_cloudwatch_data(config)
        logger.info("Fetched %d rows of CloudWatch data", len(df))
    except Exception as exc:
        logger.error("Failed to fetch CloudWatch data: %s", exc)
        raise

    if df.empty:
        logger.error("No data fetched from CloudWatch")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No data available for training"}),
        }

    pipeline = TrainingPipeline(config)

    try:
        result = pipeline.run(df)
        logger.info("Training complete. Metrics: %s", result["metrics"])
    except Exception as exc:
        logger.error("Training pipeline failed: %s", exc)
        raise

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Retraining completed successfully",
                "artifact_location": result["artifact_location"],
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "recall_at_optimal": result["metrics"]
                .get("at_optimal_threshold", {})
                .get("recall"),
                "roc_auc": result["metrics"].get("roc_auc"),
                "timestamp": datetime.utcnow().isoformat(),
            }
        ),
    }
