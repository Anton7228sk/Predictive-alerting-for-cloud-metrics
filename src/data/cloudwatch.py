import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


def get_cloudwatch_client(region_name: str = "us-east-1"):
    return boto3.client("cloudwatch", region_name=region_name)


def fetch_metric_statistics(
    namespace: str,
    metric_name: str,
    dimensions: list[dict[str, str]],
    start_time: datetime,
    end_time: datetime,
    period_seconds: int = 60,
    stat: str = "Average",
    region_name: str = "us-east-1",
) -> pd.DataFrame:
    client = get_cloudwatch_client(region_name)

    response = client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=dimensions,
        StartTime=start_time,
        EndTime=end_time,
        Period=period_seconds,
        Statistics=[stat] if stat not in ("p50", "p90", "p95", "p99") else [],
        ExtendedStatistics=[stat] if stat in ("p50", "p90", "p95", "p99") else [],
    )

    datapoints = response.get("Datapoints", [])
    if not datapoints:
        logger.warning(
            "No datapoints returned for %s/%s", namespace, metric_name
        )
        return pd.DataFrame(columns=["timestamp", metric_name])

    records = []
    for dp in datapoints:
        value = dp.get(stat, dp.get("ExtendedStatistics", {}).get(stat))
        records.append({"timestamp": dp["Timestamp"], metric_name: value})

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    return df


def list_metrics(
    namespace: Optional[str] = None,
    metric_name: Optional[str] = None,
    region_name: str = "us-east-1",
) -> list[dict[str, Any]]:
    client = get_cloudwatch_client(region_name)
    kwargs: dict[str, Any] = {}
    if namespace:
        kwargs["Namespace"] = namespace
    if metric_name:
        kwargs["MetricName"] = metric_name

    metrics = []
    paginator = client.get_paginator("list_metrics")
    for page in paginator.paginate(**kwargs):
        metrics.extend(page.get("Metrics", []))

    return metrics


def fetch_metrics_dataframe(
    metric_configs: list[dict[str, Any]],
    lookback_days: int = 30,
    region_name: str = "us-east-1",
) -> pd.DataFrame:
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)

    frames = []
    for cfg in metric_configs:
        try:
            df = fetch_metric_statistics(
                namespace=cfg["namespace"],
                metric_name=cfg["metric_name"],
                dimensions=cfg.get("dimensions", []),
                start_time=start_time,
                end_time=end_time,
                period_seconds=cfg.get("period_seconds", 60),
                stat=cfg.get("stat", "Average"),
                region_name=region_name,
            )
            col_name = cfg.get("alias", cfg["metric_name"])
            df.columns = [col_name]
            frames.append(df)
        except Exception as exc:
            logger.error(
                "Failed to fetch %s/%s: %s",
                cfg.get("namespace"),
                cfg.get("metric_name"),
                exc,
            )

    if not frames:
        return pd.DataFrame()

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.join(frame, how="outer")

    combined = combined.sort_index()
    return combined


def fetch_alarm_states(
    alarm_names: Optional[list[str]] = None,
    region_name: str = "us-east-1",
) -> pd.DataFrame:
    client = get_cloudwatch_client(region_name)
    kwargs: dict[str, Any] = {}
    if alarm_names:
        kwargs["AlarmNames"] = alarm_names

    response = client.describe_alarms(**kwargs)
    alarms = response.get("MetricAlarms", [])

    records = [
        {
            "alarm_name": a["AlarmName"],
            "state": a["StateValue"],
            "reason": a.get("StateReason", ""),
            "updated_at": a.get("StateUpdatedTimestamp"),
        }
        for a in alarms
    ]

    return pd.DataFrame(records)
