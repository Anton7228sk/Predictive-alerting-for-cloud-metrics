import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import boto3

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    timestamp: datetime
    score: float
    threshold: float
    alert_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.alert_id:
            self.alert_id = f"alert_{self.timestamp.strftime('%Y%m%dT%H%M%S')}"

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


class AlertManager:
    def __init__(
        self,
        threshold: float,
        cooldown_minutes: int = 5,
    ) -> None:
        self.threshold = threshold
        self.cooldown_minutes = cooldown_minutes
        self._active_alerts: list[Alert] = []
        self._last_alert_time: Optional[datetime] = None

    def _in_cooldown(self, timestamp: datetime) -> bool:
        if self._last_alert_time is None:
            return False
        return timestamp - self._last_alert_time < timedelta(minutes=self.cooldown_minutes)

    def check_and_alert(
        self,
        timestamp: datetime,
        score: float,
        metadata: Optional[dict] = None,
    ) -> Optional[Alert]:
        if score < self.threshold:
            return None

        if self._in_cooldown(timestamp):
            logger.debug("Score %.3f above threshold but in cooldown period", score)
            return None

        alert = Alert(
            timestamp=timestamp,
            score=score,
            threshold=self.threshold,
            metadata=metadata or {},
        )
        self._active_alerts.append(alert)
        self._last_alert_time = timestamp
        logger.warning(
            "ALERT triggered at %s: score=%.3f (threshold=%.3f)",
            timestamp,
            score,
            self.threshold,
        )
        return alert

    def get_active_alerts(self) -> list[Alert]:
        return list(self._active_alerts)

    def clear_alerts(self) -> None:
        self._active_alerts.clear()

    def send_alert(self, alert: Alert, channel: str = "sns") -> bool:
        if channel == "sns":
            return self._send_sns(alert)
        elif channel == "webhook":
            return self._send_webhook(alert)
        else:
            logger.error("Unknown alert channel: %s", channel)
            return False

    def _send_sns(self, alert: Alert) -> bool:
        try:
            sns = boto3.client("sns")
            topic_arn = alert.metadata.get("sns_topic_arn")
            if not topic_arn:
                logger.error("No SNS topic ARN found in alert metadata")
                return False

            sns.publish(
                TopicArn=topic_arn,
                Subject=f"Predictive Alert: Incident Risk Detected",
                Message=json.dumps(alert.to_dict(), indent=2),
            )
            logger.info("SNS alert sent: %s", alert.alert_id)
            return True
        except Exception as exc:
            logger.error("Failed to send SNS alert: %s", exc)
            return False

    def _send_webhook(self, alert: Alert) -> bool:
        try:
            import urllib.request

            webhook_url = alert.metadata.get("webhook_url")
            if not webhook_url:
                logger.error("No webhook URL found in alert metadata")
                return False

            payload = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info("Webhook alert sent: %s", alert.alert_id)
                    return True
                logger.error("Webhook returned status %d", response.status)
                return False
        except Exception as exc:
            logger.error("Failed to send webhook alert: %s", exc)
            return False
