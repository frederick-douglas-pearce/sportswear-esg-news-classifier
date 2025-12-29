"""Webhook-based alerting for drift detection and training events."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from .config import mlops_settings

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts that can be sent."""

    DRIFT_DETECTED = "drift_detected"
    TRAINING_COMPLETE = "training_complete"
    MODEL_PROMOTED = "model_promoted"
    DATA_QUALITY_ISSUE = "data_quality_issue"


@dataclass
class Alert:
    """Represents an alert to be sent."""

    alert_type: AlertType
    classifier_type: str
    message: str
    details: dict[str, Any] | None = None
    severity: str = "warning"  # info, warning, error
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AlertSender:
    """Send alerts via webhook (Slack/Discord compatible)."""

    def __init__(self, webhook_url: str | None = None):
        """Initialize alert sender.

        Args:
            webhook_url: Webhook URL. If None, uses settings.
        """
        self.webhook_url = webhook_url or mlops_settings.alert_webhook_url
        self.enabled = self.webhook_url is not None

    def send(self, alert: Alert) -> bool:
        """Send an alert via webhook.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Alerts disabled, skipping: {alert.message}")
            return False

        payload = self._format_payload(alert)

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                logger.info(f"Alert sent: {alert.alert_type.value}")
                return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    def _format_payload(self, alert: Alert) -> dict[str, Any]:
        """Format alert as Slack-compatible webhook payload.

        Args:
            alert: Alert to format

        Returns:
            Webhook payload dict
        """
        # Emoji based on severity
        emoji_map = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
        }
        emoji = emoji_map.get(alert.severity, ":bell:")

        # Color based on severity
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9800",
            "error": "#dc3545",
        }
        color = color_map.get(alert.severity, "#808080")

        # Format details as fields
        fields = []
        if alert.details:
            for key, value in alert.details.items():
                # Format numbers nicely
                if isinstance(value, float):
                    display_value = f"{value:.4f}"
                elif isinstance(value, (dict, list)):
                    display_value = json.dumps(value, indent=2)
                else:
                    display_value = str(value)

                fields.append({
                    "title": key.replace("_", " ").title(),
                    "value": display_value,
                    "short": len(str(display_value)) < 30,
                })

        # Slack block format
        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} ESG Classifier Alert",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*{alert.alert_type.value.replace('_', ' ').title()}*\n{alert.message}",
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Classifier: `{alert.classifier_type}` | {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                                },
                            ],
                        },
                    ],
                    "fields": fields if fields else None,
                },
            ],
        }

        return payload


def send_drift_alert(
    classifier_type: str,
    drift_score: float,
    threshold: float,
    details: dict[str, Any] | None = None,
) -> bool:
    """Send a drift detection alert.

    Args:
        classifier_type: Type of classifier (fp, ep, esg)
        drift_score: Measured drift score
        threshold: Threshold that was exceeded
        details: Additional details

    Returns:
        True if sent successfully
    """
    if not mlops_settings.alert_on_drift:
        return False

    alert = Alert(
        alert_type=AlertType.DRIFT_DETECTED,
        classifier_type=classifier_type,
        message=f"Drift detected! Score: {drift_score:.4f} (threshold: {threshold:.4f})",
        details={
            "drift_score": drift_score,
            "threshold": threshold,
            **(details or {}),
        },
        severity="warning",
    )

    sender = AlertSender()
    return sender.send(alert)


def send_training_alert(
    classifier_type: str,
    metrics: dict[str, float],
    model_version: str | None = None,
) -> bool:
    """Send a training completion alert.

    Args:
        classifier_type: Type of classifier
        metrics: Training metrics
        model_version: Optional model version

    Returns:
        True if sent successfully
    """
    if not mlops_settings.alert_on_training:
        return False

    alert = Alert(
        alert_type=AlertType.TRAINING_COMPLETE,
        classifier_type=classifier_type,
        message=f"Training complete. F2: {metrics.get('test_f2', 0):.4f}",
        details={
            "version": model_version,
            **metrics,
        },
        severity="info",
    )

    sender = AlertSender()
    return sender.send(alert)
