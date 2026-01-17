"""Unified notification system for agent workflows.

Supports multiple notification channels:
- Email (SMTP)
- Webhooks (Slack/Discord compatible)
"""

import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

from .config import agent_settings

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications."""

    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_FAILED = "workflow_failed"
    LABELING_SUMMARY = "labeling_summary"
    DRIFT_DETECTED = "drift_detected"
    HIGH_ERROR_RATE = "high_error_rate"
    DAILY_REPORT = "daily_report"


@dataclass
class Notification:
    """Represents a notification to be sent."""

    notification_type: NotificationType
    subject: str
    message: str
    details: dict[str, Any] | None = None
    severity: str = "info"  # info, warning, error
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.notification_type.value,
            "subject": self.subject,
            "message": self.message,
            "details": self.details,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
        }


class EmailNotifier:
    """Send notifications via email."""

    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        sender: str | None = None,
        recipient: str | None = None,
        password: str | None = None,
    ):
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            sender: Sender email address
            recipient: Recipient email address
            password: SMTP password or app password for authentication
        """
        self.smtp_host = smtp_host or agent_settings.smtp_host
        self.smtp_port = smtp_port or agent_settings.smtp_port
        self.sender = sender or agent_settings.email_sender
        self.recipient = recipient or agent_settings.email_recipient
        self.password = password or agent_settings.smtp_password
        self.enabled = agent_settings.email_enabled and self.recipient is not None

    def send(self, notification: Notification) -> bool:
        """Send email notification.

        Args:
            notification: Notification to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Email disabled, skipping: {notification.subject}")
            return False

        if not self.sender or not self.recipient:
            logger.warning("Email sender or recipient not configured")
            return False

        try:
            msg = self._format_email(notification)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                # Use TLS for secure connection
                server.starttls()

                # Authenticate if password is provided
                if self.password:
                    server.login(self.sender, self.password)

                server.sendmail(self.sender, self.recipient, msg.as_string())

            logger.info(f"Email sent: {notification.subject}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            logger.error("For Gmail, use an App Password (not your regular password)")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"Failed to send email: {e}")
            return False
        except Exception as e:
            logger.error(f"Email error: {e}")
            return False

    def _format_email(self, notification: Notification) -> MIMEMultipart:
        """Format notification as email message."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ESG Agent] {notification.subject}"
        msg["From"] = self.sender
        msg["To"] = self.recipient

        # Plain text version
        text_content = self._format_text(notification)
        msg.attach(MIMEText(text_content, "plain"))

        # HTML version
        html_content = self._format_html(notification)
        msg.attach(MIMEText(html_content, "html"))

        return msg

    def _format_text(self, notification: Notification) -> str:
        """Format notification as plain text."""
        lines = [
            f"ESG Agent Notification: {notification.notification_type.value}",
            "",
            notification.message,
            "",
            f"Timestamp: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Severity: {notification.severity}",
        ]

        if notification.details:
            lines.append("")
            lines.append("Details:")
            lines.append("-" * 40)
            for key, value in notification.details.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def _format_html(self, notification: Notification) -> str:
        """Format notification as HTML."""
        severity_colors = {
            "info": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
        }
        color = severity_colors.get(notification.severity, "#6c757d")

        details_html = ""
        if notification.details:
            rows = []
            for key, value in notification.details.items():
                if isinstance(value, float):
                    display = f"{value:.4f}"
                elif isinstance(value, dict):
                    display = "<br>".join(f"{k}: {v}" for k, v in value.items())
                else:
                    display = str(value)
                rows.append(f"<tr><td><strong>{key}</strong></td><td>{display}</td></tr>")
            details_html = f"""
            <h3>Details</h3>
            <table style="border-collapse: collapse; width: 100%;">
                {"".join(rows)}
            </table>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ padding: 20px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px; }}
                table {{ width: 100%; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .footer {{ color: #6c757d; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>ESG Agent: {notification.notification_type.value.replace('_', ' ').title()}</h2>
            </div>
            <div class="content">
                <p>{notification.message}</p>
                {details_html}
            </div>
            <div class="footer">
                <p>Timestamp: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        </body>
        </html>
        """


class ResendNotifier:
    """Send notifications via Resend API (recommended for email)."""

    def __init__(
        self,
        api_key: str | None = None,
        sender: str | None = None,
        recipient: str | None = None,
    ):
        """Initialize Resend notifier.

        Args:
            api_key: Resend API key
            sender: Sender email address (or use Resend's test sender)
            recipient: Recipient email address
        """
        self.api_key = api_key or agent_settings.resend_api_key
        self.sender = sender or agent_settings.email_sender or "ESG Agent <onboarding@resend.dev>"
        self.recipient = recipient or agent_settings.email_recipient
        self.enabled = (
            agent_settings.email_enabled
            and self.api_key is not None
            and self.recipient is not None
        )

    def send(self, notification: Notification) -> bool:
        """Send email notification via Resend.

        Args:
            notification: Notification to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Resend disabled, skipping: {notification.subject}")
            return False

        if not self.recipient:
            logger.warning("Email recipient not configured")
            return False

        try:
            import resend

            resend.api_key = self.api_key

            response = resend.Emails.send({
                "from": self.sender,
                "to": [self.recipient],
                "subject": f"[ESG Agent] {notification.subject}",
                "html": self._format_html(notification),
                "text": self._format_text(notification),
            })

            logger.info(f"Email sent via Resend: {notification.subject} (id: {response.get('id', 'unknown')})")
            return True

        except Exception as e:
            logger.error(f"Failed to send email via Resend: {e}")
            return False

    def _format_text(self, notification: Notification) -> str:
        """Format notification as plain text."""
        lines = [
            f"ESG Agent Notification: {notification.notification_type.value}",
            "",
            notification.message,
            "",
            f"Timestamp: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Severity: {notification.severity}",
        ]

        if notification.details:
            lines.append("")
            lines.append("Details:")
            lines.append("-" * 40)
            for key, value in notification.details.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                elif isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def _format_html(self, notification: Notification) -> str:
        """Format notification as HTML."""
        severity_colors = {
            "info": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
        }
        color = severity_colors.get(notification.severity, "#6c757d")

        details_html = ""
        if notification.details:
            rows = []
            for key, value in notification.details.items():
                if isinstance(value, float):
                    display = f"{value:.4f}"
                elif isinstance(value, dict):
                    display = "<br>".join(f"{k}: {v}" for k, v in value.items())
                else:
                    display = str(value)
                rows.append(f"<tr><td><strong>{key}</strong></td><td>{display}</td></tr>")
            details_html = f"""
            <h3>Details</h3>
            <table style="border-collapse: collapse; width: 100%;">
                {"".join(rows)}
            </table>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ padding: 20px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px; }}
                table {{ width: 100%; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .footer {{ color: #6c757d; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>ESG Agent: {notification.notification_type.value.replace('_', ' ').title()}</h2>
            </div>
            <div class="content">
                <p>{notification.message}</p>
                {details_html}
            </div>
            <div class="footer">
                <p>Timestamp: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        </body>
        </html>
        """


class WebhookNotifier:
    """Send notifications via webhook (Slack/Discord compatible)."""

    def __init__(self, webhook_url: str | None = None):
        """Initialize webhook notifier.

        Args:
            webhook_url: Webhook URL. Uses ALERT_WEBHOOK_URL if not provided.
        """
        # Import here to avoid circular dependency
        from src.mlops.config import mlops_settings

        self.webhook_url = webhook_url or mlops_settings.alert_webhook_url
        self.enabled = self.webhook_url is not None

    def send(self, notification: Notification) -> bool:
        """Send webhook notification.

        Args:
            notification: Notification to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Webhook disabled, skipping: {notification.subject}")
            return False

        try:
            import httpx

            payload = self._format_payload(notification)

            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

            logger.info(f"Webhook sent: {notification.subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False

    def _format_payload(self, notification: Notification) -> dict[str, Any]:
        """Format notification as Slack-compatible webhook payload."""
        emoji_map = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
        }
        emoji = emoji_map.get(notification.severity, ":bell:")

        color_map = {
            "info": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
        }
        color = color_map.get(notification.severity, "#6c757d")

        fields = []
        if notification.details:
            for key, value in notification.details.items():
                if isinstance(value, float):
                    display = f"{value:.4f}"
                elif isinstance(value, dict):
                    display = ", ".join(f"{k}: {v}" for k, v in value.items())
                else:
                    display = str(value)
                fields.append({
                    "title": key.replace("_", " ").title(),
                    "value": display,
                    "short": len(str(display)) < 30,
                })

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} {notification.subject}",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": notification.message,
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"{notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                                },
                            ],
                        },
                    ],
                    "fields": fields if fields else None,
                },
            ],
        }


class NotificationManager:
    """Manages sending notifications through multiple channels."""

    def __init__(self):
        """Initialize notification manager with configured channels.

        Prefers Resend API for email when RESEND_API_KEY is set,
        falls back to SMTP if only SMTP settings are configured.
        """
        # Use Resend if API key is configured, otherwise fall back to SMTP
        if agent_settings.resend_api_key:
            self.email_notifier = ResendNotifier()
        else:
            self.email_notifier = EmailNotifier()
        self.webhook_notifier = WebhookNotifier()

    def send(
        self,
        notification: Notification,
        channels: list[str] | None = None,
    ) -> dict[str, bool]:
        """Send notification through specified channels.

        Args:
            notification: Notification to send
            channels: List of channels ('email', 'webhook'). If None, uses all enabled.

        Returns:
            Dict mapping channel name to success status
        """
        results = {}

        if channels is None:
            channels = []
            if self.email_notifier.enabled:
                channels.append("email")
            if self.webhook_notifier.enabled:
                channels.append("webhook")

        if not channels:
            logger.info("No notification channels enabled, logging to console")
            self._log_notification(notification)
            return {"console": True}

        if "email" in channels:
            results["email"] = self.email_notifier.send(notification)

        if "webhook" in channels:
            results["webhook"] = self.webhook_notifier.send(notification)

        return results

    def _log_notification(self, notification: Notification) -> None:
        """Log notification to console when no channels enabled."""
        print(f"\n{'=' * 60}")
        print(f"NOTIFICATION: {notification.subject}")
        print(f"{'=' * 60}")
        print(f"Type: {notification.notification_type.value}")
        print(f"Severity: {notification.severity}")
        print(f"Timestamp: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"\n{notification.message}")

        if notification.details:
            print(f"\nDetails:")
            for key, value in notification.details.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        print(f"{'=' * 60}\n")


# Convenience functions


def send_workflow_notification(
    workflow_name: str,
    status: str,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, bool]:
    """Send workflow completion/failure notification.

    Args:
        workflow_name: Name of the workflow
        status: Workflow status (completed, failed)
        summary: Optional summary data
        error: Optional error message

    Returns:
        Dict of channel results
    """
    if status == "completed":
        notification_type = NotificationType.WORKFLOW_COMPLETE
        severity = "info"
        subject = f"Workflow Complete: {workflow_name}"
        message = f"The {workflow_name} workflow completed successfully."
    else:
        notification_type = NotificationType.WORKFLOW_FAILED
        severity = "error"
        subject = f"Workflow Failed: {workflow_name}"
        message = f"The {workflow_name} workflow failed: {error or 'Unknown error'}"

    notification = Notification(
        notification_type=notification_type,
        subject=subject,
        message=message,
        details=summary,
        severity=severity,
    )

    manager = NotificationManager()
    return manager.send(notification)


def send_labeling_summary(
    articles_processed: int,
    articles_labeled: int,
    false_positives: int,
    articles_failed: int,
    estimated_cost: float | None = None,
    additional_details: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Send daily labeling summary notification.

    Args:
        articles_processed: Total articles processed
        articles_labeled: Articles with ESG labels
        false_positives: False positive count
        articles_failed: Failed articles count
        estimated_cost: Estimated LLM cost in USD
        additional_details: Additional details to include

    Returns:
        Dict of channel results
    """
    # Determine severity based on error rate
    error_rate = articles_failed / articles_processed if articles_processed > 0 else 0
    if error_rate > 0.1:
        severity = "warning"
    else:
        severity = "info"

    details = {
        "articles_processed": articles_processed,
        "articles_labeled": articles_labeled,
        "false_positives": false_positives,
        "articles_failed": articles_failed,
        "error_rate": f"{error_rate:.1%}",
    }

    if estimated_cost is not None:
        details["estimated_cost"] = f"${estimated_cost:.4f}"

    if additional_details:
        details.update(additional_details)

    notification = Notification(
        notification_type=NotificationType.LABELING_SUMMARY,
        subject="Daily Labeling Summary",
        message=f"Processed {articles_processed} articles: {articles_labeled} labeled, {false_positives} FP, {articles_failed} failed",
        details=details,
        severity=severity,
    )

    manager = NotificationManager()
    return manager.send(notification)


def send_drift_notification(
    classifier_type: str,
    drift_score: float,
    threshold: float,
    details: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Send drift detection notification.

    Args:
        classifier_type: Type of classifier (fp, ep, esg)
        drift_score: Measured drift score
        threshold: Threshold that was exceeded
        details: Additional details

    Returns:
        Dict of channel results
    """
    notification = Notification(
        notification_type=NotificationType.DRIFT_DETECTED,
        subject=f"Drift Detected: {classifier_type.upper()} Classifier",
        message=f"Data drift detected for {classifier_type} classifier. Score: {drift_score:.4f} exceeds threshold {threshold:.4f}. Consider retraining.",
        details={
            "classifier": classifier_type,
            "drift_score": drift_score,
            "threshold": threshold,
            **(details or {}),
        },
        severity="warning",
    )

    manager = NotificationManager()
    return manager.send(notification)
