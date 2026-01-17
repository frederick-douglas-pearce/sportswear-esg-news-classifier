"""Tests for agent notification system."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agent.notifications import (
    EmailNotifier,
    Notification,
    NotificationManager,
    NotificationType,
    WebhookNotifier,
    send_drift_notification,
    send_labeling_summary,
    send_workflow_notification,
)


class TestNotification:
    """Tests for Notification dataclass."""

    def test_default_timestamp(self):
        """Test default timestamp is set."""
        notification = Notification(
            notification_type=NotificationType.WORKFLOW_COMPLETE,
            subject="Test",
            message="Test message",
        )

        assert notification.timestamp is not None
        assert isinstance(notification.timestamp, datetime)

    def test_custom_values(self):
        """Test custom values are preserved."""
        now = datetime.now(timezone.utc)
        notification = Notification(
            notification_type=NotificationType.DRIFT_DETECTED,
            subject="Drift Alert",
            message="Drift detected in FP classifier",
            details={"classifier": "fp", "score": 0.15},
            severity="warning",
            timestamp=now,
        )

        assert notification.notification_type == NotificationType.DRIFT_DETECTED
        assert notification.subject == "Drift Alert"
        assert notification.details["classifier"] == "fp"
        assert notification.severity == "warning"
        assert notification.timestamp == now

    def test_to_dict(self):
        """Test conversion to dictionary."""
        notification = Notification(
            notification_type=NotificationType.LABELING_SUMMARY,
            subject="Daily Summary",
            message="Processed 50 articles",
            details={"count": 50},
        )

        data = notification.to_dict()

        assert data["type"] == "labeling_summary"
        assert data["subject"] == "Daily Summary"
        assert data["message"] == "Processed 50 articles"
        assert data["details"]["count"] == 50
        assert "timestamp" in data


class TestEmailNotifier:
    """Tests for EmailNotifier."""

    def test_disabled_when_email_not_enabled(self):
        """Test notifier is disabled when email not enabled."""
        with patch("src.agent.notifications.agent_settings") as mock_settings:
            mock_settings.email_enabled = False
            mock_settings.smtp_host = "localhost"
            mock_settings.smtp_port = 25
            mock_settings.email_sender = None
            mock_settings.email_recipient = None

            notifier = EmailNotifier()
            assert notifier.enabled is False

    def test_disabled_when_no_recipient(self):
        """Test notifier is disabled when no recipient configured."""
        with patch("src.agent.notifications.agent_settings") as mock_settings:
            mock_settings.email_enabled = True
            mock_settings.smtp_host = "localhost"
            mock_settings.smtp_port = 25
            mock_settings.email_sender = "sender@test.com"
            mock_settings.email_recipient = None

            notifier = EmailNotifier()
            assert notifier.enabled is False

    def test_send_returns_false_when_disabled(self):
        """Test send returns False when disabled."""
        with patch("src.agent.notifications.agent_settings") as mock_settings:
            mock_settings.email_enabled = False
            mock_settings.smtp_host = "localhost"
            mock_settings.smtp_port = 25
            mock_settings.email_sender = None
            mock_settings.email_recipient = None

            notifier = EmailNotifier()
            notification = Notification(
                notification_type=NotificationType.WORKFLOW_COMPLETE,
                subject="Test",
                message="Test",
            )

            result = notifier.send(notification)
            assert result is False

    @patch("smtplib.SMTP")
    def test_send_success(self, mock_smtp):
        """Test successful email sending."""
        with patch("src.agent.notifications.agent_settings") as mock_settings:
            mock_settings.email_enabled = True
            mock_settings.smtp_host = "smtp.test.com"
            mock_settings.smtp_port = 587
            mock_settings.email_sender = "sender@test.com"
            mock_settings.email_recipient = "recipient@test.com"

            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            notifier = EmailNotifier()
            notification = Notification(
                notification_type=NotificationType.LABELING_SUMMARY,
                subject="Daily Report",
                message="50 articles processed",
                details={"count": 50},
            )

            result = notifier.send(notification)

            assert result is True
            mock_server.sendmail.assert_called_once()

    def test_format_text(self):
        """Test plain text formatting."""
        with patch("src.agent.notifications.agent_settings") as mock_settings:
            mock_settings.email_enabled = False
            mock_settings.smtp_host = "localhost"
            mock_settings.smtp_port = 25
            mock_settings.email_sender = None
            mock_settings.email_recipient = None

            notifier = EmailNotifier()
            notification = Notification(
                notification_type=NotificationType.DRIFT_DETECTED,
                subject="Drift Alert",
                message="Drift detected",
                details={"score": 0.15, "threshold": 0.1},
            )

            text = notifier._format_text(notification)

            assert "drift_detected" in text
            assert "Drift detected" in text
            assert "score: 0.1500" in text


class TestWebhookNotifier:
    """Tests for WebhookNotifier."""

    def test_disabled_when_no_url(self):
        """Test notifier is disabled when no webhook URL."""
        with patch("src.mlops.config.mlops_settings") as mock_settings:
            mock_settings.alert_webhook_url = None

            notifier = WebhookNotifier()
            assert notifier.enabled is False

    def test_send_returns_false_when_disabled(self):
        """Test send returns False when disabled."""
        with patch("src.mlops.config.mlops_settings") as mock_settings:
            mock_settings.alert_webhook_url = None

            notifier = WebhookNotifier()
            notification = Notification(
                notification_type=NotificationType.WORKFLOW_COMPLETE,
                subject="Test",
                message="Test",
            )

            result = notifier.send(notification)
            assert result is False

    @patch("httpx.Client")
    def test_send_success(self, mock_client):
        """Test successful webhook sending."""
        with patch("src.mlops.config.mlops_settings") as mock_settings:
            mock_settings.alert_webhook_url = "https://hooks.slack.com/test"

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            notifier = WebhookNotifier()
            notification = Notification(
                notification_type=NotificationType.DRIFT_DETECTED,
                subject="Drift Alert",
                message="Drift detected",
            )

            result = notifier.send(notification)

            assert result is True

    def test_format_payload(self):
        """Test Slack payload formatting."""
        with patch("src.mlops.config.mlops_settings") as mock_settings:
            mock_settings.alert_webhook_url = None

            notifier = WebhookNotifier()
            notification = Notification(
                notification_type=NotificationType.HIGH_ERROR_RATE,
                subject="High Error Rate",
                message="Error rate exceeded threshold",
                severity="error",
                details={"rate": 0.15},
            )

            payload = notifier._format_payload(notification)

            assert "attachments" in payload
            assert payload["attachments"][0]["color"] == "#dc3545"  # error color


class TestNotificationManager:
    """Tests for NotificationManager."""

    def test_logs_to_console_when_no_channels(self, capsys):
        """Test notification logged when no channels enabled."""
        with patch("src.agent.notifications.agent_settings") as mock_agent:
            with patch("src.mlops.config.mlops_settings") as mock_mlops:
                mock_agent.email_enabled = False
                mock_agent.email_recipient = None
                mock_mlops.alert_webhook_url = None

                manager = NotificationManager()
                notification = Notification(
                    notification_type=NotificationType.DAILY_REPORT,
                    subject="Daily Report",
                    message="50 articles processed",
                )

                result = manager.send(notification)

                assert "console" in result
                captured = capsys.readouterr()
                assert "Daily Report" in captured.out

    def test_send_to_specified_channels(self):
        """Test sending to specified channels only."""
        with patch("src.agent.notifications.agent_settings") as mock_agent:
            with patch("src.mlops.config.mlops_settings") as mock_mlops:
                mock_agent.email_enabled = False
                mock_agent.email_recipient = None
                mock_mlops.alert_webhook_url = None

                manager = NotificationManager()
                manager.email_notifier = MagicMock()
                manager.email_notifier.send.return_value = True
                manager.webhook_notifier = MagicMock()
                manager.webhook_notifier.send.return_value = True

                notification = Notification(
                    notification_type=NotificationType.WORKFLOW_COMPLETE,
                    subject="Test",
                    message="Test",
                )

                # Only send to email
                result = manager.send(notification, channels=["email"])

                manager.email_notifier.send.assert_called_once()
                manager.webhook_notifier.send.assert_not_called()


class TestConvenienceFunctions:
    """Tests for convenience notification functions."""

    def test_send_workflow_notification_completed(self):
        """Test workflow completion notification."""
        with patch(
            "src.agent.notifications.NotificationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.send.return_value = {"console": True}
            mock_manager_class.return_value = mock_manager

            result = send_workflow_notification(
                workflow_name="daily_labeling",
                status="completed",
                summary={"articles": 50},
            )

            assert result == {"console": True}
            call_args = mock_manager.send.call_args
            notification = call_args[0][0]
            assert notification.notification_type == NotificationType.WORKFLOW_COMPLETE
            assert "daily_labeling" in notification.subject
            assert notification.severity == "info"

    def test_send_workflow_notification_failed(self):
        """Test workflow failure notification."""
        with patch(
            "src.agent.notifications.NotificationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.send.return_value = {"console": True}
            mock_manager_class.return_value = mock_manager

            result = send_workflow_notification(
                workflow_name="daily_labeling",
                status="failed",
                error="Database connection failed",
            )

            call_args = mock_manager.send.call_args
            notification = call_args[0][0]
            assert notification.notification_type == NotificationType.WORKFLOW_FAILED
            assert notification.severity == "error"
            assert "failed" in notification.subject.lower()

    def test_send_labeling_summary(self):
        """Test labeling summary notification."""
        with patch(
            "src.agent.notifications.NotificationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.send.return_value = {"console": True}
            mock_manager_class.return_value = mock_manager

            result = send_labeling_summary(
                articles_processed=100,
                articles_labeled=80,
                false_positives=15,
                articles_failed=5,
                estimated_cost=0.50,
            )

            call_args = mock_manager.send.call_args
            notification = call_args[0][0]
            assert notification.notification_type == NotificationType.LABELING_SUMMARY
            assert "100" in notification.message
            assert "80" in notification.message

    def test_send_labeling_summary_high_error_rate(self):
        """Test labeling summary with high error rate triggers warning."""
        with patch(
            "src.agent.notifications.NotificationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.send.return_value = {"console": True}
            mock_manager_class.return_value = mock_manager

            # 20% error rate should trigger warning
            send_labeling_summary(
                articles_processed=100,
                articles_labeled=60,
                false_positives=20,
                articles_failed=20,
            )

            call_args = mock_manager.send.call_args
            notification = call_args[0][0]
            assert notification.severity == "warning"

    def test_send_drift_notification(self):
        """Test drift notification."""
        with patch(
            "src.agent.notifications.NotificationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.send.return_value = {"console": True}
            mock_manager_class.return_value = mock_manager

            result = send_drift_notification(
                classifier_type="fp",
                drift_score=0.15,
                threshold=0.1,
                details={"method": "evidently"},
            )

            call_args = mock_manager.send.call_args
            notification = call_args[0][0]
            assert notification.notification_type == NotificationType.DRIFT_DETECTED
            assert "FP" in notification.subject
            assert notification.severity == "warning"
            assert notification.details["drift_score"] == 0.15
