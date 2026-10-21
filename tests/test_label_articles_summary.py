"""Tests for the run summary scripts/label_articles.py prints (#192)."""

from unittest.mock import MagicMock, patch

import pytest

from scripts import label_articles
from src.labeling.pipeline import LabelingStats


def _run_main(stats, monkeypatch, *flags):
    monkeypatch.setattr("sys.argv", ["label_articles.py", *flags])
    pipeline = MagicMock()
    pipeline.label_articles.return_value = stats
    pipeline.get_stats.return_value = {}
    settings = MagicMock(anthropic_api_key="key", openai_api_key="key")
    with (
        patch.object(label_articles, "LabelingPipeline", return_value=pipeline),
        patch.object(label_articles, "labeling_settings", settings),
        patch.object(label_articles, "setup_logging"),
    ):
        label_articles.main()


class TestRunSummary:
    @pytest.mark.parametrize("flags", [("--dry-run",), ()], ids=["dry-run", "live"])
    @pytest.mark.parametrize("count", [3, 0])
    def test_incomplete_replies_line_is_always_printed(self, count, flags, monkeypatch, capsys):
        """The operator's view of the counter on both paths; printed even when it is zero."""
        _run_main(LabelingStats(articles_processed=4, incomplete_replies=count), monkeypatch, *flags)
        assert f"Incomplete replies:     {count}\n" in capsys.readouterr().out
