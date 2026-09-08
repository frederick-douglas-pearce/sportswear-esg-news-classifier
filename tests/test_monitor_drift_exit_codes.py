"""Tests for the scripts/monitor_drift.py exit-code contract (issue #71).

Before this contract existed the script returned 1 for both "drift detected"
and "the analysis raised", so the agent workflow could not tell a result from a
failure and fell back to scraping stdout.

AC3 ("the logged error message is non-empty") is asserted here, at its source,
rather than through the workflow's log line: the defect was that the message
went to the wrong *stream*, which is only observable where it is written.
"""

import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from src.mlops.exit_codes import (
    EXIT_DRIFT_DETECTED,
    EXIT_INDETERMINATE,
    EXIT_NO_DRIFT,
    NON_RETRYABLE_EXIT_CODES,
)
from src.mlops.monitoring import DriftReport

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "monitor_drift.py"


@pytest.fixture(scope="module")
def monitor_drift():
    """Import monitor_drift.py as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location("monitor_drift_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_report(drift_detected=False, indeterminate=False, error=None, score=0.05):
    details = {"reference_size": 300, "current_size": 70}
    if error:
        details["error"] = error
    return DriftReport(
        classifier_type="fp",
        timestamp=datetime(2026, 9, 7, 12, 0, 0),
        drift_detected=drift_detected,
        drift_score=score,
        threshold=0.1,
        details=details,
        indeterminate=indeterminate,
    )


class TestExitCodeMapping:
    def test_no_drift(self, monitor_drift):
        assert monitor_drift.exit_code_for(make_report()) == EXIT_NO_DRIFT

    def test_drift_detected(self, monitor_drift):
        report = make_report(drift_detected=True, score=0.2)
        assert monitor_drift.exit_code_for(report) == EXIT_DRIFT_DETECTED

    def test_indeterminate(self, monitor_drift):
        report = make_report(
            indeterminate=True, error="Insufficient data for drift analysis"
        )
        assert monitor_drift.exit_code_for(report) == EXIT_INDETERMINATE

    def test_indeterminate_wins_over_no_drift(self, monitor_drift):
        """The ordering IS the fix.

        An indeterminate report carries drift_detected=False because nothing
        was measured. Testing drift first would map "the check never ran" to
        exit 0 — which is how EP reported Healthy on every run it ever made.
        """
        report = make_report(
            drift_detected=False,
            indeterminate=True,
            error="Insufficient data for drift analysis",
        )

        assert report.drift_detected is False
        assert monitor_drift.exit_code_for(report) == EXIT_INDETERMINATE
        assert monitor_drift.exit_code_for(report) != EXIT_NO_DRIFT


class TestSummaryJson:
    def test_summary_carries_every_field_the_workflow_requires(
        self, monitor_drift, capsys
    ):
        from src.agent.workflows.drift_monitoring import REQUIRED_SUMMARY_FIELDS

        report = make_report()
        monitor_drift.print_summary_json(report, EXIT_NO_DRIFT)

        out = capsys.readouterr().out.strip()
        assert monitor_drift.SUMMARY_LABEL in out
        summary = json.loads(out.splitlines()[-1])

        for required in REQUIRED_SUMMARY_FIELDS:
            assert required in summary, f"summary is missing {required}"
        assert summary["exit_code"] == EXIT_NO_DRIFT
        assert summary["classifier"] == "fp"

    def test_summary_is_readable_by_the_runners_own_parser(self, monitor_drift, capsys):
        """The end-to-end contract, asserted against the REAL parser.

        The workflow reads `ScriptResult.parsed_output`, which the runner fills
        with `_parse_json_from_output`. That parser works line by line, so a
        summary emitted as `label: {...}` yields None here and every verdict
        silently degrades to `unknown`. Asserting against the real parser rather
        than `json.loads` is what makes this test notice.
        """
        from src.agent.runner import _parse_json_from_output

        report = make_report(indeterminate=True, error="No columns available")
        monitor_drift.print_report(report, verbose=True)
        monitor_drift.print_summary_json(report, EXIT_INDETERMINATE)

        parsed = _parse_json_from_output(capsys.readouterr().out)

        assert parsed["indeterminate"] is True
        assert parsed["exit_code"] == EXIT_INDETERMINATE


class TestPrintReport:
    def test_indeterminate_is_not_reported_as_healthy(self, monitor_drift, capsys):
        report = make_report(
            indeterminate=True, error="Insufficient data for drift analysis"
        )

        monitor_drift.print_report(report)

        out = capsys.readouterr().out
        assert "INDETERMINATE" in out
        assert "Healthy" not in out

    def test_healthy_still_reports_healthy(self, monitor_drift, capsys):
        """Control: the indeterminate branch has not swallowed the healthy one."""
        monitor_drift.print_report(make_report())

        assert "Healthy" in capsys.readouterr().out


class TestErrorStream:
    """AC3: the error message must land where the workflow reads it."""

    def test_analysis_error_goes_to_stderr_and_exits_indeterminate(self, tmp_path):
        """Runs the real script in a subprocess with a poisoned analysis.

        A subprocess is the only way to observe the true stream split and the
        real process exit code, which is what the agent runner sees.
        """
        harness = tmp_path / "harness.py"
        harness.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "import src.mlops as mlops\n"
            "def boom(*a, **k):\n"
            "    raise KeyError('novelty_score')\n"
            "mlops.run_drift_analysis = boom\n"
            "import runpy\n"
            "sys.argv = ['monitor_drift.py', '--classifier', 'fp', '--from-db']\n"
            f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')\n"
        )

        proc = subprocess.run(
            [sys.executable, str(harness)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,
        )

        assert proc.returncode == EXIT_INDETERMINATE
        # The MESSAGE must be present on stderr, not merely something mentioning
        # novelty_score: `traceback.print_exc` alone satisfies that, so without
        # this line deleting the `print(..., file=sys.stderr)` entirely leaves
        # every other assertion here passing (issue #71 review).
        assert "Error running drift analysis" in proc.stderr
        assert "novelty_score" in proc.stderr
        # The defect was that this went to stdout, where nothing read it.
        assert "Error running drift analysis" not in proc.stdout
        # Not merely "an error occurred": the traceback is what names the frame,
        # and it is what the workflow now carries into the run archive.
        assert "Traceback" in proc.stderr


class TestRetryContract:
    def test_drift_detected_is_not_retried(self):
        """A determinate result. Before #71 it was retried with backoff."""
        assert EXIT_DRIFT_DETECTED in NON_RETRYABLE_EXIT_CODES

    def test_indeterminate_is_retried(self):
        """Its causes include transient ones, e.g. a briefly unreachable database."""
        assert EXIT_INDETERMINATE not in NON_RETRYABLE_EXIT_CODES

    def test_success_is_not_in_the_set(self):
        assert EXIT_NO_DRIFT not in NON_RETRYABLE_EXIT_CODES

    def test_runner_passes_the_drift_contract_not_labelings(self):
        from unittest.mock import patch

        from src.agent.runner import run_monitor_drift

        with patch("src.agent.runner.run_uv_script") as mock_run:
            run_monitor_drift(classifier="fp", days=7)

        kwargs = mock_run.call_args.kwargs
        assert kwargs["non_retryable_exit_codes"] == NON_RETRYABLE_EXIT_CODES
        assert kwargs["parse_json_output"] is True


class TestMainWiring:
    """`main()` must return the same code it prints, and always print one.

    `exit_code_for` and `print_summary_json` are each tested in isolation
    above; nothing tied them together. A hardcoded
    `print_summary_json(report, EXIT_NO_DRIFT)` would pass every other test in
    this file while silently degrading every non-zero run to `unknown`, because
    the workflow rejects a summary whose exit_code disagrees with the process.
    """

    @pytest.mark.parametrize(
        "drift_detected,indeterminate,expected",
        [
            (False, False, EXIT_NO_DRIFT),
            (True, False, EXIT_DRIFT_DETECTED),
            (False, True, EXIT_INDETERMINATE),
        ],
    )
    def test_returned_code_matches_the_printed_summary(
        self, monitor_drift, capsys, monkeypatch, drift_detected, indeterminate, expected
    ):
        report = make_report(
            drift_detected=drift_detected,
            indeterminate=indeterminate,
            error="nothing to compare" if indeterminate else None,
        )
        monkeypatch.setattr(monitor_drift, "run_drift_analysis", lambda **kw: report)
        monkeypatch.setattr(
            sys, "argv", ["monitor_drift.py", "--classifier", "fp", "--from-db"]
        )

        returned = monitor_drift.main()

        summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert returned == expected
        assert summary["exit_code"] == returned


class TestCreateReferenceExitCodes:
    """`--create-reference` must obey the same contract.

    Returning 1 there would mean "drift detected" — and it is non-retryable, so
    `scripts/cron_monitor.sh` would log a failed reference build as
    "WARNING: Drift detected".
    """

    def test_failure_is_indeterminate_not_drift(
        self, monitor_drift, capsys, monkeypatch
    ):
        def boom(**kwargs):
            raise ValueError("No prediction data found for fp")

        monkeypatch.setattr(monitor_drift, "create_reference_dataset", boom)
        monkeypatch.setattr(
            sys,
            "argv",
            ["monitor_drift.py", "--classifier", "fp", "--from-db", "--create-reference"],
        )

        returned = monitor_drift.main()

        assert returned == EXIT_INDETERMINATE
        assert returned != EXIT_DRIFT_DETECTED
        captured = capsys.readouterr()
        assert "No prediction data found" in captured.err
        assert "No prediction data found" not in captured.out

    def test_non_value_errors_are_caught_too(self, monitor_drift, monkeypatch):
        """An OSError writing the parquet would otherwise escape and exit 1."""

        def boom(**kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(monitor_drift, "create_reference_dataset", boom)
        monkeypatch.setattr(
            sys,
            "argv",
            ["monitor_drift.py", "--classifier", "fp", "--from-db", "--create-reference"],
        )

        assert monitor_drift.main() == EXIT_INDETERMINATE

    def test_success_returns_no_drift(self, monitor_drift, monkeypatch, tmp_path):
        monkeypatch.setattr(
            monitor_drift, "create_reference_dataset", lambda **kw: tmp_path / "ref.parquet"
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["monitor_drift.py", "--classifier", "fp", "--from-db", "--create-reference"],
        )

        assert monitor_drift.main() == EXIT_NO_DRIFT

    def test_reference_stats_without_a_reference_is_not_success(
        self, monitor_drift, monkeypatch
    ):
        monkeypatch.setattr(monitor_drift, "get_reference_stats", lambda c: None)
        monkeypatch.setattr(
            sys,
            "argv",
            ["monitor_drift.py", "--classifier", "fp", "--reference-stats"],
        )

        assert monitor_drift.main() == EXIT_INDETERMINATE


class TestOutputJsonCarriesIndeterminate:
    """`.github/workflows/monitoring.yml` reads this file with `jq`.

    Without the field the CI step summary prints "✅ Healthy" for a check that
    measured nothing.
    """

    def test_output_report_includes_indeterminate(
        self, monitor_drift, monkeypatch, tmp_path
    ):
        report = make_report(indeterminate=True, error="Insufficient data")
        monkeypatch.setattr(monitor_drift, "run_drift_analysis", lambda **kw: report)
        out = tmp_path / "drift_report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "monitor_drift.py", "--classifier", "fp", "--from-db",
                "--output", str(out),
            ],
        )

        monitor_drift.main()

        written = json.loads(out.read_text())
        assert written["indeterminate"] is True
        assert written["details"]["error"] == "Insufficient data"
