"""Regression tests for ``scripts/backup_db.sh`` failure handling (issue #89).

The defect these guard: the script ran ``pg_dump | gzip > file`` under ``set -e``
without ``pipefail``. A pipeline reports its *last* command's status, so a
``pg_dump`` that died mid-stream was masked by ``gzip`` exiting 0 -- a truncated
archive was recorded as a good backup. The ``if [ $? -eq 0 ]`` handler that would
have removed it was unreachable, because a non-zero pipeline under ``set -e``
aborts the function before ``$?`` can be read.

Hermetic: no Docker and no PostgreSQL. A fake ``docker`` executable is placed
first on ``PATH``; its ``pg_dump`` and ``psql`` exit codes are driven by
environment variables so a mid-pipeline failure can be forced deterministically.
"""

import gzip
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = REPO_ROOT / "scripts" / "backup_db.sh"

# Emits partial SQL on stdout *before* exiting non-zero when asked to fail, which
# is what makes this a genuine mid-pipeline failure: gzip downstream still
# receives bytes and still exits 0.
FAKE_DOCKER = """#!/bin/bash
case "$1" in
    ps)
        if [ -n "${FAKE_DOCKER_DAEMON_DOWN:-}" ]; then
            echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?" >&2
            exit 1
        fi
        echo "${CONTAINER_NAME:-esg_news_db}"
        # Optional filler AFTER the match, so a consumer that stops reading at the
        # first match leaves this side still writing -- which is what provokes
        # SIGPIPE. `exec` is load-bearing: it replaces this shell with the writer,
        # so the writer's death by SIGPIPE (141) becomes `docker`'s exit status.
        # Run as a plain command instead, and this script would carry on to its own
        # `exit 0`, swallowing the 141 and making the guard below vacuous.
        if [ -n "${FAKE_DOCKER_PS_PADDING:-}" ]; then
            exec seq 1 "$FAKE_DOCKER_PS_PADDING"
        fi
        exit 0
        ;;
    exec)
        for arg in "$@"; do
            case "$arg" in
                pg_dump)
                    printf -- '-- partial dump\\nCREATE TABLE articles (id integer);\\n'
                    exit "${FAKE_PGDUMP_EXIT:-0}"
                    ;;
                psql)
                    # Drain stdin so the upstream side of the pipe is not blocked.
                    cat >/dev/null 2>&1
                    exit "${FAKE_PSQL_EXIT:-0}"
                    ;;
            esac
        done
        exit 0
        ;;
esac
exit 0
"""


@dataclass
class Harness:
    environ: dict
    backup_dir: Path

    def run(self, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(BACKUP_SCRIPT), *args],
            env=self.environ,
            input=stdin,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

    @property
    def daily_archives(self) -> list[Path]:
        daily = self.backup_dir / "daily"
        return sorted(daily.glob("*.sql.gz")) if daily.is_dir() else []


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(FAKE_DOCKER)
    fake_docker.chmod(0o755)

    child_env = os.environ.copy()
    child_env["PATH"] = f"{bin_dir}{os.pathsep}{child_env['PATH']}"
    child_env["BACKUP_DIR"] = str(tmp_path / "backups")
    child_env["CONTAINER_NAME"] = "esg_news_db"
    return Harness(environ=child_env, backup_dir=tmp_path / "backups")


def test_backup_failure_exits_nonzero_and_removes_partial_archive(harness: Harness):
    """AC1 + AC2: a mid-pipeline pg_dump failure must fail *and* leave nothing behind.

    Both assertions are required and neither is redundant:

    * drop ``pipefail`` and the pipeline reports gzip's 0, the success branch
      runs, and the exit-code assertion fails;
    * drop the ``rm -f`` and the else branch still returns 1, so only the
      leftover-archive assertion fails.

    A test asserting the exit code alone would pass against a fix that leaves the
    truncated file on disk -- which is the half that actually loses data.
    """
    harness.environ["FAKE_PGDUMP_EXIT"] = "1"

    result = harness.run("backup")

    assert result.returncode != 0, (
        "a failed pg_dump must fail the backup; "
        f"got exit 0.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert harness.daily_archives == [], (
        "a failed backup must leave no archive behind, but found: "
        f"{[p.name for p in harness.daily_archives]}"
    )


def test_backup_success_creates_archive(harness: Harness):
    """Control: the failure test above means nothing if backups never succeed.

    Without this, a script that failed unconditionally would satisfy
    ``test_backup_failure_exits_nonzero_and_removes_partial_archive``.
    """
    harness.environ["FAKE_PGDUMP_EXIT"] = "0"

    result = harness.run("backup")

    assert result.returncode == 0, (
        f"a successful pg_dump must succeed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert len(harness.daily_archives) == 1


def test_restore_reports_failure_on_corrupt_archive(harness: Harness):
    """AC3: gunzip failing on a corrupt archive must not report success.

    The third assertion -- that the failure-guidance branch actually ran -- is the
    one that guards the ``if <pipeline>`` restructure. Asserting only "non-zero
    exit" and "success line absent" is satisfied by ``pipefail`` alone: reverting
    the restore to a bare pipeline plus ``if [ $? -eq 0 ]`` still aborts at the
    pipeline and still never prints the success line, so that mutant would
    survive and the operator-facing recovery guidance would be unguarded.
    """
    # The pre-restore safety copy must SUCCEED, or the script aborts there and
    # never reaches the gunzip path this test exists to exercise. And psql must
    # exit 0, so gunzip's failure is the only thing failing the pipeline --
    # otherwise the test would pass even without pipefail.
    harness.environ["FAKE_PGDUMP_EXIT"] = "0"
    harness.environ["FAKE_PSQL_EXIT"] = "0"

    corrupt = harness.backup_dir / "daily" / "esg_news_20260906_000000.sql.gz"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"this is not gzip data")

    result = harness.run("restore", "--file", str(corrupt), stdin="yes\n")
    output = result.stdout + result.stderr

    assert result.returncode != 0, f"a corrupt archive must fail the restore.\n{output}"
    assert "Restore completed successfully!" not in output
    assert "Restore failed!" in output, (
        f"the failure-guidance branch must run, not just a bare set -e abort.\n{output}"
    )
    # Match the guidance LINE, not the bare string "pre_restore_": the success path
    # of the pre-restore copy already logs "Pre-restore backup saved to: ...
    # pre_restore_...", so a substring check would hold whichever branch ran and
    # would not notice the recovery guidance being deleted.
    assert "You can restore the pre-restore backup with:" in output, (
        f"the failure branch must tell the operator how to recover.\n{output}"
    )
    guidance = [
        line for line in output.splitlines()
        if "restore --file" in line and "pre_restore_" in line
    ]
    assert guidance, (
        f"the recovery guidance must name the pre-restore archive to restore.\n{output}"
    )


def test_list_backups_survives_a_failing_glob_under_pipefail(harness: Harness):
    """AC5: ``pipefail`` must not abort ``list`` on a directory with no archives.

    ``list_backups`` guards its listing with ``ls -A`` (is the directory
    non-empty), not with the ``*.sql.gz`` glob -- so a directory holding only
    non-archive files makes the inner ``ls`` fail. Under ``pipefail`` that would
    abort the whole listing, which is why this change adds ``|| true`` there.
    This test is what guards that ``|| true``.
    """
    daily = harness.backup_dir / "daily"
    daily.mkdir(parents=True)
    (daily / "README.txt").write_text("not an archive\n")

    result = harness.run("list")

    assert result.returncode == 0, (
        "listing a directory with no *.sql.gz must not abort under pipefail.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_running_container_is_detected_in_a_long_docker_ps_list(harness: Harness):
    """A running container must be found even when ``docker ps`` output is large.

    A *latent* hazard, not an averted incident: written as
    ``docker ps ... | grep -q ...``, grep exits at the first match and can SIGPIPE
    the upstream (status 141), which ``pipefail`` turns into a false "not running".
    It needs ``docker ps`` output to exceed the pipe buffer (~64 KiB, i.e.
    thousands of containers); this project emits about a dozen bytes, so it would
    not have fired here. Note the asymmetry that made the story seductive: it can
    only bite when the container IS present, because that is when grep matches
    early and stops reading. An absent container makes grep drain the whole stream.

    Kept because the hazard is real in principle and the guard is nearly free. The
    reason the pipe actually had to go is error attribution -- see
    ``test_backup_surfaces_docker_failure_instead_of_reporting_not_running``.

    ``FAKE_DOCKER_PS_PADDING`` is 200000 lines (~1.3 MB) deliberately: measured,
    the inversion does not occur below roughly 64 KiB, so a small value would make
    this test vacuous rather than merely weaker.
    """
    harness.environ["FAKE_DOCKER_PS_PADDING"] = "200000"
    harness.environ["FAKE_PGDUMP_EXIT"] = "0"

    result = harness.run("backup")
    output = result.stdout + result.stderr

    assert "is not running" not in output, (
        f"a running container must not be reported as stopped.\n{output}"
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_restore_aborts_and_removes_partial_pre_restore_backup(harness: Harness):
    """The pre-restore safety copy must fail loudly and leave nothing behind.

    Without this the pre-restore wrap is unguarded: reverting it to a bare
    pipeline leaves every other test in this file passing, because the only other
    test touching the restore path deliberately makes the pre-restore copy
    succeed.
    """
    harness.environ["FAKE_PGDUMP_EXIT"] = "1"
    harness.environ["FAKE_PSQL_EXIT"] = "0"

    archive = harness.backup_dir / "daily" / "esg_news_20260906_000000.sql.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(gzip.compress(b"SELECT 1;\n"))

    result = harness.run("restore", "--file", str(archive), stdin="yes\n")
    output = result.stdout + result.stderr

    assert result.returncode != 0, f"a failed pre-restore copy must fail.\n{output}"
    assert "Pre-restore backup failed" in output, output
    leftovers = list(harness.backup_dir.glob("pre_restore_*"))
    assert leftovers == [], (
        "a failed pre-restore copy must leave no partial archive behind, found: "
        f"{[q.name for q in leftovers]}"
    )


def test_restore_succeeds_on_a_valid_archive(harness: Harness):
    """Control for the restore path, mirroring test_backup_success_creates_archive.

    Without it, a mutant that inverts the restore condition -- or one where the
    success branch never runs at all -- satisfies every failure assertion above.
    """
    harness.environ["FAKE_PGDUMP_EXIT"] = "0"
    harness.environ["FAKE_PSQL_EXIT"] = "0"

    archive = harness.backup_dir / "daily" / "esg_news_20260906_000000.sql.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(gzip.compress(b"SELECT 1;\n"))

    result = harness.run("restore", "--file", str(archive), stdin="yes\n")
    output = result.stdout + result.stderr

    assert result.returncode == 0, f"a valid archive must restore.\n{output}"
    assert "Restore completed successfully!" in output
    assert "Restore failed!" not in output


def test_list_backups_shows_an_existing_archive(harness: Harness):
    """The ``|| true`` guard must not mask a broken listing.

    ``|| true`` suppresses the exit status of the ``ls ... | awk`` pipeline, so
    without a positive assertion a formatting regression would print nothing under
    the DAILY header and still exit 0 -- silent success, the defect class this
    change exists to remove.
    """
    daily = harness.backup_dir / "daily"
    daily.mkdir(parents=True)
    archive = daily / "esg_news_20260906_000000.sql.gz"
    archive.write_bytes(gzip.compress(b"SELECT 1;\n"))

    result = harness.run("list")

    assert result.returncode == 0
    assert archive.name in result.stdout, (
        f"an existing archive must actually be listed.\n{result.stdout}"
    )


def test_backup_surfaces_docker_failure_instead_of_reporting_not_running(harness: Harness):
    """A docker/daemon failure must surface docker's own message, not collapse.

    The exit code cannot tell the fix from the bug here -- both exit non-zero, so
    ``assert returncode != 0`` passes against the very code this guards. What
    distinguishes them is the *mechanism*: the buggy form discarded docker's
    diagnostic (``2>/dev/null``) and its status (``|| true``), leaving the operator
    with "not running / start it with docker compose up" -- guidance that fails the
    same way the daemon just did. The fix branches on the status and re-emits the
    message.

    So the load-bearing assertions are the presence of docker's own text and the
    absence of the misattributed guidance.
    """
    harness.environ["FAKE_DOCKER_DAEMON_DOWN"] = "1"

    result = harness.run("backup")
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "Cannot connect to the Docker daemon" in output, (
        "docker's own diagnostic must reach the operator, not be hidden by "
        f"2>/dev/null.\n{output}"
    )
    assert "Start it with: docker compose up -d postgres" not in output, (
        "a daemon failure must not be reported as a stopped container, which sends "
        f"the operator at a command that fails identically.\n{output}"
    )
    assert harness.daily_archives == [], "no archive should be written"
