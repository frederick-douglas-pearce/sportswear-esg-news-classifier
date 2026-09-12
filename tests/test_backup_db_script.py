"""Regression tests for ``scripts/backup_db.sh`` failure handling (issues #89, #93).

Two defects, in the same family: a failure that reported itself as success.

**#89 -- a truncated archive recorded as a good backup.** The script ran
``pg_dump | gzip > file`` under ``set -e`` without ``pipefail``. A pipeline
reports its *last* command's status, so a ``pg_dump`` that died mid-stream was
masked by ``gzip`` exiting 0.

**#93 -- four Docker states collapsed into one verdict.** ``check_container``
reported "Container is not running", and offered ``docker compose up -d
postgres``, whether the daemon was down, the caller was outside the ``docker``
group, ``docker`` was absent from ``PATH``, or the container was genuinely
stopped -- and that remediation is right only in the last case. The tests for it
begin at ``test_the_script_declares_the_documented_exit_codes`` and assert the
*message* rather than the exit code, because all four states exit non-zero.

The ``if [ $? -eq 0 ]`` handler that would have removed it was unreachable two
different ways, and only the second is a ``set -e`` abort: when ``pg_dump`` died
the pipeline reported ``gzip``'s 0, so ``$?`` was read, was 0, and the *success*
branch ran; when ``gzip`` itself failed the pipeline was non-zero and ``set -e``
aborted before ``$?`` could be read. The first is the case #89 is about.

Hermetic: no Docker and no PostgreSQL. Fake ``docker`` and ``du`` executables are
placed first on ``PATH``; their exit codes are driven by ``FAKE_*`` environment
variables so a mid-pipeline failure can be forced deterministically. The ``du``
stub passes through to the real binary unless asked to fail.
"""

import gzip
import os
import re
import shlex
import shutil
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
        # FAKE_DOCKER_PS_EXIT drives a failed *query* -- the daemon down, or a
        # caller outside the `docker` group. FAKE_DOCKER_PS_NAMES drives what a
        # successful query lists.
        #
        # `-` rather than `:-` on NAMES is deliberate: set-but-empty means "the
        # daemon answered and nothing is running", which has to stay distinct
        # from unset (the default listing). With `:-` the two would collapse,
        # which is the same mistake the script under test is being fixed for.
        if [ -n "${FAKE_DOCKER_PS_EXIT:-}" ]; then
            printf '%s\\n' "${FAKE_DOCKER_PS_STDERR-}" >&2
            exit "$FAKE_DOCKER_PS_EXIT"
        fi
        printf '%s\\n' "${FAKE_DOCKER_PS_NAMES-${CONTAINER_NAME:-esg_news_db}}"
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
                    # `status`'s size query is the one psql call whose OUTPUT the
                    # script reads, so it has to render a value; without this the
                    # size line would be blank in every test and a control test
                    # asserting exit 0 would bless the blank.
                    case "$*" in
                        *pg_size_pretty*)
                            # Leading spaces as `psql -t` emits them, so the
                            # script's whitespace strip is exercised rather than
                            # assumed. `-` again: set-but-empty is "psql exited
                            # 0 and said nothing", a distinct state.
                            printf '%s\\n' "${FAKE_DB_SIZE-   42 MB}"
                            exit "${FAKE_DB_SIZE_EXIT:-0}"
                            ;;
                    esac
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


# Pass-through stub for `du`: the script's only plain assignment with a PIPELINE on
# the right-hand side runs through it, so `pipefail` makes a `du` failure abort the
# success branch unless guarded. The real binary's absolute path is resolved from
# the parent's PATH (which this fixture never modifies -- only the child's copy) and
# baked in, so the stub cannot recurse into itself.
FAKE_DU = """#!/bin/bash
if [ -n "${FAKE_DU_EXIT:-}" ]; then
    exit "$FAKE_DU_EXIT"
fi
exec %s "$@"
"""


@dataclass
class Harness:
    environ: dict
    backup_dir: Path

    def run(
        self,
        *args: str,
        stdin: str | None = None,
        env: dict | None = None,
    ) -> subprocess.CompletedProcess:
        # `env` replaces the whole environment rather than merging, because the
        # one test that needs it is testing what happens when `docker` is absent
        # from PATH -- which a merge could not express.
        return subprocess.run(
            ["bash", str(BACKUP_SCRIPT), *args],
            env=self.environ if env is None else env,
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

    real_du = shutil.which("du")
    assert real_du, "du must exist on PATH for these tests"
    fake_du = bin_dir / "du"
    fake_du.write_text(FAKE_DU % shlex.quote(real_du))
    fake_du.chmod(0o755)

    child_env = os.environ.copy()
    # Drop any FAKE_* the caller happens to export: they drive the stub, so an
    # ambient one would silently change what these tests exercise.
    for key in [k for k in child_env if k.startswith("FAKE_")]:
        del child_env[key]
    child_env["PATH"] = f"{bin_dir}{os.pathsep}{child_env['PATH']}"
    child_env["BACKUP_DIR"] = str(tmp_path / "backups")
    child_env["CONTAINER_NAME"] = "esg_news_db"
    # `list_backups` formats with `ls -lh | awk '{print $9 ...}'`, so the filename
    # is only in field 9 under the default time format. An exported TIME_STYLE
    # (common in dotfiles) or a differing locale shifts the date field count and
    # turns the listing assertion red for no code reason.
    child_env["TIME_STYLE"] = "locale"
    child_env["LC_ALL"] = "C"
    # GNU ls honours QUOTING_STYLE independently of LC_ALL, and `shell-always`
    # wraps the name in quotes, which the listing regex would reject.
    child_env["QUOTING_STYLE"] = "literal"
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
    """AC1 fallout: ``pipefail`` must not abort ``list`` on a directory with no archives.

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
    # Assert the RENDERED SHAPE, not just that the name appears somewhere: raw
    # `ls -lh` output also contains the path, so a bare substring check survives
    # deleting the `| awk` formatting entirely.
    # The size field is matched as a SIZE, not as `\S+`: `ls -lh`'s neighbouring
    # columns are also non-space, so a loose pattern still passes when the awk
    # field index slips (`$5` -> `$4` prints the group name instead).
    assert re.search(
        rf"^  \S*{re.escape(archive.name)} \(\d[\d.]*[BKMGT]?\)$",
        result.stdout,
        re.MULTILINE,
    ), f"the archive must be listed in the formatted shape.\n{result.stdout}"


def test_backup_survives_a_du_failure_after_the_archive_is_written(harness: Harness):
    """A cosmetic `du` failure must not fail a backup that actually worked.

    ``BACKUP_SIZE=$(du -h "$DAILY_PATH" | cut -f1)`` is the only plain (non-``local``)
    assignment in the script whose right-hand side is a *pipeline* -- other plain
    assignments exist, but ``pipefail`` has nothing to reach in them -- so
    ``pipefail`` propagates a ``du`` failure to ``set -e`` -- which aborts *inside* the
    success branch, after the archive is
    safely written. Without the guard the result is the AC2 invariant inverted: a
    good archive on disk, a non-zero exit, no success line, no rotation, and the
    ``rm -f`` unreachable in the ``else``. This hazard did not exist before
    ``pipefail`` was added, so this PR owns it.

    Asserting exit 0 alone would kill the mutant only so long as the stub really
    fires; the final ``"(unknown)"`` assertion is what pins that, and the success
    line and surviving archive are what say the backup was *kept* rather than that
    the script merely exited quietly.
    """
    harness.environ["FAKE_PGDUMP_EXIT"] = "0"
    harness.environ["FAKE_DU_EXIT"] = "1"

    result = harness.run("backup")
    output = result.stdout + result.stderr

    assert result.returncode == 0, (
        f"a du failure must not fail a backup that succeeded.\n{output}"
    )
    assert len(harness.daily_archives) == 1, "the good archive must be kept"
    assert "Backup created successfully" in output, output
    # Pins that the stub actually fired. Exit 0 + an archive + the success line all
    # hold identically if the stub was never reached and the REAL du ran, in which
    # case this test would no longer fail if the guard were deleted. "(unknown)" is
    # the guard's own observable signature and only appears when du failed.
    assert "(unknown)" in output, (
        f"the du stub must actually have failed, or this test proves nothing.\n{output}"
    )


# --------------------------------------------------------------------------- #
# issue #93: check_container collapsed four Docker states into one verdict.
#
# The defect: `docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"`
# reported "Container is not running" whether the daemon was down, the caller was
# outside the `docker` group, `docker` was absent from PATH, or the container was
# genuinely stopped -- and the remediation it printed
# (`docker compose up -d postgres`) is correct only in the last case. docker's own
# message went into the pipe and was discarded.
#
# What the exit code CANNOT do, which is why every test below asserts the
# message: all four states exit non-zero, so `assert returncode != 0` passes
# against the unfixed script.
#
# The script's own failure branch had no test at all before this: every existing
# test above reaches `check_container` on the success path, because the `ps` stub
# echoed the container name unconditionally.
# --------------------------------------------------------------------------- #

# The contract #80 will consume. Hardcoded rather than parsed out of the script,
# so that changing the script's constants is a test failure rather than something
# the tests silently follow; `test_the_script_declares_the_documented_exit_codes`
# pins the other direction.
EXIT_CANNOT_CHECK = 2
EXIT_CONTAINER_ABSENT = 3

# The hint is correct only for a genuinely stopped container, so its presence on
# any other path is the misattribution AC2 forbids.
STOPPED_HINT = "docker compose up -d postgres"
STOPPED_VERDICT = "is not running"


def test_the_script_declares_the_documented_exit_codes():
    """The two integers are a cross-language contract, so pin them in both places.

    `src/agent/runner.py` will eventually map these to health verdicts, and bash
    cannot import a Python constant. The assertions above use the literals; this
    one asserts the script agrees, so a rename or a renumber cannot pass by
    changing only one side.
    """
    script = BACKUP_SCRIPT.read_text()
    assert f"EXIT_CANNOT_CHECK={EXIT_CANNOT_CHECK}" in script
    assert f"EXIT_CONTAINER_ABSENT={EXIT_CONTAINER_ABSENT}" in script
    # 1 stays the generic failure code. If a future edit names it for a specific
    # cause, "backup file not found" and "unknown command" start reporting that
    # cause -- a new signal collapse inside the fix for one.
    assert "EXIT_CONTAINER_ABSENT=1" not in script


@pytest.mark.parametrize(
    ("label", "exit_code", "message"),
    [
        (
            "daemon-down",
            1,
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
            "Is the docker daemon running?",
        ),
        (
            "permission-denied",
            1,
            "permission denied while trying to connect to the Docker daemon socket",
        ),
    ],
)
def test_a_failed_docker_query_is_reported_distinctly_from_a_stopped_container(
    harness: Harness, label: str, exit_code: int, message: str
):
    """AC1 + AC2: "could not check" must not wear "not running"'s verdict.

    Both of these states exit 1 from docker, so no exit code can tell them apart
    -- which is exactly why AC1 asks for docker's *output* to be surfaced and why
    this change does not invent a third code for "which cause". The operator
    distinguishes them by reading the message; the script asserts nothing about
    the cause.

    Four assertions, none redundant:

    * docker's own text is present -- delete the `printf "$listing"` and only this
      fails;
    * the stopped-container verdict is absent -- revert to the old single branch
      and only this fails;
    * the remediation hint is absent -- this is the one PR #92's draft failed, by
      printing "Is the Docker daemon running?" for every non-zero status and
      reproducing the misattribution one level down;
    * the reported exit number is docker's. This one pins a defect that was in
      this change's own first draft: the failure branch read `$?` inside
      `if ! listing=$(...); then`, where it is the status of the *negation* --
      always 0 -- so every failure reported "exited 0". Without this assertion
      that mutant survives, because the other three still hold.
    """
    harness.environ["FAKE_DOCKER_PS_EXIT"] = str(exit_code)
    harness.environ["FAKE_DOCKER_PS_STDERR"] = message

    result = harness.run("backup")

    assert result.returncode == EXIT_CANNOT_CHECK, (
        f"a failed Docker query must exit {EXIT_CANNOT_CHECK} (could not check), "
        f"not {EXIT_CONTAINER_ABSENT} (checked, absent) and not 1 (generic); "
        f"got {result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert message in result.stdout, (
        f"docker's own message is the only thing that makes {label} diagnosable, "
        f"so it must reach the operator.\nstdout:\n{result.stdout}"
    )
    assert STOPPED_VERDICT not in result.stdout, (
        f"{label} must not be reported as a stopped container.\nstdout:\n{result.stdout}"
    )
    assert STOPPED_HINT not in result.stdout, (
        f"the remediation hint assumes a cause the script has not established; "
        f"it must not appear for {label}.\nstdout:\n{result.stdout}"
    )
    assert f"docker exited {exit_code}" in result.stdout, (
        "the reported exit status must be docker's own. Reading `$?` inside an "
        "`if ! ...; then` body yields the negation's status (0), which would "
        f"print 'docker exited 0' for a real failure.\nstdout:\n{result.stdout}"
    )
    # The header and its evidence must land on ONE stream. `log_error` is a bare
    # `echo`, i.e. stdout; writing docker's text to stderr would split the two
    # across ScriptResult.stdout/.stderr, and a test concatenating both would
    # never notice.
    assert "Docker reported:" in result.stdout and message in result.stdout, (
        "the evidence must be on the same stream as the header that introduces it"
    )


def test_a_missing_docker_binary_surfaces_the_shell_s_own_diagnostic(
    harness: Harness, tmp_path: Path
):
    """AC1, the third state -- and the only one no stub can fake.

    daemon-down and permission-denied are emitted by docker, so the `ps` stub is
    faithful for them. A missing binary is emitted by **bash**, not by docker, so
    driving it through the same stub would assert only that the script prints a
    stub's stderr -- which the two rows above already cover. It would be one test
    wearing three labels, and the mechanism it is supposed to pin (that `2>&1`
    inside the command substitution captures a *shell*-emitted diagnostic) would
    go unasserted.

    So induce it for real: a PATH with no `docker` on it anywhere. `create_backup`
    calls `check_container` before anything else, and the only external the script
    needs before that point is `dirname` in its own header, so a one-symlink PATH
    is enough to reach the branch.

    Note what this also rules out: a *non-executable* `docker` first on PATH does
    NOT produce this. Bash skips it and keeps searching, so the real binary
    further along PATH answers and the test would pass while exercising nothing.
    """
    minimal_bin = tmp_path / "minimal_bin"
    minimal_bin.mkdir()
    # Exactly what is needed to reach the branch, and nothing else: `bash` to run
    # the script (Python resolves the executable against the env it is *given*,
    # so this PATH has to carry it) and `dirname` for the script's own header.
    # `docker` is deliberately absent, which is the whole point.
    for name in ("bash", "dirname"):
        real = shutil.which(name)
        assert real, f"{name} must exist on PATH for this test"
        (minimal_bin / name).symlink_to(real)
    assert shutil.which("docker", path=str(minimal_bin)) is None, (
        "this test is meaningless if `docker` is reachable on the minimal PATH"
    )

    env = {
        "PATH": str(minimal_bin),
        "BACKUP_DIR": str(harness.backup_dir),
        "CONTAINER_NAME": "esg_news_db",
    }

    result = harness.run("backup", env=env)

    assert result.returncode == EXIT_CANNOT_CHECK, (
        f"a missing docker binary is 'could not check', not 'not running'.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "docker exited 127" in result.stdout, (
        f"command-not-found is 127 and the script must report it.\nstdout:\n{result.stdout}"
    )
    assert "command not found" in result.stdout, (
        "the shell's own diagnostic is the evidence here, and capturing it is the "
        "whole reason the query is a command substitution with 2>&1 rather than a "
        f"pipeline.\nstdout:\n{result.stdout}"
    )
    assert STOPPED_VERDICT not in result.stdout, result.stdout
    assert STOPPED_HINT not in result.stdout, result.stdout
    assert harness.daily_archives == [], "no backup may be taken when the check never ran"


@pytest.mark.parametrize(
    ("label", "listing"),
    [
        ("another container is running", "other_db\nunrelated_db"),
        ("the daemon answered and nothing is running", ""),
    ],
)
def test_a_stopped_container_is_reported_as_stopped(
    harness: Harness, label: str, listing: str
):
    """AC5: the container-genuinely-absent branch, which had no test at all.

    Replacing that `if` with `if false` passed the whole suite before this, so the
    one verdict the script got *right* was the one nothing checked.

    The empty-listing row matters separately: it is the state where the daemon
    answered normally and no containers are running, which must read as absent
    rather than as a failed query. The stub uses `${VAR-default}` so that
    set-but-empty stays distinct from unset.
    """
    harness.environ["FAKE_DOCKER_PS_NAMES"] = listing

    result = harness.run("backup")

    assert result.returncode == EXIT_CONTAINER_ABSENT, (
        f"{label}: a container Docker confirms is absent must exit "
        f"{EXIT_CONTAINER_ABSENT}, not {EXIT_CANNOT_CHECK} (which would say the "
        f"state was never established).\nstdout:\n{result.stdout}"
    )
    assert STOPPED_VERDICT in result.stdout, result.stdout
    # Here -- and only here -- the hint is correct, because the cause IS
    # established. Asserting its presence is what keeps AC2's absence assertions
    # above from being satisfiable by deleting the hint outright.
    assert STOPPED_HINT in result.stdout, (
        f"the remediation is correct for a genuinely stopped container and must "
        f"still be offered.\nstdout:\n{result.stdout}"
    )
    assert harness.daily_archives == []


def test_container_name_is_matched_as_a_fixed_whole_line(harness: Harness):
    """AC6: the name was interpolated into a regex, so `.` matched any character.

    Reverting the match to `grep -q "^${CONTAINER_NAME}$"` makes this fail and
    nothing else: `esg.news_db` matches the line `esgxnews_db`, the script decides
    the container is present, and the backup proceeds against a container that is
    not there -- failing later with a worse message. Unasserted before this.
    """
    harness.environ["CONTAINER_NAME"] = "esg.news_db"
    harness.environ["FAKE_DOCKER_PS_NAMES"] = "esgxnews_db"

    result = harness.run("backup")

    assert result.returncode == EXIT_CONTAINER_ABSENT, (
        "'esg.news_db' must not match the line 'esgxnews_db'; a regex match "
        f"would treat '.' as a wildcard and report the container present.\n"
        f"stdout:\n{result.stdout}"
    )
    assert harness.daily_archives == [], (
        "no archive may be produced for a container that was never found"
    )


def test_a_container_name_holding_a_glob_metacharacter_does_not_match(harness: Harness):
    """The mutant that `grep -qxF` would have killed and an unquoted match does not.

    The match is `[[ $'\\n'"$listing"$'\\n' != *$'\\n'"$CONTAINER_NAME"$'\\n'* ]]`,
    and the quoting around `$CONTAINER_NAME` is what makes it literal. Unquoted,
    the `*` below is a glob that matches any listing at all, so the script would
    report *every* container present -- including when none is.

    CONTAINER_NAME is operator-supplied (`${CONTAINER_NAME:-esg_news_db}`), so
    this is reachable by configuration rather than theoretical. It is also why
    `grep -qxF --` was not used instead: `-F` reads a newline in the pattern as a
    list of alternative patterns, which is the same false-positive class in a
    different alphabet.
    """
    harness.environ["CONTAINER_NAME"] = "*"
    harness.environ["FAKE_DOCKER_PS_NAMES"] = "esg_news_db\nother_db"

    result = harness.run("backup")

    assert result.returncode == EXIT_CONTAINER_ABSENT, (
        "a container literally named '*' is not in the listing, so it must read "
        "as absent. An unquoted pattern would glob-match the whole listing and "
        f"report it present.\nstdout:\n{result.stdout}"
    )
    assert harness.daily_archives == []


# --------------------------------------------------------------------------- #
# issue #93, AC7: `status` is read-only, so a Docker problem must not throw away
# the on-disk facts -- and must not report success either.
#
# `show_status` makes TWO Docker calls. Fixing only the first would satisfy the
# criterion's letter and invert its point, because the second one was the quieter
# defect: `local db_size=$(docker exec ... | tr -d ' ')` returned `local`'s status
# rather than the pipeline's, and `2>/dev/null` discarded psql's reason. A
# container that was up while Postgres refused connections printed
# "Current database size: " -- an empty value rendered as a fact -- and exited 0.
# --------------------------------------------------------------------------- #

COULD_NOT_DETERMINE = "Current database size: could not be determined"


def _populate_backups(harness: Harness) -> Path:
    """Give `status` real on-disk facts to report.

    Without this the test would pass for the wrong reason: `show_status` returns
    early when the backup directory does not exist, *before* either Docker call,
    and exits 0 -- so the assertions below would never reach the code under test.
    """
    daily = harness.backup_dir / "daily"
    daily.mkdir(parents=True)
    archive = daily / "esg_news_20260906_000000.sql.gz"
    archive.write_bytes(gzip.compress(b"SELECT 1;\n"))
    return archive


def test_status_reports_the_database_size_when_docker_can_be_queried(harness: Harness):
    """Control, and it asserts the RENDERED value rather than just exit 0.

    Asserting only `returncode == 0` would have passed against the defect this
    change removes: the size line was blank and the script exited 0, so a
    permissive control would have converted that defect into a guarded
    invariant. The value is what distinguishes "the query worked" from "the query
    silently produced nothing".

    It also pins the whitespace strip: `psql -t` pads its output, the stub emits
    that padding, and `42MB` is what the script must render.
    """
    archive = _populate_backups(harness)

    result = harness.run("status")

    assert result.returncode == 0, (
        f"a status query with Docker reachable must succeed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Current database size: 42MB" in result.stdout, (
        "the size must be rendered from psql's output, with psql -t's padding "
        f"stripped.\nstdout:\n{result.stdout}"
    )
    assert COULD_NOT_DETERMINE not in result.stdout
    assert archive.name in result.stdout, "the on-disk facts must be reported too"


def test_status_reports_partial_results_when_docker_cannot_be_queried(harness: Harness):
    """AC7: keep the on-disk facts, name what could not be established, still fail.

    Three things have to hold at once, and each is a separate mutant:

    * the on-disk facts survive -- exiting at the container check, as `backup`
      does, would discard the counts the operator came for;
    * the exit code is `EXIT_CANNOT_CHECK`, not 0. `ScriptResult.success` is
      `exit_code == 0`, so exiting 0 would make "Docker unreachable"
      indistinguishable from a clean status to the only consumer that will read
      it -- this epic's defect, reintroduced at the boundary it is being removed
      from;
    * no remediation hint, because the cause is not established.
    """
    archive = _populate_backups(harness)
    harness.environ["FAKE_DOCKER_PS_EXIT"] = "1"
    harness.environ["FAKE_DOCKER_PS_STDERR"] = (
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock."
    )

    result = harness.run("status")

    assert archive.name in result.stdout, (
        "a read-only query must still report the on-disk facts it established.\n"
        f"stdout:\n{result.stdout}"
    )
    assert result.returncode == EXIT_CANNOT_CHECK, (
        f"reporting partial results must not mean exiting 0; got "
        f"{result.returncode}.\nstdout:\n{result.stdout}"
    )
    assert COULD_NOT_DETERMINE in result.stdout, (
        f"the missing value must be named, not left blank.\nstdout:\n{result.stdout}"
    )
    assert "Cannot connect to the Docker daemon" in result.stdout, result.stdout
    assert STOPPED_HINT not in result.stdout, result.stdout


@pytest.mark.parametrize(
    ("label", "size_exit", "size_output"),
    [
        ("psql exits non-zero", "2", "psql: error: connection to server failed"),
        ("psql exits 0 saying nothing", "0", ""),
    ],
)
def test_status_reports_when_the_database_size_cannot_be_determined(
    harness: Harness, label: str, size_exit: str, size_output: str
):
    """The second Docker call in `show_status`, which had no test and failed silently.

    The container is present here -- `docker ps` succeeds -- so this isolates the
    size query. Both rows must fail, for different reasons:

    * a non-zero psql was invisible because `local db_size=$(...)` returned
      `local`'s status, so the script could not tell a refused connection from a
      successful query;
    * an empty value on exit 0 is also a failure: no database's size renders as
      the empty string, so printing it asserts a fact that was never established.
      This row is what stops the fix from being "check the exit code" alone.

    Reverting either half -- the status capture or the `-z` guard -- leaves one of
    these two rows green and the other red, so neither is redundant.
    """
    _populate_backups(harness)
    harness.environ["FAKE_DB_SIZE_EXIT"] = size_exit
    harness.environ["FAKE_DB_SIZE"] = size_output

    result = harness.run("status")

    assert result.returncode == EXIT_CANNOT_CHECK, (
        f"{label}: an undetermined database size must not exit 0.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert COULD_NOT_DETERMINE in result.stdout, (
        f"{label}: the size must be named as undetermined rather than rendered "
        f"blank.\nstdout:\n{result.stdout}"
    )
    # The blank-line signature of the old defect. `$` anchors it so the assertion
    # cannot be satisfied by the "could not be determined" line above.
    assert not re.search(r"^Current database size:\s*$", result.stdout, re.MULTILINE), (
        f"{label}: an empty size must never be printed as though it were a "
        f"value.\nstdout:\n{result.stdout}"
    )
    if size_output:
        assert size_output in result.stdout, (
            f"{label}: psql's own reason is the evidence and must be surfaced.\n"
            f"stdout:\n{result.stdout}"
        )
