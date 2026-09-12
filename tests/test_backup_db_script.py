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
stopped -- and that remediation is right only in the last case.

The tests for it start below the ``issue #93`` banner. They assert the rendered
*message* as well as the specific exit code, because on the unfixed script all
four states exit non-zero: ``assert returncode != 0`` passes against the defect.
The 2-vs-3 split this change introduces is what makes an exit-code assertion
meaningful, so two of the name-matching tests legitimately assert only that.

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
                            # STDERR first, and it matters which stream each
                            # goes to: real psql writes errors and NOTICEs to
                            # stderr, and the script captures with `2>&1`. A
                            # stub that put its error on stdout would leave that
                            # capture unexercised -- reverting the script's
                            # `2>&1` to `2>/dev/null` would keep the suite
                            # green, which is how the welded-warning defect got
                            # through review once already.
                            # STREAM ORDER IS A PARAMETER, and it has to be:
                            # `2>&1` merges the two in write order, so emitting
                            # stderr first exercises only a LEADING weld, and a
                            # trailing one is a distinct case.
                            #
                            # Leading spaces on the value as `psql -t` emits
                            # them, so the script's whitespace strip is
                            # exercised rather than assumed. `-` again:
                            # set-but-empty is "psql exited 0 and said nothing",
                            # a distinct state.
                            if [ "${FAKE_DB_SIZE_STDERR_AFTER:-}" = "1" ]; then
                                printf '%s\\n' "${FAKE_DB_SIZE-   42 MB}"
                                printf '%s\\n' "${FAKE_DB_SIZE_STDERR-}" >&2
                            else
                                printf '%s\\n' "${FAKE_DB_SIZE_STDERR-}" >&2
                                printf '%s\\n' "${FAKE_DB_SIZE-   42 MB}"
                            fi
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
# (`docker compose up -d postgres`) is correct only in the last case. Only
# docker's STDOUT entered the pipe, so its error text still reached the terminal
# on stderr -- unlabelled, on another stream, and not attributable to this
# check, which is what capturing it fixes.
#
# What the exit code CANNOT do on the UNFIXED script, which is why these tests
# assert the rendered message: all four states exited non-zero there, so
# `assert returncode != 0` passes against the defect. This change splits them
# into 2 and 3, which is what makes an exit-code assertion meaningful -- so the
# two name-matching tests assert the code alone, and that is sufficient.
#
# The script's own failure branch had no test at all before this: every existing
# test above reaches `check_container` on the success path, because the `ps` stub
# echoed the container name unconditionally.
# --------------------------------------------------------------------------- #

# The contract as the script declares it. No open issue owns the Python side of
# it (see D011), so these are hardcoded rather than parsed out of the script:
# changing the script's constants should be a test failure, not something the
# tests silently follow. `test_the_script_declares_the_documented_exit_codes`
# pins the other direction.
EXIT_CANNOT_CHECK = 2
EXIT_CONTAINER_ABSENT = 3

# The hint is correct only for a genuinely stopped container, so its presence on
# any other path is the misattribution AC2 forbids.
STOPPED_HINT = "docker compose up -d postgres"
STOPPED_VERDICT = "is not running"

# AC2 forbids asserting a cause the script has not established, and naming one
# specific forbidden string would not enforce it -- PR #92's draft printed "Is
# the Docker daemon running?", but any other guess would be the same defect. The
# cannot-check branch reaches no `log_info` at all, and `create_backup` calls the
# check before it logs anything, so on that path the ABSENCE OF ANY `[INFO]`
# line is the enforceable property. Asserting that is what makes the guard
# resistant to a hint nobody thought to forbid.
INFO_PREFIX = "[INFO]"


def test_the_script_declares_the_documented_exit_codes():
    """The two integers cross a language boundary, so pin them in both places.

    Bash cannot import a Python constant, and no open issue owns the Python side
    of this contract (D011 records it as unowned), so the numbers exist in the
    script and in `docs/DATABASE.md`. The other tests in this block use the
    module constants above; this one asserts the script agrees with them, so a
    renumber cannot pass by changing only one side.

    It also pins the doc, which nothing else does: renumbering the script AND
    these constants together would otherwise leave `docs/DATABASE.md` stale, and
    that table is what a future consumer reads.
    """
    script = BACKUP_SCRIPT.read_text()
    assert f"EXIT_CANNOT_CHECK={EXIT_CANNOT_CHECK}" in script
    assert f"EXIT_CONTAINER_ABSENT={EXIT_CONTAINER_ABSENT}" in script

    # Scoped to the Exit Codes section, not the whole file: `| `2` |` also
    # appears in the environment-variable table further down (as
    # SCRAPE_DELAY_SECONDS' default), so a whole-file search is satisfied by a
    # row that has nothing to do with this contract -- the assertion would pass
    # with the exit-code row deleted.
    database_doc = (REPO_ROOT / "docs" / "DATABASE.md").read_text()
    start = database_doc.index("### Exit Codes")
    exit_code_section = database_doc[start : database_doc.index("###", start + 1)]
    for code in (EXIT_CANNOT_CHECK, EXIT_CONTAINER_ABSENT):
        assert f"| `{code}` |" in exit_code_section, (
            f"docs/DATABASE.md's Exit Codes table is the contract a future "
            f"consumer reads; it must carry the same numbers as the script, and "
            f"`{code}` is missing from it"
        )


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
        # The gated `else` branch. Without this row, deleting the
        # `if [ -n "$listing" ]` gate: without a row that reaches the `else`,
        # "a header followed by nothing renders absence as presence" would be a
        # claim with nothing behind it.
        ("docker said nothing", 1, ""),
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
    * the stopped-container verdict is absent. Deleting the whole cannot-check
      block fails several assertions here, the exit code first; this one covers
      the narrower case where cannot-check reuses "is not running" as its
      wording;
    * **no `[INFO]` line is emitted at all.** Naming one forbidden string would
      not enforce AC2: PR #92's draft printed "Is the Docker daemon running?",
      but any other guess at the cause is the same defect, and asserting the
      absence of that one phrase leaves every other guess green. `create_backup`
      calls the check before it logs anything and the cannot-check branch
      reaches no `log_info`, so the absence of the prefix is the enforceable
      property;
    * the reported exit number is docker's. `$?` read inside
      `if ! listing=$(...); then` is the status of the *negation* -- always 0 --
      so a failure would report "exited 0" while every other assertion here
      still held.
    """
    harness.environ["FAKE_DOCKER_PS_EXIT"] = str(exit_code)
    harness.environ["FAKE_DOCKER_PS_STDERR"] = message

    result = harness.run("backup")

    if message:
        # The header is asserted separately from the text: deleting
        # `log_error "Docker reported:"` leaves the text in place, and deleting
        # the `printf` leaves the header in place.
        assert "Docker reported:" in result.stdout, result.stdout
    else:
        assert "Docker produced no output." in result.stdout, (
            "with no output to show, the script must say so rather than print a "
            f"header over nothing.\nstdout:\n{result.stdout}"
        )
        assert "Docker reported:" not in result.stdout, result.stdout

    assert result.returncode == EXIT_CANNOT_CHECK, (
        f"a failed Docker query must exit {EXIT_CANNOT_CHECK} (could not check), "
        f"not {EXIT_CONTAINER_ABSENT} (checked, absent) and not 1 (generic); "
        f"got {result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    if message:
        assert message in result.stdout, (
            f"docker's own message is the only thing that makes {label} "
            f"diagnosable, so it must reach the operator.\nstdout:\n{result.stdout}"
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
    # AC2, enforced as a property rather than as a blocklist. See INFO_PREFIX.
    assert INFO_PREFIX not in result.stdout, (
        "the cannot-check branch must offer no guidance at all, because it has "
        "established no cause. Any `[INFO]` line here is a guess -- PR #92's "
        f"draft guessed the daemon.\nstdout:\n{result.stdout}"
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
    needs before that point is `dirname` in its own header, so a PATH carrying
    only `bash` and `dirname` is enough to reach the branch.

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
    assert "docker: command not found" in result.stdout, (
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
    # The hint is offered here and nowhere else. Asserting its presence is what
    # keeps AC2's absence assertions above from being satisfiable by deleting the
    # hint outright.
    assert STOPPED_HINT in result.stdout, (
        f"the remediation belongs on the one path where the container is "
        f"confirmed absent.\nstdout:\n{result.stdout}"
    )
    # But it is hedged, and the listing is shown, because "not in the running
    # list" is ALSO what a misconfigured CONTAINER_NAME or a different compose
    # prefix looks like -- states where starting the container changes nothing
    # and the operator would otherwise loop. The listing is what lets them tell
    # which state they are in.
    assert "If it is stopped" in result.stdout, (
        "the hint must not assert that the container is stopped, which is only "
        f"one of the states this branch covers.\nstdout:\n{result.stdout}"
    )
    if listing:
        assert "Running containers:" in result.stdout, result.stdout
        for name in listing.splitlines():
            assert name in result.stdout, (
                "the running containers must be shown, or a name mismatch is "
                f"indistinguishable from a stopped container.\nstdout:\n{result.stdout}"
            )
    else:
        assert "No containers are running." in result.stdout, result.stdout
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


@pytest.mark.parametrize(
    ("label", "container_name"),
    [
        ("a glob star", "*"),
        ("a glob single-char", "esg_news_d?"),
        ("a glob class", "esg_news_d[b]"),
        # `grep -qxF --` satisfies AC3 and was rejected for exactly this: `-F`
        # reads a newline in the PATTERN as a list of alternative patterns, so
        # this name matches the line `esg_news_db` and reports the container
        # present. The shipped `[[ ]]` requires the pattern's lines to appear
        # ADJACENTLY, which they do not here.
        ("a newline, non-adjacent", "zzz_nonexistent\nesg_news_db"),
    ],
)
def test_a_hostile_container_name_is_not_matched(
    harness: Harness, label: str, container_name: str
):
    """CONTAINER_NAME is operator-supplied, so its metacharacters are reachable.

    The match is `[[ $'\\n'"$listing"$'\\n' != *$'\\n'"$CONTAINER_NAME"$'\\n'* ]]`
    and the quoting around `$CONTAINER_NAME` is what makes it literal. None of
    these names is in the listing, so every row must read as absent.

    What this does NOT claim: that the construct is immune to every hazard.
    Measured, it is narrower than `grep -qxF` on the newline case rather than
    immune to it -- a name whose lines appear *adjacently* in the listing still
    matches. That is unreachable through Docker, whose names cannot contain a
    newline, and needs a deliberately hostile CONTAINER_NAME.
    """
    harness.environ["CONTAINER_NAME"] = container_name
    harness.environ["FAKE_DOCKER_PS_NAMES"] = "esg_news_db\nother_db"

    result = harness.run("backup")

    assert result.returncode == EXIT_CONTAINER_ABSENT, (
        f"{label}: no container by this name is in the listing, so it must read "
        f"as absent.\nstdout:\n{result.stdout}"
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


# Every unit `pg_size_pretty` emits, measured off a live server (PostgreSQL
# 16.11) rather than recalled: `SELECT pg_size_pretty(x::bigint)` over the
# magnitude boundaries. Note "1 bytes" -- plural even at 1, so there is no
# singular spelling to allow.
#
# A unit missing from the shape guard's list would make it reject a legitimate
# size, so `status` would report "could not be determined" for a database of
# that magnitude. `PB` was omitted when the guard was first written.
PG_SIZE_PRETTY_UNITS = [
    ("1 bytes", "1bytes"),
    ("10 kB", "10kB"),
    ("214 MB", "214MB"),
    ("1024 GB", "1024GB"),
    ("1024 TB", "1024TB"),
    ("1024 PB", "1024PB"),
]


@pytest.mark.parametrize(("psql_output", "rendered"), PG_SIZE_PRETTY_UNITS)
def test_status_reports_the_database_size_when_docker_can_be_queried(
    harness: Harness, psql_output: str, rendered: str
):
    """Control, and it asserts the RENDERED value rather than just exit 0.

    Asserting only `returncode == 0` would have passed against the defect this
    change removes: the size line was blank and the script exited 0, so a
    permissive control would have converted that defect into a guarded
    invariant. The value is what distinguishes "the query worked" from "the query
    silently produced nothing".

    It pins two things beyond that: the whitespace strip, since `psql -t` pads
    its output and the stub emits that padding; and the shape guard's unit list,
    since a unit missing from it turns a legitimate size into a permanent
    "could not be determined".
    """
    archive = _populate_backups(harness)
    harness.environ["FAKE_DB_SIZE"] = f"   {psql_output}"

    result = harness.run("status")

    assert result.returncode == 0, (
        f"a status query with Docker reachable must succeed, and {psql_output!r} "
        f"is a value pg_size_pretty really emits.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert f"Current database size: {rendered}" in result.stdout, (
        "the size must be rendered from psql's output, with psql -t's padding "
        f"stripped.\nstdout:\n{result.stdout}"
    )
    assert COULD_NOT_DETERMINE not in result.stdout, (
        f"{psql_output!r} is a legitimate size and must not be rejected by the "
        f"shape guard.\nstdout:\n{result.stdout}"
    )
    assert archive.name in result.stdout, "the on-disk facts must be reported too"


@pytest.mark.parametrize(
    ("label", "env", "expected_exit", "evidence"),
    [
        (
            "the query failed",
            {
                "FAKE_DOCKER_PS_EXIT": "1",
                "FAKE_DOCKER_PS_STDERR": (
                    "Cannot connect to the Docker daemon at unix:///var/run/docker.sock."
                ),
            },
            EXIT_CANNOT_CHECK,
            "Cannot connect to the Docker daemon",
        ),
        # The other half of the 2-vs-3 distinction, on the `status` path:
        # reporting a confirmed-stopped container as "never established" is the
        # collapse this issue is about.
        (
            "the container is absent",
            {"FAKE_DOCKER_PS_NAMES": "other_db"},
            EXIT_CONTAINER_ABSENT,
            "other_db",
        ),
    ],
)
def test_status_reports_partial_results_rather_than_aborting(
    harness: Harness, label: str, env: dict, expected_exit: int, evidence: str
):
    """AC7: keep the on-disk facts, name what could not be established, still fail.

    Three things have to hold at once:

    * the on-disk facts survive. `show_status` prints them *before* the check,
      so a read-only query that aborts there loses the counts the operator came
      for;
    * the exit code is the SPECIFIC one -- 2 when the query failed, 3 when Docker
      confirmed the container absent. Not 0: `ScriptResult.success` is
      `exit_code == 0`, so exiting 0 would make "Docker unreachable"
      indistinguishable from a clean status to the only consumer that will read
      it. And not a single code for both, which would rebuild the collapse;
    * the missing value is named rather than rendered blank;
    """
    archive = _populate_backups(harness)
    harness.environ.update(env)

    result = harness.run("status")

    assert archive.name in result.stdout, (
        f"{label}: a read-only query must still report the on-disk facts it "
        f"established.\nstdout:\n{result.stdout}"
    )
    assert result.returncode == expected_exit, (
        f"{label}: expected exit {expected_exit}; got {result.returncode}. "
        f"Reporting partial results must not mean exiting 0, and the two states "
        f"must not share a code.\nstdout:\n{result.stdout}"
    )
    assert COULD_NOT_DETERMINE in result.stdout, (
        f"{label}: the missing value must be named, not left blank.\n"
        f"stdout:\n{result.stdout}"
    )
    assert evidence in result.stdout, (
        f"{label}: the evidence for the verdict must be shown.\nstdout:\n{result.stdout}"
    )
    if expected_exit == EXIT_CANNOT_CHECK:
        # Only the hint is asserted absent here, not `INFO_PREFIX`: `show_status`
        # legitimately logs "[INFO] Backup Status" before either check, so the
        # absence-of-any-INFO property used on the `backup` path does not
        # transfer to this one.
        assert STOPPED_HINT not in result.stdout, result.stdout


@pytest.mark.parametrize(
    ("label", "size_exit", "stdout_text", "stderr_text", "stderr_after"),
    [
        # `5`, not `2`: psql's own status is echoed into the message, and if it
        # equalled EXIT_CANNOT_CHECK then mutating the branch's
        # `exit "$EXIT_CANNOT_CHECK"` to `exit "$status"` would be
        # indistinguishable here and survive on this row.
        ("psql exits non-zero", "5", "", "psql: error: connection to server failed", False),
        ("psql exits 0 saying nothing", "0", "", "", False),
        # The welded-warning case. psql exits 0 and DOES return the size, but a
        # warning on stderr is folded into the same capture by `2>&1`; with only
        # a non-emptiness guard the script printed
        # "Current database size: WARNING:...42MB" and exited 0 -- an
        # unestablished value rendered as a fact, which is this change's own
        # defect class. The shape guard is what rejects it.
        (
            "psql exits 0 with a warning before the size",
            "0",
            "   42 MB",
            "WARNING: there is no transaction in progress",
            False,
        ),
        # The same weld, the other way round. `2>&1` merges the two streams in
        # write order, so a warning landing after the value is a distinct case
        # from one landing before it.
        (
            "psql exits 0 with a warning after the size",
            "0",
            "   214 MB",
            "WARNING: terminal is not fully functional",
            True,
        ),
        # A trailing tail of LETTERS ONLY. Punctuation is rejected by any shape
        # check at all, so a tail without it is what exercises the unit list
        # rather than the surrounding anchors.
        (
            "psql exits 0 with an unpunctuated tail after the size",
            "0",
            "   214 MB",
            "extra data ignored",
            True,
        ),
    ],
)
def test_status_reports_when_the_database_size_cannot_be_determined(
    harness: Harness,
    label: str,
    size_exit: str,
    stdout_text: str,
    stderr_text: str,
    stderr_after: bool,
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

    The rows cover psql failing, psql succeeding with nothing to say, and a
    warning welded onto a real value from either side.
    """
    _populate_backups(harness)
    harness.environ["FAKE_DB_SIZE_EXIT"] = size_exit
    harness.environ["FAKE_DB_SIZE"] = stdout_text
    harness.environ["FAKE_DB_SIZE_STDERR"] = stderr_text
    if stderr_after:
        harness.environ["FAKE_DB_SIZE_STDERR_AFTER"] = "1"

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
    # Nor may any OTHER unvalidated string be rendered as the size. This is what
    # catches the welded-warning row specifically: the only size line permitted
    # is the "could not be determined" one.
    rendered = re.findall(r"^Current database size: (.+)$", result.stdout, re.MULTILINE)
    assert rendered == ["could not be determined"], (
        f"{label}: the only size line may be the could-not-determine one; "
        f"found {rendered!r}.\nstdout:\n{result.stdout}"
    )
    if stderr_text:
        assert stderr_text in result.stdout, (
            f"{label}: psql's own text is the evidence and must be surfaced, "
            f"unmangled by the whitespace strip.\nstdout:\n{result.stdout}"
        )
        assert "psql reported:" in result.stdout, result.stdout
    else:
        assert "psql produced no output." in result.stdout, (
            f"{label}: with nothing to show, the script must say so rather than "
            f"print a header over nothing.\nstdout:\n{result.stdout}"
        )
    if size_exit != "0":
        # Pins the reported number, the way the docker side does. Without it, a
        # message naming the wrong status survives.
        assert f"(psql exited {size_exit})" in result.stdout, (
            f"{label}: the reported status must be psql's own.\nstdout:\n{result.stdout}"
        )
    else:
        # psql exited 0 here, so naming an exit status as the reason would assert
        # a cause that did not fire.
        assert "psql exited 0 but returned no usable size" in result.stdout, (
            f"{label}: a shape/emptiness failure must not be reported as an "
            f"exit-status failure.\nstdout:\n{result.stdout}"
        )
