# Loop config — sportswear-esg-news-classifier

Per-project bindings for the supervised dev loop. `loop-engine.md` (the generic engine, bundled in
the plugin) references every value below **by parameter name**; this file is the only surface this
project edits. The engine is never edited here.

See `${CLAUDE_PLUGIN_ROOT}/skills/dev-loop/loop-engine.md` for the operating procedure and semantics.

> Written by hand 2026-09-06 against the repo as it stands, not generated. Review before the first
> run; a binding that looks wrong is a finding to hand to a human, never something the orchestrator
> rewrites mid-run.

---

## 1. Project parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `BACKLOG_SOURCE` | GitHub milestone **`silent-success`** | `gh issue list --milestone silent-success --state open`. Ten rows: #71 (live instance), #72 (epic tracker — see §3), #73–#80. |
| `SCOPE_AGENT` | the **`pm`** subagent | user-global; scope/priority/requirements. |
| `DESIGN_AGENT` | the **`architect`** subagent | user-global; reviews plans pre-implementation. **Records its outcome in `.claude/specs/decisions.md`** as the next `## D0NN:` entry — that file, not issue comments, is this project's decision surface (the engine's Resume step asks which surface a project uses). |
| `CODE_REVIEW` | parallel finder subagents over `git diff main...HEAD` **+ the issue's acceptance criteria**, angles chosen per the diff's risk surface, then a pass confirming each finding | The orchestrator runs this itself. `/code-review <effort>` is model-invocable and could be bound instead; the fan-out is preferred here because this repo's risk surface is unusually heterogeneous (workflow control flow, cron boundaries, LLM output parsing, SQL) and per-lens attribution is worth more than a flat finding list. |
| `SECURITY_REVIEW` | the **`/security-review`** skill (local, model-invocable), scoped per §4 | No labeled security workflow exists; `.github/workflows/` holds `ci.yml`, `deploy.yml`, `monitoring.yml`. Local path only. |
| `VERIFY` | the affected entry point's **dry-run**, where it has one — e.g. `uv run python scripts/collect_news.py --dry-run --max-calls 5`, `uv run python scripts/label_articles.py --dry-run --batch-size 5`. Where the change is a scheduled workflow, run that workflow and **read its recorded status**, not its exit code | ⚠ **This repo's whole epic is that exit codes lie** (#72). A `VERIFY` that accepts a zero exit reproduces the defect under review. Verify the *output* — rows written, status recorded, feed non-empty. |
| `PRIORITY_LABELS` | ⚠ **no `priority:*` label is set on any milestone row.** Selection order is the **implementation-order table in #72's third comment** (`0 → #80/pipefail, 1 → #71, 2 → #73, 3 → #74, 4 → #76, 5 → #75`, then wave 3), then `Depends on:`/`Blocked by`, tiebreak issue-number ascending | The labels `priority:high|medium|low` exist in the repo but are unused on #71–#80. The engine's selection step assumes a *label* ordering; this backlog encodes it in an epic comment. Treat the table as authoritative and re-read it — it is human-owned and may be amended. |
| `ARCHITECT_TRIGGERS` | see §2 | |
| `SOURCE_LAYOUT` | see §3 | |
| `TEST_CMD` | `uv run pytest` | The required CI check runs `uv run pytest --tb=short --junitxml=junit.xml`; the plain form is the local equivalent. |
| `LINT_CMD` | `—` — **no linter is configured in this project** | A deliberate not-applicable, not a blank. No ruff/flake8 config exists in `pyproject.toml` or a pre-commit config. Journal as `n/a: no linter configured`. |
| `TYPE_CMD` | `—` — **no type checker is configured in this project** | Same: no mypy/pyright config. Journal as `n/a: no type checker configured`. |
| `HERMETIC_TEST_CMD` | `—` — **no offline/hermetic test tier is declared in this project** | ⚠ **A deliberate not-applicable, and the reason is load-bearing — do not reduce this to a bare `—`, and never delete the row** (an absent row reads as *unknown, and unknown is due*). Discharged by reading, 2026-09-06: no tier is declared in `pyproject.toml`, `tests/conftest.py` or `CONTRIBUTING.md`, and no socket-level block exists. **`uv run --frozen` is not one** — CONTRIBUTING's "no network resolution" is about *dependency* resolution in the cron window, not test-time isolation. `RUN_DB_TESTS=1` is an opt-in *extra* tier, not an offline one. **If a hermetic tier is ever added, bind it here and verify the block is socket-level** with a direct-IP connect, per the engine — a proxy still resolves DNS. |
| `CI_STATUS_CMD` | `gh pr checks <PR>` | The required check is **`test`**. `deploy.yml` and `monitoring.yml` are not PR gates. |
| `BRANCH_FMT` | `<type>/<issue-number>-<description>` — e.g. `fix/71-drift-monitoring-status` | Prefixes are `feature/`, `fix/`, `docs/`, `refactor/` (`CLAUDE.md` → Development Workflow). |
| `COMMIT_CONV` | **imperative mood**, e.g. `Add…`, `Fix…`, `Update…` — **not** Conventional Commits | ⚠ Differs from the engine's usual example. Do not write `fix:`/`feat:` prefixes here. |
| `PR_TEMPLATE` | `.github/pull_request_template.md` — Summary / `Closes #N` / Changes / Test plan | Replicate all four sections. |
| `MERGE_METHOD` | `gh pr merge <PR> --squash --delete-branch` | ⚠ **Squash**, so the engine's "set the squash `--subject` scope explicitly" discipline **is** live here — but the scope is imperative-mood prose, not a Conventional Commit type. `delete_branch_on_merge` is already `true`; passing `--delete-branch` is harmless and explicit. |
| `APPEND_ONLY_FILES` | **`.claude/specs/decisions.md`** — entries are `## D0NN: <title>` | Wired in `.claude/loop.append-guard.json`. The loop **writes to this file itself** (`DESIGN_AGENT` outcomes), which is exactly why it is protected: a full-file `Write` that drops an existing decision is blocked. |
| `PERMISSION_POSTURE` | subagents are **read-only/validate-only**; the parent thread performs every mutation (Write/Edit/commit/merge) | The engine names this parameter but does not currently read it. |
| `LEDGER_ROOT` | `.claude/loop/` | **gitignored** — local working state, never committed. |
| `RELEASE_SCHEME` | **no package release cycle.** `version = "0.1.0"` in `pyproject.toml` is inert — nothing publishes it and there are no tags. **The release is the deployment** (`deploy.yml` → GCP). | The merge gate reads "≤ patch bump or no bump": an ordinary change produces **no** release-artifact bump and qualifies on that clause. ⚠ **But a change that alters what `deploy.yml` ships, or a model artifact, is release-visible** — treat it as an always-escalate "risky/irreversible" change and take the human merge gate. |

---

## 2. `ARCHITECT_TRIGGERS`

Fire `DESIGN_AGENT` (the `architect` subagent) when a plan hits any of these, **or** when the
orchestrator is unsure. Skip for pure `docs/` and `README` edits.

- **A change to `src/agent/workflows/base.py`, or to how step-level failure becomes workflow
  status.** This is the shared mechanism the whole `silent-success` epic is built on; #73 and #74
  land here and every wave-3 story depends on them. A partial change leaves some workflows
  reporting on the old contract and some on the new.
- **A new health/status vocabulary term** — `unknown`, `check_failed`, or any successor. The epic's
  own thesis is that a missing signal must never collapse into "healthy", and a term introduced in
  one workflow and not the others reintroduces exactly that.
- **A change to what a scheduled job asserts about its output** (freshness, non-emptiness, row
  counts, score present) as opposed to its exit code. Getting this wrong is the defect the epic
  exists to fix.
- **Anything touching the run archive under `~/.esg-agent/history/`** — its format, or what reads
  it. #75 and #76 both consume it, and it is 1,000+ existing runs that cannot be re-created.
- **A change to the labeling or prompt contract** (`prompts/labeling/`, LLM output parsing). Output
  shape changes are silently absorbed by permissive parsers and surface as label drift much later.
- **A schema or migration change** (`migrations/`).
- **A change that would require the engine to know something project-specific.** That is the signal
  to introduce a new `CAPS` parameter here instead.

---

## 3. `SOURCE_LAYOUT` — router signals

- **Package layout:** `src/` holds the product (`src/agent/`, `src/data_collection/`,
  `src/labeling/`, `src/mlops/`). `scripts/` holds entry points run by cron and by hand. `tests/`
  is pytest. `migrations/`, `models/`, `mlruns/`, `prompts/`, `queries/` are supporting artifacts.

- **`code` route (default):** anything under `src/`, `scripts/`, `tests/`, `migrations/`,
  `prompts/`, `.github/`, or `Dockerfile`/`docker-compose.yml`. Full pipeline, all gates.

- **`research` route — notebooks and data artifacts.** `notebooks/`, and analysis whose output is a
  finding rather than shipped behavior. ⚠ **A notebook is not a `docs` change**: `.ipynb` is JSON
  carrying embedded outputs, so its diff is large and mostly noise. If a change to a notebook also
  changes `src/`, route the whole row `code` and review the `src/` half — never let the notebook's
  diff volume push the row to a weaker gate set.

- **`docs` route:** `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `docs/`, `social/`, and
  typo/link/formatting fixes anywhere. Skips architect and security; light review. The loop still
  goes commit → PR → review; it never pushes to `main` (`CLAUDE.md`: "Do NOT commit directly to
  main").

- **`stub-defer` marker:** an **epic tracker** issue — title beginning `Epic:` — is not a work row.
  Carry it terminal (`deferred`); the loop does not close epics. **#72 is such a row.**

- **Data files are never AC evidence on their own.** `data/*.jsonl` is timestamped run output and
  churns constantly. A criterion is satisfied by the code that produces the file, not by the file.

---

## 4. Security routing

The host is GitHub. There is **no labeled security workflow**, so the labeled path does not exist
here and the local `/security-review` path is the whole story.

> ### ⛔ Precondition — `origin/HEAD` must be set, or this gate dies before it runs
>
> `/security-review` opens by shelling out to `` !`git diff --name-only origin/HEAD...` ``. On a
> fresh clone `origin/HEAD` is unset and the gate exits with
> `fatal: ambiguous argument 'origin/HEAD...'`, reviewing nothing.
>
> **Repair (idempotent):**
> ```bash
> git symbolic-ref -q refs/remotes/origin/HEAD >/dev/null || git remote set-head origin -a
> ```
>
> Treat a `fatal: ambiguous argument` from this gate as a missing ref, not a clean review: **an
> erroring gate is not a passing gate** — escalate, never journal it as clean.

**Run the local `/security-review`** when the diff touches any of:

- **credential handling or environment loading** — this repo holds `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, database credentials and GCP deploy credentials, and already runs
  `block_secret_reads.py` / `detect_secrets_in_output.py` hooks because of it;
- **anything parsing external input** — `src/data_collection/` (news APIs, scraped article HTML),
  and LLM responses in `src/labeling/`. Scraped HTML and model output are both untrusted input;
- **SQL construction** anywhere, and `migrations/`;
- **`Dockerfile`, `docker-compose.yml`, `.github/workflows/deploy.yml`** — the deployed surface;
- **`scripts/backup_db.sh`** and any cron-invoked shell.

**Skip** for changes confined to `tests/`, `notebooks/`, `docs/`, `social/`, `README.md`,
`CLAUDE.md`, `CONTRIBUTING.md`. Journal the skip and why.

---

## 5. Class B at the merge gate

This repo has 43 test files and a real suite, so the acceptance gate's mutation pass is **usually
due and usually runnable** — unlike a prose-only project. Run it.

Two project-specific cautions:

1. **Mutating LLM-dependent code needs a mocked path.** 41 of 43 test files already mock; a mutation
   whose kill depends on a live API call is not a guard, it is a flake. If a mutation cannot be
   killed without the network, that is a finding about the test, not about the mutation.
2. **The epic's own subject matter is the limit case.** A change that alters what a scheduled job
   asserts, while adding no test that would fail if the assertion were removed, is precisely
   `mutation-survivors=1 (no guard added)` — and in this repo that is the defect under repair, not a
   bookkeeping note. Do not merge it on a hand-wave.

**Do not reach for a new `n/a` reason.** The engine's list is closed at two and this file does not
reopen it.
