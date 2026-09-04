# Contributing

## Development Workflow

1. **Create or pick a GitHub issue** describing the work
2. **Create a feature branch** from `main`
3. **Make commits** on the branch
4. **Push and open a PR** linking the issue
5. **CI must pass** (full test suite runs automatically)
6. **Squash merge** into `main`

## Branch Naming

```
<type>/<issue-number>-<short-description>
```

| Type | Use for |
|------|---------|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring without behavior change |

Examples: `feature/12-add-linting`, `fix/15-scraper-timeout`, `docs/9-update-readme`

## Commit Conventions

- Use imperative mood: "Add feature" not "Added feature"
- Start with an action verb: Add, Fix, Update, Remove, Refactor
- Keep the first line under 72 characters
- Add detail in the body when the "why" isn't obvious

## PR Process

1. Link the issue with `Closes #N` in the PR description
2. CI runs automatically when the PR is opened or updated — all tests must pass. It runs again on `main` after the squash merge
3. Use squash merge to keep `main` history clean:
   ```bash
   gh pr merge --squash --delete-branch
   ```

## Testing

```bash
# Run all tests (required to pass before merge)
uv run pytest

# Run with verbose output
uv run pytest -v

# Database tests (requires PostgreSQL, not run in CI)
RUN_DB_TESTS=1 uv run pytest tests/test_database.py
```

## Dependencies & Scheduled Jobs

Cron jobs (collection, scraping, agent workflows, drift monitoring) and the
agent's in-process script calls run `uv run --frozen`. `--frozen` uses the
already-provisioned virtual environment with **no network resolution**, so the
early-morning cron window does not fail trying to re-fetch the direct-URL
spaCy model dependency (`en-core-web-sm`) from GitHub when DNS is flaky
(see issue #45). The trade-off: cron will **not** auto-resolve dependency
changes — it fails loudly on a stale lockfile instead.

**After changing dependencies** (editing `pyproject.toml`), reconcile the env
manually so the next scheduled run picks up the change:

```bash
uv lock      # update uv.lock to match pyproject.toml
uv sync      # install the resolved dependencies into .venv
```

If a scheduled job logs a `uv run --frozen` lockfile error, the venv is out of
sync — run the two commands above.

## Quick Reference

```bash
# 1. Create issue
gh issue create --title "Short description" --body "Details"

# 2. Create branch (using issue number from step 1)
git checkout -b feature/12-short-description main

# 3. Work and commit
git add <files>
git commit -m "Add the thing"

# 4. Push and create PR
git push -u origin feature/12-short-description
gh pr create --title "Add the thing" --body "Closes #12"

# 5. Wait for CI, then merge
gh pr checks
gh pr merge --squash --delete-branch
```
