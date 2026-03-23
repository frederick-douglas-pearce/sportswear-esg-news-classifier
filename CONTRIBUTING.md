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
2. CI runs automatically on push — all tests must pass
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
