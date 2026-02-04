---
name: export-feed
description: Export ESG news feed to website repo and push to trigger rebuild
---

# Export Feed Skill

This skill exports labeled ESG news articles to the website repository as JSON and Atom feeds, then commits and pushes to trigger a site rebuild.

## Configuration

```
PROJECT_DIR: /home/fdpearce/Documents/Courses/DataTalksClub/projects/machine-learning-zoomcamp/sportswear-esg-news-classifier
WEBSITE_DIR: /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io
JSON_OUTPUT: ${WEBSITE_DIR}/_data/esg_news.json
ATOM_OUTPUT: ${WEBSITE_DIR}/assets/feeds/esg_news.atom
```

## Step 1: Run the Export Script

Execute the export script from the project directory:

```bash
cd /home/fdpearce/Documents/Courses/DataTalksClub/projects/machine-learning-zoomcamp/sportswear-esg-news-classifier && \
uv run python scripts/export_website_feed.py --format both \
  --json-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/_data/esg_news.json \
  --atom-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/assets/feeds/esg_news.atom
```

The script will output:
- Number of articles exported
- Scorecard summary (top/back performers)
- File paths written

## Step 2: Check for Changes in Website Repo

```bash
cd /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io && \
git status --short _data/esg_news.json assets/feeds/esg_news.atom
```

If no changes are shown, inform the user that the feed is already up to date and skip the remaining steps.

## Step 3: Commit Changes

If there are changes, commit them:

```bash
cd /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io && \
git add _data/esg_news.json assets/feeds/esg_news.atom && \
git commit -m "$(cat <<'EOF'
Update ESG news feed

Automated export from sportswear-esg-news-classifier
EOF
)"
```

## Step 4: Push to Remote

Push the commit to trigger a GitHub Pages rebuild:

```bash
cd /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io && \
git push
```

## Step 5: Summary

After completing all steps, provide a summary:
- Number of articles exported
- Commit hash
- Confirm the push was successful and site rebuild will be triggered

## Optional Arguments

The user may specify optional arguments after `/export-feed`:

- `--dry-run`: Run export without committing (skip steps 3-4)
- `--no-scorecard`: Export without scorecard calculation
- `--scorecard-period-days N`: Custom scorecard period (default: 14)

If the user specifies `--dry-run`, only run steps 1-2 and report what would be committed.
