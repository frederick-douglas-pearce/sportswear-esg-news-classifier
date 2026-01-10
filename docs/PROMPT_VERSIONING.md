# Prompt Versioning

This document describes the prompt versioning system for the ESG labeling pipeline.

## Overview

The labeling pipeline uses versioned prompts stored in text files under `prompts/labeling/`. This approach provides:

- **Reproducibility**: Track exactly which prompts were used for each labeling run
- **A/B Testing**: Compare results between prompt versions
- **Rollback**: Easily revert to a previous prompt version if issues arise
- **Audit Trail**: Database records link each label to the prompt version used

## Directory Structure

```
prompts/
  labeling/
    registry.json           # Version registry with production pointer
    v1.0.0/
      system_prompt.txt     # System prompt template
      user_prompt.txt       # User prompt template
      config.json           # Version configuration
    v1.1.0/
      system_prompt.txt
      user_prompt.txt
      config.json
```

## Registry Format

The `registry.json` file tracks all prompt versions and designates the production version:

```json
{
  "prompt_type": "esg_labeling",
  "description": "ESG classification prompts for sportswear brand news articles",
  "production": "v1.0.0",
  "versions": {
    "v1.0.0": {
      "created_at": "2025-01-09T00:00:00Z",
      "commit_message": "Initial prompt version",
      "description": "Original prompts with brand verification and ESG guidance",
      "model_recommendations": {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "temperature": 0.0
      },
      "tags": {
        "author": "initial",
        "status": "production"
      }
    }
  }
}
```

## Version Configuration

Each version directory contains a `config.json` that specifies:

```json
{
  "version": "v1.0.0",
  "system_prompt_file": "system_prompt.txt",
  "user_prompt_file": "user_prompt.txt",
  "variables": {
    "system": ["brands"],
    "user": ["title", "published_at", "source_name", "brands", "content"]
  },
  "model_config": {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 2000,
    "temperature": 0.0
  }
}
```

## CLI Usage

### List Available Versions

```bash
uv run python scripts/label_articles.py --list-prompts
```

Output:
```
=== Available Prompt Versions ===

  v1.0.0 (production)
    Initial prompt version - baseline extracted from config.py
    Created: 2025-01-09

  v1.1.0
    Add Substantive Coverage Test for clearer false positive classification
    Created: 2025-01-09
```

### Use a Specific Version

```bash
# Test v1.1.0 with dry-run
uv run python scripts/label_articles.py --prompt-version v1.1.0 --dry-run --batch-size 5

# Use v1.1.0 for production labeling
uv run python scripts/label_articles.py --prompt-version v1.1.0 --batch-size 10
```

### Use Production Version (Default)

```bash
# Uses version specified in registry.json "production" field
uv run python scripts/label_articles.py --batch-size 10
```

### Environment Variable Override

```bash
# Set default prompt version via environment
LABELING_PROMPT_VERSION=v1.1.0 uv run python scripts/label_articles.py --batch-size 10
```

## Database Tracking

Prompt version information is stored in two tables:

### labeling_runs Table

Tracks the prompt version used for each labeling batch:

| Column | Type | Description |
|--------|------|-------------|
| prompt_version | VARCHAR(20) | Version string (e.g., 'v1.0.0') |
| prompt_system_hash | VARCHAR(12) | SHA256 hash prefix of system prompt |
| prompt_user_hash | VARCHAR(12) | SHA256 hash prefix of user prompt |

### brand_labels Table

Links each label to the prompt version that created it:

| Column | Type | Description |
|--------|------|-------------|
| prompt_version | VARCHAR(20) | Version string (e.g., 'v1.0.0') |

### Migration

Run the migration to add prompt versioning columns:

```bash
psql $DATABASE_URL -f migrations/003_prompt_versioning.sql
```

## Creating a New Prompt Version

### Step 1: Create Version Directory

```bash
mkdir -p prompts/labeling/v1.2.0
```

### Step 2: Copy and Modify Prompts

```bash
# Copy from existing version
cp prompts/labeling/v1.1.0/* prompts/labeling/v1.2.0/

# Edit the prompts
nano prompts/labeling/v1.2.0/system_prompt.txt
```

### Step 3: Update config.json

```bash
# Update version number
nano prompts/labeling/v1.2.0/config.json
```

### Step 4: Register the Version

Add entry to `prompts/labeling/registry.json`:

```json
"v1.2.0": {
  "created_at": "2025-01-15T00:00:00Z",
  "commit_message": "Description of changes",
  "description": "Detailed description of this version",
  "model_recommendations": {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 2000,
    "temperature": 0.0
  },
  "tags": {
    "author": "your-name",
    "status": "testing"
  }
}
```

### Step 5: Test the New Version

```bash
# Dry run to verify prompts load correctly
uv run python scripts/label_articles.py --prompt-version v1.2.0 --dry-run --batch-size 3
```

### Step 6: Promote to Production

Update `registry.json` to set the new version as production:

```json
{
  "production": "v1.2.0",
  ...
}
```

Also update the version's status tag from "testing" to "production":

```json
"v1.2.0": {
  ...
  "tags": {
    "author": "your-name",
    "status": "production"
  }
}
```

### Step 7: Verify the Promotion

```bash
# Confirm new version is marked as production
uv run python scripts/label_articles.py --list-prompts

# Test that default labeling uses the new version
uv run python scripts/label_articles.py --dry-run --batch-size 1
```

Look for `Loaded prompt version v1.2.0` in the output.

### Rollback

If issues arise after promotion, revert by changing `"production"` back to the previous version in `registry.json`:

```json
{
  "production": "v1.1.0",
  ...
}
```

Existing labels retain their `prompt_version` field for traceability - only future labeling is affected by the rollback.

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v1.0.0 | 2025-01-09 | Initial version - baseline prompts from config.py |
| v1.1.0 | 2025-01-09 | Added Substantive Coverage Test for false positive clarity |

## Prompt Variables

### System Prompt Variables

| Variable | Description |
|----------|-------------|
| `{brands}` | Comma-separated list of target sportswear brands |

### User Prompt Variables

| Variable | Description |
|----------|-------------|
| `{title}` | Article title |
| `{published_at}` | Publication date (YYYY-MM-DD format) |
| `{source_name}` | Source/publisher name |
| `{brands}` | Comma-separated brands to analyze in this article |
| `{content}` | Article content text |

## Hash Verification

Each prompt version has SHA256 hash prefixes computed at load time. These hashes are stored in the database to verify that the exact same prompts were used:

```python
from src.labeling.prompt_manager import prompt_manager

version = prompt_manager.load_version("v1.0.0")
print(f"System hash: {version.system_prompt_hash}")
print(f"User hash: {version.user_prompt_hash}")
```

## Fallback Behavior

If versioned prompts cannot be loaded (e.g., files missing), the labeler falls back to hardcoded prompts in `src/labeling/config.py`. The version will be reported as "legacy" in this case.

## Querying Labels by Prompt Version

```sql
-- Count labels by prompt version
SELECT prompt_version, COUNT(*)
FROM brand_labels
GROUP BY prompt_version
ORDER BY prompt_version;

-- Find labels created with a specific version
SELECT a.title, bl.brand, bl.prompt_version
FROM brand_labels bl
JOIN articles a ON a.id = bl.article_id
WHERE bl.prompt_version = 'v1.1.0';

-- Compare labeling runs by version
SELECT prompt_version, prompt_system_hash, COUNT(*) as runs, SUM(brands_labeled) as total_labels
FROM labeling_runs
GROUP BY prompt_version, prompt_system_hash;
```
