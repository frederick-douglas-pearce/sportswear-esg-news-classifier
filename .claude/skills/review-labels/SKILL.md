---
name: review-labels
description: Review recent labeling runs for potential errors and inconsistencies
---

# Review Labels Skill

This skill reviews articles labeled since the last review (or last 3 days if no prior review) and flags potential labeling errors for attention.

## Step 1: Get Review Window

First, check the state file for the last review timestamp:

```bash
cat .claude/skills/review-labels/last_run.txt 2>/dev/null || echo "No previous run"
```

If no previous run exists, use 3 days ago as the start date.

## Step 2: Run the Review Query

Run this Python script to analyze recent labeling:

```bash
uv run python -c "
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

# Calculate review window - use last run or default to 3 days
state_file = '.claude/skills/review-labels/last_run.txt'
try:
    with open(state_file) as f:
        since_date = f.read().strip()
except FileNotFoundError:
    since_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

print('=' * 70)
print(f'LABELING REVIEW: Articles labeled since {since_date}')
print('=' * 70)

with engine.connect() as conn:
    # Summary of recent labeling
    result = conn.execute(text('''
        SELECT COUNT(DISTINCT a.id) as articles,
               COUNT(DISTINCT bl.id) as brand_labels
        FROM articles a
        JOIN brand_labels bl ON a.id = bl.article_id
        WHERE a.labeled_at >= :since_date
    '''), {'since_date': since_date})
    row = result.fetchone()
    print(f'\nArticles labeled: {row.articles}, Brand labels: {row.brand_labels}')

    # Count FP-classifier-flagged articles in review window
    result = conn.execute(text('''
        SELECT COUNT(*) as fp_count
        FROM articles
        WHERE labeling_status = 'false_positive'
          AND created_at >= :since_date
    '''), {'since_date': since_date})
    fp_row = result.fetchone()
    print(f'FP-classifier flagged: {fp_row.fp_count} articles')

    # 1. Brand sentiment breakdown
    print('\n' + '-' * 70)
    print('BRAND SENTIMENT BREAKDOWN')
    print('-' * 70)
    result = conn.execute(text('''
        SELECT bl.brand,
               COUNT(*) as labels,
               SUM(CASE WHEN bl.environmental_sentiment = 1 THEN 1 ELSE 0 END) as env_pos,
               SUM(CASE WHEN bl.environmental_sentiment = -1 THEN 1 ELSE 0 END) as env_neg,
               SUM(CASE WHEN bl.social_sentiment = 1 THEN 1 ELSE 0 END) as soc_pos,
               SUM(CASE WHEN bl.social_sentiment = -1 THEN 1 ELSE 0 END) as soc_neg,
               SUM(CASE WHEN bl.governance_sentiment = 1 THEN 1 ELSE 0 END) as gov_pos,
               SUM(CASE WHEN bl.governance_sentiment = -1 THEN 1 ELSE 0 END) as gov_neg,
               SUM(CASE WHEN bl.digital_sentiment = 1 THEN 1 ELSE 0 END) as dig_pos,
               SUM(CASE WHEN bl.digital_sentiment = -1 THEN 1 ELSE 0 END) as dig_neg
        FROM articles a
        JOIN brand_labels bl ON a.id = bl.article_id
        WHERE a.labeled_at >= :since_date
        GROUP BY bl.brand
        ORDER BY labels DESC
    '''), {'since_date': since_date})
    for r in result:
        print(f'  {r.brand}: {r.labels} labels | E(+{r.env_pos}/-{r.env_neg}) S(+{r.soc_pos}/-{r.soc_neg}) G(+{r.gov_pos}/-{r.gov_neg}) D(+{r.dig_pos}/-{r.dig_neg})')

    # 2. Articles with negative sentiment (potential issues to review)
    print('\n' + '-' * 70)
    print('ARTICLES WITH NEGATIVE SENTIMENT (Review for accuracy)')
    print('-' * 70)
    result = conn.execute(text('''
        SELECT a.title, bl.brand,
               bl.environmental_sentiment as env,
               bl.social_sentiment as soc,
               bl.governance_sentiment as gov,
               bl.digital_sentiment as dig,
               a.url
        FROM articles a
        JOIN brand_labels bl ON a.id = bl.article_id
        WHERE a.labeled_at >= :since_date
          AND (bl.environmental_sentiment = -1
               OR bl.social_sentiment = -1
               OR bl.governance_sentiment = -1
               OR bl.digital_sentiment = -1)
        ORDER BY a.labeled_at DESC
        LIMIT 10
    '''), {'since_date': since_date})
    rows = result.fetchall()
    if rows:
        for r in rows:
            cats = []
            if r.env == -1: cats.append('E-')
            if r.soc == -1: cats.append('S-')
            if r.gov == -1: cats.append('G-')
            if r.dig == -1: cats.append('D-')
            print(f'  [{r.brand}] {\" \".join(cats)}: {r.title[:60]}...')
            print(f'    URL: {r.url}')
    else:
        print('  No negative sentiment articles in this period')

    # 3. Articles marked false_positive (spot check)
    # Note: FP-classifier-flagged articles have labeled_at=NULL, so use created_at
    # Content preview uses full_content column (see queries/article_queries.sql for reference)
    print('\n' + '-' * 70)
    print('RECENT FALSE POSITIVES (Spot check for missed ESG content)')
    print('-' * 70)
    result = conn.execute(text('''
        SELECT a.id, a.title, a.source_name, a.url, a.brands_mentioned,
               LEFT(a.full_content, 500) as content_preview
        FROM articles a
        WHERE a.labeling_status = 'false_positive'
          AND a.created_at >= :since_date
        ORDER BY a.created_at DESC
        LIMIT 10
    '''), {'since_date': since_date})
    rows = result.fetchall()
    if rows:
        for r in rows:
            brands = ', '.join(r.brands_mentioned) if r.brands_mentioned else 'unknown'
            print(f'  [{r.source_name}] ({brands}) {r.title[:60]}...')
            print(f'    ID: {r.id}')
            print(f'    URL: {r.url}')
            if r.content_preview:
                print(f'    Preview: {r.content_preview[:200]}...')
    else:
        print('  No false positives in this period')

    # 4. Substantive analyst articles mislabeled as false_positive
    # Per docs/LABELING.md v1.7.0: boilerplate template stock articles = false_positive (correct)
    # but substantive analyst articles (named analysts, specific commentary) = skipped
    # Exclude known boilerplate aggregator sources that are correctly false_positive
    print('\n' + '-' * 70)
    print('SUBSTANTIVE ANALYST ARTICLES (false_positive -> should be skipped)')
    print('-' * 70)
    result = conn.execute(text('''
        SELECT id, title, source_name,
               LEFT(full_content, 500) as content_preview
        FROM articles
        WHERE labeling_status = 'false_positive'
          AND created_at >= :since_date
          AND (
            LOWER(title) LIKE '%analyst%'
            OR LOWER(title) LIKE '%rating%'
            OR LOWER(title) LIKE '%upgrade%'
            OR LOWER(title) LIKE '%downgrade%'
            OR LOWER(title) LIKE '%price target%'
            OR LOWER(title) LIKE '%raises%target%'
            OR LOWER(title) LIKE '%lowers%target%'
            OR LOWER(title) LIKE '%forecasters%'
            OR LOWER(title) LIKE '%consensus%'
          )
          AND (
            LOWER(title) LIKE '%columbia sportswear%'
            OR LOWER(title) LIKE '%under armour%'
            OR LOWER(title) LIKE '%nike%'
            OR LOWER(title) LIKE '%nke%'
            OR LOWER(title) LIKE '%lululemon%'
            OR LOWER(title) LIKE '%lulu%'
            OR LOWER(title) LIKE '%adidas%'
            OR LOWER(title) LIKE '%addyy%'
            OR LOWER(title) LIKE '%puma se%'
            OR LOWER(title) LIKE '%pumsy%'
            OR LOWER(title) LIKE '%deckers%'
            OR LOWER(title) LIKE '%deck%'
            OR LOWER(title) LIKE '%anta%'
            OR LOWER(title) LIKE '%asics%'
          )
          AND LOWER(source_name) NOT IN (
            'marketbeat', 'defenseworld net', 'defenseworld.net',
            'americanbankingnews', 'themarketsdaily.com', 'markets daily',
            'the markets daily', 'daily political', 'dailypolitical.com',
            'the lincolnian online', 'tickerreport.com', 'ticker report',
            'bbns', 'zolmax', 'thelincolnianonline'
          )
        ORDER BY created_at DESC
        LIMIT 10
    '''), {'since_date': since_date})
    rows = result.fetchall()
    if rows:
        print(f'  Found {len(rows)} potentially mislabeled articles:')
        for r in rows:
            print(f'  [{r.source_name}] {r.title[:60]}...')
            print(f'    ID: {r.id}')
            if r.content_preview:
                print(f'    Preview: {r.content_preview[:200]}...')
        print('  → Verify these contain substantive analyst commentary (not just templates)')
        print('  → To view full article: uv run python scripts/fix_label.py show <ID>')
        print('  → To fix: uv run python scripts/fix_label.py update <ID> --status skipped')
    else:
        print('  ✓ No mislabeled substantive analyst articles found')

    # 5. Articles with multiple brands (check consistency)
    print('\n' + '-' * 70)
    print('MULTI-BRAND ARTICLES (Check for consistent labeling)')
    print('-' * 70)
    result = conn.execute(text('''
        SELECT a.title,
               STRING_AGG(bl.brand || ':' ||
                   COALESCE(CASE WHEN bl.environmental_sentiment IS NOT NULL
                            THEN 'E' || bl.environmental_sentiment END, '') ||
                   COALESCE(CASE WHEN bl.social_sentiment IS NOT NULL
                            THEN 'S' || bl.social_sentiment END, '') ||
                   COALESCE(CASE WHEN bl.governance_sentiment IS NOT NULL
                            THEN 'G' || bl.governance_sentiment END, '') ||
                   COALESCE(CASE WHEN bl.digital_sentiment IS NOT NULL
                            THEN 'D' || bl.digital_sentiment END, ''),
                   ' | ') as brand_labels,
               a.url
        FROM articles a
        JOIN brand_labels bl ON a.id = bl.article_id
        WHERE a.labeled_at >= :since_date
        GROUP BY a.id, a.title, a.url
        HAVING COUNT(DISTINCT bl.brand) > 1
        ORDER BY a.labeled_at DESC
        LIMIT 5
    '''), {'since_date': since_date})
    rows = result.fetchall()
    if rows:
        for r in rows:
            print(f'  {r.title[:55]}...')
            print(f'    Labels: {r.brand_labels}')
    else:
        print('  No multi-brand articles in this period')

print('\n' + '=' * 70)
"
```

## Step 3: Update State File

After completing the review, update the last run timestamp:

```bash
date +%Y-%m-%d > .claude/skills/review-labels/last_run.txt
```

## Step 4: Analysis

After running the queries:

1. **Brand Sentiment Breakdown**: Look for unexpected patterns (e.g., a brand with mostly negative coverage when they typically have positive ESG stories)

2. **Negative Sentiment Articles**: Click through to verify the negative sentiment is justified by the article content

3. **False Positives**: Spot check using the content preview. Remember:
   - `false_positive` = not actually about the sportswear brand (e.g., "Vans" = vehicles)
   - `skipped` = genuinely about the brand, but no ESG content (e.g., product reviews, analyst articles)
   - If an article IS about the brand but was marked `false_positive`, it should be `skipped`

4. **Substantive Analyst Articles**: Per [docs/LABELING.md](../../docs/LABELING.md#stock-article-classification) v1.7.0:
   - **Boilerplate template articles** (MarketBeat, DefenseWorld, DailyPolitical, etc.) → correctly `false_positive` — these are auto-generated aggregations with no original analysis
   - **Substantive analyst articles** (named analysts with specific commentary, original analysis from reputable sources like Reuters, Bloomberg, Seeking Alpha) → should be `skipped`, not `false_positive`
   - The query excludes known boilerplate aggregator sources, so flagged articles are likely genuine mislabels

5. **Multi-Brand Articles**: Verify brands in the same article got consistent treatment

## Step 5: Apply Corrections

Use `scripts/fix_label.py` to verify and correct any flagged articles:

```bash
# View full article details and content for verification
uv run python scripts/fix_label.py show <article-id>

# Correct one or more articles
uv run python scripts/fix_label.py update <id> [<id>...] --status skipped

# List valid statuses and their meanings
uv run python scripts/fix_label.py statuses
```

For ad-hoc queries, see `queries/article_queries.sql` and `queries/labeling_queries.sql` for reference SQL using the correct schema (e.g., `full_content` column for article text).

Summarize findings and flag any articles that may need relabeling or further investigation.
