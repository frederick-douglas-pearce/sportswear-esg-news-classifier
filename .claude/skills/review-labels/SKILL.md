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
    print(f'\nArticles reviewed: {row.articles}, Brand labels: {row.brand_labels}')

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
    print('\n' + '-' * 70)
    print('RECENT FALSE POSITIVES (Spot check for missed ESG content)')
    print('-' * 70)
    result = conn.execute(text('''
        SELECT title, source_name, url
        FROM articles
        WHERE labeling_status = 'false_positive'
          AND labeled_at >= :since_date
        ORDER BY labeled_at DESC
        LIMIT 5
    '''), {'since_date': since_date})
    rows = result.fetchall()
    if rows:
        for r in rows:
            print(f'  [{r.source_name}] {r.title[:65]}...')
            print(f'    URL: {r.url}')
    else:
        print('  No false positives in this period')

    # 4. Mislabeled analyst articles (should be skipped, not false_positive)
    # Per docs/LABELING.md: analyst ratings/price targets have substantive content
    print('\n' + '-' * 70)
    print('MISLABELED ANALYST ARTICLES (false_positive -> should be skipped)')
    print('-' * 70)
    result = conn.execute(text('''
        SELECT id, title, source_name
        FROM articles
        WHERE labeling_status = 'false_positive'
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
        ORDER BY created_at DESC
        LIMIT 10
    '''))
    rows = result.fetchall()
    if rows:
        print(f'  ⚠️  Found {len(rows)} articles that may need correction:')
        for r in rows:
            print(f'  [{r.source_name}] {r.title[:60]}...')
            print(f'    ID: {r.id}')
        print('  → These have analyst content and should likely be skipped, not false_positive')
    else:
        print('  ✓ No mislabeled analyst articles found')

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

3. **False Positives**: Spot check a few to ensure they truly aren't ESG-relevant

4. **Mislabeled Analyst Articles**: Per [docs/LABELING.md](../../docs/LABELING.md#stock-article-classification), articles with analyst ratings, price targets, or substantive financial commentary should be `skipped` (sent to LLM), not `false_positive`. If any are found:
   - Verify they contain substantive content (named analysts, specific ratings/targets)
   - Update status from `false_positive` to `skipped` using:
     ```sql
     UPDATE articles SET labeling_status = 'skipped', skipped_at = NOW()
     WHERE id = 'article-uuid-here';
     ```

5. **Multi-Brand Articles**: Verify brands in the same article got consistent treatment

Summarize findings and flag any articles that may need relabeling or further investigation.
