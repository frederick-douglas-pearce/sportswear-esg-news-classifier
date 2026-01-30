---
name: esg-status
description: Show ESG news classifier project status: collection stats, labeling progress, and recent runs
---

# ESG Project Status

Run this command to get the current project status:

```bash
uv run python -c "
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    print('=' * 60)
    print('ESG NEWS CLASSIFIER STATUS')
    print('=' * 60)

    # Collection stats (last 7 days)
    result = conn.execute(text('''
        SELECT COUNT(*) as runs,
               COALESCE(SUM(articles_fetched), 0) as fetched,
               COALESCE(SUM(articles_scraped), 0) as scraped,
               COALESCE(SUM(articles_scrape_failed), 0) as failed
        FROM collection_runs WHERE started_at >= NOW() - INTERVAL '7 days'
    '''))
    row = result.fetchone()
    print(f'\nCOLLECTION (Last 7 days)')
    print(f'  Runs: {row.runs}, Fetched: {row.fetched}, Scraped: {row.scraped}, Failed: {row.failed}')

    # Labeling status
    print('\nLABELING STATUS')
    total = conn.execute(text('SELECT COUNT(*) FROM articles')).scalar()
    for status in ['labeled', 'false_positive', 'skipped', 'pending', 'unlabelable']:
        count = conn.execute(text(f\"SELECT COUNT(*) FROM articles WHERE labeling_status = '{status}'\")).scalar()
        pct = count/total*100 if total > 0 else 0
        print(f'  {status:<16} {count:>6} ({pct:.1f}%)')
    print(f'  TOTAL            {total:>6}')

    # Recent labeling runs
    print('\nRECENT LABELING RUNS')
    runs = conn.execute(text('''
        SELECT started_at::date as date,
               COUNT(*) as runs,
               SUM(articles_processed) as processed,
               SUM(brands_labeled) as brands,
               ROUND(SUM(estimated_cost_usd)::numeric, 2) as cost
        FROM labeling_runs
        WHERE started_at >= NOW() - INTERVAL '7 days'
        GROUP BY started_at::date
        ORDER BY date DESC
        LIMIT 5
    ''')).fetchall()
    for r in runs:
        print(f'  {r.date}: {r.runs} runs, {r.processed} articles, {r.brands} brands, \${r.cost}')
    if not runs:
        print('  No labeling runs in last 7 days')
"
```

After showing the status, summarize the key metrics and offer to drill down into:
- Collection details (scrape errors, sources)
- Labeling errors or anomalies
- Specific brand coverage
- Model/classifier status
