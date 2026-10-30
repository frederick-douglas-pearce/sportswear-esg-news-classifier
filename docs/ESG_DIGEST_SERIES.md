# ESG Digest Blog Series: Research and Plan

Status: **planning** (2026-10-29; owner decisions recorded 2026-10-30). Nothing here is built yet.
This records what was found when surveying the three sibling repos that publish to the Pages site,
what the database can support, and the recommended shape of the series. Decisions and the questions
still open are at the end.

## The idea

A biweekly post summarizing the sportswear ESG news the pipeline labeled in the past two weeks,
with a longer-window trends section (2 to 3 months). Posts are published to the Jekyll Pages site
(`frederick-douglas-pearce.github.io`, blog name "Ground Truth") with the same folder structure,
CI gates and publish-on-merge workflow the sibling repos use. The summarization prompt is versioned
because it will need iteration, and again as the data changes.

## What the data can support (measured 2026-10-29)

Counts are by `published_at`, `labeling_status = 'labeled'`, before deduplication.

| Window                     | Labeled articles | Notes                                                                      |
| -------------------------- | ---------------- | -------------------------------------------------------------------------- |
| Last 14 days               | 10               | 6 brands; Nike 4, Puma 2; 0 Environmental, 7 Social, 5 Governance labels   |
| Last 90 days               | 128              | Weekly counts range 1 to 20; median two-week window is about 15            |
| Same 90 days, all statuses | 1,150            | 463 skipped, 316 false positive, 229 unlabelable, 10 deduplicated, 4 pending |

Daily `scorecard_snapshots` exist from 2026-02-10 to today (253 rows), with per-brand ranks in
`scorecard_brand_scores`.

What this means for the series:

- **The two-week section is a short digest, not a survey.** Expect 5 to 20 distinct stories after
  deduplication, often concentrated on two or three brands. Some windows will have no Environmental
  story at all. The format has to work when the window is thin, and say so when it is, rather than
  pad.
- **The trends section is where the volume is,** but 128 articles over 50 brands means most brands
  have n of 3 or fewer. Trend claims need per-brand counts printed next to them, and brand-level
  "trends" below a stated minimum n should be left out. That fits the voice already ("n=2 is below
  threshold"), and the scorecard history gives a deterministic rank series to anchor it.
- **The corpus is "what this pipeline saw," not "the industry's ESG news."** Collection is GDELT plus
  NewsData.io with a domain blocklist, so coverage is a sample with its own biases. Every post should
  say so once, plainly.
- **Labels are still being corrected.** #192's backfill moved rows between `skipped`,
  `false_positive` and `labeled`, and #230 (policy rulings from the review) is next. A digest written
  now summarizes a labeled set that is still changing, and the 463 skipped rows in the 90-day window
  are exactly the cohort the cleanup is re-deciding. See open question 1.
- **Frozen snapshots vs relabeled data.** A scorecard snapshot records ranks as they were computed
  that day. After a relabel, a trend read from snapshots and a trend recomputed from today's labels
  can disagree. The data pack (below) should pick one source per number and name it.
- **A prompt change is a step in the series.** v1.12.0 (production since 2026-10-23) gives product
  and financial articles no ESG categories, so it labels fewer articles than v1.10.0 did. A 90-day
  window currently spans that change. The pack records the prompt versions behind each window's
  labels, and the trends section says when it compares across one.

### Data density is part of the story, stated as a funnel

The 14-day window that produced 10 labeled articles started from 150 collected: 66 skipped,
36 unlabelable, 34 false positive, 10 labeled, 4 pending. "Not much happened" is a real statement,
but only about the stage that measured it:

| Stage                                 | What a low count means                                       |
| ------------------------------------- | ------------------------------------------------------------ |
| Collected                             | Little coverage of these brands (or a collection problem)    |
| Unlabelable / false positive          | Pipeline loss or brand-name noise, says nothing about ESG    |
| Judged sportswear, no ESG content     | **The real "quiet" signal:** the brands were in the news, not for ESG |
| Labeled                               | ESG news, with categories and sentiment                      |

So each issue opens with the funnel ("150 articles mentioned these brands; N were about the
brands' own business; K had ESG content"), and compares K with the 90-day median. A quiet fortnight
then reads as a finding with a baseline, not as an excuse. The pack has to split `skipped` by
reason to get the third row, which the provenance code (`src/data_collection/provenance.py`)
already distinguishes.

## How the sibling repos publish

Three repos publish into the same Pages `_posts/` namespace, each with its own series category:
`claude-code-sessions`, `us_presidential_vote_analysis` and `claude-code-loop`
(`display_categories` in the site's `_config.yml`). The pattern is the same in all three, ported
sessions → vote (#132) → loop. Each repo's `posts/README.md` is the authoritative spec.

### The pieces

| Piece                                    | What it does                                                                                                                                                                       | When it runs                                                       |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `tooling/publish-to-pages.py`            | Transforms frontmatter (strips upstream-only fields), resolves and copies each post's OG card, writes only changed bytes, refuses to overwrite a card another repo published (#157) | Called by `pages-sync.yml`; runnable locally with `--dry-run`      |
| `.github/workflows/pages-sync.yml`       | Cross-repo checkout with a fine-grained PAT, then commit and a bounded reconcile-retry push to Pages `main`                                                                        | Push to `main` touching `posts/**`; `workflow_dispatch` dry run    |
| `tooling/check-og-cards.py`              | PR guard: runs the publisher's own `build_plan`, so a post without a resolvable card fails on the PR                                                                               | `og-card-guard.yml`, PRs and `main` pushes                         |
| `.github/workflows/prettier.yml`         | PR guard: `posts/` must be Prettier-clean under the site's exact pinned versions                                                                                                  | Every PR and `main` push                                           |
| `tooling/check-humanizer-pass.py`        | PR guard: every post records `humanizer_pass: vX.Y.Z` or `none`                                                                                                                    | `humanizer-guard.yml`                                              |
| `tooling/render-og-card.py`              | Renders a 1200×630 card from a committed TOML brief (PEP 723 header, Inkscape on PATH)                                                                                             | By hand, before the PR                                             |
| `tooling/attestation.py` (loop only)     | One definition of the `vX.Y.Z`-or-`none` grammar shared by the frontmatter test and the humanizer guard                                                                            | Imported                                                           |
| `tests/test_posts_frontmatter.py` (loop) | Every field required and non-empty, date matches filename                                                                                                                          | The repo's required test job                                       |

Supporting files: `package.json` + `package-lock.json` (Prettier and
`@shopify/prettier-plugin-liquid` pinned exactly), `.prettierrc`, a `.prettierignore` that ignores
everything except `posts/`, `AI-DISCLOSURE.md`, `LICENSE-prose.md` (CC BY 4.0 for `posts/`, with the
root `LICENSE` excluding it).

**Shipping a post is a merge.** Draft in gitignored `social/drafts/`, move the finished post into
`posts/` on a branch, run the humanizer pass as its own commit, open a PR, let the guards run,
squash-merge. `pages-sync` publishes from `main` only; a dispatch from another ref must be a dry run.

### Which repo to port from: `claude-code-loop`

| Difference          | sessions                | vote                     | loop                           |
| ------------------- | ----------------------- | ------------------------ | ------------------------------ |
| Prettier pin        | 3.9.5                   | 3.9.5                    | **3.9.6, matches the site**    |
| Namespace guard     | not ported              | yes (#157)               | yes                            |
| OG card location    | `social/images/`        | `social/images/` carve-out | `posts/images/<slug>/`       |
| Frontmatter test    | no                      | no                       | yes, plus shared attestation grammar |
| `claims_verified`   | no                      | no                       | yes                            |
| Publisher wiring tests | partial              | partial                  | yes (#199)                     |

The site now pins Prettier **3.9.6**; sessions and vote still pin 3.9.5. That is the drift their
own READMEs call "the one way it can pass here and still fail there." Out of scope for this repo,
but worth fixing in those two.

Port from loop, with these adaptations:

- **OG cards in `posts/images/<slug>/`.** This repo's `.gitignore` already ignores `social/*`, the
  same reasoning loop used.
- **Tests go in `tests/` under pytest**, so they run in the existing `ci.yml` (loop uses
  `unittest` on bare `python3`; keep the tooling itself stdlib-only so the guards need no `uv sync`).
- **This repo has no `LICENSE` file.** The prose license split needs a root license to cut against.
  Decide before the first post.
- **Keep the namespace guard.** This repo is the fourth publisher, and it is also the source of the
  site's daily ESG-feed commits (`website_export`, from the `-feed` worktree). The reconcile-retry
  loop already handles that race; the feed commit subjects (`Update ESG news feed - …`) do not look
  like a sync subject, so provenance stays unambiguous.

### Owner setup (one-time, outside this repo)

1. Fine-grained PAT, contents read/write on the Pages repo only, named for this repo.
2. GitHub Environment `pages-sync` in this repo with secret `PAGES_SYNC_TOKEN`.
3. Add the new category slug to the site's `display_categories`, or the filter chip never appears.
4. Verify with a `dry_run` dispatch before the first real publish.

## Recommended architecture for the digest

Three layers, each doing the thing it is good at.

### 1. A deterministic data pack (code, tested)

`scripts/build_digest_pack.py` writes one JSON file per issue: the window, every labeled article in
it (title, source, URL, date, brands, categories, sentiment, top evidence excerpts by rerank score),
deduplicated with the scorecard's own embedding dedup (0.56), plus the trend tables (per-brand,
per-category counts and sentiment over the long window, and scorecard rank movement), each with its
n. The pack records its query parameters, row counts and a content hash.

Why code and not prompt: every number in a post must be re-derivable, and an LLM counting rows is
the failure mode. The prompt receives numbers; it never computes them. This is also what makes a
prompt change testable: the same pack under two prompt versions isolates the prompt.

### 2. A versioned summarization prompt

`prompts/digest/` with the same layout as `prompts/labeling/`: `registry.json` with a `production`
pointer and `model_recommendations`, and `vX.Y.Z/` directories holding `system_prompt.txt`,
`user_prompt.txt`, `config.json`. `PromptManager` already takes a `prompts_dir`, so it can load
these without changes.

A version is only reproducible if the model and its parameters are pinned with it. That argues for
calling the prompt through the API from a script (`scripts/draft_digest.py --pack … --version …`),
not running it as free-form instructions to whatever model the Claude Code session is on. Each post
records the prompt version, model and pack hash in upstream-only frontmatter (stripped on publish,
like `og_card_source`), so any post can be regenerated.

Iteration loop: keep the raw prompt output as `social/drafts/<date>-<slug>.raw.md` beside the edited
version, as the sibling repos do for the marketer. The raw-vs-shipped diff is the eval signal for the
next prompt version. A lighter version of D036 applies: compare two versions on the same packs before
moving `production`, judged on grounding (every claim traces to a pack row), coverage, and edit
distance to what shipped. Do not build a scoring harness until there are several issues to compare.

### 3. A skill that orchestrates, and the marketer for the framing

A project skill (`.claude/skills/esg-digest/SKILL.md`) runs one issue end to end: build the pack,
show its summary (counts, thin categories, below-threshold brands) for a go/no-go, draft with the
production prompt, save raw and working copies to `social/drafts/`, run the claim check against the
pack, then hand off to the post PR. The skill holds the procedure; the prompt version holds the
words. Putting the prompt text inside `SKILL.md` would version it only through git history, with no
pinned model and no registry.

**The marketer is the wrong tool for the digest body, and the right one around it.** Its voice
signatures are first-person narrative: self-deprecation openers, contrast pairs, optimization
framing. A news digest reports what third parties did, with citations, in a neutral register. It
also has no database access. Use it for:

- the series launch post (why this series exists, what the pipeline is, its limits),
- the one-paragraph editorial framing at the top of each issue, if the owner wants a personal note,
- the LinkedIn post announcing each issue,
- series planning (`social/series-outline.md`).

`marketer-retrospective` then calibrates those pieces, and the digest prompt's own versions
calibrate the body.

## Editorial guardrails (proposed)

The vote repo's guardrail is "nothing critical of a data source is published." This series has a
different risk: it summarizes news, much of it negative, about named companies.

- **Every factual claim links to the source article,** and says what the article reported, not what
  happened ("Reuters reported that …"). No claim that is not in a cited article.
- **Short quotes only,** from the stored evidence excerpts, linked to the source. No reproducing
  article text.
- **Sentiment is the pipeline's label, presented as such.** "The classifier labeled this negative on
  worker rights," not "Brand X behaved badly."
- **`claims_verified` is required,** as in the loop series: the date every claim was checked against
  the pack and its source URL. LLM summarization of 15 articles can still attribute a claim to the
  wrong brand.
- **State the window, the n, and the collection caveat** in every issue.

## Series shape (draft)

- **Cadence:** every two weeks, window aligned to the scorecard's default 14-day period, so the post
  and the `/esg-news` page agree.
- **Category:** one slug, for example `sportswear-esg` (to add to `display_categories`). Tags carry
  the kind (`digest`, `trends`, `launch`) and topics (`esg`, `sustainability`, brand names).
- **Sections:** a short first-person note; the funnel; what happened (by ESG category, brands named,
  cited); notable trend changes since the previous issue over the trend window (90 days as a
  placeholder, decision 5; narrative, with n, below-threshold changes left out); a methods footer (window, collection sources, prompt
  version, link to the live `/esg-news` page and the repo).
- **Charts:** the site has Chart.js, ECharts and Plotly enabled via post frontmatter, so trend charts
  can be data in the markdown, with no image assets for the publisher to handle. The publisher
  copies only the OG card today; inline images would need an extension.
- **OG card:** one series chassis with a per-issue specimen (for example the window's category-mix
  bar), rendered from a committed TOML brief.

## Build order (proposed)

1. Port the publishing machinery from `claude-code-loop` (one PR, no post; `pages-sync` stays inert
   until a dated post exists). Owner setup in parallel.
2. Data pack script with tests.
3. `prompts/digest/v0.1.0` and the draft script; generate a few back-issues from past windows as
   the prompt's first iteration set (not published).
4. The `esg-digest` skill.
5. Launch post (marketer), then the first digest.

## Decisions (owner, 2026-10-30)

1. **Build now, publish after the label cleanup (#230) settles.**
2. **Prompt runs through the API with a pinned model,** so it can be automated later (cron or an
   agent workflow). Automation drafts; it does not publish. The cron builds the pack, drafts the
   issue and opens a PR; `humanizer_pass` and `claims_verified` are human attestations, so merging
   stays a person's act. A `digest_drafting` workflow under `src/agent/workflows/` is the natural
   home once the scripts exist.
3. **Two-week window, biweekly.** A week would be more actionable but the data flow cannot support
   it; four weeks would be stale. The pack script takes the window as a parameter, so if collection
   volume grows the cadence can tighten without code changes. Data density is reported in every
   issue as the funnel above.
4. **The trends section reports only notable changes since the previous issue,** as narrative: for
   example, coverage of a brand's environmental record turning positive after a new initiative,
   following months of negative coverage. It is not a standing table repeated every issue. Two
   consequences:
   - "Since the previous issue" means the previous issue's pack as published, not a recomputation
     (relabels would change it). Packs must be stored where the next issue can read them; where
     (committed or not, given they hold titles, URLs and excerpts) is for the epic to settle.
   - A "turnaround" claim is causal and the data is thin. The prompt states only what coverage
     showed, with n on both sides ("coverage shifted from 4 negative to 3 positive items after X
     launched"), never why a brand changed, and a change below a minimum n is not reported as a
     trend. The minimum is to be set from the packs.
5. **Trend window length is unsettled; 90 days is a placeholder.** 90 days is a quarter but shifts
   only about 15% between biweekly issues; 60 days shifts about 25%, so changes surface sooner, and
   it is simpler to explain. The trade-off is roughly two-thirds the n per brand. Decide from back-issue
   packs generated at both lengths.
6. **MIT `LICENSE` for the repo,** with `LICENSE-prose.md` (CC BY 4.0) for `posts/`, as in the
   siblings.
7. **A short first-person note opens each issue, with some optimism.** The optimism has to be
   sourced: the pack lists the window's positive-sentiment items (and the 90-day ones as a
   fallback), and the note or a "bright spot" section draws from them with a citation. When the
   window has none, the note says so rather than inventing one. Under cron automation the body is
   drafted by the pinned prompt, and the note is written at PR time (by the owner, or the marketer
   in an interactive pass), so the first-person voice stays the owner's.

## Still open

- **Trend window length** (decision 5).
- **Series title and category slug.** Candidates are below; the slug can be settled independently
  (for example `sportswear-esg`).

| Title                                              | Note                                                                 |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| What the Labels Say                                | Clothing labels and ESG labels; honest that it is a classifier's view |
| Footprints                                         | Shoes and carbon footprint; short, works as a card wordmark          |
| Split Times                                        | Running splits, a two-week interval                                  |
| Running Tally                                      | A count kept as it goes; fits the scorecard                          |
| What's Happening in Sportswear Sustainability?     | Owner's draft; clear, long for a card, and promises breadth the data may not have |
