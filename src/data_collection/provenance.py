"""Where an article's FP training label came from, derived at query time (#154, D030).

Nothing here is stored: the source is a pure function of the article's status, its stored reason,
the latest FP prediction for it, and whether a human verified it. A stored column would have to be
re-synced on every relabel or correction.

Sources:
- ``llm``: the labeling LLM decided the label. For ``skipped`` rows this rests on the stored reason
  being one the pipeline writes after a judgment. For ``false_positive`` and ``labeled`` rows it is
  inferred from the presence of an FP prediction: a status moved by hand (``fix_label.py`` changes
  the status, never the reason) also reads as ``llm`` until #143's audit log records such moves.
- ``pre_classifier``: as ``llm``, but no FP prediction is recorded for the article: it was labeled
  before the FP classifier existed, or with the pre-filter off and no novelty score to record.
- ``fp_classifier``: a ``false_positive`` whose latest FP prediction is a skip (``skipped_llm``)
  and whose stored reason is that skip's own reason, so nothing has relabeled it since. Excluded
  from training by default.
- ``human``: a human ruled on it.
- ``unjudged``: a ``skipped`` row whose reply, or lack of one, does not establish whether the
  article has a sportswear brand (``UNJUDGED_SKIP_REASONS``). Excluded by default.
- ``unattributed``: a ``skipped`` row whose reason no pipeline path writes (no reason, or a reason
  left behind by a status move). Kept, and counted separately.
- ``rule``: a ``false_positive`` written by ``cleanup_false_positives.py``'s keyword sweep. Excluded
  by default.

An owner-attested cutoff ("articles before this date were reviewed") is deliberately not a source.
It is weaker than a per-article human ruling; the export offers it as a separate opt-in tier.

The FP export's query (``fp_export_query``) lives here, and so does the #192 backfill's cohort
(``select_backfill_cohort``, D039), which reads the same rows and sources, plus ``failed`` rows.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, NamedTuple

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import load_only

from .models import (
    FP_ACTION_SKIPPED_LLM,
    FP_ACTIONS,
    FP_REASON_AUTO_DETECTED_PREFIX,
    FP_REASON_MANUAL,
    JUDGED_SKIP_REASONS,
    SKIP_INSUFFICIENT_CONTENT,
    SKIP_NO_BRANDS,
    UNJUDGED_SKIP_REASONS,
    Article,
    BrandJudgment,
    BrandLabel,
    ClassifierPrediction,
)

LLM = "llm"
PRE_CLASSIFIER = "pre_classifier"
FP_CLASSIFIER = "fp_classifier"
HUMAN = "human"
UNJUDGED = "unjudged"
UNATTRIBUTED = "unattributed"
RULE = "rule"

LABEL_SOURCES = frozenset({LLM, PRE_CLASSIFIER, FP_CLASSIFIER, HUMAN, UNJUDGED, UNATTRIBUTED, RULE})
# Sources the FP training export leaves out unless asked; the export reads this set.
EXCLUDED_BY_DEFAULT = frozenset({FP_CLASSIFIER, UNJUDGED, RULE})

# Labeling statuses the FP export reads, and the class each one is.
FP_POSITIVE_STATUSES = ("labeled", "skipped")
FP_NEGATIVE_STATUS = "false_positive"
# Not an FP training label: the backfill's arm X reads it (#86)
FAILED_STATUS = "failed"

# Human rulings with no audit log yet: article id -> the status the human ruled. A ruling counts
# only while the article still has that status, so a later relabel is not called human.
# Source: #182's comments of 2026-10-09 (corrections, skipped -> false_positive) and 2026-10-10
# (rows confirmed as correctly labeled). #143 backfills these into its audit log and deletes this.
HUMAN_RULINGS: dict[str, str] = {
    # 2026-10-09: corrected to false_positive
    "aabc7f6e-8fc5-4c02-9880-b9e404bd840a": "false_positive",
    "f69140d1-cf99-48a2-8663-22dbae05af1a": "false_positive",
    "16497d46-e8cc-49b6-bd2d-2f0c297d5f5a": "false_positive",
    "d2b995a3-fa6d-4e45-9c42-2d35f6cd1c3f": "false_positive",
    "9ea08c90-be9a-48b1-a7e0-9c3405875fde": "false_positive",
    "852d660f-850c-45e5-9e0c-e8a8cf944208": "false_positive",
    "ad4950e8-8edc-4978-ba0e-ebf6448105a5": "false_positive",
    "477bff74-57b0-4f9c-b552-55a6bba2d611": "false_positive",
    "f5f9f1d3-ab3a-4e05-bafe-709650482b94": "false_positive",
    "0d56c213-d05a-4134-a8fa-08c6b4002076": "false_positive",
    "39a66305-56cf-4b24-9ba9-180d0498cb15": "false_positive",
    "4944b79e-e5af-43e6-b4b8-04624c58f75f": "false_positive",
    "a48bbba6-28aa-46cc-8d91-a54ef93df934": "false_positive",
    "382f2cd1-1b78-453c-8e7f-d91901e4a9f6": "false_positive",
    "3054e143-3cac-4864-854a-3a4b0723b7e6": "false_positive",
    "ae913840-b9ad-4403-a886-73a331f26abf": "false_positive",
    "2029c74e-2b8c-4f63-a200-8520d9ef8d94": "false_positive",
    # 2026-10-10: confirmed as labeled correctly
    "fbefe459-5ddf-4e05-b092-1bbc31b823cb": "skipped",
    "21825f8a-d923-4451-97e3-c305a0490a1e": "skipped",
    "eed1f353-8d0b-40c8-b9ce-cfacb6d865d0": "skipped",
    "0a21c3cd-b005-4401-9c44-21000e795e43": "skipped",
    "80a3cee0-a752-42ca-a593-a575846782e6": "skipped",
    "d8b66c00-ca1b-4e3f-9f6a-ed9509975566": "skipped",
    "55b54027-afc5-4566-ab7f-f6d393bc8343": "labeled",
}


class UnknownFPActionError(ValueError):
    """An FP prediction carries an action_taken this module does not know how to attribute."""


def human_verified_ids(candidates: Iterable[tuple[str, str]]) -> set[str]:
    """Ids, among ``(article_id, labeling_status)`` pairs, whose status a human verified.

    Today the only source is ``HUMAN_RULINGS``. #143's audit log and #146's confirmed review samples
    are further sources; they need a database query, so this signature takes a session when they
    land (D030).
    """
    return {
        article_id
        for article_id, status in candidates
        if HUMAN_RULINGS.get(article_id) == status
    }


def derive_label_source(
    labeling_status: str,
    labeling_error: str | None,
    latest_fp_action: str | None,
    latest_fp_skip_reason: str | None = None,
    *,
    human_verified: bool = False,
    article_id: Any = None,
) -> str:
    """Return the source of an FP training label. See the module docstring for the vocabulary.

    ``latest_fp_action`` and ``latest_fp_skip_reason`` are the ``action_taken`` and ``skip_reason``
    of the article's most recent FP prediction, or ``None`` if it has none. Reasons are matched by
    exact text (prefix for the keyword sweep).

    Raises:
        UnknownFPActionError: ``latest_fp_action`` is not one of ``FP_ACTIONS``.
        ValueError: ``labeling_status`` is not one the FP export reads.
    """
    if latest_fp_action is not None and latest_fp_action not in FP_ACTIONS:
        raise UnknownFPActionError(
            f"Unrecognized FP action_taken {latest_fp_action!r} on article {article_id}"
        )
    llm_era = LLM if latest_fp_action is not None else PRE_CLASSIFIER

    if human_verified:
        return HUMAN

    if labeling_status == FP_NEGATIVE_STATUS:
        if labeling_error == FP_REASON_MANUAL:
            return HUMAN
        if labeling_error and labeling_error.startswith(FP_REASON_AUTO_DETECTED_PREFIX):
            return RULE
        # The pipeline stores an FP skip's reason as the prediction's skip_reason. A different
        # stored reason means something relabeled the article after the skip without recording a
        # new prediction (an LLM run with the pre-filter off and no novelty score), so the skip
        # did not decide this label.
        if latest_fp_action == FP_ACTION_SKIPPED_LLM and labeling_error == latest_fp_skip_reason:
            return FP_CLASSIFIER
        return llm_era

    if labeling_status == "skipped":
        # The FP classifier never writes 'skipped', so the latest FP action says nothing about
        # who chose this status; only the stored reason does.
        if labeling_error in UNJUDGED_SKIP_REASONS:
            return UNJUDGED
        if labeling_error in JUDGED_SKIP_REASONS:
            return llm_era
        return UNATTRIBUTED

    if labeling_status == "labeled":
        return llm_era

    raise ValueError(f"No FP training label for labeling_status {labeling_status!r}")


def latest_prediction_subquery(classifier_type: str):
    """Subquery of each article's most recent prediction from one classifier.

    Columns: ``article_id``, ``action_taken``, ``skip_reason``, ``created_at``. Outer-join it to
    ``Article`` on ``article_id``; an article with no prediction gets NULLs. Ties on ``created_at``
    are broken by ``id``, which is arbitrary but fixed.
    """
    ranked = (
        select(
            ClassifierPrediction.article_id,
            ClassifierPrediction.action_taken,
            ClassifierPrediction.skip_reason,
            ClassifierPrediction.created_at,
            func.row_number()
            .over(
                partition_by=ClassifierPrediction.article_id,
                order_by=(
                    ClassifierPrediction.created_at.desc().nulls_last(),
                    ClassifierPrediction.id.desc(),
                ),
            )
            .label("rn"),
        )
        .where(ClassifierPrediction.classifier_type == classifier_type)
        .subquery()
    )
    return (
        select(
            ranked.c.article_id, ranked.c.action_taken, ranked.c.skip_reason, ranked.c.created_at
        )
        .where(ranked.c.rn == 1)
        .subquery("latest_prediction")
    )


def fp_export_query(session, since: datetime | None = None, *, also_failed: bool = False):
    """Articles the FP export reads, each with its latest FP prediction, in a fixed order.

    The order matters: the training splits use ``train_test_split(random_state=42)``, which
    assigns rows by position, so an unordered query would split the same data differently. The
    #192 backfill selects its cohort from the same rows (``select_backfill_cohort``), plus the
    ``failed`` rows (``also_failed``), which the export never reads.
    """
    base_filter = Article.scrape_status == "success"
    if since:
        base_filter = and_(base_filter, Article.created_at >= since)
    statuses = [*FP_POSITIVE_STATUSES, FP_NEGATIVE_STATUS, *([FAILED_STATUS] if also_failed else [])]

    latest = latest_prediction_subquery("fp")
    return (
        session.query(Article, latest.c.action_taken, latest.c.skip_reason, latest.c.created_at)
        .outerjoin(latest, latest.c.article_id == Article.id)
        .filter(base_filter)
        .filter(Article.labeling_status.in_(statuses))
        .order_by(Article.id)
    )


# --- The #192 backfill cohort (D039) ---------------------------------------------------------
# Rows whose stored status the LLM decided, or could decide, without a stored judgment for each
# brand, and failed calls the LLM never judged (#86): the backfill relabels them under the
# production prompt. One arm table, so a later backfill (#224) adds an arm rather than a second
# selection.

ARM_UNJUDGED = "unjudged"  # U: a skip that never judged sportswear status, at a reason a reply writes
ARM_FALSE_POSITIVE = "false_positive"  # F: an LLM-decided negative
ARM_JUDGED_SKIP = "judged_skip"  # J: a judged skip written since the production version's first run
ARM_FAILED = "failed"  # X: a failed call the LLM never judged, most with no reply at all (#86)
BACKFILL_ARMS = (ARM_UNJUDGED, ARM_FALSE_POSITIVE, ARM_JUDGED_SKIP, ARM_FAILED)
# Statuses an arm reads
BACKFILL_STATUSES = ("skipped", FP_NEGATIVE_STATUS, FAILED_STATUS)

# Unjudged reasons a reply writes, or wrote (SKIP_NO_SPORTSWEAR_BRAND: the pipeline before #200).
# The other two are written before any LLM call, and a relabel would write them again.
LLM_REACHABLE_UNJUDGED_REASONS = UNJUDGED_SKIP_REASONS - {SKIP_INSUFFICIENT_CONTENT, SKIP_NO_BRANDS}

# Why a row is left out of the cohort. Reported, never selected.
EXCLUDED_DETERMINISTIC = "deterministic_skip"  # an unjudged skip a pipeline check writes
EXCLUDED_HUMAN = "human"  # a HUMAN_RULINGS key whatever its status, or a human-sourced label
EXCLUDED_HUMAN_LABELS = "human_reviewed_labels"  # a brand label a human reviewed or wrote
EXCLUDED_POSSIBLE_HAND_MOVE = "possible_hand_move"
EXCLUDED_NO_BRANDS = "no_brands"
EXCLUDED_INSUFFICIENT_CONTENT = "insufficient_content"
EXCLUDED_HAS_JUDGMENTS = "has_judgments"  # a brand_judgments row exists: relabeled, or decided since D038
EXCLUDED_SKIPPED_AFTER_BOUND = "skipped_after_bound"  # skipped at or after the bound (U and J only)
EXCLUDED_NOT_SCRAPED = "not_scraped"  # at a cohort status, but outside the FP export's scrape filter
EXCLUDED_FAILED_IN_BACKFILL = "failed_in_backfill"  # a tagged run wrote this failure (X only)


def _brand_key(name: str) -> str:
    """``normalize_brand``'s key (``src/labeling/models.py``), restated so this module does not
    load the labeling package."""
    return name.strip().casefold()


def possible_hand_move(labeling_error: str | None, brands: Sequence[str]) -> bool:
    """Whether a ``false_positive`` row's reason is not in the form an LLM reply writes, for a
    brand the article mentions.

    A reply's negative is ``"<brand>: <reason>"`` per brand, joined by ``"; "`` (``dispose_reply``;
    every pipeline version since false-positive detection), so the first part names a brand that
    was sent. Fail-closed: every other reason counts as a possible hand move, since
    ``fix_label.py update`` changes the status and keeps the reason. That covers a move from
    ``failed`` (the labeler's error), from ``skipped`` (a skip reason) or from ``deduplicated``,
    a reason a human wrote, none, and an FP classifier skip's reason on a row an LLM did not
    decide. It stands in for #143's audit log, which will record the move itself.
    """
    if not labeling_error:
        return True
    name, separator, _ = labeling_error.split("; ", 1)[0].partition(": ")
    return not separator or _brand_key(name) not in {_brand_key(b) for b in brands}


class BackfillDecision(NamedTuple):
    """A row's arm, or why it is left out. Both None: not a backfill candidate at all."""

    arm: str | None
    excluded: str | None = None


NOT_A_CANDIDATE = BackfillDecision(None)


def as_utc(moment: datetime) -> datetime:
    """A naive timestamp is taken as UTC, so it compares with an aware one."""
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def backfill_arm(
    *,
    article_id: str,
    labeling_status: str,
    labeling_error: str | None,
    source: str,
    skipped_at: datetime | None,
    brands: Sequence[str],
    sufficient_content: bool,
    has_judgments: bool,
    human_labels: bool,
    judged_skip_cutoff: datetime | None,
    bound: datetime,
    tried_reason: str | None = None,
) -> BackfillDecision:
    """The backfill arm of one row, or why it is left out (D039).

    ``source`` is ``derive_label_source``'s, with ``human_verified`` from ``human_verified_ids``
    (None for a ``failed`` row, which has no FP training label).
    ``judged_skip_cutoff`` is the production prompt version's first run (None: no J arm).
    ``bound`` ends the pass: a U or J row skipped at or after it is not selected, so a row the
    backfill skipped again, and a skip the cron writes meanwhile, are not retried. A NULL
    ``skipped_at`` is before every bound.
    ``tried_reason`` is the reason a tagged run recorded for the row before relabeling it (None:
    no tagged run has). It ends arm X's pass, since a ``failed`` write records no timestamp.
    """
    if labeling_status not in BACKFILL_STATUSES:
        return NOT_A_CANDIDATE
    if source == HUMAN or article_id in HUMAN_RULINGS:
        return BackfillDecision(None, EXCLUDED_HUMAN)

    if labeling_status == FAILED_STATUS:
        arm = ARM_FAILED
    elif labeling_status == "skipped" and source == UNJUDGED:
        if labeling_error not in LLM_REACHABLE_UNJUDGED_REASONS:
            return BackfillDecision(None, EXCLUDED_DETERMINISTIC)
        arm = ARM_UNJUDGED
    elif labeling_status == FP_NEGATIVE_STATUS and source in (LLM, PRE_CLASSIFIER):
        arm = ARM_FALSE_POSITIVE
    elif (
        labeling_status == "skipped"
        and source in (LLM, PRE_CLASSIFIER)
        and judged_skip_cutoff is not None
        and skipped_at is not None
        and as_utc(skipped_at) >= as_utc(judged_skip_cutoff)
    ):
        arm = ARM_JUDGED_SKIP
    else:
        return NOT_A_CANDIDATE

    if human_labels:
        return BackfillDecision(None, EXCLUDED_HUMAN_LABELS)
    if not brands:
        return BackfillDecision(None, EXCLUDED_NO_BRANDS)
    # After the brand check: with no brands, no reason can name one
    if arm == ARM_FALSE_POSITIVE and possible_hand_move(labeling_error, brands):
        return BackfillDecision(None, EXCLUDED_POSSIBLE_HAND_MOVE)
    if not sufficient_content:
        return BackfillDecision(None, EXCLUDED_INSUFFICIENT_CONTENT)
    if has_judgments:
        return BackfillDecision(None, EXCLUDED_HAS_JUDGMENTS)
    # F needs no bound: every reply writes judgments or another status, and a row with no reply
    # keeps its status to be retried
    if (
        arm in (ARM_UNJUDGED, ARM_JUDGED_SKIP)
        and skipped_at is not None
        and as_utc(skipped_at) >= as_utc(bound)
    ):
        return BackfillDecision(None, EXCLUDED_SKIPPED_AFTER_BOUND)
    # X: a row a tagged run tried that is now failed with another reason was failed by that run
    # (a reply the labeler could not use), so it is not retried. A row whose status the run kept
    # (a transient failure, a call with no reply, an exception) still has the recorded reason.
    if arm == ARM_FAILED and tried_reason is not None and labeling_error != tried_reason:
        return BackfillDecision(None, EXCLUDED_FAILED_IN_BACKFILL)
    return BackfillDecision(arm)


@dataclass
class BackfillRow:
    """One row of the FP export query, as the backfill reads it."""

    article_id: str
    labeling_status: str
    labeling_error: str | None
    fp_action: str | None
    fp_skip_reason: str | None
    skipped_at: datetime | None
    created_at: datetime | None
    brands: list[str]
    content_chars: int
    sufficient_content: bool


HAND_MOVE_REASON_CHARS = 40


@dataclass
class BackfillCohort:
    """The selected ids by arm, what each had before the relabel, and what was left out."""

    judged_skip_cutoff: datetime | None
    bound: datetime
    selected: dict[str, list[str]] = field(default_factory=lambda: {a: [] for a in BACKFILL_ARMS})
    # article id -> [status, reason, arm] before the relabel
    prior: dict[str, list[str | None]] = field(default_factory=dict)
    content_chars: dict[str, int] = field(default_factory=dict)
    excluded: Counter = field(default_factory=Counter)
    # (arm, stored reason) for U and J; (arm, label source) for F, whose reasons are free text;
    # (arm, the reason's first HAND_MOVE_REASON_CHARS) for X, whose errors carry request ids
    by_reason: Counter = field(default_factory=Counter)
    # (arm, UTC year-month the article was collected)
    by_month: Counter = field(default_factory=Counter)
    # selected articles that still carry non-human brand_labels rows
    stale_labels: int = 0
    # article id -> why it was left out, for each row with an exclusion
    left_out: dict[str, str] = field(default_factory=dict)
    # the first HAND_MOVE_REASON_CHARS of each possible hand move's reason, so the owner sees what
    # the fail-closed test left out
    hand_move_reasons: Counter = field(default_factory=Counter)
    # article id -> the reason the first tagged run to relabel it recorded (arm X's pass end)
    tried: dict[str, str | None] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        """Every selected id, arm by arm in BACKFILL_ARMS order, by id within an arm."""
        return [i for arm in BACKFILL_ARMS for i in self.selected[arm]]


def build_backfill_cohort(
    rows: Iterable[BackfillRow],
    *,
    judged_ids: set[str],
    human_label_ids: set[str],
    label_ids: set[str],
    judged_skip_cutoff: datetime | None,
    bound: datetime,
    tried: dict[str, str | None] | None = None,
) -> BackfillCohort:
    """The cohort over rows in id order; ``backfill_arm`` decides each row. ``tried`` is
    ``tried_reasons`` over the tagged runs so far."""
    rows = list(rows)
    tried = dict(tried or {})
    cohort = BackfillCohort(judged_skip_cutoff=judged_skip_cutoff, bound=bound, tried=tried)
    verified = human_verified_ids((r.article_id, r.labeling_status) for r in rows)
    for r in rows:
        if r.labeling_status not in BACKFILL_STATUSES:
            continue
        source = None if r.labeling_status == FAILED_STATUS else derive_label_source(
            r.labeling_status,
            r.labeling_error,
            r.fp_action,
            r.fp_skip_reason,
            human_verified=r.article_id in verified,
            article_id=r.article_id,
        )
        decision = backfill_arm(
            article_id=r.article_id,
            labeling_status=r.labeling_status,
            labeling_error=r.labeling_error,
            source=source,
            skipped_at=r.skipped_at,
            brands=r.brands,
            sufficient_content=r.sufficient_content,
            has_judgments=r.article_id in judged_ids,
            human_labels=r.article_id in human_label_ids,
            judged_skip_cutoff=judged_skip_cutoff,
            bound=bound,
            tried_reason=tried.get(r.article_id),
        )
        if decision.excluded:
            cohort.excluded[decision.excluded] += 1
            cohort.left_out[r.article_id] = decision.excluded
        if decision.excluded == EXCLUDED_POSSIBLE_HAND_MOVE:
            cohort.hand_move_reasons[(r.labeling_error or "")[:HAND_MOVE_REASON_CHARS]] += 1
        if decision.arm is None:
            continue
        arm = decision.arm
        cohort.selected[arm].append(r.article_id)
        cohort.prior[r.article_id] = [r.labeling_status, r.labeling_error, arm]
        cohort.content_chars[r.article_id] = r.content_chars
        if arm == ARM_FALSE_POSITIVE:
            reason_key = source
        elif arm == ARM_FAILED:
            reason_key = (r.labeling_error or "")[:HAND_MOVE_REASON_CHARS]
        else:
            reason_key = r.labeling_error
        cohort.by_reason[(arm, reason_key)] += 1
        month = as_utc(r.created_at).strftime("%Y-%m") if r.created_at else "unknown"
        cohort.by_month[(arm, month)] += 1
        if r.article_id in label_ids:
            cohort.stale_labels += 1
    return cohort


def tried_reasons(run_configs: Iterable[dict | None]) -> dict[str, str | None]:
    """Article id -> the reason the first tagged run to relabel it recorded, from the runs'
    configs in start order. A retried article's first record is its reason from before the
    backfill."""
    tried: dict[str, str | None] = {}
    for config in run_configs:
        for article_id, values in (((config or {}).get("run_config") or {}).get("prior") or {}).items():
            tried.setdefault(article_id, values[1])
    return tried


def _cohort_from_db(
    session,
    *,
    judged_skip_cutoff: datetime | None,
    bound: datetime,
    tried: dict[str, str | None] | None = None,
    ids: list[str] | None = None,
) -> BackfillCohort:
    """``build_backfill_cohort`` over the FP export's rows and the ``failed`` rows, or only
    those with the given ids."""
    # Imported here: the labeling package's __init__ loads the whole pipeline, and the FP export
    # imports this module without needing it
    from src.labeling.article_input import article_content, has_sufficient_content

    query = fp_export_query(session, also_failed=True).options(
        load_only(
            Article.id,
            Article.labeling_status,
            Article.labeling_error,
            Article.skipped_at,
            Article.created_at,
            Article.brands_mentioned,
            Article.full_content,
            Article.description,
        )
    )
    if ids is not None:
        query = query.filter(Article.id.in_(ids))
    rows = []
    for article, fp_action, fp_skip_reason, _ in query.all():
        content = article_content(article.full_content, article.description)
        rows.append(
            BackfillRow(
                article_id=str(article.id),
                labeling_status=article.labeling_status,
                labeling_error=article.labeling_error,
                fp_action=fp_action,
                fp_skip_reason=fp_skip_reason,
                skipped_at=article.skipped_at,
                created_at=article.created_at,
                brands=list(article.brands_mentioned or []),
                content_chars=len(content),
                sufficient_content=has_sufficient_content(content),
            )
        )

    def article_ids(column, *criteria) -> set[str]:
        query = session.query(column).filter(*criteria)
        if ids is not None:
            query = query.filter(column.in_(ids))
        return {str(i) for (i,) in query.distinct()}

    human = or_(BrandLabel.human_reviewed.is_(True), BrandLabel.labeled_by == "human")
    return build_backfill_cohort(
        rows,
        judged_ids=article_ids(BrandJudgment.article_id),
        human_label_ids=article_ids(BrandLabel.article_id, human),
        label_ids=article_ids(BrandLabel.article_id),
        judged_skip_cutoff=judged_skip_cutoff,
        bound=bound,
        tried=tried,
    )


def select_backfill_cohort(
    session,
    *,
    judged_skip_cutoff: datetime | None,
    bound: datetime,
    tried: dict[str, str | None] | None = None,
) -> BackfillCohort:
    """The #192 backfill cohort from the database (D039): the FP export's rows and the
    ``failed`` rows (#86), decided by ``backfill_arm``, plus a count of rows at a cohort status
    outside the export's scrape filter.
    """
    cohort = _cohort_from_db(session, judged_skip_cutoff=judged_skip_cutoff, bound=bound, tried=tried)
    not_scraped = (
        session.query(func.count(Article.id))
        .filter(Article.labeling_status.in_(BACKFILL_STATUSES))
        .filter(or_(Article.scrape_status.is_(None), Article.scrape_status != "success"))
        .scalar()
    )
    if not_scraped:
        cohort.excluded[EXCLUDED_NOT_SCRAPED] += not_scraped
    return cohort


def recheck_backfill_batch(
    session, cohort: BackfillCohort, ids: list[str]
) -> tuple[list[str], dict[str, str]]:
    """The batch's ids still selected as they were, and why each other one was dropped.

    The cohort is selected once per invocation, which can run for hours, so each batch is decided
    again just before it is labeled, with the same cutoff and bound. A row whose arm, status or
    reason is not what the cohort recorded is dropped, so a human's move during the run is not
    overwritten. Why: its exclusion, or ``changed``.
    """
    fresh = _cohort_from_db(
        session,
        judged_skip_cutoff=cohort.judged_skip_cutoff,
        bound=cohort.bound,
        tried=cohort.tried,
        ids=ids,
    )
    kept, dropped = [], {}
    for i in ids:
        if fresh.prior.get(i) == cohort.prior[i]:
            kept.append(i)
        else:
            dropped[i] = fresh.left_out.get(i, "changed")
    return kept, dropped
