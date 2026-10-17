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
"""

from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, select

from .models import (
    FP_ACTION_SKIPPED_LLM,
    FP_ACTIONS,
    FP_REASON_AUTO_DETECTED_PREFIX,
    FP_REASON_MANUAL,
    JUDGED_SKIP_REASONS,
    UNJUDGED_SKIP_REASONS,
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
