"""Data models for experiment logging and ML agent learning."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Experiment Entry (RL-parallel: state → action → observation → reward → reflection) ---


class ExperimentState(BaseModel):
    """Snapshot of the world before the experiment (RL: state)."""

    production_version: str | None = None
    production_metrics: dict[str, float] = Field(default_factory=dict)
    training_data: dict[str, Any] = Field(default_factory=dict)
    feature_config: dict[str, Any] = Field(default_factory=dict)
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    known_issues: list[str] = Field(default_factory=list)


class ExperimentChange(BaseModel):
    """A single change within an action."""

    component: str  # feature_config | model_config | training_data | labels
    field: str
    old_value: Any = None
    new_value: Any = None
    rationale: str = ""


class ExperimentAction(BaseModel):
    """What was changed (RL: action)."""

    type: str  # feature_engineering | hyperparameter_tuning | data_change | etc.
    summary: str
    changes: list[ExperimentChange] = Field(default_factory=list)
    hypothesis: str = ""


class ExperimentObservation(BaseModel):
    """Raw results (RL: observation)."""

    metrics: dict[str, float] = Field(default_factory=dict)
    delta: dict[str, Any] = Field(default_factory=dict)
    error_analysis: dict[str, Any] = Field(default_factory=dict)
    feature_importance: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class ExperimentReward(BaseModel):
    """Evaluation against goals (RL: reward)."""

    primary_metric: str = "test_f2"
    primary_value: float | None = None
    primary_target: float | None = None
    target_met: bool = False
    constraints_met: dict[str, bool] = Field(default_factory=dict)
    promoted: bool = False
    promoted_as: str | None = None
    promotion_reason: str = ""


class ExperimentReflection(BaseModel):
    """Qualitative analysis — the learning (RL: policy update)."""

    summary: str = ""
    hypothesis_result: str = ""  # confirmed | refuted | inconclusive
    hypothesis_evidence: str = ""
    surprises: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    confidence: str = "medium"  # low | medium | high
    confidence_notes: str = ""


class ExperimentEntry(BaseModel):
    """Complete experiment log entry."""

    experiment_id: str
    classifier: str  # fp | ep | esg
    created_at: datetime
    completed_at: datetime | None = None
    status: str = "running"  # running | completed | failed | abandoned
    parent_experiment: str | None = None
    tags: list[str] = Field(default_factory=list)

    state: ExperimentState = Field(default_factory=ExperimentState)
    action: ExperimentAction
    observation: ExperimentObservation = Field(default_factory=ExperimentObservation)
    reward: ExperimentReward = Field(default_factory=ExperimentReward)
    reflection: ExperimentReflection = Field(default_factory=ExperimentReflection)
    extensions: dict[str, Any] = Field(default_factory=dict)


# --- Decision (judgment calls within experiments) ---


class DecisionOption(BaseModel):
    """A candidate option considered during a decision."""

    id: str
    description: str
    expected_outcome: str = ""
    risk: str = "medium"  # low | medium | high


class Decision(BaseModel):
    """Tracks an individual judgment call within an experiment."""

    decision_id: str
    experiment_id: str
    classifier: str
    timestamp: datetime
    phase: str  # eda | feature_engineering | model_selection | ...
    trigger_type: str  # metric_observation | error_pattern | ...
    trigger_description: str
    trigger_evidence: list[str] = Field(default_factory=list)
    options: list[DecisionOption] = Field(default_factory=list)
    chosen: str
    reasoning: str
    precedents: list[dict[str, str]] = Field(default_factory=list)
    outcome_success: bool | None = None
    outcome_result: str = ""
    lesson_learned: str = ""


# --- Knowledge Base (accumulated learnings) ---


class Pattern(BaseModel):
    """An observed pattern from experiments."""

    id: str
    category: str  # feature_engineering | error_analysis | data_quality
    pattern: str
    confidence: str = "medium"
    evidence: list[dict[str, str]] = Field(default_factory=list)
    first_observed: str = ""
    last_confirmed: str = ""
    contradictions: list[dict[str, str]] = Field(default_factory=list)


class Heuristic(BaseModel):
    """A learned rule of thumb derived from patterns."""

    id: str
    trigger: str
    action: str
    confidence: str = "medium"
    derived_from: list[str] = Field(default_factory=list)
    times_applied: int = 0
    times_successful: int = 0


class DiagnosticStep(BaseModel):
    """A single step in a diagnostic tree."""

    check: str
    if_yes: str = ""
    if_no: str = ""


class DiagnosticTree(BaseModel):
    """A decision tree for diagnosing issues."""

    id: str
    name: str
    trigger: str
    steps: list[DiagnosticStep] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)


class KnowledgeBase(BaseModel):
    """Per-classifier accumulated knowledge."""

    classifier: str  # "fp", "ep", etc.
    last_updated: datetime
    version: int = 1
    patterns: list[Pattern] = Field(default_factory=list)
    heuristics: list[Heuristic] = Field(default_factory=list)
    baselines: dict[str, Any] = Field(default_factory=dict)
    diagnostic_trees: list[DiagnosticTree] = Field(default_factory=list)


class SharedPattern(BaseModel):
    """Cross-classifier pattern in shared_knowledge.yaml."""

    id: str
    pattern: str
    applies_to: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    evidence: list[dict[str, str]] = Field(default_factory=list)
    source_classifier: str = ""


class SharedKnowledgeBase(BaseModel):
    """Knowledge that generalizes across classifiers."""

    last_updated: datetime
    version: int = 1
    patterns: list[SharedPattern] = Field(default_factory=list)


# --- Index (quick lookup) ---


class IndexEntry(BaseModel):
    """Summary entry for quick lookup without reading full experiment files."""

    id: str
    status: str
    action_type: str
    action_summary: str
    test_f2: float | None = None
    delta_f2: float | None = None
    promoted: bool = False
    tags: list[str] = Field(default_factory=list)


class ClassifierSummary(BaseModel):
    """Aggregate stats for a classifier."""

    total_experiments: int = 0
    total_promoted: int = 0
    current_production: str = ""
    best_test_f2_ever: float = 0.0


class ExperimentIndex(BaseModel):
    """Index of all experiments for quick lookup."""

    last_updated: datetime
    experiments: dict[str, list[IndexEntry]] = Field(default_factory=dict)
    summary: dict[str, ClassifierSummary] = Field(default_factory=dict)
