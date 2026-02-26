"""YAML-based storage for experiment logs and knowledge bases."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import (
    ClassifierSummary,
    Decision,
    ExperimentEntry,
    ExperimentIndex,
    IndexEntry,
    KnowledgeBase,
    SharedKnowledgeBase,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path("data/experiments")


class ExperimentStore:
    """Manages experiment log storage with YAML persistence."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or DEFAULT_BASE_DIR

    # --- Directory helpers ---

    def _classifier_dir(self, classifier: str) -> Path:
        """Get (and create) classifier experiment directory."""
        d = self.base_dir / classifier
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _decisions_dir(self, classifier: str) -> Path:
        """Get (and create) decisions subdirectory for a classifier."""
        d = self._classifier_dir(classifier) / "decisions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _knowledge_dir(self) -> Path:
        """Get (and create) knowledge directory."""
        d = self.base_dir / "knowledge"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _index_path(self) -> Path:
        return self.base_dir / "index.yaml"

    # --- Atomic write helper ---

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        """Write YAML atomically via temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = path.with_suffix(".yaml.tmp")
        with open(temp_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        temp_file.rename(path)

    def _load_yaml(self, path: Path) -> dict[str, Any] | None:
        """Load YAML file, returning None if missing."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    # --- Experiment CRUD ---

    def _experiment_path(self, experiment_id: str, classifier: str) -> Path:
        """Get file path for an experiment."""
        # ID format: {classifier}_{YYYYMMDD_HHMMSS}
        # File: data/experiments/{classifier}/exp_{YYYYMMDD_HHMMSS}.yaml
        suffix = experiment_id.removeprefix(f"{classifier}_")
        return self._classifier_dir(classifier) / f"exp_{suffix}.yaml"

    def create_experiment(self, entry: ExperimentEntry) -> Path:
        """Write a new experiment entry and update the index."""
        path = self._experiment_path(entry.experiment_id, entry.classifier)
        if path.exists():
            raise FileExistsError(f"Experiment file already exists: {path}")
        self._atomic_write(path, entry.model_dump(mode="json"))
        self._update_index(entry)
        logger.info(f"Created experiment {entry.experiment_id} at {path}")
        return path

    def load_experiment(self, experiment_id: str) -> ExperimentEntry:
        """Load an experiment by ID.

        Raises:
            FileNotFoundError: If experiment file doesn't exist.
        """
        # Extract classifier from ID prefix
        classifier = experiment_id.split("_")[0]
        path = self._experiment_path(experiment_id, classifier)
        data = self._load_yaml(path)
        if data is None:
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        return ExperimentEntry.model_validate(data)

    def update_experiment(self, entry: ExperimentEntry) -> None:
        """Update an existing experiment entry and index."""
        path = self._experiment_path(entry.experiment_id, entry.classifier)
        self._atomic_write(path, entry.model_dump(mode="json"))
        self._update_index(entry)
        logger.info(f"Updated experiment {entry.experiment_id}")

    def list_experiments(
        self, classifier: str, limit: int = 10
    ) -> list[IndexEntry]:
        """List experiments from index (no full file reads), newest first."""
        index = self.load_index()
        entries = index.experiments.get(classifier, [])
        return entries[:limit]

    # --- Index ---

    def load_index(self) -> ExperimentIndex:
        """Load the experiment index, creating empty one if missing."""
        data = self._load_yaml(self._index_path())
        if data is None:
            return ExperimentIndex(last_updated=datetime.now(timezone.utc))
        return ExperimentIndex.model_validate(data)

    def _update_index(self, entry: ExperimentEntry) -> None:
        """Update the index with an experiment entry."""
        index = self.load_index()
        classifier = entry.classifier

        # Build index entry
        idx_entry = IndexEntry(
            id=entry.experiment_id,
            status=entry.status,
            action_type=entry.action.type,
            action_summary=entry.action.summary,
            test_f2=entry.observation.metrics.get("test_f2"),
            delta_f2=entry.observation.delta.get("test_f2"),
            promoted=entry.reward.promoted,
            tags=entry.tags,
        )

        # Update or insert in classifier list
        if classifier not in index.experiments:
            index.experiments[classifier] = []

        entries = index.experiments[classifier]
        # Replace existing entry if present
        entries = [e for e in entries if e.id != entry.experiment_id]
        entries.insert(0, idx_entry)
        # Keep sorted newest-first (by ID which contains timestamp)
        entries.sort(key=lambda e: e.id, reverse=True)
        index.experiments[classifier] = entries

        # Update summary
        summary = index.summary.get(classifier, ClassifierSummary())
        summary.total_experiments = len(entries)
        summary.total_promoted = sum(1 for e in entries if e.promoted)
        if idx_entry.test_f2 is not None:
            summary.best_test_f2_ever = max(
                summary.best_test_f2_ever,
                idx_entry.test_f2,
            )
        if entry.reward.promoted and entry.reward.promoted_as:
            summary.current_production = entry.reward.promoted_as
        index.summary[classifier] = summary

        index.last_updated = datetime.now(timezone.utc)
        self._atomic_write(self._index_path(), index.model_dump(mode="json"))

    # --- Knowledge base ---

    def load_knowledge(self, classifier: str) -> KnowledgeBase:
        """Load per-classifier knowledge base, returning empty if missing."""
        path = self._knowledge_dir() / f"{classifier}_knowledge.yaml"
        data = self._load_yaml(path)
        if data is None:
            return KnowledgeBase(
                classifier=classifier,
                last_updated=datetime.now(timezone.utc),
            )
        return KnowledgeBase.model_validate(data)

    def save_knowledge(self, knowledge: KnowledgeBase) -> None:
        """Save per-classifier knowledge base, incrementing version."""
        knowledge.version += 1
        knowledge.last_updated = datetime.now(timezone.utc)
        path = self._knowledge_dir() / f"{knowledge.classifier}_knowledge.yaml"
        self._atomic_write(path, knowledge.model_dump(mode="json"))
        logger.info(
            f"Saved {knowledge.classifier} knowledge v{knowledge.version}"
        )

    def load_shared_knowledge(self) -> SharedKnowledgeBase:
        """Load shared cross-classifier knowledge base."""
        path = self._knowledge_dir() / "shared_knowledge.yaml"
        data = self._load_yaml(path)
        if data is None:
            return SharedKnowledgeBase(
                last_updated=datetime.now(timezone.utc),
            )
        return SharedKnowledgeBase.model_validate(data)

    def save_shared_knowledge(self, knowledge: SharedKnowledgeBase) -> None:
        """Save shared knowledge base, incrementing version."""
        knowledge.version += 1
        knowledge.last_updated = datetime.now(timezone.utc)
        path = self._knowledge_dir() / "shared_knowledge.yaml"
        self._atomic_write(path, knowledge.model_dump(mode="json"))
        logger.info(f"Saved shared knowledge v{knowledge.version}")

    # --- Decisions ---

    def save_decision(self, decision: Decision) -> Path:
        """Save a decision to the decisions subdirectory."""
        d = self._decisions_dir(decision.classifier)
        path = d / f"{decision.decision_id}.yaml"
        self._atomic_write(path, decision.model_dump(mode="json"))
        logger.info(f"Saved decision {decision.decision_id}")
        return path

    def load_decisions(self, experiment_id: str) -> list[Decision]:
        """Load all decisions for a given experiment."""
        classifier = experiment_id.split("_")[0]
        d = self._decisions_dir(classifier)
        decisions: list[Decision] = []
        for path in sorted(d.glob("*.yaml")):
            data = self._load_yaml(path)
            if data and data.get("experiment_id") == experiment_id:
                decisions.append(Decision.model_validate(data))
        return decisions

    # --- Queries ---

    def get_recent_experiments(
        self, classifier: str, n: int = 5
    ) -> list[ExperimentEntry]:
        """Load full entries for the N most recent experiments."""
        index_entries = self.list_experiments(classifier, limit=n)
        results: list[ExperimentEntry] = []
        for idx_entry in index_entries:
            try:
                results.append(self.load_experiment(idx_entry.id))
            except FileNotFoundError:
                logger.warning(f"Index references missing experiment: {idx_entry.id}")
        return results

    def get_parent_chain(
        self, experiment_id: str, depth: int = 3
    ) -> list[ExperimentEntry]:
        """Follow parent_experiment links up to depth."""
        chain: list[ExperimentEntry] = []
        current_id: str | None = experiment_id
        for _ in range(depth):
            if current_id is None:
                break
            try:
                entry = self.load_experiment(current_id)
            except FileNotFoundError:
                break
            chain.append(entry)
            current_id = entry.parent_experiment
        return chain

    def find_experiments_by_tag(
        self, classifier: str, tag: str
    ) -> list[IndexEntry]:
        """Filter index entries by tag."""
        index = self.load_index()
        entries = index.experiments.get(classifier, [])
        return [e for e in entries if tag in e.tags]
