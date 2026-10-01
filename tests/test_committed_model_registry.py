"""The committed model artifacts against the registry pointer (#176, D028, D029).

Three checks with different lifetimes:

- The version/hash check asserts the committed config's stamp names the pointer and
  hashes the committed joblib. #176 proposed (handoff comment on #172) that #172's
  build guard, once it runs against the committed tree in CI, replace this check with
  a call to its guard function.
- The load-and-predict check loads the committed FP artifact under today's code and
  compares its probabilities with values captured from the served container. The guard
  D027 describes binds the model bytes, not the code the pickle imports (D028), so this
  check is not proposed for replacement.
- The retired-EP check asserts that EP has no production pointer, keeps its v1.0.0
  record unchanged, and has no tracked artifact (#180, D029). Any change to
  `registry["ep"]` fails it, including registering an EP version that is not promoted,
  so it is updated when a retrained EP is registered. A companion check keeps the EP
  compose service opt-in, so a plain `docker compose up` does not build it.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from src.deployment.fp import FPClassifier
from src.deployment.versioning import file_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# models/registry.json's EP entry as #180 left it: no pointer, and the v1.0.0 record
# kept unchanged as the reference for a retrained EP.
RETIRED_EP_REGISTRY = {
    "production": None,
    "versions": {
        "v1.0.0": {
            "created_at": "2024-12-23T00:00:00Z",
            "trained_on": "data/ep_training_data.jsonl",
            "model_name": "LR_tuned",
            "transformer_method": "tfidf_lsa",
            "threshold": 0.7237,
            "metrics": {
                "cv_f2": 0.9311,
                "cv_recall": 1.0,
                "cv_precision": 0.7299,
                "test_f2": 0.9311,
                "test_recall": 1.0,
                "test_precision": 0.7299,
            },
        }
    },
}


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _git_tracked_files_under(directory: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", directory],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return [name for name in result.stdout.split("\0") if name]


@pytest.mark.parametrize("classifier", ["fp"])
def test_committed_config_stamp_names_the_pointer_and_hashes_the_joblib(classifier):
    registry = _read_json(MODELS_DIR / "registry.json")
    config = _read_json(MODELS_DIR / f"{classifier}_classifier_config.json")
    pipeline = MODELS_DIR / f"{classifier}_classifier_pipeline.joblib"

    assert config.get("version") == registry[classifier]["production"]
    assert config.get("pipeline_sha256") == file_sha256(pipeline)


def test_ep_pointer_is_retired_and_no_ep_artifact_is_committed():
    registry = _read_json(MODELS_DIR / "registry.json")

    assert registry["ep"] == RETIRED_EP_REGISTRY
    # What git tracks (the index; HEAD in CI), not the working tree: a local notebook or
    # train.py run writes models/ep_* without adding it. One listing, filtered here rather
    # than by a git pathspec, so registry.json's presence in it shows the listing is real
    # and the empty ep_ selection below is evidence of absence.
    tracked = _git_tracked_files_under("models")
    assert "models/registry.json" in tracked
    assert [name for name in tracked if name.startswith("models/ep_")] == []


# Captured 2026-09-30 from the local fp-classifier-api container's /predict/batch
# (the image the labeling pipeline calls, serving registry FP v2.4.0's bytes). The
# request, response, container and in-container hashes are recorded on #176:
# https://github.com/frederick-douglas-pearce/sportswear-esg-news-classifier/issues/176#issuecomment-5922595169
SERVED_FP_PREDICTIONS = [
    (
        {
            "title": "Nike expands recycled polyester use across its running apparel",
            "content": "Nike said on Tuesday that more of its running apparel will use recycled polyester, part of a plan to cut carbon emissions in its supply chain. The sportswear company also reported progress on factory worker safety audits.",
            "brands": ["Nike"],
            "source_name": "Reuters",
            "category": ["business"],
        },
        0.9989012521926607,
    ),
    (
        {
            "title": "Puma spotted near hiking trail prompts park warning",
            "content": "Wildlife officials closed a hiking trail after a puma was seen near a campground. Rangers said the mountain lion appeared healthy and advised visitors to keep children and pets close.",
            "brands": ["Puma"],
            "source_name": "Local News",
            "category": ["environment"],
        },
        0.6007323772645654,
    ),
    (
        {
            "title": "Patagonia glacier tourism grows as visitors flock south",
            "content": "Tourism in the Patagonia region of Argentina and Chile rose this year, with visitors hiking to glaciers and national parks. Local guides said demand for trekking tours has never been higher.",
            "brands": ["Patagonia"],
            "source_name": "Travel Weekly",
            "category": ["travel"],
        },
        0.25,
    ),
    (
        {
            "title": "Under Armour shares fall after weak quarterly forecast",
            "content": "Shares of Under Armour dropped after the athletic apparel maker cut its annual sales outlook, citing softer demand in North America and higher discounting. Analysts said the brand faces pressure from Nike and Adidas.",
            "brands": ["Under Armour"],
            "source_name": "MarketWatch",
            "category": ["business"],
        },
        0.991411492206625,
    ),
]


@pytest.fixture(scope="module")
def committed_fp_classifier() -> FPClassifier:
    return FPClassifier(
        pipeline_path=str(MODELS_DIR / "fp_classifier_pipeline.joblib"),
        config_path=str(MODELS_DIR / "fp_classifier_config.json"),
    )


def test_committed_fp_artifact_reports_the_pointer_version(committed_fp_classifier):
    registry = _read_json(MODELS_DIR / "registry.json")

    assert committed_fp_classifier.version == registry["fp"]["production"]


@pytest.mark.parametrize(
    ("article", "served_probability"),
    SERVED_FP_PREDICTIONS,
    ids=[article["brands"][0] for article, _ in SERVED_FP_PREDICTIONS],
)
def test_committed_fp_artifact_reproduces_the_served_probabilities(
    committed_fp_classifier, article, served_probability
):
    result = committed_fp_classifier.predict_from_fields(**article)

    assert result["probability"] == pytest.approx(served_probability, abs=1e-9)


def test_ep_compose_service_is_opt_in():
    with open(REPO_ROOT / "docker-compose.yml") as f:
        compose = yaml.safe_load(f)

    assert compose["services"]["ep-classifier-api"]["profiles"] == ["ep"]
