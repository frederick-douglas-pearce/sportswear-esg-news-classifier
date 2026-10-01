"""The committed model artifacts against the registry pointer (#176, D028).

Two checks with different lifetimes:

- The version/hash check asserts the committed config's stamp names the pointer and
  hashes the committed joblib. #176 proposed (handoff comment on #172) that #172's
  build guard, once it runs against the committed tree in CI, replace this check with
  a call to its guard function.
- The load-and-predict check loads the committed FP artifact under today's code and
  compares its probabilities with values captured from the served container. The guard
  D027 describes binds the model bytes, not the code the pickle imports (D028), so this
  check is not proposed for replacement.
"""

import json
from pathlib import Path

import pytest

from src.deployment.fp import FPClassifier
from src.deployment.versioning import file_sha256

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize(
    "classifier",
    [
        "fp",
        pytest.param(
            "ep",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="#180: EP's registered bytes are not recoverable; the committed EP artifact is unregistered",
            ),
        ),
    ],
)
def test_committed_config_stamp_names_the_pointer_and_hashes_the_joblib(classifier):
    registry = _read_json(MODELS_DIR / "registry.json")
    config = _read_json(MODELS_DIR / f"{classifier}_classifier_config.json")
    pipeline = MODELS_DIR / f"{classifier}_classifier_pipeline.joblib"

    assert config.get("version") == registry[classifier]["production"]
    assert config.get("pipeline_sha256") == file_sha256(pipeline)


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
