"""Tests for reporting the loaded artifact's model version (#115, D026).

The version recorded in classifier_predictions must describe the artifact the
API loaded: stamped by every registry writer, verified against the loaded
pipeline's hash, carried through /model/info, and recorded by the pipeline
with a distinct value for each way it can be missing.
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from src.deployment.versioning import (
    DISABLED,
    NON_VERSION_SENTINELS,
    UNAVAILABLE,
    UNKNOWN,
    UNREPORTED,
    UNVERSIONED,
    file_sha256,
    stamp_artifact_version,
)
from src.labeling.classifier_client import ClassifierClient, FPPredictionResult
from src.labeling.pipeline import LabelingPipeline

PIPELINE_BYTES = b"pretend joblib bytes"
BASE_CONFIG = {
    "threshold": 0.6,
    "model_name": "TestModel",
    "transformer_method": "tfidf",
    "test_f2": 0.93,
}


@pytest.fixture
def artifact(tmp_path):
    """An unregistered artifact: a pipeline file and a config with no version."""
    pipeline_path = tmp_path / "fp_classifier_pipeline.joblib"
    pipeline_path.write_bytes(PIPELINE_BYTES)
    config_path = tmp_path / "fp_classifier_config.json"
    config_path.write_text(json.dumps(BASE_CONFIG))
    return pipeline_path, config_path


def _read(path):
    return json.loads(Path(path).read_text())


def _load_fp_classifier(pipeline_path, config_path):
    from src.deployment.fp.classifier import FPClassifier

    with patch("joblib.load", return_value=MagicMock()):
        return FPClassifier(pipeline_path=str(pipeline_path), config_path=str(config_path))


class TestSentinels:
    def test_sentinels_are_distinct(self):
        values = [UNVERSIONED, UNREPORTED, UNAVAILABLE, DISABLED, UNKNOWN]
        assert len(set(values)) == len(values)
        assert NON_VERSION_SENTINELS == frozenset(values)

    def test_sentinel_strings_are_pinned(self):
        """These are persisted in classifier_predictions and queried later (#145)."""
        assert UNVERSIONED == "unversioned"
        assert UNREPORTED == "unreported"
        assert UNAVAILABLE == "unavailable"
        assert DISABLED == "disabled"
        assert UNKNOWN == "unknown"


class TestStampArtifactVersion:
    def test_writes_version_and_pipeline_hash(self, artifact):
        pipeline_path, config_path = artifact
        stamp_artifact_version(config_path, "v3.1.0", pipeline_path)

        config = _read(config_path)
        assert config["version"] == "v3.1.0"
        assert config["pipeline_sha256"] == hashlib.sha256(PIPELINE_BYTES).hexdigest()
        # Existing fields are preserved.
        assert config["threshold"] == BASE_CONFIG["threshold"]

    def test_warns_when_restamping_a_different_version(self, artifact, caplog):
        pipeline_path, config_path = artifact
        stamp_artifact_version(config_path, "v3.1.0", pipeline_path)
        with caplog.at_level(logging.WARNING, logger="src.deployment.versioning"):
            stamp_artifact_version(config_path, "v3.2.0", pipeline_path)

        assert "already carried version v3.1.0" in caplog.text
        assert _read(config_path)["version"] == "v3.2.0"


class TestLoadedArtifactVersion:
    def test_reports_stamped_version_when_hash_matches(self, artifact):
        pipeline_path, config_path = artifact
        stamp_artifact_version(config_path, "v3.1.0", pipeline_path)

        info = _load_fp_classifier(pipeline_path, config_path).get_model_info()

        assert info["version"] == "v3.1.0"
        assert info["artifact_sha256"] == hashlib.sha256(PIPELINE_BYTES).hexdigest()[:12]

    def test_unregistered_artifact_is_unversioned(self, artifact):
        pipeline_path, config_path = artifact

        info = _load_fp_classifier(pipeline_path, config_path).get_model_info()

        assert info["version"] == UNVERSIONED

    def test_version_without_pipeline_hash_is_unversioned(self, artifact, caplog):
        pipeline_path, config_path = artifact
        config_path.write_text(json.dumps({**BASE_CONFIG, "version": "v3.1.0"}))

        with caplog.at_level(logging.ERROR, logger="src.deployment.base"):
            info = _load_fp_classifier(pipeline_path, config_path).get_model_info()

        assert info["version"] == UNVERSIONED
        assert "is missing or does not match the loaded pipeline" in caplog.text

    def test_pipeline_hash_without_version_is_unversioned(self, artifact):
        pipeline_path, config_path = artifact
        config_path.write_text(
            json.dumps({**BASE_CONFIG, "pipeline_sha256": file_sha256(pipeline_path)})
        )

        info = _load_fp_classifier(pipeline_path, config_path).get_model_info()

        assert info["version"] == UNVERSIONED

    def test_config_beside_a_different_pipeline_is_unversioned(self, artifact, caplog):
        pipeline_path, config_path = artifact
        stamp_artifact_version(config_path, "v3.1.0", pipeline_path)
        # A new pipeline lands beside the old, stamped config (e.g. an interrupted notebook save).
        pipeline_path.write_bytes(b"a different model")

        with caplog.at_level(logging.ERROR, logger="src.deployment.base"):
            info = _load_fp_classifier(pipeline_path, config_path).get_model_info()

        assert info["version"] == UNVERSIONED
        assert "is missing or does not match the loaded pipeline" in caplog.text

    def test_model_info_endpoint_carries_loaded_version(self, artifact):
        """The version must survive /model/info's response_model (one of the missing links behind #115)."""
        from fastapi.testclient import TestClient

        import scripts.predict as predict_module

        pipeline_path, config_path = artifact
        stamp_artifact_version(config_path, "v3.1.0", pipeline_path)
        classifier = _load_fp_classifier(pipeline_path, config_path)

        with patch("scripts.predict.create_classifier", return_value=classifier), patch(
            "scripts.predict.ENABLE_PREDICTION_LOGGING", False
        ):
            with TestClient(predict_module.app) as client:
                predict_module.classifier = classifier
                data = client.get("/model/info").json()

        assert data["version"] == "v3.1.0"
        assert data["artifact_sha256"] == file_sha256(pipeline_path)[:12]


class TestRegistryWritersStamp:
    """Every script that writes a registry version stamps the artifact config."""

    @pytest.fixture
    def models_dir(self, tmp_path, monkeypatch):
        models = tmp_path / "models"
        models.mkdir()
        (models / "fp_classifier_pipeline.joblib").write_bytes(PIPELINE_BYTES)
        (models / "fp_classifier_config.json").write_text(json.dumps(BASE_CONFIG))
        (models / "registry.json").write_text(json.dumps({"fp": {"production": None, "versions": {}}}))
        monkeypatch.chdir(tmp_path)
        return models

    def _run_register(self, monkeypatch, *args):
        import scripts.register_model as register_model

        monkeypatch.setattr(sys, "argv", ["register_model.py", "--classifier", "fp", *args])
        with patch.object(register_model, "register_mlflow", return_value=(None, None)):
            register_model.main()

    def test_register_model_stamps_on_update_registry(self, models_dir, monkeypatch):
        self._run_register(monkeypatch, "--version", "v3.1.0", "--update-registry")

        config = _read(models_dir / "fp_classifier_config.json")
        assert config["version"] == "v3.1.0"
        assert config["pipeline_sha256"] == file_sha256(models_dir / "fp_classifier_pipeline.joblib")
        assert "v3.1.0" in _read(models_dir / "registry.json")["fp"]["versions"]

    def test_register_model_dry_run_does_not_stamp(self, models_dir, monkeypatch):
        self._run_register(monkeypatch, "--version", "v3.1.0", "--update-registry", "--dry-run")

        assert "version" not in _read(models_dir / "fp_classifier_config.json")

    def test_register_model_without_update_registry_does_not_stamp(self, models_dir, monkeypatch):
        self._run_register(monkeypatch, "--version", "v3.1.0")

        assert "version" not in _read(models_dir / "fp_classifier_config.json")

    def test_register_model_stamps_only_after_registry_write(self, models_dir, monkeypatch):
        import scripts.register_model as register_model

        with patch.object(register_model, "update_registry", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                self._run_register(monkeypatch, "--version", "v3.1.0", "--update-registry")

        assert "version" not in _read(models_dir / "fp_classifier_config.json")

    def test_register_model_refuses_when_pipeline_missing(self, models_dir, monkeypatch):
        (models_dir / "fp_classifier_pipeline.joblib").unlink()
        registry_before = _read(models_dir / "registry.json")

        with pytest.raises(SystemExit) as exc:
            self._run_register(monkeypatch, "--version", "v3.1.0", "--update-registry")

        assert exc.value.code != 0
        assert _read(models_dir / "registry.json") == registry_before
        assert "version" not in _read(models_dir / "fp_classifier_config.json")

    def _run_promote(self, models_dir, monkeypatch, *extra):
        import scripts.promote_model as promote_model

        monkeypatch.setattr(
            sys, "argv",
            ["promote_model.py", "--classifier", "fp", "--version", "v3.1.0",
             "--models-dir", str(models_dir), *extra],
        )
        return promote_model.main()

    def test_promote_model_stamps(self, models_dir, monkeypatch):
        assert self._run_promote(models_dir, monkeypatch) == 0

        config = _read(models_dir / "fp_classifier_config.json")
        assert config["version"] == "v3.1.0"
        assert config["pipeline_sha256"] == hashlib.sha256(PIPELINE_BYTES).hexdigest()
        assert "v3.1.0" in _read(models_dir / "registry.json")["fp"]["versions"]

    def test_promote_model_dry_run_does_not_stamp(self, models_dir, monkeypatch):
        registry_before = _read(models_dir / "registry.json")

        assert self._run_promote(models_dir, monkeypatch, "--dry-run") == 0

        assert "version" not in _read(models_dir / "fp_classifier_config.json")
        assert _read(models_dir / "registry.json") == registry_before

    def test_promote_model_stamps_only_after_registry_write(self, models_dir, monkeypatch):
        import scripts.promote_model as promote_model

        with patch.object(promote_model, "save_registry", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                self._run_promote(models_dir, monkeypatch)

        assert "version" not in _read(models_dir / "fp_classifier_config.json")

    def test_promote_model_refuses_when_pipeline_missing(self, models_dir, monkeypatch):
        (models_dir / "fp_classifier_pipeline.joblib").unlink()
        registry_before = _read(models_dir / "registry.json")

        assert self._run_promote(models_dir, monkeypatch) == 1

        assert _read(models_dir / "registry.json") == registry_before
        assert "version" not in _read(models_dir / "fp_classifier_config.json")

    def test_retrain_promote_version_stamps_the_promoted_copy(self, models_dir, tmp_path):
        import scripts.retrain as retrain

        output_dir = tmp_path / "candidate"
        output_dir.mkdir()
        (output_dir / "fp_classifier_pipeline.joblib").write_bytes(b"candidate model")
        (output_dir / "fp_classifier_config.json").write_text(json.dumps(BASE_CONFIG))

        registry_path = models_dir / "registry.json"
        config_path = models_dir / "fp_classifier_config.json"
        real_stamp = retrain.stamp_artifact_version

        def stamp_after_registry(*args, **kwargs):
            # The registry entry must already be on disk when the config is stamped.
            assert "v3.1.0" in _read(registry_path)["fp"]["versions"]
            return real_stamp(*args, **kwargs)

        def deploy_after_stamp(*args, **kwargs):
            assert _read(config_path)["version"] == "v3.1.0"

        with patch.object(retrain, "stamp_artifact_version", side_effect=stamp_after_registry) as stamp, \
                patch.object(retrain, "trigger_deploy_workflow", side_effect=deploy_after_stamp) as deploy:
            retrain.promote_version(
                classifier="fp",
                version="v3.1.0",
                new_metrics={"threshold": 0.6},
                data_path="data/fp_training_data.jsonl",
                output_dir=output_dir,
                models_dir=models_dir,
                registry_path=registry_path,
            )

        assert stamp.call_count == 1
        assert deploy.call_count == 1
        config = _read(config_path)
        assert config["version"] == "v3.1.0"
        assert config["pipeline_sha256"] == hashlib.sha256(b"candidate model").hexdigest()
        # The candidate's own config is left as it was.
        assert "version" not in _read(output_dir / "fp_classifier_config.json")


class TestClientFallback:
    def test_failed_fetch_reports_unavailable_and_is_not_cached(self):
        client = ClassifierClient("http://localhost:1")
        http = MagicMock()
        http.get.side_effect = httpx.ConnectError("refused")
        client._client = http

        assert client.get_model_info()["version"] == UNAVAILABLE
        assert client._model_info is None


class TestPipelineRecordsVersion:
    @pytest.fixture
    def article(self):
        return {
            "id": uuid4(),
            "title": "Nike releases new running shoe",
            "full_content": "Nike announced a new running shoe...",
            "description": "",
            "brands_mentioned": ["Nike"],
            "published_at": datetime.now(timezone.utc),
            "source_name": "ESPN",
            "category": ["sports"],
        }

    def _run(self, article, model_info=None, batch_error=None):
        with patch("src.labeling.pipeline.db"), patch(
            "src.labeling.pipeline.labeling_settings"
        ) as settings:
            settings.fp_classifier_enabled = True
            settings.fp_skip_llm_threshold = 0.3
            fp_client = MagicMock()
            if batch_error is not None:
                fp_client.predict_fp_batch.side_effect = batch_error
            else:
                fp_client.predict_fp_batch.return_value = [
                    FPPredictionResult(
                        is_sportswear=True, probability=0.9, confidence_level="low", threshold=0.3
                    )
                ]
            fp_client.get_model_info.return_value = model_info
            pipeline = LabelingPipeline(fp_client=fp_client)
            _, prediction = pipeline._run_fp_prefilter(article, dry_run=True)
        return prediction

    def test_records_the_version_the_api_reports(self, article, caplog):
        with caplog.at_level(logging.WARNING, logger="src.labeling.pipeline"):
            prediction = self._run(article, model_info={"version": "v3.1.0"})

        assert prediction.model_version == "v3.1.0"
        assert "reported no model version" not in caplog.text

    def test_api_without_version_field_is_unreported(self, article, caplog):
        with caplog.at_level(logging.WARNING, logger="src.labeling.pipeline"):
            prediction = self._run(article, model_info={"model_name": "RF_tuned"})

        assert prediction.model_version == UNREPORTED
        assert "reported no model version (unreported)" in caplog.text

    def test_failed_model_info_fetch_is_unavailable(self, article, caplog):
        """A real client whose /model/info request fails while the batch call succeeds."""
        fp_client = ClassifierClient("http://localhost:1")
        http = MagicMock()
        http.get.side_effect = httpx.ConnectError("refused")
        fp_client._client = http

        with patch("src.labeling.pipeline.db"), patch(
            "src.labeling.pipeline.labeling_settings"
        ) as settings, patch.object(
            fp_client,
            "predict_fp_batch",
            return_value=[
                FPPredictionResult(
                    is_sportswear=True, probability=0.9, confidence_level="low", threshold=0.3
                )
            ],
        ), caplog.at_level(logging.WARNING):
            settings.fp_classifier_enabled = True
            settings.fp_skip_llm_threshold = 0.3
            pipeline = LabelingPipeline(fp_client=fp_client)
            _, prediction = pipeline._run_fp_prefilter(article, dry_run=True)

        assert prediction.model_version == UNAVAILABLE
        assert "Failed to get model info" in caplog.text
        assert f"reported no model version ({UNAVAILABLE})" in caplog.text

    def test_unversioned_artifact_is_recorded_as_such(self, article, caplog):
        with caplog.at_level(logging.WARNING, logger="src.labeling.pipeline"):
            prediction = self._run(article, model_info={"version": UNVERSIONED})

        assert prediction.model_version == UNVERSIONED
        assert f"reported no model version ({UNVERSIONED})" in caplog.text

    def test_one_version_warning_per_batch(self, article, caplog):
        articles = [dict(article, id=uuid4()) for _ in range(3)]
        with patch("src.labeling.pipeline.db"), patch(
            "src.labeling.pipeline.labeling_settings"
        ) as settings:
            settings.fp_classifier_enabled = True
            settings.fp_skip_llm_threshold = 0.3
            fp_client = MagicMock()
            fp_client.predict_fp_batch.return_value = [
                FPPredictionResult(
                    is_sportswear=True, probability=0.9, confidence_level="low", threshold=0.3
                )
                for _ in articles
            ]
            fp_client.get_model_info.return_value = {}
            pipeline = LabelingPipeline(fp_client=fp_client)
            with caplog.at_level(logging.WARNING, logger="src.labeling.pipeline"):
                results = pipeline._run_fp_prefilter_batch(articles, dry_run=True)

        warnings = [r for r in caplog.records if "reported no model version" in r.getMessage()]
        assert len(warnings) == 1
        assert [results[a["id"]][1].model_version for a in articles] == [UNREPORTED] * 3

    def test_disabled_classifier_records_disabled(self, article):
        with patch("src.labeling.pipeline.db"), patch(
            "src.labeling.pipeline.labeling_settings"
        ) as settings:
            settings.fp_classifier_enabled = False
            pipeline = LabelingPipeline()
            with patch.object(pipeline, "_save_classifier_prediction") as save:
                pipeline._run_fp_prefilter_batch(
                    [article], novelty_scores={article["id"]: (0.4, 2)}, dry_run=False
                )

        assert save.call_count == 1
        assert save.call_args.args[1].model_version == "disabled"

    def test_failed_batch_is_unavailable_not_unknown(self, article):
        prediction = self._run(article, batch_error=RuntimeError("connection refused"))

        assert prediction.action_taken == "failed"
        assert prediction.model_version == UNAVAILABLE
