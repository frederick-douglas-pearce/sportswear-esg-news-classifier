"""Tests for the registry version rules (#189, D033).

A registered version name is never rewritten, a new version is numbered from the
highest registered version, every write records the pipeline's hash once in both the
entry and the config stamp, and the pointer moves alone only onto bytes it can check.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import src.deployment.registry as registry_module
from src.deployment.registry import (
    PointerMoveRefused,
    VersionExistsError,
    check_pointer_move,
    move_pointer,
    next_version,
    parse_version,
    require_new_version,
    write_version,
)
from src.deployment.versioning import ArtifactRefused, check_build_artifact

PIPELINE_BYTES = b"pretend joblib bytes"
PIPELINE_SHA256 = hashlib.sha256(PIPELINE_BYTES).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _registry(production="v1.0.0", versions=("v1.0.0", "v1.2.0", "v1.1.0")):
    return {"fp": {"production": production, "versions": {v: {"threshold": 0.5} for v in versions}}}


class TestParseVersion:
    def test_parses_semantic_name(self):
        assert parse_version("v2.10.3") == (2, 10, 3)
        assert parse_version("v1.0.0") == (1, 0, 0)

    @pytest.mark.parametrize("name", ["v2", "2.1.0", "v2.1", "v2.1.0-rc1", "v2.1.0\n", "", None])
    def test_rejects_anything_else(self, name):
        with pytest.raises(ValueError, match="vMAJOR.MINOR.PATCH"):
            parse_version(name)

    # A second spelling of a registered version would be a second registry key for it.
    # Fullwidth digits after the first position are rejected only by re.ASCII.
    @pytest.mark.parametrize(
        "name",
        ["v02.4.0", "v2.04.0", "v2.4.00", "v\uff12.4.0", "v1\uff10.0.0", "v2.1\uff10.0",
         "v0.0.0", "v0.1.0"],
    )
    def test_rejects_non_canonical_spellings_and_major_zero(self, name):
        with pytest.raises(ValueError, match="vMAJOR.MINOR.PATCH"):
            parse_version(name)


class TestNextVersion:
    def test_bumps_from_highest_version_not_the_pointer(self):
        # The pointer is v1.0.0; a pointer-based minor bump would give the taken v1.1.0.
        assert next_version(_registry(), "fp", "minor") == "v1.3.0"

    def test_each_bump(self):
        registry = _registry()
        assert next_version(registry, "fp", "major") == "v2.0.0"
        assert next_version(registry, "fp", "minor") == "v1.3.0"
        assert next_version(registry, "fp", "patch") == "v1.2.1"

    def test_each_bump_resets_the_lower_parts(self):
        registry = _registry(versions=("v1.0.0", "v1.2.3"))
        assert next_version(registry, "fp", "major") == "v2.0.0"
        assert next_version(registry, "fp", "minor") == "v1.3.0"
        assert next_version(registry, "fp", "patch") == "v1.2.4"

    def test_compares_numerically_not_as_strings(self):
        registry = _registry(versions=("v1.9.0", "v1.10.0"))
        assert next_version(registry, "fp", "minor") == "v1.11.0"

    def test_first_version(self):
        assert next_version({}, "fp", "minor") == "v1.0.0"
        assert next_version({"fp": {"production": None, "versions": {}}}, "fp", "major") == "v1.0.0"

    def test_unparseable_registered_name_raises(self):
        with pytest.raises(ValueError, match="'v2'"):
            next_version(_registry(versions=("v1.0.0", "v2")), "fp", "minor")

    def test_unknown_bump_raises(self):
        with pytest.raises(ValueError, match="bump"):
            next_version(_registry(), "fp", "minor-ish")


class TestRequireNewVersion:
    def test_new_name_passes(self):
        require_new_version(_registry(), "fp", "v1.3.0", "minor")

    def test_existing_name_names_the_next_free_version_for_the_bump(self):
        with pytest.raises(VersionExistsError) as exc:
            require_new_version(_registry(), "fp", "v1.1.0", "patch")

        message = str(exc.value)
        assert "fp v1.1.0 is already registered" in message
        assert "next free version is v1.2.1 (patch bump" in message

    def test_malformed_name_is_refused(self):
        with pytest.raises(ValueError, match="'v2'"):
            require_new_version(_registry(), "fp", "v2", "minor")


@pytest.fixture
def artifact(tmp_path):
    pipeline_path = tmp_path / "fp_classifier_pipeline.joblib"
    pipeline_path.write_bytes(PIPELINE_BYTES)
    config_path = tmp_path / "fp_classifier_config.json"
    config_path.write_text(json.dumps({"threshold": 0.6}))
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(_registry()))
    return registry_path, config_path, pipeline_path


class TestWriteVersion:
    def test_records_one_hash_in_entry_and_stamp(self, artifact):
        registry_path, config_path, pipeline_path = artifact

        with patch.object(
            registry_module, "file_sha256", wraps=registry_module.file_sha256
        ) as hasher:
            recorded = write_version(
                registry_path, "fp", "v1.3.0", {"threshold": 0.6}, config_path, pipeline_path, "minor"
            )

        assert hasher.call_count == 1
        assert recorded == PIPELINE_SHA256
        entry = _read(registry_path)["fp"]["versions"]["v1.3.0"]
        assert entry == {"threshold": 0.6, "pipeline_sha256": PIPELINE_SHA256}
        config = _read(config_path)
        assert config["version"] == "v1.3.0"
        assert config["pipeline_sha256"] == PIPELINE_SHA256

    def test_sets_production_only_when_asked(self, artifact):
        registry_path, config_path, pipeline_path = artifact

        write_version(registry_path, "fp", "v1.3.0", {}, config_path, pipeline_path, "minor")
        assert _read(registry_path)["fp"]["production"] == "v1.0.0"

        write_version(
            registry_path, "fp", "v1.4.0", {}, config_path, pipeline_path, "minor", set_production=True
        )
        assert _read(registry_path)["fp"]["production"] == "v1.4.0"

    def test_creates_a_missing_registry_and_classifier(self, tmp_path, artifact):
        _, config_path, pipeline_path = artifact
        registry_path = tmp_path / "new_registry.json"

        write_version(registry_path, "ep", "v1.0.0", {}, config_path, pipeline_path, "minor")

        assert _read(registry_path) == {
            "ep": {"production": None, "versions": {"v1.0.0": {"pipeline_sha256": PIPELINE_SHA256}}}
        }

    @pytest.mark.parametrize("hashed", [False, True], ids=["hashless", "hashed"])
    def test_refuses_an_existing_name_and_writes_nothing(self, artifact, hashed):
        registry_path, config_path, pipeline_path = artifact
        registry = _registry()
        if hashed:
            registry["fp"]["versions"]["v1.1.0"]["pipeline_sha256"] = PIPELINE_SHA256
        registry_path.write_text(json.dumps(registry))
        registry_bytes = registry_path.read_bytes()
        config_bytes = config_path.read_bytes()

        with pytest.raises(VersionExistsError):
            write_version(registry_path, "fp", "v1.1.0", {}, config_path, pipeline_path, "minor")

        assert registry_path.read_bytes() == registry_bytes
        assert config_path.read_bytes() == config_bytes

    def test_stamps_after_the_registry_is_saved(self, artifact):
        """D026: the entry and its hash are on disk before the config claims the version."""
        registry_path, config_path, pipeline_path = artifact
        real_stamp = registry_module.stamp_artifact_version

        def stamp_after_save(*args, **kwargs):
            entry = _read(registry_path)["fp"]["versions"]["v1.3.0"]
            assert entry["pipeline_sha256"] == PIPELINE_SHA256
            return real_stamp(*args, **kwargs)

        with patch.object(
            registry_module, "stamp_artifact_version", side_effect=stamp_after_save
        ) as stamp:
            write_version(registry_path, "fp", "v1.3.0", {}, config_path, pipeline_path, "minor")

        assert stamp.call_count == 1
        assert stamp.call_args.args == (config_path, "v1.3.0", PIPELINE_SHA256)

    def test_no_stamp_when_the_registry_save_fails(self, artifact):
        registry_path, config_path, pipeline_path = artifact
        config_bytes = config_path.read_bytes()

        with patch.object(registry_module, "save_registry", side_effect=OSError("disk full")), \
                patch.object(registry_module, "stamp_artifact_version") as stamp:
            with pytest.raises(OSError):
                write_version(registry_path, "fp", "v1.3.0", {}, config_path, pipeline_path, "minor")

        assert stamp.call_count == 0
        assert config_path.read_bytes() == config_bytes


@pytest.fixture
def promotable(artifact):
    """A registry whose v1.1.0 records the hash of the bytes in models/, with a matching stamp."""
    registry_path, config_path, pipeline_path = artifact
    registry = _registry()
    registry["fp"]["versions"]["v1.1.0"]["pipeline_sha256"] = PIPELINE_SHA256
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    config_path.write_text(
        json.dumps({"threshold": 0.6, "version": "v1.1.0", "pipeline_sha256": PIPELINE_SHA256})
    )
    return registry_path, config_path, pipeline_path


class TestMovePointer:
    def test_changes_only_production(self, promotable):
        registry_path, config_path, pipeline_path = promotable
        before = _read(registry_path)
        config_bytes = config_path.read_bytes()

        previous = move_pointer(registry_path, "fp", "v1.1.0", pipeline_path, config_path)

        after = _read(registry_path)
        assert previous == "v1.0.0"
        assert after["fp"]["production"] == "v1.1.0"
        before["fp"]["production"] = "v1.1.0"
        assert after == before
        # The config is never re-stamped.
        assert config_path.read_bytes() == config_bytes

    def _assert_refused(self, promotable, version, match):
        registry_path, config_path, pipeline_path = promotable
        registry_bytes = registry_path.read_bytes()
        config_bytes = config_path.read_bytes() if config_path.exists() else None

        with pytest.raises(PointerMoveRefused, match=match):
            move_pointer(registry_path, "fp", version, pipeline_path, config_path)

        assert registry_path.read_bytes() == registry_bytes
        assert (config_path.read_bytes() if config_path.exists() else None) == config_bytes

    def test_refuses_unknown_version(self, promotable):
        self._assert_refused(promotable, "v9.0.0", "not a registered version")

    def test_refuses_hash_mismatch(self, promotable):
        _, _, pipeline_path = promotable
        pipeline_path.write_bytes(b"a different model")

        self._assert_refused(promotable, "v1.1.0", "is not fp v1.1.0's bytes")

    def test_refuses_bytes_and_stamp_that_agree_on_a_hash_the_entry_does_not_record(
        self, promotable
    ):
        # Same case as the build guard's (tests/test_build_guard.py): the joblib and
        # stamp agree, the entry records another hash.
        registry_path, config_path, _ = promotable
        registry = _read(registry_path)
        registry["fp"]["versions"]["v1.0.0"]["pipeline_sha256"] = "0" * 64
        registry_path.write_text(json.dumps(registry, indent=2) + "\n")
        config_path.write_text(
            json.dumps({"threshold": 0.6, "version": "v1.0.0", "pipeline_sha256": PIPELINE_SHA256})
        )

        self._assert_refused(promotable, "v1.0.0", "is not fp v1.0.0's bytes")

    def test_refuses_hashless_entry(self, promotable):
        # v1.2.0 records no hash, even though the bytes in models/ could be anything.
        self._assert_refused(promotable, "v1.2.0", "no recorded pipeline_sha256")

    @pytest.mark.parametrize(
        "stamp",
        [
            {"version": "v1.0.0", "pipeline_sha256": PIPELINE_SHA256},
            {"version": "v1.1.0", "pipeline_sha256": "0" * 64},
            {},
        ],
        ids=["other-version", "other-hash", "unstamped"],
    )
    def test_refuses_config_stamp_that_does_not_match(self, promotable, stamp):
        _, config_path, _ = promotable
        config_path.write_text(json.dumps({"threshold": 0.6, **stamp}))

        self._assert_refused(promotable, "v1.1.0", "is stamped")

    def test_refuses_missing_pipeline(self, promotable):
        _, _, pipeline_path = promotable
        pipeline_path.unlink()

        self._assert_refused(promotable, "v1.1.0", "Pipeline not found")

    def test_refuses_missing_config(self, promotable):
        _, config_path, _ = promotable
        config_path.unlink()

        self._assert_refused(promotable, "v1.1.0", "Config not found")

    def test_check_alone_writes_nothing(self, promotable):
        registry_path, config_path, pipeline_path = promotable
        registry_bytes = registry_path.read_bytes()

        check_pointer_move(_read(registry_path), "fp", "v1.1.0", pipeline_path, config_path)

        assert registry_path.read_bytes() == registry_bytes

    @pytest.mark.parametrize(
        "break_tree",
        [
            None,
            lambda registry, config, pipeline: pipeline.write_bytes(b"a different model"),
            lambda registry, config, pipeline: config.write_text(json.dumps({"version": "v1.0.0"})),
            lambda registry, config, pipeline: (pipeline.unlink(), pipeline.mkdir()),
            lambda registry, config, pipeline: config.write_text("[]"),
            lambda registry, config, pipeline: config.write_bytes(b"\xff\xfe binary"),
            lambda registry, config, pipeline: registry["fp"]["versions"].update({"v1.1.0": None}),
        ],
        ids=["intact", "other-bytes", "other-stamp", "directory", "config-not-object", "config-binary", "entry-not-object"],
    )
    def test_allows_a_move_exactly_where_the_build_guard_passes(self, promotable, break_tree):
        """A move is allowed exactly when the build guard would then pass on these files (D034)."""
        registry_path, config_path, pipeline_path = promotable
        registry = _read(registry_path)
        if break_tree is not None:
            break_tree(registry, config_path, pipeline_path)
        # The registry as the move would leave it, for the guard to read.
        guarded = json.loads(json.dumps(registry))
        guarded["fp"]["production"] = "v1.1.0"
        registry_path.write_text(json.dumps(guarded))

        if break_tree is None:
            assert check_build_artifact(registry_path, "fp", pipeline_path, config_path) == "v1.1.0"
            check_pointer_move(registry, "fp", "v1.1.0", pipeline_path, config_path)
            return

        with pytest.raises(ArtifactRefused) as guard:
            check_build_artifact(registry_path, "fp", pipeline_path, config_path)
        with pytest.raises(PointerMoveRefused) as move:
            check_pointer_move(registry, "fp", "v1.1.0", pipeline_path, config_path)
        # Same refusal: the guard's message is the move's, prefixed with the pointer.
        assert str(guard.value) == f"fp production pointer is v1.1.0: {move.value}"


# The fields the writer scripts print from an artifact config.
WRITER_CONFIG = {"threshold": 0.6, "model_name": "TestModel", "transformer_method": "tfidf", "test_f2": 0.93}


class TestWritersRefuseExistingNames:
    """Each registry writer refuses an existing name before any side effect.

    The malformed-name refusals are tested for register_model.py and promote_model.py; retrain.py
    numbers its own names with ``next_version``.
    """

    @pytest.fixture
    def models_dir(self, tmp_path, monkeypatch):
        models = tmp_path / "models"
        models.mkdir()
        (models / "fp_classifier_pipeline.joblib").write_bytes(PIPELINE_BYTES)
        (models / "fp_classifier_config.json").write_text(json.dumps(WRITER_CONFIG))
        self._write_registry(models, hashed=False)
        monkeypatch.chdir(tmp_path)
        return models

    @staticmethod
    def _write_registry(models, hashed):
        entry = {"threshold": 0.5}
        if hashed:
            entry["pipeline_sha256"] = "f" * 64
        registry = {"fp": {"production": "v3.0.0", "versions": {"v3.0.0": entry}}}
        (models / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")

    @staticmethod
    def _snapshot(models):
        return {path.name: path.read_bytes() for path in sorted(models.iterdir())}

    def _run_register(self, monkeypatch, *args):
        import scripts.register_model as register_model

        monkeypatch.setattr("sys.argv", ["register_model.py", "--classifier", "fp", *args])
        with patch.object(register_model, "register_mlflow", return_value=(None, None)) as mlflow_reg:
            try:
                register_model.main()
            finally:
                self.mlflow_reg = mlflow_reg

    @pytest.mark.parametrize("hashed", [False, True], ids=["hashless", "hashed"])
    @pytest.mark.parametrize("dry_run", [False, True], ids=["write", "dry-run"])
    def test_register_model_refuses_existing_name(self, models_dir, monkeypatch, capsys, hashed, dry_run):
        self._write_registry(models_dir, hashed)
        before = self._snapshot(models_dir)
        extra = ["--dry-run"] if dry_run else []

        with pytest.raises(SystemExit) as exc:
            self._run_register(monkeypatch, "--version", "v3.0.0", "--update-registry", *extra)

        assert exc.value.code == 1
        assert self.mlflow_reg.call_count == 0
        assert self._snapshot(models_dir) == before
        err = capsys.readouterr().err
        assert "fp v3.0.0 is already registered" in err
        assert "next free version is v3.0.1 (patch bump" in err

    def test_register_model_refuses_malformed_name(self, models_dir, monkeypatch, capsys):
        before = self._snapshot(models_dir)

        with pytest.raises(SystemExit) as exc:
            self._run_register(monkeypatch, "--version", "v4", "--update-registry")

        assert exc.value.code == 1
        assert self.mlflow_reg.call_count == 0
        assert self._snapshot(models_dir) == before
        assert "vMAJOR.MINOR.PATCH" in capsys.readouterr().err

    def test_register_model_mlflow_only_on_existing_name_leaves_registry(self, models_dir, monkeypatch, capsys):
        before = self._snapshot(models_dir)

        self._run_register(monkeypatch, "--version", "v3.0.0")

        assert self.mlflow_reg.call_count == 1
        assert self._snapshot(models_dir) == before
        out = capsys.readouterr().out
        assert "the registry is not written" in out
        assert "the next free version is v3.0.1" in out
        assert "overwrite" not in out

    def test_register_model_auto_version_bumps_from_highest(self, models_dir, monkeypatch):
        registry = {
            "fp": {
                "production": "v3.0.0",
                "versions": {"v3.0.0": {}, "v3.1.0": {}},
            }
        }
        (models_dir / "registry.json").write_text(json.dumps(registry))

        # The model_training workflow's promote step runs this command.
        self._run_register(monkeypatch, "--bump", "minor", "--update-registry")

        written = _read(models_dir / "registry.json")["fp"]["versions"]
        assert sorted(written) == ["v3.0.0", "v3.1.0", "v3.2.0"]
        assert written["v3.2.0"]["pipeline_sha256"] == PIPELINE_SHA256
        assert _read(models_dir / "fp_classifier_config.json")["version"] == "v3.2.0"

    def _run_promote(self, models_dir, monkeypatch, *args):
        import scripts.promote_model as promote_model

        monkeypatch.setattr(
            "sys.argv",
            ["promote_model.py", "--classifier", "fp", "--models-dir", str(models_dir), *args],
        )
        with patch("builtins.input", side_effect=AssertionError("prompted")) as prompt:
            result = promote_model.main()
        assert prompt.call_count == 0
        return result

    @pytest.mark.parametrize("hashed", [False, True], ids=["hashless", "hashed"])
    def test_promote_model_refuses_existing_name(self, models_dir, monkeypatch, capsys, hashed):
        self._write_registry(models_dir, hashed)
        before = self._snapshot(models_dir)

        assert self._run_promote(models_dir, monkeypatch, "--version", "v3.0.0") == 1

        assert self._snapshot(models_dir) == before
        err = capsys.readouterr().err
        assert "fp v3.0.0 is already registered" in err
        assert "next free version is v3.1.0 (minor bump" in err

    def test_promote_model_refuses_malformed_name(self, models_dir, monkeypatch):
        before = self._snapshot(models_dir)

        assert self._run_promote(models_dir, monkeypatch, "--version", "v4") == 1

        assert self._snapshot(models_dir) == before

    @pytest.fixture
    def rollback_target(self, models_dir):
        """v3.0.0 records the hash of the bytes in models/ and the config is stamped with it."""
        registry = {
            "fp": {
                "production": "v3.1.0",
                "versions": {
                    "v3.0.0": {"threshold": 0.5, "pipeline_sha256": PIPELINE_SHA256},
                    "v3.1.0": {"threshold": 0.5},
                },
            }
        }
        (models_dir / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
        (models_dir / "fp_classifier_config.json").write_text(
            json.dumps({**WRITER_CONFIG, "version": "v3.0.0", "pipeline_sha256": PIPELINE_SHA256})
        )
        return registry

    def test_promote_model_pointer_only_moves_only_production(self, models_dir, monkeypatch, rollback_target):
        config_before = (models_dir / "fp_classifier_config.json").read_bytes()

        assert self._run_promote(models_dir, monkeypatch, "--version", "v3.0.0", "--pointer-only") == 0

        rollback_target["fp"]["production"] = "v3.0.0"
        assert _read(models_dir / "registry.json") == rollback_target
        assert (models_dir / "fp_classifier_config.json").read_bytes() == config_before

    def test_promote_model_pointer_only_dry_run_writes_nothing(self, models_dir, monkeypatch, rollback_target):
        before = self._snapshot(models_dir)

        assert self._run_promote(
            models_dir, monkeypatch, "--version", "v3.0.0", "--pointer-only", "--dry-run"
        ) == 0

        assert self._snapshot(models_dir) == before

    @pytest.mark.parametrize(
        "args",
        [
            ["--version", "v3.1.0", "--pointer-only"],
            ["--version", "v9.0.0", "--pointer-only"],
            ["--version", "v3.0.0", "--pointer-only", "--data-path", "data/x.jsonl"],
            ["--version", "v3.0.0", "--pointer-only", "--production"],
        ],
        ids=["hashless", "unknown", "with-data-path", "with-production"],
    )
    def test_promote_model_pointer_only_refusals(self, models_dir, monkeypatch, rollback_target, args):
        before = self._snapshot(models_dir)

        assert self._run_promote(models_dir, monkeypatch, *args) == 1

        assert self._snapshot(models_dir) == before

    @pytest.mark.parametrize("hashed", [False, True], ids=["hashless", "hashed"])
    def test_retrain_promote_version_refuses_existing_name(self, models_dir, tmp_path, hashed):
        import scripts.retrain as retrain

        self._write_registry(models_dir, hashed)
        output_dir = tmp_path / "candidate"
        output_dir.mkdir()
        (output_dir / "fp_classifier_pipeline.joblib").write_bytes(b"candidate model")
        (output_dir / "fp_classifier_config.json").write_text(json.dumps({"threshold": 0.5}))
        before = self._snapshot(models_dir)

        with patch.object(retrain.shutil, "copy") as copy, \
                patch.object(retrain, "trigger_deploy_workflow") as deploy:
            with pytest.raises(VersionExistsError, match="next free version is v3.1.0"):
                retrain.promote_version(
                    classifier="fp",
                    version="v3.0.0",
                    new_metrics={"threshold": 0.5},
                    data_path="data/fp_training_data.jsonl",
                    output_dir=output_dir,
                    models_dir=models_dir,
                    registry_path=models_dir / "registry.json",
                )

        assert copy.call_count == 0
        assert deploy.call_count == 0
        assert self._snapshot(models_dir) == before

    def test_retrain_main_refuses_before_training(self, models_dir, monkeypatch, capsys):
        import scripts.retrain as retrain

        monkeypatch.setattr(
            "sys.argv",
            ["retrain.py", "--classifier", "fp", "--data", "data/x.jsonl", "--models-dir", str(models_dir)],
        )
        # next_version never returns a registered name, so a taken name is injected here.
        with patch.object(retrain, "get_next_version", return_value="v3.0.0"), \
                patch.object(retrain, "train_new_version") as train:
            assert retrain.main() == 1

        assert train.call_count == 0
        assert "fp v3.0.0 is already registered" in capsys.readouterr().err

    def test_retrain_main_prints_the_pointer_not_the_highest_version(self, models_dir, monkeypatch, capsys):
        import scripts.retrain as retrain

        registry_path = models_dir / "registry.json"
        registry = _read(registry_path)
        registry["fp"]["production"] = None
        registry_path.write_text(json.dumps(registry))
        monkeypatch.setattr(
            "sys.argv",
            ["retrain.py", "--classifier", "fp", "--data", "data/x.jsonl", "--models-dir", str(models_dir)],
        )
        with patch.object(retrain, "train_new_version", return_value={"success": False, "error": "stop"}):
            assert retrain.main() == 1

        # get_current_version would report v3.0.0, the highest version, as production.
        assert "Production version: None" in capsys.readouterr().out
