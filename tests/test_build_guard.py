"""Tests for the build guard: an image is built only from the pointer's registered bytes (#172, D034).

The guard function and its CLI are tested on temporary registries and artifacts. The
Dockerfile wiring is checked by reading the Dockerfile: the guard runs in the builder
stage, the runtime stage takes the files the guard checked, and the bypass is off by
default and passed by no supported build.
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.deployment.versioning import ArtifactRefused, check_build_artifact

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSIONING_PY = REPO_ROOT / "src" / "deployment" / "versioning.py"
DOCKERFILE = REPO_ROOT / "Dockerfile"

PIPELINE_BYTES = b"pretend joblib bytes"
PIPELINE_SHA256 = hashlib.sha256(PIPELINE_BYTES).hexdigest()
OTHER_SHA256 = "0" * 64


@pytest.fixture
def models_dir(tmp_path):
    """A models/ whose fp pointer, v1.1.0, records the joblib's hash and matches its stamp."""
    models = tmp_path / "models"
    models.mkdir()
    registry = {
        "fp": {
            "production": "v1.1.0",
            "versions": {
                "v1.0.0": {"threshold": 0.5, "pipeline_sha256": OTHER_SHA256},
                "v1.1.0": {"threshold": 0.6, "pipeline_sha256": PIPELINE_SHA256},
                "v0.9.0": {"threshold": 0.5},
            },
        },
        "ep": {"production": None, "versions": {"v1.0.0": {"threshold": 0.7}}},
    }
    (models / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    (models / "fp_classifier_pipeline.joblib").write_bytes(PIPELINE_BYTES)
    _write_stamp(models, {"version": "v1.1.0", "pipeline_sha256": PIPELINE_SHA256})
    return models


def _write_stamp(models, stamp, classifier="fp"):
    config = {"threshold": 0.6, "transformer_method": "tfidf", **stamp}
    (models / f"{classifier}_classifier_config.json").write_text(json.dumps(config))


def _set_registry(models, update):
    registry = json.loads((models / "registry.json").read_text())
    update(registry)
    (models / "registry.json").write_text(json.dumps(registry))


def _check(models, classifier="fp"):
    return check_build_artifact(
        models / "registry.json",
        classifier,
        models / f"{classifier}_classifier_pipeline.joblib",
        models / f"{classifier}_classifier_config.json",
    )


class TestCheckBuildArtifact:
    def test_returns_the_pointer_when_the_files_are_its_registered_bytes(self, models_dir):
        assert _check(models_dir) == "v1.1.0"

    def test_refuses_an_unstamped_config(self, models_dir):
        _write_stamp(models_dir, {})

        with pytest.raises(ArtifactRefused, match=r"pointer is v1\.1\.0: .*stamped version=None"):
            _check(models_dir)

    def test_refuses_a_stamp_naming_another_version(self, models_dir):
        _write_stamp(models_dir, {"version": "v1.0.0", "pipeline_sha256": PIPELINE_SHA256})

        with pytest.raises(ArtifactRefused, match=r"pointer is v1\.1\.0: .*stamped version='v1\.0\.0'"):
            _check(models_dir)

    def test_refuses_a_stamp_with_another_hash(self, models_dir):
        # Beyond AC1 (D034): with the right version but the wrong hash, the API would
        # report `unversioned` from an image the guard let through.
        _write_stamp(models_dir, {"version": "v1.1.0", "pipeline_sha256": OTHER_SHA256})

        with pytest.raises(ArtifactRefused, match=f"pipeline_sha256='{OTHER_SHA256}'"):
            _check(models_dir)

    def test_refuses_a_joblib_that_is_not_the_pointers_bytes(self, models_dir):
        (models_dir / "fp_classifier_pipeline.joblib").write_bytes(b"a different model")
        actual = hashlib.sha256(b"a different model").hexdigest()

        with pytest.raises(ArtifactRefused) as refused:
            _check(models_dir)

        message = str(refused.value)
        assert "pointer is v1.1.0" in message
        assert actual in message
        assert PIPELINE_SHA256 in message

    def test_refuses_a_pointer_entry_without_a_hash(self, models_dir):
        _set_registry(models_dir, lambda r: r["fp"].update(production="v0.9.0"))
        _write_stamp(models_dir, {"version": "v0.9.0", "pipeline_sha256": PIPELINE_SHA256})

        with pytest.raises(ArtifactRefused, match=r"pointer is v0\.9\.0: .*no recorded pipeline_sha256"):
            _check(models_dir)

    def test_refuses_a_null_pointer_whatever_files_are_present(self, models_dir):
        # EP files that would pass for a registered version are present; the null
        # pointer is refused anyway.
        _set_registry(
            models_dir,
            lambda r: r["ep"]["versions"]["v1.0.0"].update(pipeline_sha256=PIPELINE_SHA256),
        )
        (models_dir / "ep_classifier_pipeline.joblib").write_bytes(PIPELINE_BYTES)
        _write_stamp(models_dir, {"version": "v1.0.0", "pipeline_sha256": PIPELINE_SHA256}, "ep")

        with pytest.raises(ArtifactRefused, match="ep has no production version"):
            _check(models_dir, "ep")

    def test_refuses_an_unregistered_pointer(self, models_dir):
        _set_registry(models_dir, lambda r: r["fp"].update(production="v9.0.0"))

        with pytest.raises(ArtifactRefused, match=r"pointer is v9\.0\.0: .*not a registered version"):
            _check(models_dir)

    def test_refuses_a_classifier_missing_from_the_registry(self, models_dir):
        with pytest.raises(ArtifactRefused, match="esg is not in"):
            _check(models_dir, "esg")

    def test_refuses_a_missing_joblib(self, models_dir):
        (models_dir / "fp_classifier_pipeline.joblib").unlink()

        with pytest.raises(ArtifactRefused, match="Pipeline not found"):
            _check(models_dir)

    def test_refuses_a_missing_config(self, models_dir):
        (models_dir / "fp_classifier_config.json").unlink()

        with pytest.raises(ArtifactRefused, match="Config not found"):
            _check(models_dir)

    def test_refuses_a_missing_registry(self, models_dir):
        (models_dir / "registry.json").unlink()

        with pytest.raises(ArtifactRefused, match="Registry not found"):
            _check(models_dir)

    @pytest.mark.parametrize("name", ["registry.json", "fp_classifier_config.json"])
    def test_refuses_a_file_that_is_not_valid_json(self, models_dir, name):
        (models_dir / name).write_text("{not json")

        with pytest.raises(ArtifactRefused, match=f"{name} is not valid JSON"):
            _check(models_dir)


class TestCli:
    """The CLI as the Dockerfile runs it: this one file, by path, under ``python -I``."""

    def _run(self, tmp_path, *args):
        # A lone copy, run from a directory with no src/, shows the guard needs
        # nothing but the standard library.
        script = tmp_path / "versioning.py"
        shutil.copy(VERSIONING_PY, script)
        return subprocess.run(
            [sys.executable, "-I", str(script), *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

    def test_exits_zero_naming_the_pointer_and_hash(self, models_dir, tmp_path):
        result = self._run(tmp_path, "--classifier", "fp", "--models-dir", str(models_dir))

        assert result.returncode == 0, result.stderr
        assert "production v1.1.0" in result.stdout
        assert PIPELINE_SHA256 in result.stdout

    def test_exits_one_with_the_refusal_on_stderr(self, models_dir, tmp_path):
        result = self._run(tmp_path, "--classifier", "ep", "--models-dir", str(models_dir))

        assert result.returncode == 1
        assert "ep has no production version" in result.stderr
        assert result.stdout == ""

    def test_exits_two_on_a_usage_error(self, tmp_path):
        result = self._run(tmp_path, "--classifier", "fp")

        assert result.returncode == 2
        assert "--models-dir" in result.stderr


def _dockerfile_stages() -> list[list[str]]:
    """The Dockerfile's instructions, continuation lines joined, split into stages at FROM."""
    instructions, current = [], ""
    for line in DOCKERFILE.read_text().splitlines():
        stripped = line.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        instructions.append(current + stripped)
        current = ""

    stages = []
    for instruction in instructions:
        if instruction.upper().startswith("FROM "):
            stages.append([])
        if stages:
            stages[-1].append(instruction)
    return stages


def _index(stage, predicate, what):
    matches = [i for i, instruction in enumerate(stage) if predicate(instruction)]
    assert len(matches) == 1, f"expected one {what}, found {[stage[i] for i in matches]}"
    return matches[0]


def _guard_run(builder):
    return builder[_index(builder, lambda s: s.startswith("RUN ") and "versioning.py" in s, "guard RUN")]


class TestDockerfileWiring:
    def test_the_builder_runs_the_guard_before_it_downloads_anything(self):
        builder, _ = _dockerfile_stages()

        guard = _index(
            builder,
            lambda s: s.startswith("RUN ") and "python -I /build/versioning.py" in s,
            "guard RUN",
        )
        detection = _index(builder, lambda s: s.startswith("RUN ") and "transformer_method" in s, "detection RUN")
        assert guard < detection
        # Detection reads the config the guard checked.
        assert "/build/models/${CLASSIFIER_TYPE}_classifier_config.json" in builder[detection]

    def test_the_builder_copies_the_registry_and_the_named_artifacts_only(self):
        builder, _ = _dockerfile_stages()

        copy = builder[_index(builder, lambda s: s.startswith("COPY ") and "models/" in s, "models COPY")]
        assert copy.split()[1:] == [
            "models/registry.json",
            "models/${CLASSIFIER_TYPE}_classifier_pipeline.jobli[b]",
            "models/${CLASSIFIER_TYPE}_classifier_config.jso[n]",
            "/build/models/",
        ]

    def test_the_runtime_takes_the_checked_files_from_the_builder(self):
        _, runtime = _dockerfile_stages()

        copy = runtime[_index(runtime, lambda s: s.startswith("COPY ") and "classifier_" in s, "model COPY")]
        assert copy.split()[1] == "--from=builder"
        assert "/build/models/${CLASSIFIER_TYPE}_classifier_pipeline.joblib" in copy
        assert "/build/models/${CLASSIFIER_TYPE}_classifier_config.json" in copy

    def test_the_runtime_takes_nothing_from_models_in_the_context_and_no_registry(self):
        _, runtime = _dockerfile_stages()

        for instruction in runtime:
            assert "registry.json" not in instruction
            if instruction.startswith("COPY ") and "--from=" not in instruction:
                assert "models/" not in instruction, instruction

    def test_the_bypass_is_off_by_default(self):
        builder, _ = _dockerfile_stages()

        assert "ARG SKIP_MODEL_GUARD=false" in builder
        assert builder.index("ARG SKIP_MODEL_GUARD=false") < builder.index(_guard_run(builder))

    @pytest.mark.parametrize(
        ("value", "guard_runs"),
        [("true", False), ("false", True), ("TRUE", True), ("1", True), ("yes", True), ("", True)],
    )
    def test_only_exactly_true_skips_the_guard(self, models_dir, value, guard_runs):
        # The builder's guard command, run by sh against a tree the guard refuses (EP's
        # null pointer): it fails exactly when the guard ran.
        command = _guard_run(_dockerfile_stages()[0])[len("RUN "):]
        command = command.replace("python -I /build/versioning.py", f"{sys.executable} -I {VERSIONING_PY}")
        command = command.replace("/build/models", str(models_dir))

        result = subprocess.run(
            ["sh", "-c", command],
            env={"SKIP_MODEL_GUARD": value, "CLASSIFIER_TYPE": "ep", "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )

        if guard_runs:
            assert result.returncode == 1, result.stdout
            assert "ep has no production version" in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert "WARNING: SKIP_MODEL_GUARD=true" in result.stdout

    @pytest.mark.parametrize(
        "path", [".github/workflows/deploy.yml", "docker-compose.yml", "scripts/deploy_cloudrun.sh"]
    )
    def test_no_supported_build_passes_the_bypass(self, path):
        assert not re.search("SKIP_MODEL_GUARD", (REPO_ROOT / path).read_text())
