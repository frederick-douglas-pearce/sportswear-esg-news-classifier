"""Agent processes do not build the labeling settings (#211).

`src/agent/runner.py` and `src/agent/workflows/daily_labeling.py` import
`src.labeling.exit_codes`, which runs `src/labeling/__init__.py`. That file imported every
labeling submodule, so `labeling_settings` was built in every agent process and one unparseable
labeling setting stopped every agent workflow, `run_audit` included.

Each check runs in a fresh interpreter: this test process has already imported the package.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# `LabelingSettings` reads it with int(), so this value raises at import of src.labeling.config.
UNPARSEABLE = {"LABELING_BATCH_SIZE": "abc"}

REPORT_CONFIG_LOADED = (
    "import sys; print('src.labeling.config' in sys.modules)"
)


def _run(code: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={**os.environ, **(extra_env or {})},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_unparseable_value_stops_an_import_of_the_labeling_config():
    """Control: without it, the agent imports below could pass because the value parses."""
    result = _run("import src.labeling.config", UNPARSEABLE)

    assert result.returncode != 0
    assert "invalid literal for int()" in result.stderr


@pytest.mark.parametrize("module", ["src.agent.runner", "src.agent.workflows"])
def test_the_agent_imports_without_the_labeling_config(module):
    """`src.agent.workflows` registers every workflow, daily_labeling and run_audit included."""
    result = _run(f"import {module}; {REPORT_CONFIG_LOADED}", UNPARSEABLE)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_the_exit_codes_import_without_the_labeling_config():
    result = _run(f"import src.labeling.exit_codes; {REPORT_CONFIG_LOADED}", UNPARSEABLE)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


class TestLazyPackageAttributes:
    def test_every_exported_name_is_its_submodules_object(self):
        import importlib

        import src.labeling as package

        assert package.__all__, "the package exports nothing"
        for name in package.__all__:
            module = importlib.import_module(f"src.labeling.{package._EXPORTS[name]}")
            assert getattr(package, name) is getattr(module, name), name

    def test_from_import_of_a_package_name_works(self):
        from src.labeling import LabelingPipeline, labeling_settings
        from src.labeling.config import labeling_settings as settings
        from src.labeling.pipeline import LabelingPipeline as pipeline_class

        assert LabelingPipeline is pipeline_class
        assert labeling_settings is settings

    def test_an_unknown_name_raises_attribute_error(self):
        import src.labeling as package

        with pytest.raises(AttributeError, match="no_such_name"):
            package.no_such_name

    def test_dir_lists_the_exported_names(self):
        import src.labeling as package

        assert set(package.__all__) <= set(dir(package))
