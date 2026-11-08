"""Importing the agent does not build the labeling settings (#211).

`src/agent/runner.py` and `src/agent/workflows/daily_labeling.py` import
`src.labeling.exit_codes`, which runs `src/labeling/__init__.py`. That file imported the
submodules behind its exports, config among them, so `labeling_settings` was built at import in
every agent process and one unparseable labeling setting stopped every agent workflow, `run_audit`
included.

The import checks run in a fresh interpreter: this test process may already have imported the
package, and a cached module would hide what an import loads.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, sentinel

import pytest

REPO = Path(__file__).resolve().parent.parent

# `LabelingSettings` reads it with int(), so this value raises at import of src.labeling.config.
UNPARSEABLE = {"LABELING_BATCH_SIZE": "abc"}

# The default environment: an import attempted and swallowed under UNPARSEABLE leaves no module
# behind, so the agent imports are also checked where the config would import cleanly.
ENVIRONMENTS = pytest.mark.parametrize(
    "extra_env", [UNPARSEABLE, {}], ids=["unparseable", "default"]
)

REPORT_CONFIG_LOADED = "import sys; print('src.labeling.config' in sys.modules)"

# The package's exports before #211, written out so a name dropped from `_EXPORTS` is caught.
EXPORTED_NAMES = {
    "ArticleChunker",
    "ArticleLabeler",
    "BrandAnalysis",
    "CategoryLabel",
    "Chunk",
    "CrossEncoderReranker",
    "ESG_CATEGORIES",
    "EvidenceMatch",
    "EvidenceMatcher",
    "LABELING_SYSTEM_PROMPT",
    "LabelingDatabase",
    "LabelingPipeline",
    "LabelingResponse",
    "LabelingResult",
    "LabelingStats",
    "OpenAIEmbedder",
    "RerankCandidate",
    "RerankResult",
    "labeling_db",
    "labeling_settings",
    "match_all_evidence",
}


def _run(code: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={**os.environ, **(extra_env or {})},
        capture_output=True,
        text=True,
        timeout=120,
    )


def _last_line(result: subprocess.CompletedProcess) -> str:
    return result.stdout.strip().splitlines()[-1]


def test_the_unparseable_value_stops_an_import_of_the_labeling_config():
    """Control: without it, the agent imports below could pass because the value parses."""
    result = _run("import src.labeling.config", UNPARSEABLE)

    assert result.returncode != 0
    assert "invalid literal for int()" in result.stderr


def test_the_unparseable_value_stops_a_read_of_a_package_attribute():
    """The package attribute surfaces the config's error rather than hiding it."""
    result = _run("from src.labeling import labeling_settings", UNPARSEABLE)

    assert result.returncode != 0
    assert "invalid literal for int()" in result.stderr


@ENVIRONMENTS
@pytest.mark.parametrize(
    "module",
    ["src.agent.runner", "src.agent.workflows", "src.agent.__main__", "src.labeling.exit_codes"],
)
def test_the_agent_imports_without_the_labeling_config(module, extra_env):
    """`src.agent.workflows` registers every workflow, daily_labeling and run_audit included;
    `src.agent.__main__` is what the cron runs."""
    result = _run(f"import {module}; {REPORT_CONFIG_LOADED}", extra_env)

    assert result.returncode == 0, result.stderr
    assert _last_line(result) == "False"


def test_dir_lists_the_exported_names_before_any_is_read():
    result = _run(
        "import src.labeling as p; print(set(p.__all__) <= set(dir(p)))"
    )

    assert result.returncode == 0, result.stderr
    assert _last_line(result) == "True"


class TestLazyPackageAttributes:
    def test_the_package_exports_the_same_names_as_before(self):
        import src.labeling as package

        assert set(package.__all__) == EXPORTED_NAMES

    def test_every_exported_name_is_its_submodules_object(self):
        import importlib

        import src.labeling as package

        for name in EXPORTED_NAMES:
            module = importlib.import_module(f"src.labeling.{package._EXPORTS[name]}")
            assert getattr(package, name) is getattr(module, name), name

    def test_from_import_of_a_package_name_works(self):
        from src.labeling import LabelingPipeline, labeling_settings
        from src.labeling.config import labeling_settings as settings
        from src.labeling.pipeline import LabelingPipeline as pipeline_class

        assert LabelingPipeline is pipeline_class
        assert labeling_settings is settings

    def test_a_patch_on_the_submodule_is_not_kept_by_the_package(self):
        import src.labeling as package
        import src.labeling.config as config

        real = config.labeling_settings
        with patch.object(config, "labeling_settings", sentinel.patched):
            assert package.labeling_settings is sentinel.patched

        assert package.labeling_settings is real

    def test_an_unknown_name_raises_attribute_error(self):
        import src.labeling as package

        with pytest.raises(AttributeError, match="no_such_name"):
            package.no_such_name
