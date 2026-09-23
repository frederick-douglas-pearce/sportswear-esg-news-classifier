"""Pin the suite-wide isolation of the agent state dir (#124).

`tests/conftest.py` keeps every test out of the real `~/.esg-agent`, whose
`history/` is the archive `run_audit` reads to decide whether a scheduled
workflow is still running. These tests fail if that isolation regresses.

Every workflow name here is synthetic on purpose: if the isolation ever
breaks, these tests must not themselves deposit a production-named record in
the real archive.
"""

from pathlib import Path

import pytest

from src.agent import state as state_module
from src.agent.config import AgentSettings, agent_settings
from src.agent.state import StateManager

REAL_DEFAULT_STATE_DIR = Path.home() / ".esg-agent"

PROBE_WORKFLOW = "isolation_probe"


def _is_under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def test_state_paths_resolve_inside_pytest_temp_root(tmp_path_factory):
    base = tmp_path_factory.getbasetemp()

    assert _is_under(agent_settings.state_dir, base)
    assert _is_under(agent_settings.history_dir, base)
    assert _is_under(agent_settings.state_file, base)


def test_state_manager_singleton_uses_the_isolated_state_file(tmp_path_factory):
    base = tmp_path_factory.getbasetemp()

    assert state_module.state_manager.state_file == agent_settings.state_file
    assert _is_under(state_module.state_manager.state_file, base)
    assert state_module.state_manager.list_workflows() == []


def test_fresh_agent_settings_resolves_to_the_isolated_dir():
    assert AgentSettings().state_dir == agent_settings.state_dir


def test_archive_without_a_local_fixture_lands_in_the_isolated_history(
    tmp_path_factory,
):
    manager = StateManager()
    manager.create_workflow(name=PROBE_WORKFLOW, steps=["only"])
    manager.start_workflow(PROBE_WORKFLOW)
    manager.start_step(PROBE_WORKFLOW, "only")
    manager.complete_step(PROBE_WORKFLOW, "only")
    manager.complete_workflow(PROBE_WORKFLOW)

    archives = list(agent_settings.history_dir.glob(f"{PROBE_WORKFLOW}_*.yaml"))
    assert len(archives) == 1
    assert archives[0].parent == agent_settings.history_dir
    assert _is_under(archives[0], tmp_path_factory.getbasetemp())


@pytest.mark.parametrize("probe", ["first", "second"])
def test_each_test_starts_with_an_empty_state_dir(probe):
    """A dir shared across tests would hold the other parametrization's archive."""
    assert list(agent_settings.state_dir.iterdir()) == []

    manager = StateManager()
    name = f"{PROBE_WORKFLOW}_{probe}"
    manager.create_workflow(name=name, steps=["only"])
    manager.start_workflow(name)
    manager.complete_workflow(name)

    assert list(agent_settings.history_dir.glob(f"{name}_*.yaml"))


def test_import_time_singletons_were_built_outside_the_real_state_dir(
    request, monkeypatch
):
    """The session floor: `pytest_configure` redirects before `src.agent` is imported.

    `monkeypatch` is the same instance the autouse fixture patched through, so
    undoing it exposes the values the singletons were built with at import.
    Nothing is written after the undo.
    """
    from tests.conftest import _session_state_dir

    session_dir = request.config.stash[_session_state_dir]
    monkeypatch.undo()

    assert agent_settings.state_dir == session_dir
    assert state_module.state_manager.state_file == session_dir / "state.yaml"
    assert not _is_under(agent_settings.state_dir, REAL_DEFAULT_STATE_DIR)
