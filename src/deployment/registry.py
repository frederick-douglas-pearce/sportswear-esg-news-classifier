"""Registry version rules: immutable names, numbering, and the pointer (#189, D033).

A version name in ``models/registry.json`` names one set of bytes for good. Every
registry writer goes through :func:`write_version`, which refuses an existing name,
records the pipeline's ``pipeline_sha256`` and the config's ``threshold`` in the entry,
and stamps the same hash into the artifact config (D026, D035). A new version is
numbered from the highest registered version, never from the ``production`` pointer.
The pointer moves on its own through :func:`move_pointer`, which changes only
``production``.

The tooling assumes one writer at a time. Nothing locks the registry between a load
and a save, so concurrent writers are unsupported (D033).
"""

import json
import re
from pathlib import Path
from typing import Union

from .versioning import (
    ArtifactRefused,
    check_registered_bytes,
    file_sha256,
    read_config_threshold,
    stamp_artifact_version,
)

BUMPS = ("major", "minor", "patch")

# Canonical names only: ASCII digits, no leading zeros, MAJOR at least 1. Each version
# then has one spelling, so ``v02.4.0`` cannot register a second key for ``v2.4.0``.
_VERSION_RE = re.compile(r"v([1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", re.ASCII)


class VersionExistsError(ValueError):
    """The version name is already in the registry, so it cannot be written again."""


class PointerMoveRefused(ValueError):
    """The ``production`` pointer cannot be moved to the requested version."""


def parse_version(name: str) -> tuple[int, int, int]:
    """Parse a registry version name of the form ``vMAJOR.MINOR.PATCH``.

    Raises:
        ValueError: ``name`` is not of that form in canonical spelling (``v2``,
            ``2.1.0``, ``v02.4.0`` and ``v0.1.0`` are not).
    """
    match = _VERSION_RE.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        raise ValueError(
            f"Invalid version name {name!r}: expected vMAJOR.MINOR.PATCH with no leading "
            "zeros and MAJOR at least 1, e.g. v2.7.0"
        )
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def next_version(registry: dict, classifier: str, bump: str) -> str:
    """Return the next version name for ``classifier``, bumped from its highest version.

    The highest registered version is the base, not the ``production`` pointer, so
    the result is never a name already in the registry. A classifier with no
    versions gets ``v1.0.0``.

    Raises:
        ValueError: ``bump`` is not one of :data:`BUMPS`, or a registered name does
            not parse (the highest version cannot then be known).
    """
    if bump not in BUMPS:
        raise ValueError(f"Invalid bump {bump!r}: expected one of {', '.join(BUMPS)}")

    versions = registry.get(classifier, {}).get("versions", {})
    if not versions:
        return "v1.0.0"

    major, minor, patch = max(parse_version(name) for name in versions)
    if bump == "major":
        return f"v{major + 1}.0.0"
    if bump == "minor":
        return f"v{major}.{minor + 1}.0"
    return f"v{major}.{minor}.{patch + 1}"


def require_new_version(registry: dict, classifier: str, version: str, bump: str) -> None:
    """Refuse a version name that is malformed or already registered.

    Callers run this before any side effect: before files are copied into
    ``models/``, before training, and, for ``register_model.py --update-registry``,
    before MLflow registration.

    Args:
        registry: The loaded registry.
        classifier: Classifier type (fp, ep, esg).
        version: The version name about to be written.
        bump: The caller's bump, used to name the next free version in the refusal.

    Raises:
        ValueError: ``version`` is not canonical ``vMAJOR.MINOR.PATCH`` (see
            :func:`parse_version`).
        VersionExistsError: ``version`` is already registered for ``classifier``.
    """
    parse_version(version)
    if version in registry.get(classifier, {}).get("versions", {}):
        free = next_version(registry, classifier, bump)
        raise VersionExistsError(
            f"{classifier} {version} is already registered, and a registered version "
            f"is never rewritten (D033). The next free version is {free} ({bump} bump "
            "from the highest registered version), or use any higher unregistered version."
        )


def load_registry(registry_path: Union[str, Path]) -> dict:
    """Load the registry, or return an empty one if the file does not exist."""
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return {}
    with open(registry_path) as f:
        return json.load(f)


def save_registry(registry_path: Union[str, Path], registry: dict) -> None:
    """Write the registry as indented JSON with a trailing newline."""
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")


def write_version(
    registry_path: Union[str, Path],
    classifier: str,
    version: str,
    entry: dict,
    config_path: Union[str, Path],
    pipeline_path: Union[str, Path],
    bump: str,
    set_production: bool = False,
) -> str:
    """Add a new version entry to the registry and stamp the artifact config.

    This is the only code that writes a registry entry. The pipeline is hashed once,
    and that hash is written both into the entry as ``pipeline_sha256`` and into the
    config stamp, so the two cannot differ. The entry's ``threshold`` is read from the
    config being stamped, replacing any ``threshold`` in ``entry``, so the build guard's
    threshold check passes on what this writes (D035).

    Args:
        registry_path: The registry JSON file.
        classifier: Classifier type (fp, ep, esg).
        version: The new version name.
        entry: The entry's other fields; the caller decides its shape.
        config_path: The artifact config to stamp.
        pipeline_path: The artifact joblib whose bytes the version names.
        bump: The caller's bump, for the refusal message.
        set_production: Also point ``production`` at the new version.

    Returns:
        The recorded ``pipeline_sha256``.

    Raises:
        ValueError: ``version`` is malformed.
        VersionExistsError: ``version`` is already registered. Nothing is written.
        ArtifactRefused: the config cannot be read as a JSON object, or holds no numeric
            threshold. Nothing is written.
    """
    registry = load_registry(registry_path)
    # Checked again on the registry as loaded here. This narrows, and does not close,
    # the window between a caller's check and this write; concurrent writers are
    # unsupported (D033).
    require_new_version(registry, classifier, version, bump)
    threshold = read_config_threshold(config_path)

    pipeline_sha256 = file_sha256(pipeline_path)
    record = registry.setdefault(classifier, {"production": None, "versions": {}})
    record.setdefault("versions", {})[version] = {
        **entry,
        "threshold": threshold,
        "pipeline_sha256": pipeline_sha256,
    }
    if set_production:
        record["production"] = version

    save_registry(registry_path, registry)
    # After the registry save, so the config never claims a version the registry
    # lacks (D026). If the stamp fails, the name is already spent; recovering is a
    # hand edit (D033).
    stamp_artifact_version(config_path, version, pipeline_sha256)
    return pipeline_sha256


def check_pointer_move(
    registry: dict,
    classifier: str,
    version: str,
    pipeline_path: Union[str, Path],
    config_path: Union[str, Path],
) -> None:
    """Refuse a pointer move unless ``models/`` already holds ``version``'s bytes.

    The move is allowed only when the entry records a ``pipeline_sha256``, the joblib
    at ``pipeline_path`` has that hash, and the config at ``config_path`` is stamped
    with ``version`` and that hash, and its ``threshold`` equals the entry's (D035).
    The check is the build guard's
    (``versioning.check_registered_bytes``, D034), so a move is allowed exactly
    when the build guard would then pass on these files.

    Raises:
        PointerMoveRefused: with the reason.
    """
    try:
        check_registered_bytes(registry, classifier, version, pipeline_path, config_path)
    except ArtifactRefused as e:
        raise PointerMoveRefused(str(e)) from e


def move_pointer(
    registry_path: Union[str, Path],
    classifier: str,
    version: str,
    pipeline_path: Union[str, Path],
    config_path: Union[str, Path],
) -> str | None:
    """Point ``production`` at an existing version without rewriting any entry.

    Runs :func:`check_pointer_move` first and never re-stamps the config. It saves the
    registry as loaded here, so an entry another writer added since the load would be
    lost; concurrent writers are unsupported (D033).

    Returns:
        The previous ``production`` value.

    Raises:
        PointerMoveRefused: the move is refused; nothing is written.
    """
    registry = load_registry(registry_path)
    check_pointer_move(registry, classifier, version, pipeline_path, config_path)

    previous = registry[classifier].get("production")
    registry[classifier]["production"] = version
    save_registry(registry_path, registry)
    return previous
