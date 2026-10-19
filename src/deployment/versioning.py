"""Model version stamping and the values recorded when no version is known.

A classifier's version describes the artifact the API actually loaded (D026).
Every registry write (``src.deployment.registry.write_version``) stamps
``version`` and ``pipeline_sha256`` into the artifact's config via
:func:`stamp_artifact_version`, with the same hash it records in the registry
entry; the API reports that version only when the hash matches the joblib it
loaded.

When no version can be reported, ``classifier_predictions.model_version``
records one of the values below. Each points at a different remedy, so they
are never merged.

The build guard (#172, D034) is :func:`check_build_artifact`. The Dockerfile's builder stage
runs this file by path (``python -I /build/versioning.py``), so it imports only
the standard library and nothing from ``src``.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# The API loaded an artifact with no valid registered version: the config has
# no version, or its pipeline_sha256 is missing or does not match the loaded
# joblib. Remedy: register the artifact, then rebuild the API image from a tree
# holding the stamped config (deploy.yml builds from the committed tree).
UNVERSIONED = "unversioned"

# The API responded but its /model/info carries no version field: the image
# predates this field. Remedy: rebuild the image.
UNREPORTED = "unreported"

# The model-info fetch failed (the client logs why), or the FP batch step failed
# (API call, result handling, or saving the prediction; that row is
# action_taken='failed' and its error_message says which).
UNAVAILABLE = "unavailable"

# The FP classifier is turned off in configuration.
DISABLED = "disabled"

# Legacy value: rows written before version reporting existed. Never written now.
UNKNOWN = "unknown"

NON_VERSION_SENTINELS = frozenset({UNVERSIONED, UNREPORTED, UNAVAILABLE, DISABLED, UNKNOWN})


def file_sha256(path: Union[str, Path]) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp_artifact_version(
    config_path: Union[str, Path],
    version: str,
    pipeline_sha256: str,
) -> None:
    """Write ``version`` and the pipeline's sha256 into an artifact config.

    Called by ``src.deployment.registry.write_version`` after it saves the
    registry entry for ``version``, with the hash it recorded in that entry.

    Args:
        config_path: The artifact's ``<type>_classifier_config.json``.
        version: The registry version string recorded for this artifact.
        pipeline_sha256: The artifact joblib's sha256, which binds the version to it.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = json.load(f)

    previous = config.get("version")
    if previous is not None and previous != version:
        logger.warning(
            f"{config_path} already carried version {previous}; re-stamping as {version}"
        )

    config["version"] = version
    config["pipeline_sha256"] = pipeline_sha256

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


class ArtifactRefused(ValueError):
    """The files are not the registered bytes of the version they are checked against."""


def _load_json_object(path: Path) -> dict:
    """Load a JSON object, refusing a file that is unreadable, not JSON, or not an object."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ArtifactRefused(f"{path} is not valid JSON: {e}") from e
    except OSError as e:
        raise ArtifactRefused(f"{path} cannot be read: {e}") from e
    if not isinstance(data, dict):
        raise ArtifactRefused(f"{path} does not hold a JSON object.")
    return data


def _is_file(path: Path) -> bool:
    """``path.is_file()``, refusing a path that cannot be checked (e.g. a permission error)."""
    try:
        return path.is_file()
    except OSError as e:
        raise ArtifactRefused(f"{path} cannot be read: {e}") from e


def check_registered_bytes(
    registry: dict,
    classifier: str,
    version: str,
    pipeline_path: Union[str, Path],
    config_path: Union[str, Path],
) -> str:
    """Refuse unless the files are ``version``'s registered, stamped bytes.

    The entry must record a ``pipeline_sha256``, the joblib at ``pipeline_path`` must
    have that hash, and the config at ``config_path`` must be stamped with ``version``
    and that hash. Both the build guard (:func:`check_build_artifact`) and a pointer
    move (``src.deployment.registry.check_pointer_move``) use this check (D034).

    Returns:
        The entry's recorded ``pipeline_sha256``.

    Raises:
        ArtifactRefused: with the reason.
    """
    record = registry.get(classifier, {}) if isinstance(registry, dict) else None
    versions = record.get("versions", {}) if isinstance(record, dict) else None
    if not isinstance(versions, dict):
        raise ArtifactRefused(f"{classifier}'s registry record has no versions object.")
    if not isinstance(version, str) or version not in versions:
        raise ArtifactRefused(f"{classifier} {version!r} is not a registered version.")
    entry = versions[version]
    if not isinstance(entry, dict):
        raise ArtifactRefused(f"{classifier} {version}'s registry entry is not an object.")

    recorded = entry.get("pipeline_sha256")
    if not recorded:
        raise ArtifactRefused(
            f"{classifier} {version} has no recorded pipeline_sha256, so its bytes cannot "
            "be checked. Record its hash through its own reconciliation first (D033)."
        )

    pipeline_path = Path(pipeline_path)
    if not _is_file(pipeline_path):
        raise ArtifactRefused(f"Pipeline not found: {pipeline_path}")
    try:
        actual = file_sha256(pipeline_path)
    except OSError as e:
        raise ArtifactRefused(f"{pipeline_path} cannot be read: {e}") from e
    if actual != recorded:
        raise ArtifactRefused(
            f"{pipeline_path} is not {classifier} {version}'s bytes: its sha256 is "
            f"{actual}, and the registry records {recorded}."
        )

    config_path = Path(config_path)
    if not _is_file(config_path):
        raise ArtifactRefused(f"Config not found: {config_path}")
    config = _load_json_object(config_path)
    if config.get("version") != version or config.get("pipeline_sha256") != recorded:
        raise ArtifactRefused(
            f"{config_path} is stamped version={config.get('version')!r}, "
            f"pipeline_sha256={config.get('pipeline_sha256')!r}; {classifier} {version} "
            f"needs version={version!r} and pipeline_sha256={recorded!r}."
        )
    return recorded


def check_build_artifact(
    registry_path: Union[str, Path],
    classifier: str,
    pipeline_path: Union[str, Path],
    config_path: Union[str, Path],
) -> str:
    """Refuse to build unless the files are the registry pointer's registered bytes.

    A classifier whose ``production`` pointer is null is refused before any file is
    looked at, so artifacts present in ``models/`` never let it pass (D029).

    Returns:
        The pointer (the version the files are).

    Raises:
        ArtifactRefused: with the reason, naming the pointer once it has been read.
    """
    registry_path = Path(registry_path)
    if not _is_file(registry_path):
        raise ArtifactRefused(f"Registry not found: {registry_path}")
    registry = _load_json_object(registry_path)

    if classifier not in registry:
        raise ArtifactRefused(
            f"{classifier} is not in {registry_path}; nothing may be built for it."
        )
    record = registry[classifier]
    if not isinstance(record, dict):
        raise ArtifactRefused(f"{classifier}'s record in {registry_path} is not an object.")
    if "production" not in record:
        raise ArtifactRefused(
            f"{classifier} in {registry_path} has no production key; nothing may be built for it."
        )
    pointer = record["production"]
    if pointer is None:
        raise ArtifactRefused(
            f"{classifier} has no production version (its pointer is null in "
            f"{registry_path}); nothing may be built for it, whatever "
            f"{classifier}_classifier_* files are present."
        )
    if not isinstance(pointer, str):
        raise ArtifactRefused(
            f"{classifier}'s production pointer in {registry_path} is {pointer!r}, not a "
            "version name."
        )

    try:
        check_registered_bytes(registry, classifier, pointer, pipeline_path, config_path)
    except ArtifactRefused as e:
        raise ArtifactRefused(f"{classifier} production pointer is {pointer}: {e}") from e
    return pointer


def main(argv: Union[list[str], None] = None) -> int:
    """Run the build guard on ``<models-dir>``: exit 0 to build, 1 to refuse."""
    parser = argparse.ArgumentParser(
        description="Refuse to build a classifier image unless models/ holds the "
        "registry pointer's registered bytes (#172, D034)."
    )
    parser.add_argument("--classifier", required=True, help="Classifier type (fp, ep, esg)")
    parser.add_argument(
        "--models-dir", required=True, type=Path, help="Directory holding registry.json"
    )
    args = parser.parse_args(argv)

    # The same file names as src/deployment/config.py and the Dockerfile's COPY lines.
    pipeline_path = args.models_dir / f"{args.classifier}_classifier_pipeline.joblib"
    config_path = args.models_dir / f"{args.classifier}_classifier_config.json"
    try:
        pointer = check_build_artifact(
            args.models_dir / "registry.json", args.classifier, pipeline_path, config_path
        )
    except ArtifactRefused as e:
        print(f"Build guard refused {args.classifier}: {e}", file=sys.stderr)
        return 1

    print(
        f"Build guard: {args.classifier} artifact is production {pointer} "
        f"(pipeline_sha256 {file_sha256(pipeline_path)})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
