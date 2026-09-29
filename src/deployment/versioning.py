"""Model version stamping and the values recorded when no version is known.

A classifier's version describes the artifact the API actually loaded (D026).
Every registry writer stamps ``version`` and ``pipeline_sha256`` into the
artifact's config via :func:`stamp_artifact_version`; the API reports that
version only when the hash matches the joblib it loaded.

When no version can be reported, ``classifier_predictions.model_version``
records one of the values below. Each points at a different remedy, so they
are never merged.
"""

import hashlib
import json
import logging
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
    pipeline_path: Union[str, Path],
) -> None:
    """Write ``version`` and the pipeline's sha256 into an artifact config.

    Call this only where a registry entry for ``version`` is written for the
    artifact at ``pipeline_path``/``config_path``.

    Args:
        config_path: The artifact's ``<type>_classifier_config.json``.
        version: The registry version string recorded for this artifact.
        pipeline_path: The artifact's joblib, whose hash binds the version to it.
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
    config["pipeline_sha256"] = file_sha256(pipeline_path)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
