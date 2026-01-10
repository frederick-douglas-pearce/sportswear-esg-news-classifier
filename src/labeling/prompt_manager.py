"""Prompt version management for the labeling pipeline.

This module provides versioned prompt management for the ESG labeling pipeline.
Prompts are stored in text files under prompts/labeling/ with version directories.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default prompts directory relative to project root
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "labeling"


@dataclass
class PromptVersion:
    """A versioned prompt configuration.

    Attributes:
        version: Semantic version string (e.g., 'v1.0.0')
        system_prompt: The system prompt template text
        user_prompt: The user prompt template text
        system_prompt_hash: SHA256 hash prefix for verification
        user_prompt_hash: SHA256 hash prefix for verification
        model_config: Recommended model configuration
        created_at: ISO timestamp when version was created
        commit_message: Description of changes in this version
        description: Longer description of the prompt version
        tags: Key-value metadata tags
    """

    version: str
    system_prompt: str
    user_prompt: str
    system_prompt_hash: str
    user_prompt_hash: str
    model_config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    commit_message: str = ""
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "version": self.version,
            "system_prompt_hash": self.system_prompt_hash,
            "user_prompt_hash": self.user_prompt_hash,
            "model_config": self.model_config,
            "created_at": self.created_at,
            "commit_message": self.commit_message,
            "description": self.description,
            "tags": self.tags,
        }

    def to_metadata(self) -> dict[str, Any]:
        """Get metadata for database storage (excludes prompt text)."""
        return {
            "version": self.version,
            "system_prompt_hash": self.system_prompt_hash,
            "user_prompt_hash": self.user_prompt_hash,
        }


class PromptManager:
    """Manages prompt versions for the labeling pipeline.

    This class handles loading, caching, and formatting of versioned prompts.
    Prompts are stored in text files under prompts/labeling/{version}/.

    Example:
        >>> manager = PromptManager()
        >>> version = manager.load_version("v1.0.0")
        >>> system = manager.get_formatted_system_prompt(brands=["Nike", "Adidas"])
        >>> user = manager.get_formatted_user_prompt(
        ...     title="Nike sustainability report",
        ...     content="Article text...",
        ...     brands="Nike",
        ... )
    """

    def __init__(self, prompts_dir: Path | None = None):
        """Initialize the prompt manager.

        Args:
            prompts_dir: Path to prompts directory. Defaults to prompts/labeling/.
        """
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self._registry: dict | None = None
        self._cache: dict[str, PromptVersion] = {}

    def _load_registry(self) -> dict:
        """Load the prompt registry.

        Returns:
            Registry dictionary with version metadata.

        Raises:
            FileNotFoundError: If registry.json doesn't exist.
        """
        if self._registry is None:
            registry_path = self.prompts_dir / "registry.json"
            if registry_path.exists():
                with open(registry_path) as f:
                    self._registry = json.load(f)
            else:
                raise FileNotFoundError(f"Prompt registry not found: {registry_path}")
        return self._registry

    def get_production_version(self) -> str:
        """Get the current production prompt version.

        Returns:
            Version string (e.g., 'v1.0.0').
        """
        registry = self._load_registry()
        return registry.get("production", "v1.0.0")

    def list_versions(self) -> list[str]:
        """List all available prompt versions.

        Returns:
            List of version strings, sorted by version number.
        """
        registry = self._load_registry()
        versions = list(registry.get("versions", {}).keys())
        # Sort by version number
        return sorted(versions, key=lambda v: [int(x) for x in v.lstrip("v").split(".")])

    def get_version_info(self, version: str | None = None) -> dict[str, Any]:
        """Get metadata for a specific version.

        Args:
            version: Version to get info for. Defaults to production version.

        Returns:
            Dictionary with version metadata.

        Raises:
            ValueError: If version doesn't exist.
        """
        if version is None:
            version = self.get_production_version()

        registry = self._load_registry()
        versions = registry.get("versions", {})

        if version not in versions:
            raise ValueError(f"Unknown prompt version: {version}")

        return {
            "version": version,
            "is_production": version == registry.get("production"),
            **versions[version],
        }

    def load_version(self, version: str | None = None) -> PromptVersion:
        """Load a specific prompt version.

        Args:
            version: Version to load. Defaults to production version.

        Returns:
            PromptVersion instance with loaded prompts.

        Raises:
            ValueError: If version doesn't exist.
            FileNotFoundError: If prompt files are missing.
        """
        if version is None:
            version = self.get_production_version()

        # Return cached version if available
        if version in self._cache:
            return self._cache[version]

        registry = self._load_registry()
        versions = registry.get("versions", {})

        if version not in versions:
            available = ", ".join(self.list_versions())
            raise ValueError(
                f"Unknown prompt version: {version}. Available: {available}"
            )

        version_dir = self.prompts_dir / version
        config_path = version_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Version config not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Load prompt files
        system_path = version_dir / config["system_prompt_file"]
        user_path = version_dir / config["user_prompt_file"]

        if not system_path.exists():
            raise FileNotFoundError(f"System prompt not found: {system_path}")
        if not user_path.exists():
            raise FileNotFoundError(f"User prompt not found: {user_path}")

        with open(system_path) as f:
            system_prompt = f.read()
        with open(user_path) as f:
            user_prompt = f.read()

        # Get version metadata
        version_meta = versions[version]

        # Compute hashes for verification
        system_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]
        user_hash = hashlib.sha256(user_prompt.encode()).hexdigest()[:12]

        prompt_version = PromptVersion(
            version=version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            system_prompt_hash=system_hash,
            user_prompt_hash=user_hash,
            model_config=config.get("model_config", {}),
            created_at=version_meta.get("created_at", ""),
            commit_message=version_meta.get("commit_message", ""),
            description=version_meta.get("description", ""),
            tags=version_meta.get("tags", {}),
        )

        # Cache the loaded version
        self._cache[version] = prompt_version
        logger.debug(f"Loaded prompt version {version} (system: {system_hash}, user: {user_hash})")

        return prompt_version

    def get_formatted_system_prompt(
        self,
        version: str | None = None,
        brands: list[str] | None = None,
    ) -> str:
        """Get the system prompt with variables filled in.

        Args:
            version: Prompt version to use. Defaults to production.
            brands: List of brand names to include in prompt.

        Returns:
            Formatted system prompt string.
        """
        prompt_version = self.load_version(version)
        brands_str = ", ".join(brands) if brands else ""
        return prompt_version.system_prompt.format(brands=brands_str)

    def get_formatted_user_prompt(
        self,
        version: str | None = None,
        title: str = "",
        published_at: str = "Unknown",
        source_name: str = "Unknown",
        brands: str = "",
        content: str = "",
    ) -> str:
        """Get the user prompt with variables filled in.

        Args:
            version: Prompt version to use. Defaults to production.
            title: Article title.
            published_at: Publication date string.
            source_name: Source/publisher name.
            brands: Comma-separated brand names to analyze.
            content: Article content text.

        Returns:
            Formatted user prompt string.
        """
        prompt_version = self.load_version(version)
        return prompt_version.user_prompt.format(
            title=title,
            published_at=published_at,
            source_name=source_name,
            brands=brands,
            content=content,
        )

    def clear_cache(self) -> None:
        """Clear the version cache.

        Call this if prompt files have been modified on disk.
        """
        self._cache.clear()
        self._registry = None
        logger.debug("Cleared prompt version cache")


# Default instance for convenience
prompt_manager = PromptManager()


def get_prompt_version(version: str | None = None) -> PromptVersion:
    """Get a prompt version using the default manager.

    Args:
        version: Version to load. Defaults to production.

    Returns:
        PromptVersion instance.
    """
    return prompt_manager.load_version(version)


def get_production_prompt_version() -> str:
    """Get the production prompt version string.

    Returns:
        Production version string (e.g., 'v1.0.0').
    """
    return prompt_manager.get_production_version()


def list_prompt_versions() -> list[str]:
    """List all available prompt versions.

    Returns:
        List of version strings.
    """
    return prompt_manager.list_versions()
