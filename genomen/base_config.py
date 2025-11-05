import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class BaseConfig:
    """Base class for all configuration classes with common functionality."""

    @contextmanager
    def temporary_config(self, **kwargs):
        """Context manager to temporarily change configuration values."""
        original_values = {}

        # Save original values and set new ones
        for key, value in kwargs.items():
            if hasattr(self, key):
                original_values[key] = getattr(self, key)
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")

        try:
            yield
        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(self, key, value)

    def save(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where to save the config
        """
        # Create directory if it doesn't exist
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        # Create config dict in the expected format
        config_dict = {self.__class__.__name__: asdict(self)}

        # Save config as YAML
        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
