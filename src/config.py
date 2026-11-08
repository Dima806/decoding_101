"""Project settings, loaded from ``config/settings.yaml`` with environment overrides.

Precedence (highest first): ``DECODING101_*`` environment variables, then the YAML file,
then the defaults below. Nested keys use ``__``: ``DECODING101_DECODING__DEFAULT_TOP_P=0.9``.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CharLMSettings(_Section):
    context: int = 8
    embed_dim: int = 24
    hidden_dim: int = 256
    train_steps: int = 1500
    batch_size: int = 256
    learning_rate: float = 3e-3


class HFSettings(_Section):
    model_id: str = "distilgpt2"
    max_new_tokens: int = 40


class OllamaSettings(_Section):
    model: str = "qwen2.5:0.5b"
    url: str = "http://localhost:11434"
    timeout_s: float = 60.0


class GenerationSettings(_Section):
    max_new_tokens: int = 300
    n_samples: int = 5


class DecodingSettings(_Section):
    temperatures: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    default_temperature: float = 0.8
    top_k_values: list[int] = [1, 3, 5, 10, 20]
    default_top_k: int = 5
    top_p_values: list[float] = [0.5, 0.8, 0.9, 0.95]
    default_top_p: float = 0.8
    beam_widths: list[int] = [1, 2, 4, 8]
    default_beam_width: int = 4
    length_penalty: float = 1.0
    repetition_penalties: list[float] = [1.0, 1.1, 1.2, 1.3]
    default_repetition_penalty: float = 1.3


class PathSettings(_Section):
    figures: Path = Path("outputs/figures")
    results: Path = Path("outputs/results")


class Settings(BaseSettings):
    """Top-level settings object. Build it with :func:`load_settings`, not directly."""

    model_config = SettingsConfigDict(
        env_prefix="DECODING101_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    seed: int = 42
    char_lm: CharLMSettings = CharLMSettings()
    hf: HFSettings = HFSettings()
    ollama: OllamaSettings = OllamaSettings()
    generation: GenerationSettings = GenerationSettings()
    decoding: DecodingSettings = DecodingSettings()
    paths: PathSettings = PathSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment beats the YAML values passed in as init kwargs.
        return (env_settings, init_settings)

    @property
    def figures_dir(self) -> Path:
        """Absolute path of the figures directory (PNGs)."""
        return _absolute(self.paths.figures)

    @property
    def results_dir(self) -> Path:
        """Absolute path of the results directory (JSON)."""
        return _absolute(self.paths.results)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: Path | str | None = None) -> Settings:
    """Read ``settings.yaml`` (or ``path``) and apply ``DECODING101_*`` overrides."""
    yaml_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    data = yaml.safe_load(yaml_path.read_text()) or {}
    return Settings(**data)


@cache
def get_settings() -> Settings:
    """Process-wide settings from the default YAML file (cached)."""
    return load_settings()
