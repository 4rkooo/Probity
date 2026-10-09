"""Process settings from environment variables (``PROBITY_*``). Secrets are never logged."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from probity.domain.enums import OperatingMode

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROBITY_", env_file=".env", extra="ignore", frozen=True
    )

    mode: OperatingMode = OperatingMode.AUTO
    data_dir: Path = REPO_ROOT / "data"
    database_url: str | None = None
    policy_path: Path = REPO_ROOT / "config" / "policy.demo.yaml"
    fixture_root: Path = REPO_ROOT / "fixtures" / "demo"
    fixture_manifest: Path = REPO_ROOT / "fixtures" / "demo" / "manifest.json"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    worker_id: str | None = None
    lease_seconds: int = Field(default=30, gt=0)
    worker_poll_seconds: float = Field(default=0.5, gt=0)
    log_level: str = "INFO"

    vast_enabled: bool = False
    vast_endpoint: str | None = None
    vast_token: SecretStr | None = None
    cosmos_enabled: bool = False
    cosmos_endpoint: str | None = None
    cosmos_token: SecretStr | None = None
    cosmos_model_id: str | None = None
    wandb_enabled: bool = False
    wandb_api_key: SecretStr | None = None
    wandb_base_url: str = "https://api.inference.wandb.ai/v1"
    wandb_project: str | None = None

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'cases.db'}"


def publish_process_secrets(settings: Settings) -> None:
    """Expose ``.env`` secrets to adapters that read ``os.environ``.

    Pytest keeps the process environment alone so offline adapter tests stay offline.
    """
    if "pytest" in sys.modules:
        return
    key = settings.wandb_api_key.get_secret_value() if settings.wandb_api_key else ""
    if key:
        os.environ.setdefault("WANDB_API_KEY", key)
        os.environ.setdefault("PROBITY_WANDB_API_KEY", key)
    if settings.wandb_base_url:
        os.environ.setdefault("WANDB_BASE_URL", settings.wandb_base_url)
    if settings.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", settings.wandb_project)
        os.environ.setdefault("WEAVE_PROJECT", settings.wandb_project)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    publish_process_secrets(settings)
    return settings
