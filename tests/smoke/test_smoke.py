from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

from probity.api.main import create_app


def test_core_modules_import() -> None:
    for name in (
        "probity.domain.models",
        "probity.domain.enums",
        "probity.domain.policy",
        "probity.domain.api",
        "probity.domain.fixtures",
        "probity.ports",
        "probity.config",
        "probity.cli",
    ):
        importlib.import_module(name)


def test_api_app_builds_and_serves_openapi() -> None:
    client = TestClient(create_app())
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/videos/{video_id}/searches" in response.json()["paths"]
