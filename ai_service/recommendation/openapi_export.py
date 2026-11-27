"""Deterministic OpenAPI contract exporter and drift verifier (NFR-14)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai_service.main import create_app
from ai_service.recommendation.settings import Settings


def default_contract_path() -> Path:
    """Return canonical repository path for the frozen AI service OpenAPI YAML contract."""
    return Path(__file__).resolve().parents[2] / "docs" / "contracts" / "ai-service.openapi.yaml"


def build_live_openapi_schema() -> dict[str, Any]:
    """Generate the live FastAPI OpenAPI dictionary independent of local .env secrets."""
    cfg = Settings(
        _env_file=None,
        ai_service_env="development",
        ai_service_token=None,
        ai_service_require_auth=None,
    )
    app = create_app(cfg)
    return app.openapi()


def render_openapi_yaml(schema: dict[str, Any]) -> str:
    """Serialize an OpenAPI schema dict deterministically to YAML."""
    header = (
        "# CoverageIQ AI Service OpenAPI Contract (Module M5 / 6A)\n"
        "# Generated automatically from FastAPI app.openapi() — do not edit manually.\n"
        "# Regenerate via: python -m ai_service.recommendation.openapi_export\n\n"
    )
    body = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True, width=100)
    return header + body


def export_openapi_contract(output_path: Path | None = None) -> Path:
    """Write the frozen OpenAPI contract YAML matching the live FastAPI schema."""
    target = output_path or default_contract_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = render_openapi_yaml(build_live_openapi_schema())
    target.write_text(yaml_text, encoding="utf-8")
    return target


def verify_openapi_contract(contract_path: Path | None = None) -> None:
    """Assert the frozen OpenAPI YAML file matches live FastAPI app.openapi() output."""
    target = contract_path or default_contract_path()
    if not target.is_file():
        raise FileNotFoundError(f"Frozen OpenAPI contract file missing: {target}")
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    live = build_live_openapi_schema()
    if loaded != live:
        raise ValueError(
            "OpenAPI contract drift detected between docs/contracts/ai-service.openapi.yaml "
            "and live FastAPI create_app().openapi()."
        )


if __name__ == "__main__":
    written_path = export_openapi_contract()
    print(f"Exported OpenAPI contract to {written_path}")
