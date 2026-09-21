"""Generate a deterministic, lockfile-bound CycloneDX component inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import uuid
from pathlib import Path
from typing import cast
from urllib.parse import quote

import yaml

ROOT = Path(__file__).resolve().parents[1]


def lock_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_version() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8")).get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml must contain a project table")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml project.version must be a non-empty string")
    return version


def npm_purl(name: str, version: str) -> str:
    if name.startswith("@") and "/" in name:
        scope, package = name[1:].split("/", 1)
        return f"pkg:npm/%40{quote(scope)}/{quote(package)}@{quote(version)}"
    return f"pkg:npm/{quote(name)}@{quote(version)}"


def python_components() -> list[dict[str, str]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    components = []
    for package in lock.get("package", []):
        name = package.get("name")
        version = package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            components.append(
                {
                    "type": "library",
                    "name": name,
                    "version": version,
                    "purl": f"pkg:pypi/{name}@{version}",
                }
            )
    return components


def node_components() -> list[dict[str, str]]:
    lock = yaml.safe_load((ROOT / "workbench/pnpm-lock.yaml").read_text(encoding="utf-8"))
    packages = lock.get("packages", {}) if isinstance(lock, dict) else {}
    components: list[dict[str, str]] = []
    if not isinstance(packages, dict):
        return components
    for locator in packages:
        if not isinstance(locator, str) or "@" not in locator[1:]:
            continue
        name, version = locator.rsplit("@", 1)
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": npm_purl(name, version),
            }
        )
    return components


def validate(document: dict[str, object]) -> None:
    if document.get("bomFormat") != "CycloneDX" or document.get("specVersion") != "1.5":
        raise ValueError("SBOM must identify CycloneDX 1.5")
    serial = document.get("serialNumber")
    if not isinstance(serial, str) or not serial.startswith("urn:uuid:"):
        raise ValueError("SBOM serialNumber must be a UUID URN")
    uuid.UUID(serial.removeprefix("urn:uuid:"))
    components = document.get("components")
    if not isinstance(components, list):
        raise ValueError("SBOM components must be a list")
    purls = [item.get("purl") for item in components if isinstance(item, dict)]
    if len(purls) != len(set(purls)):
        raise ValueError("SBOM component purls must be unique")


def build_document() -> dict[str, object]:
    uv_lock = ROOT / "uv.lock"
    pnpm_lock = ROOT / "workbench/pnpm-lock.yaml"
    components = sorted(
        python_components() + node_components(), key=lambda item: (item["purl"], item["version"])
    )
    serial = uuid.uuid5(uuid.NAMESPACE_OID, "shadowskillbench-lockfile-bound-sbom-v1")
    document: dict[str, object] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "shadowskillbench",
                "version": project_version(),
            },
            "properties": [
                {"name": "ssb:uv-lock-sha256", "value": lock_digest(uv_lock)},
                {"name": "ssb:pnpm-lock-sha256", "value": lock_digest(pnpm_lock)},
            ],
        },
        "components": components,
    }
    return document


def validate_current(document: dict[str, object]) -> None:
    """Require an SBOM to exactly match the current project and lockfiles."""

    validate(document)
    if document != build_document():
        raise ValueError("SBOM does not match the current project and lockfiles")


def self_check() -> int:
    document = build_document()
    validate(document)
    invalid_serial = dict(document, serialNumber="urn:uuid:not-a-uuid")
    try:
        validate(invalid_serial)
    except (ValueError, AttributeError):
        pass
    else:
        raise AssertionError("invalid SBOM serial was accepted")
    components = cast(list[dict[str, str]], document["components"])
    duplicate = dict(document, components=[components[0]] * 2)
    try:
        validate(duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate SBOM purl was accepted")
    metadata = cast(dict[str, object], document["metadata"])
    component = cast(dict[str, object], metadata["component"])
    stale_component = dict(component, version="0.0.0-stale")
    stale_metadata = dict(metadata, component=stale_component)
    try:
        validate_current(dict(document, metadata=stale_metadata))
    except ValueError:
        pass
    else:
        raise AssertionError("stale SBOM metadata was accepted")
    print(f"SBOM_SELF_CHECK_OK components={len(components)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--output", type=Path)
    modes.add_argument("--check", type=Path)
    modes.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    if args.check:
        validate_current(json.loads(args.check.read_text(encoding="utf-8")))
        print(f"SBOM_CHECK_OK input={args.check}")
        return 0
    document = build_document()
    validate(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    components = cast(list[dict[str, str]], document["components"])
    print(f"SBOM_OK components={len(components)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
