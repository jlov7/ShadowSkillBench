UV_CACHE_DIR ?= /private/tmp/shadowskillbench-uv-cache
UV = UV_CACHE_DIR="$(UV_CACHE_DIR)" uv
UV_RUN = UV_CACHE_DIR="$(UV_CACHE_DIR)" uv run --no-sync

.PHONY: setup lint typecheck test property-test mutation-test verify clean \
	public-surface sbom pack-manifest verify-public workbench-check workbench-test workbench-build

SBOM_OUTPUT ?= /tmp/shadowskillbench.cdx.json

setup:
	$(UV) sync --all-groups --frozen --no-editable

lint:
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

typecheck:
	$(UV_RUN) pyright

test:
	$(UV_RUN) pytest -m 'not live'

property-test:
	$(UV_RUN) pytest -m property -q

mutation-test:
	$(UV_RUN) pytest -m mutation -q

verify: lint typecheck test property-test mutation-test

public-surface:
	$(UV_RUN) python scripts/public_surface_scan.py --self-check
	$(UV_RUN) python scripts/public_surface_scan.py

sbom:
	$(UV_RUN) python scripts/generate_sbom.py --output "$(SBOM_OUTPUT)"
	$(UV_RUN) python scripts/generate_sbom.py --check "$(SBOM_OUTPUT)"
	$(UV_RUN) python scripts/generate_sbom.py --self-check

pack-manifest:
	$(UV_RUN) python scripts/verify_pack_manifest.py

workbench-check:
	CI=true pnpm --dir workbench run check

workbench-test:
	CI=true pnpm --dir workbench run test

workbench-build:
	CI=true pnpm --dir workbench run build

verify-public: public-surface sbom pack-manifest verify workbench-check workbench-test workbench-build

clean:
	$(UV_RUN) python scripts/safe_clean.py
