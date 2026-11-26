SHELL := /bin/bash
project_dir := $(shell pwd)
MIN_UV_VERSION := 0.11.16
export UV_MALWARE_CHECK ?= 1

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make <target>\n\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: check-uv-version
check-uv-version: ## Ensure uv supports audit and malware checks
	@command -v uv >/dev/null || { echo "uv $(MIN_UV_VERSION)+ required. Install uv, then rerun this target." >&2; exit 1; }
	@version="$$(uv --version | awk '{print $$2}')"; \
	python3 -c 'import sys; parse=lambda s: tuple(int(p) for p in s.split(".")[:3]); sys.exit(0 if parse(sys.argv[1]) >= parse(sys.argv[2]) else 1)' "$$version" "$(MIN_UV_VERSION)" || { \
		echo "uv $(MIN_UV_VERSION)+ required. Found: uv $$version" >&2; \
		echo "Upgrade uv with your normal installer or package manager, then rerun this target." >&2; \
		echo "Temporary fallback: UV_MALWARE_CHECK=1 uvx --from uv==0.11.26 uv sync --dev --locked" >&2; \
		exit 1; \
	}

.PHONY: dev
dev: check-uv-version ## Install dev dependencies
	cd $(project_dir)
	uv sync --dev

.PHONY: install
install: check-uv-version ## Install runtime dependencies
	cd $(project_dir)
	uv sync --no-dev

.PHONY: lint
lint: ## Run ruff linting
	cd $(project_dir)
	uv run ruff check .

.PHONY: lint-fix
lint-fix: ## Run ruff with auto-fixes
	cd $(project_dir)
	uv run ruff check --fix .

.PHONY: format
format: ## Format code with ruff
	cd $(project_dir)
	uv run ruff format .

.PHONY: format-check
format-check: ## Check formatting without changes
	cd $(project_dir)
	uv run ruff format --check .

.PHONY: typecheck
typecheck: ## Run Pyright (standard mode)
	cd $(project_dir)
	uv run basedpyright  # paths come from [tool.pyright] include

.PHONY: test
test: ## Run tests
	cd $(project_dir)
	uv run pytest -vv

.PHONY: licensecheck
licensecheck: ## Check licenses via pylic
	cd $(project_dir)
	uv run pylic check --allow-extra-safe-licenses

.PHONY: audit
audit: check-uv-version ## Audit dependencies for known vulnerabilities
	cd $(project_dir)
	uv audit --locked

.PHONY: pre-commit
pre-commit: ## Run all pre-commit hooks
	cd $(project_dir)
	uv run pre-commit run --all-files

.PHONY: gitleaks
gitleaks: ## Scan full git history for secrets (requires a git work tree)
	cd $(project_dir)
	bash scripts/gitleaks-scan.sh

.PHONY: check
check: check-uv-version pre-commit gitleaks test licensecheck audit
