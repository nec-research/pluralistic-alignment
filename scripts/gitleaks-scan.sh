#!/usr/bin/env bash
# Downloads a pinned gitleaks release if needed, then scans full git history.
# Requires a git work tree (same as pre-commit and setuptools-scm in this template).
set -euo pipefail

GITLEAKS_VERSION="${GITLEAKS_VERSION:-8.24.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${ROOT}/.cache/gitleaks/${GITLEAKS_VERSION}"
BIN="${CACHE}/gitleaks"

if [[ ! -x "${BIN}" ]]; then
  mkdir -p "${CACHE}"
  os="$(uname -s)"
  arch="$(uname -m)"
  case "${os}" in
    Linux*) os_part=linux ;;
    Darwin*) os_part=darwin ;;
    *)
      echo "gitleaks-scan: unsupported OS: ${os}" >&2
      exit 1
      ;;
  esac
  case "${arch}" in
    x86_64 | amd64) arch_part=x64 ;;
    aarch64 | arm64) arch_part=arm64 ;;
    *)
      echo "gitleaks-scan: unsupported architecture: ${arch}" >&2
      exit 1
      ;;
  esac
  url="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_${os_part}_${arch_part}.tar.gz"
  echo "gitleaks-scan: downloading ${url}" >&2
  if ! command -v curl >/dev/null 2>&1; then
    echo "gitleaks-scan: curl is required to download gitleaks" >&2
    exit 1
  fi
  curl -fsSL "${url}" | tar -xz -C "${CACHE}" gitleaks
fi

cd "${ROOT}"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "gitleaks-scan: not a git repository (this template expects a git work tree; see README)." >&2
  exit 1
fi
exec "${BIN}" git . --verbose --redact
