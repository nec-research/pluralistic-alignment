"""Repository paths and the private-storage guard for gated data (FLORES+)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"
WIKI_DATA_DIR = DATA_DIR / "wiki"
WIKI_RESULTS_DIR = RESULTS_DIR / "wiki"

PRIVATE_DIRNAME = "private"
# Blocks whose text (and every output containing it) must stay in private storage.
PRIVATE_BLOCKS = frozenset({"flores"})


class PrivateDataError(RuntimeError):
    """Raised when gated text would be written outside private storage."""


def is_private_path(
    path: Path, roots: Iterable[Path] = (DATA_DIR, RESULTS_DIR)
) -> bool:
    """True if `path` lies under `<root>/private/` for one of `roots`."""
    resolved = path.resolve()
    return any(
        resolved.is_relative_to((root / PRIVATE_DIRNAME).resolve()) for root in roots
    )


def check_output_path(
    path: Path,
    blocks: Iterable[str],
    roots: Iterable[Path] = (DATA_DIR, RESULTS_DIR),
) -> None:
    """Refuse to write gated blocks outside `data/private/` or `results/private/`."""
    private = sorted(set(blocks) & PRIVATE_BLOCKS)
    if private and not is_private_path(path, roots):
        raise PrivateDataError(
            f"blocks {private} are gated (extension plan §2): write them under "
            f"data/{PRIVATE_DIRNAME}/ or results/{PRIVATE_DIRNAME}/, not {path}"
        )
