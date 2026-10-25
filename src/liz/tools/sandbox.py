"""Path sandbox shared by the filesystem and terminal tools.

A path is allowed when, after resolving it fully (including `..` and
symlinks), it sits inside one of the configured roots. An empty root list
means unrestricted, which is what unit tests use; the CLI always passes
roots derived from `FS_ROOTS` (home + working directory by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def resolve_roots(roots: Sequence[Path | str] | None) -> list[Path]:
    """Normalize roots to resolved absolute paths."""
    if not roots:
        return []
    return [Path(root).expanduser().resolve() for root in roots]


def within_roots(
    path: Path | str, roots: Sequence[Path | str] | None
) -> tuple[bool, Path]:
    """Check a path against the roots.

    Returns ``(allowed, resolved_path)``. The path is always resolved
    first so `..` segments and symlinks cannot escape the sandbox.
    """
    resolved = Path(path).expanduser().resolve()
    normalized = resolve_roots(roots)
    if not normalized:
        return True, resolved
    for root in normalized:
        if resolved == root or root in resolved.parents:
            return True, resolved
    return False, resolved


def format_roots(roots: Sequence[Path | str] | None) -> str:
    normalized = resolve_roots(roots)
    if not normalized:
        return "unrestricted"
    return ", ".join(str(root) for root in normalized)
