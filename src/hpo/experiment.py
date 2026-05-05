"""Search-space composition over the merged HPO config.

`compose_search_config(paths)` loads the layered configs with HPOLoader,
walks the merged dict to collect every `(dotted_path, SearchSpec)` pair,
and returns the merged dict alongside the sorted search space.

The merged dict still contains `SearchSpec` instances at the searched
leaves — the runner is responsible for replacing them with concrete trial
values via `splice_value(...)`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from hpo.config import load_layered_config
from hpo.space import CategoricalSpec, FloatSpec, IntSpec, SearchSpec


def _walk_collect_specs(
    node: Any, prefix: str, out: list[tuple[str, SearchSpec]]
) -> None:
    if isinstance(node, (FloatSpec, IntSpec, CategoricalSpec)):
        out.append((prefix, node))
        return
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            _walk_collect_specs(v, child, out)
        return
    if isinstance(node, list):
        for i, v in enumerate(node):
            child = f"{prefix}[{i}]"
            _walk_collect_specs(v, child, out)


def compose_search_config(
    paths: Sequence[str],
) -> tuple[dict[str, Any], list[tuple[str, SearchSpec]]]:
    """Load layered configs (HPO mode) and collect search-space leaves.

    Returns `(merged_dict, search_space)` sorted by dotted path. The
    `hpo:` block is preserved at `merged["hpo"]` and is NOT walked for
    search-space tags (HPO knobs are concrete only).
    """
    merged = load_layered_config(paths, with_hpo=True)
    body = {k: v for k, v in merged.items() if k != "hpo"}
    search_space: list[tuple[str, SearchSpec]] = []
    _walk_collect_specs(body, "", search_space)
    search_space.sort(key=lambda kv: kv[0])
    return merged, search_space


def splice_value(
    merged: dict[str, Any], dotted_path: str, value: Any
) -> None:
    """In-place: write `value` to the leaf at `dotted_path`.

    Supports keys like `lr_mu` and bracketed list indices like
    `weights[2]` (matching how `_walk_collect_specs` builds paths).
    """
    parts: list[str | int] = []
    buf = ""
    i = 0
    while i < len(dotted_path):
        ch = dotted_path[i]
        if ch == ".":
            if buf:
                parts.append(buf)
                buf = ""
            i += 1
            continue
        if ch == "[":
            if buf:
                parts.append(buf)
                buf = ""
            close = dotted_path.find("]", i)
            if close < 0:
                raise ValueError(f"unterminated bracket in path {dotted_path!r}")
            parts.append(int(dotted_path[i + 1 : close]))
            i = close + 1
            continue
        buf += ch
        i += 1
    if buf:
        parts.append(buf)
    if not parts:
        raise ValueError(f"empty path {dotted_path!r}")

    cur: Any = merged
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def _spec_to_jsonable(spec: SearchSpec) -> dict[str, Any]:
    if isinstance(spec, FloatSpec):
        return {
            "kind": "float", "low": spec.low, "high": spec.high,
            "log": spec.log, "step": spec.step,
        }
    if isinstance(spec, IntSpec):
        return {
            "kind": "int", "low": spec.low, "high": spec.high,
            "log": spec.log, "step": spec.step,
        }
    if isinstance(spec, CategoricalSpec):
        return {"kind": "categorical", "choices": list(spec.choices)}
    raise TypeError(f"unknown SearchSpec subtype: {type(spec).__name__}")


def search_space_hash(search_space: Sequence[tuple[str, SearchSpec]]) -> str:
    """Stable hex digest of the search space, used to gate study resume."""
    payload = [
        {"path": path, **_spec_to_jsonable(spec)}
        for path, spec in sorted(search_space, key=lambda kv: kv[0])
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def search_space_to_jsonable(
    search_space: Sequence[tuple[str, SearchSpec]],
) -> list[dict[str, Any]]:
    return [
        {"path": path, **_spec_to_jsonable(spec)}
        for path, spec in search_space
    ]


def stripped_concrete_dict(merged: dict[str, Any]) -> dict[str, Any]:
    """Replace every `SearchSpec` in `merged` with a `null` placeholder."""
    def _strip(node: Any) -> Any:
        if isinstance(node, (FloatSpec, IntSpec, CategoricalSpec)):
            return None
        if isinstance(node, dict):
            return {k: _strip(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_strip(v) for v in node]
        return node
    return _strip(merged)
