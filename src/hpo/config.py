"""Layered --config loading: deep-merge YAMLs, flatten by leaf, apply to argparse.

Precedence (low → high): leftmost --config < … < rightmost --config < CLI flags.

`with_hpo=True` selects `HPOLoader` (allows `!hpo:*` tags); plain runs use
`yaml.SafeLoader` and reject HPO configs at parse time.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from hpo.tags import HPOLoader

# Top-level keys in a YAML config that are HPO-only and must NOT be flattened
# into argparse dests. Kept nested and returned alongside the flat dict.
HPO_ONLY_KEYS = frozenset({"hpo"})


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursive last-wins merge. Lists/scalars replace; dicts recurse."""
    out = dict(base)
    for k, v in over.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def _resolve_config_path(arg: str) -> Path:
    """Accept a short name (`hpo_lr_search`) or absolute/relative path.

    Short names resolve to `<repo>/configs/<name>.yaml`. Direct paths win
    if they exist.
    """
    p = Path(arg)
    if p.is_file():
        return p
    if not p.suffix:
        candidate = CONFIGS_DIR / f"{arg}.yaml"
        if candidate.is_file():
            return candidate
    candidate = CONFIGS_DIR / arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"config {arg!r} not found. Tried:\n  - {p.resolve()}\n  - {CONFIGS_DIR / (arg + '.yaml')}"
    )


def load_layered_config(
    paths: Sequence[str], *, with_hpo: bool
) -> dict[str, Any]:
    """Load each path, deep-merge left-to-right (last wins).

    `with_hpo` selects HPOLoader (HPO mode) or yaml.SafeLoader (concrete-run
    mode). In concrete-run mode, `!hpo:*` tags raise.
    """
    Loader = HPOLoader if with_hpo else yaml.SafeLoader
    merged: dict[str, Any] = {}
    for raw in paths:
        path = _resolve_config_path(raw)
        loaded = yaml.load(path.read_text(), Loader=Loader)
        if loaded is None:
            continue
        if not isinstance(loaded, dict):
            raise ValueError(
                f"{path}: expected a top-level mapping, got {type(loaded).__name__}"
            )
        merged = _deep_merge(merged, loaded)
    return merged


def _walk_leaves(
    node: Any, prefix: str, out: dict[str, tuple[str, Any]]
) -> None:
    """Collect (leaf_name → (dotted_path, value)) for every leaf in `node`.

    Skips entries that are dicts (we recurse into them). Skips lists at any
    level — argparse dests don't take nested-list overrides.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                _walk_leaves(v, child, out)
            else:
                if k in out:
                    raise ValueError(
                        f"duplicate leaf key {k!r}: previously at "
                        f"{out[k][0]!r}, now at {child!r}"
                    )
                out[k] = (child, v)


def flatten_to_argparse(
    merged: dict[str, Any], parser: argparse.ArgumentParser
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Flatten `merged` into `{argparse_dest: value}`.

    The `hpo:` block is split off and returned separately (kept nested).
    Unknown leaf names raise. Each value is validated against the matching
    argparse action's `choices` (if defined).
    """
    hpo_block = merged.get("hpo", {}) if isinstance(merged.get("hpo"), dict) else {}
    body = {k: v for k, v in merged.items() if k not in HPO_ONLY_KEYS}

    leaves: dict[str, tuple[str, Any]] = {}
    _walk_leaves(body, "", leaves)

    valid_dests: dict[str, argparse.Action] = {
        a.dest: a for a in parser._actions if a.dest != "help"
    }

    flat: dict[str, Any] = {}
    for leaf, (dotted, value) in leaves.items():
        if leaf not in valid_dests:
            raise ValueError(
                f"unknown config key {leaf!r} (at {dotted!r}); not an argparse dest"
            )
        action = valid_dests[leaf]
        if action.choices is not None and value not in action.choices:
            raise ValueError(
                f"invalid value for {leaf!r} (at {dotted!r}): {value!r} "
                f"(choices: {list(action.choices)})"
            )
        flat[leaf] = value
    return flat, hpo_block


def explicit_cli_dests(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> set[str]:
    """Return the set of argparse dests the user explicitly passed on argv.

    Done via a clone of `parser` whose every action's default is `None`
    sentinel, then re-parsing `argv`. Any dest still `None` after re-parse
    was NOT set on the CLI.
    """
    sentinel = object()
    clone = argparse.ArgumentParser(add_help=False)
    for action in parser._actions:
        if action.dest == "help":
            continue
        kwargs: dict[str, Any] = {"dest": action.dest, "default": sentinel}
        if action.option_strings:
            args = list(action.option_strings)
        else:
            args = [action.dest]
        # Replicate the action subclass behaviour so flags parse identically.
        if isinstance(action, argparse._StoreTrueAction):
            kwargs["action"] = "store_true"
            kwargs["default"] = sentinel
        elif isinstance(action, argparse._StoreFalseAction):
            kwargs["action"] = "store_false"
            kwargs["default"] = sentinel
        elif isinstance(action, argparse._StoreConstAction):
            kwargs["action"] = "store_const"
            kwargs["const"] = action.const
        elif isinstance(action, argparse._AppendAction):
            kwargs["action"] = "append"
        else:
            if action.nargs is not None:
                kwargs["nargs"] = action.nargs
            if action.type is not None:
                kwargs["type"] = action.type
            if action.choices is not None:
                kwargs["choices"] = action.choices
        try:
            clone.add_argument(*args, **kwargs)
        except (argparse.ArgumentError, TypeError, ValueError):
            # Skip args we can't faithfully clone — they'll just be treated
            # as not-on-CLI (config wins). Acceptable conservative default.
            continue
    parsed, _ = clone.parse_known_args(list(argv))
    return {
        dest: None for dest, val in vars(parsed).items() if val is not sentinel
    }.keys() | set()


def apply_to_args(
    args: argparse.Namespace,
    flat: dict[str, Any],
    explicit: set[str],
) -> None:
    """Overwrite `args.<key>` for each entry NOT in `explicit` (CLI wins)."""
    for k, v in flat.items():
        if k in explicit:
            continue
        if not hasattr(args, k):
            raise ValueError(f"argparse namespace has no attribute {k!r}")
        setattr(args, k, v)


def load_and_apply_configs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config_paths: Sequence[str],
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """End-to-end plain-run helper.

    Loads `config_paths` in concrete mode, validates leaf names, applies
    them to `args` while letting CLI flags win. Returns the `hpo:` block
    (empty for plain runs that don't have one).
    """
    if not config_paths:
        return {}
    merged = load_layered_config(config_paths, with_hpo=False)
    flat, hpo_block = flatten_to_argparse(merged, parser)
    explicit = explicit_cli_dests(parser, sys.argv[1:] if argv is None else argv)
    apply_to_args(args, flat, explicit)
    return hpo_block
