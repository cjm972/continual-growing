"""!hpo: YAML tag constructors → SearchSpec instances.

A YAML loader subclass `HPOLoader(yaml.SafeLoader)` registers four tags:

    !hpo:float       low, high [, log=true] [, step=...]
    !hpo:int         low, high [, log=true] [, step=...]
    !hpo:loguniform  low, high                # alias for !hpo:float ..., log=true
    !hpo:choice      a, b, c, ...

Scalar grammar after the tag:

    spec  := positional ("," positional)* ("," kwarg)*
    kwarg := IDENT "=" SCALAR

Plain `yaml.safe_load` (used by plain runs) sees `!hpo:float ...` as an
unknown tag and raises — by design. HPO configs MUST be loaded through
`HPOLoader`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from hpo.space import CategoricalSpec, FloatSpec, IntSpec, SearchSpec

_FLOAT_RE = re.compile(
    r"""^[+-]?(
        (\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?
    )$""",
    re.VERBOSE,
)
_INT_RE = re.compile(r"^[+-]?\d+$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _split_outside_quotes(s: str) -> list[str]:
    """Split on commas not inside quoted strings."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for ch in s:
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if quote is not None:
        raise ValueError(f"unterminated quoted string in {s!r}")
    parts.append("".join(buf).strip())
    return [p for p in parts if p != ""]


def _coerce_literal(token: str) -> Any:
    if not token:
        raise ValueError("empty token")
    first, last = token[0], token[-1]
    if first == last and first in ("'", '"') and len(token) >= 2:
        return token[1:-1]
    low = token.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none", "~"):
        return None
    if _INT_RE.match(token):
        return int(token)
    if _FLOAT_RE.match(token):
        return float(token)
    return token


def _split_kwarg(token: str) -> tuple[str, Any] | None:
    eq = token.find("=")
    if eq <= 0:
        return None
    name = token[:eq].strip()
    value_token = token[eq + 1 :].strip()
    if not _IDENT_RE.match(name):
        return None
    return name, _coerce_literal(value_token)


def _parse_spec_args(scalar: str) -> tuple[list[Any], dict[str, Any]]:
    if scalar is None:
        return [], {}
    s = scalar.strip()
    if not s:
        return [], {}
    parts = _split_outside_quotes(s)
    positional: list[Any] = []
    kwargs: dict[str, Any] = {}
    seen_kwarg = False
    for tok in parts:
        kw = _split_kwarg(tok)
        if kw is not None:
            seen_kwarg = True
            name, value = kw
            if name in kwargs:
                raise ValueError(f"duplicate kwarg {name!r} in tag scalar {scalar!r}")
            kwargs[name] = value
        else:
            if seen_kwarg:
                raise ValueError(
                    f"positional argument after kwarg in tag scalar {scalar!r}"
                )
            positional.append(_coerce_literal(tok))
    return positional, kwargs


def _expect_n_positional(label: str, positional: Sequence[Any], n: int, raw: str) -> None:
    if len(positional) != n:
        raise ValueError(
            f"!hpo:{label} expects {n} positional argument(s), got "
            f"{len(positional)}: {raw!r}"
        )


def _check_kwargs(label: str, kwargs: dict[str, Any], allowed: set[str], raw: str) -> None:
    extra = set(kwargs) - allowed
    if extra:
        raise ValueError(
            f"!hpo:{label} got unknown kwargs {sorted(extra)} "
            f"(allowed: {sorted(allowed)}): {raw!r}"
        )


def _construct_float(loader: yaml.Loader, node: yaml.ScalarNode) -> SearchSpec:
    raw = node.value
    positional, kwargs = _parse_spec_args(raw)
    _expect_n_positional("float", positional, 2, raw)
    _check_kwargs("float", kwargs, {"log", "step"}, raw)
    return FloatSpec(
        low=float(positional[0]),
        high=float(positional[1]),
        log=bool(kwargs.get("log", False)),
        step=(None if kwargs.get("step") is None else float(kwargs["step"])),
    )


def _construct_int(loader: yaml.Loader, node: yaml.ScalarNode) -> SearchSpec:
    raw = node.value
    positional, kwargs = _parse_spec_args(raw)
    _expect_n_positional("int", positional, 2, raw)
    _check_kwargs("int", kwargs, {"log", "step"}, raw)
    return IntSpec(
        low=int(positional[0]),
        high=int(positional[1]),
        log=bool(kwargs.get("log", False)),
        step=int(kwargs.get("step", 1)),
    )


def _construct_loguniform(loader: yaml.Loader, node: yaml.ScalarNode) -> SearchSpec:
    raw = node.value
    positional, kwargs = _parse_spec_args(raw)
    _expect_n_positional("loguniform", positional, 2, raw)
    _check_kwargs("loguniform", kwargs, set(), raw)
    return FloatSpec(low=float(positional[0]), high=float(positional[1]), log=True)


def _construct_choice(loader: yaml.Loader, node: yaml.ScalarNode) -> SearchSpec:
    raw = node.value
    positional, kwargs = _parse_spec_args(raw)
    if positional == [] and kwargs == {}:
        raise ValueError(f"!hpo:choice requires at least one choice: {raw!r}")
    _check_kwargs("choice", kwargs, set(), raw)
    return CategoricalSpec(choices=tuple(positional))


class HPOLoader(yaml.SafeLoader):
    """SafeLoader subclass with `!hpo:*` tag constructors registered."""


HPOLoader.add_constructor("!hpo:float", _construct_float)
HPOLoader.add_constructor("!hpo:int", _construct_int)
HPOLoader.add_constructor("!hpo:loguniform", _construct_loguniform)
HPOLoader.add_constructor("!hpo:choice", _construct_choice)


def load_yaml_with_hpo(path: str | Path) -> Any:
    return yaml.load(Path(path).read_text(), Loader=HPOLoader)
