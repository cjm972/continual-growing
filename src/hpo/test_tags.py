"""Smoke tests for the !hpo: tag loader. Run with `python -m hpo.test_tags`."""

from __future__ import annotations

import yaml

from hpo.space import CategoricalSpec, FloatSpec, IntSpec
from hpo.tags import HPOLoader


def _load(text: str):
    return yaml.load(text, Loader=HPOLoader)


def test_float_basic():
    out = _load("v: !hpo:float 0.0, 1.0")
    assert out["v"] == FloatSpec(low=0.0, high=1.0, log=False, step=None)


def test_float_with_kwargs():
    out = _load("v: !hpo:float 0.05, 1.0, log=true, step=0.05")
    assert out["v"] == FloatSpec(low=0.05, high=1.0, log=True, step=0.05)


def test_int_basic():
    out = _load("v: !hpo:int 1, 5")
    assert out["v"] == IntSpec(low=1, high=5, log=False, step=1)


def test_loguniform():
    out = _load("v: !hpo:loguniform 1.0e-4, 1.0e-1")
    assert out["v"] == FloatSpec(low=1e-4, high=1e-1, log=True, step=None)


def test_choice():
    out = _load("v: !hpo:choice tpe, cmaes, random")
    assert out["v"] == CategoricalSpec(choices=("tpe", "cmaes", "random"))


def test_bad_arity():
    try:
        _load("v: !hpo:float 0.0")
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong arity")


def test_unknown_kwarg():
    try:
        _load("v: !hpo:float 0.0, 1.0, foo=1")
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown kwarg")


if __name__ == "__main__":
    test_float_basic()
    test_float_with_kwargs()
    test_int_basic()
    test_loguniform()
    test_choice()
    test_bad_arity()
    test_unknown_kwarg()
    print("OK")
