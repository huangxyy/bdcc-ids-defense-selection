"""Import and path sanity checks for the package.

These guard the three failure modes we hit in practice:

* a module that cannot be imported at all (broken relative import, circular
  import, missing dependency);
* a lazy export in ``ids_defense_selection._EXPORTS`` that points at a name
  which no longer exists;
* the PyPI ``pathlib`` backport shadowing the standard library.
"""
from __future__ import annotations

import importlib
import pathlib
import pkgutil

import pytest

import ids_defense_selection as idsds


def test_every_submodule_imports() -> None:
    module_names = sorted(m.name for m in pkgutil.iter_modules(idsds.__path__))
    assert module_names, "the package must expose at least one submodule"
    for name in module_names:
        importlib.import_module(f"ids_defense_selection.{name}")


def test_every_public_export_resolves() -> None:
    failed = {}
    for name in idsds._EXPORTS:
        try:
            getattr(idsds, name)
        except Exception as exc:  # noqa: BLE001 - report every broken export at once
            failed[name] = f"{type(exc).__name__}: {exc}"
    assert not failed, f"broken lazy exports: {failed}"
    assert sorted(idsds.__all__) == sorted(idsds._EXPORTS)


def test_pathlib_does_not_come_from_site_packages() -> None:
    """PyPI's ``pathlib`` backport crashes on modern Python; keep it uninstalled."""
    path = pathlib.Path(pathlib.__file__).resolve()
    assert "site-packages" not in path.parts, (
        f"`pathlib` is resolving to {path}; remove the third-party `pathlib` package"
    )
    assert hasattr(pathlib.Path, "is_relative_to"), "the stdlib pathlib is required"


@pytest.mark.parametrize("attribute", ["PROJECT_ROOT", "DEFAULT_DATA_DIR", "DEFAULT_OUTPUT_ROOT"])
def test_default_paths_point_inside_the_repository(attribute: str) -> None:
    from ids_defense_selection.paths import PROJECT_ROOT

    value = getattr(importlib.import_module("ids_defense_selection.paths"), attribute)
    assert value == PROJECT_ROOT or PROJECT_ROOT in value.parents
