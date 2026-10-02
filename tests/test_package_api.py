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


def test_submodule_list_matches_the_package_directory() -> None:
    """`_SUBMODULES` must not drift when a module is added or renamed."""
    on_disk = {m.name for m in pkgutil.iter_modules(idsds.__path__)}
    assert set(idsds._SUBMODULES) == on_disk


def test_submodules_are_reachable_as_attributes() -> None:
    """`import ids_defense_selection as idsds; idsds.style` must work."""
    for name in idsds._SUBMODULES:
        module = getattr(idsds, name)
        assert module.__name__ == f"ids_defense_selection.{name}"
    assert idsds.paths.PROJECT_ROOT == idsds.DEFAULT_DATA_DIR.parent


def test_export_names_do_not_shadow_submodules() -> None:
    """A name must mean one thing; otherwise the submodule branch wins silently."""
    assert set(idsds._EXPORTS).isdisjoint(idsds._SUBMODULES)


def test_type_checking_block_declares_every_lazy_export() -> None:
    """IDEs need the flat exports re-declared under ``TYPE_CHECKING``.

    Without them, ``__getattr__ -> Any`` wins and hover/go-to-definition breaks.
    """
    import ast

    source = pathlib.Path(idsds.__file__).read_text(encoding="utf-8")
    declared: dict[str, str] = {}
    submodules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)):
            continue
        if node.test.id != "TYPE_CHECKING":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.ImportFrom) or stmt.level != 1:
                continue
            for alias in stmt.names:
                if stmt.module is None:
                    submodules.add(alias.name)
                else:
                    declared[alias.name] = f".{stmt.module}"
    assert submodules == set(idsds._SUBMODULES)
    assert declared == idsds._EXPORTS, (
        "the TYPE_CHECKING re-exports drifted from _EXPORTS; add or remove the names so both maps match"
    )


def test_unknown_attribute_suggests_the_closest_names() -> None:
    with pytest.raises(AttributeError, match="did you mean"):
        idsds.ExpermentConfig  # noqa: B018 - the typo is the point


def test_dunder_dir_lists_exports_and_submodules() -> None:
    listing = dir(idsds)
    assert set(idsds.__all__) <= set(listing)
    assert set(idsds._SUBMODULES) <= set(listing)


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
