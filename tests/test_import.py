"""Import every live module under ``hetmarl``.

Guards against dead-import regressions (e.g. a module that references a package
that no longer exists). ``setup.py`` is excluded because it calls
``setuptools.setup()`` at import time, which is packaging, not a live module.
"""
import importlib

import pytest

from conftest import REPO

PKG_ROOT = REPO / "hetmarl"


def _iter_modules():
    for path in sorted(PKG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "setup.py":
            continue
        rel = path.relative_to(REPO)
        if path.name == "__init__.py":
            yield ".".join(rel.parent.parts)
        else:
            yield ".".join(rel.with_suffix("").parts)


MODULES = list(_iter_modules())


def test_module_list_nonempty():
    assert len(MODULES) > 40, MODULES


@pytest.mark.parametrize("module", MODULES)
def test_import_module(module):
    importlib.import_module(module)
