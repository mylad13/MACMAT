"""Guard that a default MACMAT run does not require the external ``ncps`` package.

The MACMAT transformer uses only the vendored CfCCell/WiredCfCCell; the
ncps-backed exports (LTCCell/CfC/LTC) are imported lazily. This runs a fresh
interpreter with ``ncps`` import blocked and asserts the model still imports.
"""
import subprocess
import sys

from conftest import REPO

_SCRIPT = r"""
import builtins, sys
_real = builtins.__import__
def _blocked(name, *a, **k):
    if name == "ncps" or name.startswith("ncps."):
        raise ImportError("ncps blocked")
    return _real(name, *a, **k)
builtins.__import__ = _blocked
import hetmarl.algorithms.macmat.algorithm.macmat_transformer  # noqa: F401
assert "ncps" not in sys.modules, "ncps was imported during MACMAT import"
print("OK")
"""


def test_macmat_imports_without_ncps():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
