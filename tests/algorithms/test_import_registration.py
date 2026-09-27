import json
import subprocess
import sys


def test_import_optastra_registers_every_algorithm_module():
    """A fresh `import optastra` must import (and so register) every
    algorithm subpackage -- run in a subprocess so modules imported by other
    tests can't mask a missing import in optastra/algorithms/__init__.py."""
    script = (
        "import json, pkgutil, sys\n"
        "import optastra\n"
        "import optastra.algorithms as algos\n"
        "subpackages = [m.name for m in pkgutil.iter_modules(algos.__path__) if m.ispkg]\n"
        "print(json.dumps({\n"
        "    'subpackages': subpackages,\n"
        "    'imported': [p for p in subpackages if f'optastra.algorithms.{p}' in sys.modules],\n"
        "    'registered': optastra.Algorithm.list_all(),\n"
        "}))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout.strip().splitlines()[-1])

    assert {"byol", "simclr"} <= set(info["subpackages"])
    assert info["imported"] == info["subpackages"]
    assert {"byol", "simclr"} <= set(info["registered"])
