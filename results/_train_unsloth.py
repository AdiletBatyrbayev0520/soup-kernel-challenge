"""harness/_train.py, unchanged except that it lets Soup's `backend: unsloth` import Unsloth.

The harness blocks unsloth (sys.modules["unsloth"] = None) so that it cannot patch the base run;
for the Unsloth reference we need it, imported first as Unsloth asks. Same arguments as _train.py.
"""

import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "harness" / "_train.py"
BLOCK = '    sys.modules["unsloth"] = None\n    sys.modules["unsloth_zoo"] = None\n'

source = TRAIN.read_text()
assert BLOCK in source, "harness/_train.py changed: update this launcher"
code = compile(source.replace(BLOCK, "    import unsloth  # noqa: F401\n"), str(TRAIN), "exec")
sys.path.insert(0, str(TRAIN.parent))
exec(code, {"__name__": "__main__", "__file__": str(TRAIN)})
