"""One benchmark batch (base, solution, profiled solution, checks) -> rows in
results/<machine>/results.csv.

    python results/bench.py --machine 5090 <label>     # label: the branch/feature being measured

--machine (t4, 3090, 4090, 5090) is passed to solution/machines.py as SOUP_MACHINE. Reuses
dashboard/server.py's run_batch and ingest; only the CSV path, history dir and note differ.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "dashboard"))
import server  # noqa: E402
from solution.machines import PROFILES  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--machine", required=True, choices=PROFILES)
parser.add_argument("label")
args = parser.parse_args()

os.environ["SOUP_MACHINE"] = args.machine  # the harness copies os.environ into every run
(HERE / args.machine).mkdir(exist_ok=True)
server.CSV_PATH = HERE / args.machine / "results.csv"
_ingest = server.ingest
server.ingest = lambda batch, out, note="": _ingest(batch, out, f"{args.label} {note}".strip())
server.run_batch(f"{time.strftime('%Y%m%d-%H%M%S')}-{args.machine}-{re.sub(r'[^\w.-]', '_', args.label)}")
