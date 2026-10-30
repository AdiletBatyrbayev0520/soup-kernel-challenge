"""Local dashboard: a button that runs the benchmark, and runs.csv with the history of every run.

    python3 dashboard/server.py          # http://127.0.0.1:8765

Each button press runs `harness/run_task.py --profile` (base, solution, one profiled solution run)
and `harness/check_rules.py` into runs/history/<id>/, then appends one CSV row per run.
"""

import csv
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CSV_PATH = HERE / "runs.csv"
IMPORTED = HERE / "imported"  # CSVs uploaded from other machines, one file each
MAX_IMPORT_BYTES = 20 * 2**20
HISTORY = REPO / "runs" / "history"
LAUNCH = [str(REPO / "run_local.sh")] if (REPO / "run_local.sh").exists() else []

# First match wins; order matters (a "dequant" kernel is also element-wise).
CATEGORIES = [
    ("own_triton", None),  # kernels defined in solution/, matched by name below
    ("example_triton", None),  # solution/ kernels named example_*: they do not count for the rules
    ("nf4_dequant", r"dequant|nf4|kgemm_4bit|bnb"),
    ("attention", r"flash|fmha|attention|attn|sdpa|cudnn"),
    ("softmax_ce", r"softmax|nll|cross_entropy|logsumexp"),
    ("matmul", r"gemm|cutlass|cublas|xmma|matmul|wgmma|nvjet|s16816|s1688|splitk|bmm"),
    ("optimizer", r"adam|multi_tensor|foreach"),
    ("reduction", r"reduce|norm|sum|mean|max|argmax"),
    # not a bare "cast": element-wise kernels are named gpu_kernel_impl_nocast<...>
    ("cast_copy", r"converting_copy|direct_copy|copy_kernel|memcpy|memset|fill_kernel|cat_|index|gather|scatter"),
    ("elementwise", r"elementwise|vectorized|unrolled|pointwise|triton_"),
    ("other", r""),
]
BACKWARD = re.compile(r"backward|bwd|grad", re.I)
COLUMNS = (["batch_id", "started_at", "run", "side", "gpu", "exit_code", "wall_s", "train_loop_s",
            "median_step_s", "gpu_peak_mib", "torch_alloc_mib", "torch_reserved_mib", "eval_loss",
            "speedup", "memory_ratio", "quality_delta_pct", "checks_passed", "checks_total",
            "profile_steps", "gpu_ms_per_step"]
           + [f"{name}_ms_per_step" for name, _ in CATEGORIES] + ["backward_named_ms_per_step", "note"])

state = {"running": False, "batch": None, "log": None}
lock = threading.Lock()


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def categorize(kernels, own_names):
    """GPU ms per category from the profiler table (kernel name -> total self time)."""
    out = {name: 0.0 for name, _ in CATEGORIES}
    backward = 0.0
    for k in kernels:
        name, ms = k["name"], k["time_us"] / 1000
        if BACKWARD.search(name):
            backward += ms
        if name in own_names:
            out["example_triton" if name.startswith("example_") else "own_triton"] += ms
            continue
        for cat, pattern in CATEGORIES[2:]:
            if re.search(pattern, name, re.I):
                out[cat] += ms
                break
    return out, backward


def ingest(batch_id, out_dir, note=""):
    """Append one row per run of a finished (or failed) batch to runs.csv."""
    result = load(out_dir / "result.json") or {}
    checks = (load(out_dir / "checks.json") or {}).get("checks", [])
    summary = result.get("summary", {})
    ratios = summary.get("ratios", {})
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((out_dir / "run.log").stat().st_mtime
                                                                   if (out_dir / "run.log").exists()
                                                                   else time.time()))
    common = {"batch_id": batch_id, "started_at": started,
              "gpu": "; ".join(result.get("versions", {}).get("gpus", [])),
              "speedup": ratios.get("speedup"), "memory_ratio": ratios.get("memory_ratio"),
              "quality_delta_pct": ratios.get("quality_delta_pct"),
              "checks_passed": sum(c["ok"] for c in checks) if checks else None,
              "checks_total": len(checks) or None, "note": note}
    rows = []
    for side, runs in result.get("runs", {}).items():
        for run in runs:
            m = run.get("metrics") or {}
            rows.append({**common, "run": run.get("run"), "side": side,
                         "exit_code": run.get("exit_code"), "wall_s": run.get("wall_s"),
                         "train_loop_s": m.get("train_loop_s"), "median_step_s": m.get("median_step_s"),
                         "gpu_peak_mib": run.get("gpu_peak_mib"),
                         "torch_alloc_mib": m.get("torch_max_allocated_mib"),
                         "torch_reserved_mib": m.get("torch_max_reserved_mib"),
                         "eval_loss": summary.get(side, {}).get("eval_loss")})
    profile = load(out_dir / "solution_profile" / "profile.json")
    if profile:
        steps = profile["active_steps"]
        pm = load(out_dir / "solution_profile" / "metrics.json") or {}
        own = {n for n, f in pm.get("triton_files", {}).items() if f.startswith("solution/")}
        cats, backward = categorize(profile["kernels"], own)
        rows.append({**common, "run": "solution_profile", "side": "profile", "exit_code": pm.get("exit_code"),
                     "profile_steps": steps,
                     "gpu_ms_per_step": round(profile["total_gpu_us"] / 1000 / steps, 3),
                     **{f"{c}_ms_per_step": round(v / steps, 3) for c, v in cats.items()},
                     "backward_named_ms_per_step": round(backward / steps, 3)})
    if not rows:  # a failed batch still leaves a row with the reason
        rows.append({**common, "run": "-", "side": "-", "note": note or "no result.json"})
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def run_batch(batch_id):
    out = HISTORY / batch_id
    out.mkdir(parents=True)
    log_path = out / "run.log"
    note = ""
    try:
        with log_path.open("w") as log:
            for cmd in ([*LAUNCH, "python", "harness/run_task.py", "--profile", "--out", str(out)],
                        [*LAUNCH, "python", "harness/check_rules.py", "--result", str(out / "result.json")]):
                log.write(f"$ {' '.join(cmd)}\n")
                log.flush()
                code = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
                if code and "run_task" in cmd[len(LAUNCH) + 1]:
                    note = f"run_task.py exit {code}"
                    break
    except Exception as exc:  # noqa: BLE001 - still record the batch
        note = f"{exc.__class__.__name__}: {exc}"
    finally:
        ingest(batch_id, out, note)
        state["running"] = False


def safe_name(name):
    """A plain file name inside imported/: no paths, always .csv."""
    name = re.sub(r"[^\w.-]", "_", Path(name).name).lstrip(".") or "imported"
    return name if name.endswith(".csv") else name + ".csv"


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlsplit(self.path).path == "/":
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/runs.csv":
            self.send(200, CSV_PATH.read_bytes() if CSV_PATH.exists() else ",".join(COLUMNS), "text/csv")
        elif self.path.startswith("/imported/"):
            path = IMPORTED / safe_name(unquote(self.path[len("/imported/"):]))
            if path.is_file():
                self.send(200, path.read_bytes(), "text/csv")
            else:
                self.send(404, "not found", "text/plain")
        elif self.path == "/api/sources":
            names = sorted(p.name for p in IMPORTED.glob("*.csv")) if IMPORTED.exists() else []
            self.send(200, json.dumps(names))
        elif self.path == "/api/status":
            tail = ""
            if state["log"] and Path(state["log"]).exists():
                tail = "\n".join(Path(state["log"]).read_text(errors="replace").splitlines()[-15:])
            self.send(200, json.dumps({"running": state["running"], "batch": state["batch"], "log": tail}))
        else:
            self.send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path.startswith("/api/import?name="):
            return self.import_csv(safe_name(unquote(self.path.split("=", 1)[1])))
        if self.path != "/api/run":
            return self.send(404, "not found", "text/plain")
        with lock:
            if state["running"]:
                return self.send(409, json.dumps({"error": "a run is already in progress"}))
            batch_id = time.strftime("%Y%m%d-%H%M%S")
            state.update(running=True, batch=batch_id, log=str(HISTORY / batch_id / "run.log"))
        threading.Thread(target=run_batch, args=(batch_id,), daemon=True).start()
        self.send(202, json.dumps({"batch": batch_id}))

    def import_csv(self, name):
        """Save a runs.csv from another machine as dashboard/imported/<name> (same name overwrites)."""
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_IMPORT_BYTES:
            return self.send(400, json.dumps({"error": f"file must be 1 B .. {MAX_IMPORT_BYTES >> 20} MiB"}))
        try:
            text = self.rfile.read(size).decode("utf-8-sig")
            header = next(csv.reader([text.splitlines()[0]]))
        except (UnicodeDecodeError, IndexError, csv.Error):
            return self.send(400, json.dumps({"error": "not a UTF-8 CSV file"}))
        missing = {"batch_id", "run", "side"} - set(header)
        if missing:
            return self.send(400, json.dumps({"error": f"not a dashboard runs.csv: no column {', '.join(sorted(missing))}"}))
        IMPORTED.mkdir(exist_ok=True)
        (IMPORTED / name).write_text(text)
        self.send(200, json.dumps({"name": name, "rows": max(0, len(text.strip().splitlines()) - 1)}))

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    # The first start imports the run that already sits in runs/ (from a plain run_task.py).
    if not CSV_PATH.exists() and (REPO / "runs" / "result.json").exists():
        ingest("initial", REPO / "runs", "imported from runs/")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"dashboard on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
