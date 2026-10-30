"""Reference: real Unsloth (Soup's `backend: unsloth`) against base Soup and our solution, measured
the way harness/run_task.py measures (harness/_train.py, wall time and nvidia-smi peak from
outside, eval_adapter.py on held-out rows).

    ./run_local.sh python results/unsloth_reference.py --unsloth-python ../.venv-unsloth/bin/python

Unsloth lives in its own venv: it is banned from solution/ and would break check 2 in .venv.
This is a comparison only, not a submission: Unsloth changes the task (LoRA on all 7
projections with target_modules: auto, its own kernels), so its adapter is not the same LoRA.
Writes results/<machine>/unsloth_reference.csv and prints a table.
"""

import argparse
import csv
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
from common import OFFLINE_ENV, ENV, read_json, run_logged  # noqa: E402
from run_task import load_task  # noqa: E402

COLUMNS = ["batch_id", "run", "python", "backend", "lora_targets", "gradient_checkpointing",
           "exit_code", "wall_s", "train_loop_s", "median_step_s", "gpu_peak_mib",
           "torch_alloc_mib", "torch_reserved_mib", "first_loss", "mean_loss", "eval_loss",
           "speedup_vs_base", "memory_ratio_vs_base", "eval_delta_pct_vs_base", "note"]


def config_for(run_dir, overrides):
    import yaml
    config = load_task()
    config["data"]["train"] = str(REPO / "data" / "train.jsonl")
    config["output"] = str(run_dir / "adapter")
    for section, values in overrides.items():
        if isinstance(values, dict):
            config.setdefault(section, {}).update(values)
        else:
            config[section] = values
    path = run_dir / "task.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def train(name, python, overrides, out, solution=False, env=OFFLINE_ENV, timeout=1800):
    run_dir = out / name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    # Unsloth runs need _train.py without its unsloth block: see results/_train_unsloth.py
    script = REPO / ("results/_train_unsloth.py" if overrides.get("backend") == "unsloth"
                     else "harness/_train.py")
    cmd = [python, str(script), "--config", str(config_for(run_dir, overrides)),
           "--metrics", str(run_dir / "metrics.json")] + (["--solution"] if solution else [])
    env = dict(env, SOUP_DB_PATH=str(run_dir / "experiments.db"),
               TRITON_CACHE_DIR=str(run_dir / "triton_cache"),
               TORCHINDUCTOR_CACHE_DIR=str(run_dir / "inductor_cache"))
    print(f"[ref] {name}: training ...", flush=True)
    code, wall, peak, _start = run_logged(cmd, run_dir / "train.log", timeout, env)
    print(f"[ref] {name}: exit {code}, {wall} s, peak {peak} MiB", flush=True)
    return run_dir, code, wall, peak


def losses(adapter_dir):
    states = sorted(Path(adapter_dir).glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[-1]))
    logged = [e["loss"] for e in (read_json(states[-1], {}) if states else {}).get("log_history", [])
              if "loss" in e]
    return (logged[0], statistics.mean(logged)) if logged else (None, None)


def evaluate(run_dir, out):
    out_file = run_dir / "eval.json"
    code, *_ = run_logged([sys.executable, str(REPO / "harness" / "eval_adapter.py"), "--adapter",
                           str(run_dir / "adapter"), "--eval-file",
                           str(REPO / "data" / "eval_public.jsonl"), "--out", str(out_file)],
                          run_dir / "eval.log", 1800, OFFLINE_ENV)
    return read_json(out_file, {}).get("mean_loss") if code == 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsloth-python", required=True)
    parser.add_argument("--machine", default="5090")
    args = parser.parse_args()
    batch = time.strftime("%Y%m%d-%H%M%S")
    out = REPO / "runs" / "unsloth_reference" / batch
    out.mkdir(parents=True)

    unsloth_backend = {"backend": "unsloth"}
    unsloth_gc = {"backend": "unsloth", "training": {"gradient_checkpointing": True}}
    # Untimed warm-up with network: Unsloth may fetch its own pre-quantized checkpoint and
    # compile its kernels; the timed runs below are offline, like the harness.
    print("[ref] unsloth warm-up (untimed, online) ...", flush=True)
    train("warmup_unsloth", args.unsloth_python, unsloth_backend, out, env=ENV)

    runs = [("base", sys.executable, {}, False, "transformers", "q_proj,v_proj", False),
            ("ours_unsloth_branch", sys.executable, {}, True, "transformers + solution/",
             "q_proj,v_proj", False),
            ("unsloth", args.unsloth_python, unsloth_backend, False, "unsloth", "all 7 (auto)", False),
            ("unsloth_gc", args.unsloth_python, unsloth_gc, False, "unsloth", "all 7 (auto)",
             "unsloth (offloaded)")]
    rows = []
    for name, python, overrides, solution, backend, targets, gc in runs:
        run_dir, code, wall, peak = train(name, python, overrides, out, solution)
        metrics = read_json(run_dir / "metrics.json", {})
        first, mean = losses(run_dir / "adapter")
        cfg = read_json(run_dir / "adapter" / "adapter_config.json", {}) or {}
        rows.append({"batch_id": batch, "run": name, "python": Path(python).parent.parent.name,
                     "backend": backend,
                     "lora_targets": ",".join(sorted(cfg.get("target_modules") or [])) or targets,
                     "gradient_checkpointing": gc, "exit_code": code, "wall_s": wall,
                     "train_loop_s": metrics.get("train_loop_s"),
                     "median_step_s": metrics.get("median_step_s"), "gpu_peak_mib": peak,
                     "torch_alloc_mib": metrics.get("torch_max_allocated_mib"),
                     "torch_reserved_mib": metrics.get("torch_max_reserved_mib"),
                     "first_loss": first, "mean_loss": mean,
                     "eval_loss": evaluate(run_dir, out) if code == 0 else None,
                     "note": "" if code == 0 else f"failed, see {run_dir / 'train.log'}"})
    base = rows[0]
    for row in rows:
        if row["exit_code"] == 0 and base["exit_code"] == 0:
            row["speedup_vs_base"] = round(base["wall_s"] / row["wall_s"], 4)
            row["memory_ratio_vs_base"] = round(base["gpu_peak_mib"] / row["gpu_peak_mib"], 4)
            if row["eval_loss"] and base["eval_loss"]:
                row["eval_delta_pct_vs_base"] = round(
                    100 * (row["eval_loss"] - base["eval_loss"]) / base["eval_loss"], 3)
    path = REPO / "results" / args.machine / "unsloth_reference.csv"
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['run']:22} wall {row['wall_s']} s  peak {row['gpu_peak_mib']} MiB  "
              f"eval {row['eval_loss']}  x{row.get('speedup_vs_base')} time  "
              f"x{row.get('memory_ratio_vs_base')} memory")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
