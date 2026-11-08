"""Command line interface.

    python -m lcseg prepare  --lc-root /path/to/lung_colon_image_set --out outputs
    python -m lcseg group    --out outputs
    python -m lcseg generator --out outputs
    python -m lcseg train    --out outputs --variant full --seed 0 [--gpu 0]
    python -m lcseg launch   --out outputs            # all pending runs, one worker per GPU
    python -m lcseg bench    --out outputs
    python -m lcseg report   --out outputs
"""
import argparse
import os
import subprocess
import sys
import time

from .config import load_cfg


def all_runs(cfg):
    order = ["full", "unet", "att", "full_nofocal"]
    runs = []
    seeds = {v: list(cfg["RUNS"].get(v, [])) for v in order}
    # seed 0 of every configuration first, then the remaining seeds of the complete model
    for v in order:
        if 0 in seeds[v]:
            runs.append((v, 0))
    for v in order:
        runs += [(v, s) for s in seeds[v] if s != 0]
    return runs


def pending_runs(cfg, out):
    return [(v, s) for v, s in all_runs(cfg)
            if not os.path.exists(os.path.join(out, "runs", f"{v}_s{s}", "eval.npz"))]


def launch(cfg_path, out, n_gpu=None, hours=None):
    """Run all pending (variant, seed) runs, distributing them over the available GPUs."""
    cfg = load_cfg(cfg_path)
    todo = pending_runs(cfg, out)
    if not todo:
        print("[launch] nothing pending")
        return
    if n_gpu is None:
        try:
            n_gpu = len([l for l in subprocess.check_output(["nvidia-smi", "-L"]).decode().splitlines() if l.strip()])
        except Exception:
            n_gpu = 1
    n_gpu = max(1, n_gpu)
    per = [todo[i::n_gpu] for i in range(n_gpu)]
    procs = []
    for g, runs in enumerate(per):
        if not runs:
            continue
        spec = ",".join(f"{v}:{s}" for v, s in runs)
        log = open(os.path.join(out, f"worker_gpu{g}.log"), "a")
        cmd = [sys.executable, "-m", "lcseg", "worker", "--out", out, "--runs", spec, "--gpu", str(g)]
        if cfg_path:
            cmd += ["--config", cfg_path]
        if hours:
            cmd += ["--deadline", str(time.time() + hours * 3600)]
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        print("[launch]", " ".join(cmd), flush=True)
    for p in procs:
        p.wait()
    print("[launch] finished; still pending:", pending_runs(cfg, out))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lcseg")
    ap.add_argument("stage", choices=["prepare", "group", "generator", "train", "worker", "launch",
                                      "evaluate", "bench", "report"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--config", default=None, help="JSON file overriding lcseg/config.py")
    ap.add_argument("--lc-root", default=None)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--runs", default="")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--hours", type=float, default=None, help="launch: stop starting new runs after this")
    ap.add_argument("--est-hours", type=float, default=3.0, help="worker: expected duration of one run")
    a = ap.parse_args(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    cfg = load_cfg(a.config)
    os.makedirs(a.out, exist_ok=True)

    if a.stage == "prepare":
        from .data import stage_prepare
        stage_prepare(cfg, a.lc_root, a.out)
    elif a.stage == "group":
        from .grouping import stage_group
        stage_group(cfg, a.out)
    elif a.stage == "generator":
        from .evaluate import stage_generator
        stage_generator(cfg, a.out)
    elif a.stage == "train":
        from .train import run_one
        run_one(cfg, a.out, a.variant, a.seed)
    elif a.stage == "worker":
        from .train import run_one
        for spec in [s for s in a.runs.split(",") if s]:
            v, s = spec.split(":")
            if a.deadline and time.time() + a.est_hours * 3600 > a.deadline:
                print(f"[worker] not enough time left for {spec}; stopping", flush=True)
                break
            run_one(cfg, a.out, v, int(s))
    elif a.stage == "launch":
        launch(a.config, a.out, hours=a.hours)
    elif a.stage == "evaluate":
        from .evaluate import evaluate_run
        evaluate_run(cfg, a.out, a.variant, a.seed)
    elif a.stage == "bench":
        from .evaluate import stage_bench
        stage_bench(cfg, a.out, variants=("full", "unet", "att"))
    elif a.stage == "report":
        from .report import stage_report
        stage_report(cfg, a.out)


if __name__ == "__main__":
    main()
