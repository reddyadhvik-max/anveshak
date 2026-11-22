"""Back-to-back runner: works through configs/pipeline_r1.yaml one step at a time, with the panel's buttons.

  python -m anveshak.pipeline --detach          # start or resume; state in runs/pipeline/, log in runs/pipeline/train.log

Steps are commands. A step with `train: <name>` is a training stage in runs/<name>; the pipeline forwards
Pause / Stop now to it, and Resume continues it with --resume last.pt. Finished steps are recorded in
runs/pipeline/done.json and never rerun. The step list is re-read before every step, so new steps can be appended
(git pull) while it runs. Placeholders: {D} = data root ($ANVESHAK_DATA or ~/data/round1), {read:path} = file contents.
A failing step stops the pipeline as "failed"; fix the cause and press Resume to rerun that step.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from anveshak.train import REPO, set_state, spawn

RUNS = REPO / "runs"
P = RUNS / "pipeline"
STEPS = Path(os.environ.get("ANVESHAK_STEPS", REPO / "configs" / "pipeline_r1.yaml"))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def load_steps(path=None):
    from ultralytics.utils import YAML
    return YAML.load(path or STEPS)["steps"]


def expand(cmd):
    d = os.environ.get("ANVESHAK_DATA", str(Path("~/data/round1").expanduser()))
    out = []
    for c in cmd:
        c = str(c).replace("{D}", d)
        c = re.sub(r"\{read:([^}]+)\}", lambda m: (REPO / m.group(1)).read_text().strip(), c)
        out.append(sys.executable if c == "python" else c)
    return out


def done_list():
    f = P / "done.json"
    return json.loads(f.read_text()) if f.exists() else []


def kill_tree(proc):
    import psutil
    try:
        for c in psutil.Process(proc.pid).children(recursive=True):
            c.kill()
    except psutil.Error:
        pass
    proc.kill()


def run_task(step):
    with open(P / "train.log", "a") as log:
        log.write(f"\n=== {step['name']}: {' '.join(expand(step['cmd']))}\n")
        log.flush()
        proc = subprocess.Popen(expand(step["cmd"]), cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        while proc.poll() is None:
            if (P / "STOP").exists():  # tasks are short and rerunnable: kill and redo on Resume
                (P / "STOP").unlink()
                kill_tree(proc)
                return "stopped"
            time.sleep(2)
    return proc.returncode


def run_train(step):
    run = RUNS / step["train"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "PIPELINE").touch()  # the panel's Resume on this run resumes the pipeline instead
    st = json.loads((run / "state.json").read_text())["state"] if (run / "state.json").exists() else None
    if st == "finished":
        return 0
    last = run / "weights" / "last.pt"
    cmd = expand(step["cmd"]) if st is None or not last.exists() else [sys.executable, "-m", "anveshak.train", "--resume", str(last)]
    forwarded = False
    with open(run / "train.log", "a") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env={**os.environ, "YOLO_OFFLINE": "1"}, creationflags=NO_WINDOW)
        while proc.poll() is None:
            if (P / "STOP").exists():
                (P / "STOP").unlink()
                (run / "STOP").touch()
            if (P / "PAUSE").exists() and not forwarded:
                (run / "PAUSE").touch()
                forwarded = True
            elif forwarded and not (P / "PAUSE").exists():  # Cancel pause
                (run / "PAUSE").unlink(missing_ok=True)
                forwarded = False
            time.sleep(2)
    (run / "PAUSE").unlink(missing_ok=True)  # a pause asked on the final epoch is honoured before the next step
    st = json.loads((run / "state.json").read_text())["state"]
    if st in ("paused", "stopped"):
        return st
    return 0 if st == "finished" and proc.returncode == 0 else proc.returncode or 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detach", action="store_true")
    a = ap.parse_args()
    P.mkdir(parents=True, exist_ok=True)
    if a.detach:
        pid = spawn([sys.executable, "-m", "anveshak.pipeline"], P)
        print(f"pipeline started, pid {pid}; panel: http://localhost:{os.environ.get('ANVESHAK_PORT', 8790)}")
        return
    (P / "STOP").unlink(missing_ok=True)
    set_state(P, "running")
    while True:
        done = done_list()
        todo = [s for s in load_steps() if s["name"] not in done]
        if not todo:
            set_state(P, "finished")
            return
        if (P / "PAUSE").exists():
            (P / "PAUSE").unlink()
            set_state(P, "paused")
            return
        step = todo[0]
        (P / "current.json").write_text(json.dumps({"step": step["name"], "since": time.strftime("%Y-%m-%d %H:%M:%S")}))
        rc = run_train(step) if "train" in step else run_task(step)
        if rc in ("paused", "stopped"):
            (P / "PAUSE").unlink(missing_ok=True)
            set_state(P, rc)
            return
        if rc != 0:
            with open(P / "train.log", "a") as log:
                log.write(f"\n!!! step {step['name']} failed with exit code {rc}; fix it and press Resume\n")
            set_state(P, "failed")
            return
        (P / "done.json").write_text(json.dumps(done + [step["name"]], indent=1))
        (P / "current.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
