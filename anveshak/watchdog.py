"""Restart the pipeline if it died while "running" (GPU reset / TDR, crash), without waiting for someone to press
Resume. Run it every 10 minutes from Task Scheduler (no console window with pythonw):

  schtasks /Create /SC MINUTE /MO 10 /TN AnveshakWatchdog /F /TR "\"<repo>\\.venv\\Scripts\\pythonw.exe\" \"<repo>\\anveshak\\watchdog.py\""

It restarts only when the pipeline's state is "running", its pid is dead, no training process is alive, no Stop/Pause
was asked, and fewer than 3 restarts happened in the last 6 h (a crash loop needs a human). The restarted pipeline
resumes the interrupted training step from last.pt. Every decision to restart is logged in runs/pipeline/watchdog.log.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script by Task Scheduler

from anveshak.control import alive  # noqa: E402
from anveshak.pipeline import P, RUNS  # noqa: E402
from anveshak.train import spawn  # noqa: E402

MAX_RESTARTS, WINDOW = 3, 6 * 3600


def should_restart(state, runner_alive, trainer_alive, flags, recent):
    return state == "running" and not runner_alive and not trainer_alive and not flags and recent < MAX_RESTARTS


def main():
    f = P / "state.json"
    if not f.exists():
        return
    st = json.loads(f.read_text())
    trainers = [json.loads(s.read_text()) for s in RUNS.glob("*/state.json") if s.parent != P]
    trainer_alive = any(t.get("state") == "running" and alive(t.get("pid")) for t in trainers)
    log = P / "watchdog.log"
    now = time.time()
    recent = sum(1 for l in (log.read_text().splitlines() if log.exists() else []) if now - float(l.split()[0]) < WINDOW)
    flags = (P / "STOP").exists() or (P / "PAUSE").exists()
    if should_restart(st["state"], alive(st.get("pid")), trainer_alive, flags, recent):
        pid = spawn([sys.executable, "-m", "anveshak.pipeline"], P)
        with open(log, "a") as fh:
            fh.write(f"{now:.0f} {time.strftime('%Y-%m-%d %H:%M:%S')} pipeline pid {st.get('pid')} dead; restarted as {pid}\n")


if __name__ == "__main__":
    main()
