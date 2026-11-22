"""Training control panel: Pause (after the current epoch's checkpoint) and Resume buttons.

  python -m anveshak.control            # then open http://localhost:8790 (port: --port or $ANVESHAK_PORT)
  python -m anveshak.control --host 0.0.0.0   # only on a trusted network such as Tailscale: anyone who can reach it can pause or resume

Pause doesn't kill anything. It drops a PAUSE file, and train.py exits cleanly right after the next
last.pt save. Resume relaunches `anveshak.train --resume last.pt` in the background, logging to train.log.
"""
import argparse
import csv
import html
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from anveshak.train import spawn

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"


def alive(pid):
    import psutil  # Ultralytics dependency; os.kill(pid, 0) would send Ctrl+C on Windows
    try:
        return bool(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def run_info(run):
    st = json.loads((run / "state.json").read_text())
    state = st["state"]
    if state == "running" and not alive(st.get("pid")):
        state = "stopped"  # process died without updating state (crash, reboot, power loss)
    if state == "running" and (run / "STOP").exists():
        state = "stopping"
    elif state == "running" and (run / "PAUSE").exists():
        state = "pausing"
    rows = list(csv.DictReader(open(run / "results.csv"))) if (run / "results.csv").exists() else []
    last = {k.strip(): v.strip() for k, v in rows[-1].items()} if rows else {}
    epochs = "?"
    if (run / "args.yaml").exists():
        epochs = next((l.split(":", 1)[1].strip() for l in open(run / "args.yaml") if l.startswith("epochs:")), "?")
    return {"name": run.name, "state": state, "time": st["time"], "epoch": f"{len(rows)}/{epochs}",
            "mAP50": last.get("metrics/mAP50(B)", "-"), "mAP50-95": last.get("metrics/mAP50-95(B)", "-"),
            "resumable": (run / "weights" / "last.pt").exists(), "rows": rows,
            "total": int(epochs) if str(epochs).isdigit() else len(rows)}


def fmt_dur(sec):
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


def epoch_log(i):
    """Done / in progress / to do, with per-epoch metrics and an ETA."""
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in i["rows"]]
    total, done = max(i["total"], len(rows)), len(rows)
    active = i["state"] in ("running", "pausing", "stopping") and done < total
    # Ultralytics' "time" column is cumulative seconds, but it restarts from 0 after a resume.
    durs, prev = [], 0.0
    for r in rows:
        try:
            t = float(r.get("time", "") or 0)
        except ValueError:
            t = 0.0
        durs.append(t - prev if t > prev else t)
        prev = t
    known = [d for d in durs if d > 0]
    todo = total - done - (1 if active else 0)
    eta = f" &middot; about <b>{fmt_dur(sum(known) / len(known) * (total - done))}</b> left" if known and done < total and i["state"] != "finished" else ""
    pct = 100 * done / total if total else 0
    summary = (f"Epoch log: <b>{done}</b> done &middot; <b>{1 if active else 0}</b> in progress &middot; "
               f"<b>{max(todo, 0)}</b> to go{eta}")
    bar = (f'<div class="bar"><div style="width:{pct:.1f}%"></div></div>')
    trs = []
    if active:
        trs.append(f'<tr class="now"><td>{done + 1}</td><td colspan="5">in progress</td></tr>')
    for n in range(done, 0, -1):
        r, d = rows[n - 1], durs[n - 1]
        trs.append(f"<tr><td>{n}</td><td>done</td><td>{r.get('metrics/mAP50(B)', '-')}</td><td>{r.get('metrics/mAP50-95(B)', '-')}</td>"
                   f"<td>{r.get('train/box_loss', '-')}</td><td>{fmt_dur(d) if d > 0 else '-'}</td></tr>")
    if done < total:
        nxt = done + 2 if active else done + 1
        if nxt <= total:
            trs.insert(0, f'<tr class="todo"><td>{nxt}&ndash;{total}</td><td colspan="5">to do</td></tr>' if nxt < total
                       else f'<tr class="todo"><td>{total}</td><td colspan="5">to do</td></tr>')
    table = ("<table><tr><th>epoch</th><th>status</th><th>mAP50</th><th>mAP50-95</th><th>box loss</th><th>time</th></tr>"
             + "".join(trs) + "</table>")
    return f'{bar}<details open><summary>{summary}</summary><div style="max-height:320px;overflow-y:auto">{table}</div></details>'


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=5,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        # CREATE_NO_WINDOW: the panel runs under pythonw (no console), so without it Windows flashes
        # a console window for every nvidia-smi call, i.e. on every 30 s page refresh.
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi not available"


def tail(path, n=15):
    return "".join(open(path, errors="replace").readlines()[-n:]) if path.exists() else ""


def runs():
    return sorted((r for r in RUNS.glob("*") if (r / "state.json").exists()), key=lambda r: r.stat().st_mtime, reverse=True)


COLORS = {"failed": "#cf222e", "running": "#1a7f37", "pausing": "#9a6700", "paused": "#0969da", "finished": "#6e7781", "stopped": "#cf222e", "stopping": "#cf222e"}


def pipeline_card(r, i):
    from anveshak.pipeline import done_list, load_steps
    done = done_list()
    cur = json.loads((r / "current.json").read_text())["step"] if (r / "current.json").exists() else None
    try:
        steps = load_steps()
    except Exception as e:  # a half-edited step file must not take the panel down
        steps, cur = [], f"step file unreadable: {e}"
    rows = []
    for s in steps:
        n = s["name"]
        st = "done" if n in done else ("now: " + i["state"] if n == cur else "to do")
        cls = "" if n in done else ("now" if n == cur else "todo")
        rows.append(f'<tr class="{cls}"><td>{html.escape(n)}</td><td>{"training" if "train" in s else "task"}</td><td>{st}</td></tr>')
    stop = ('<button name="a" value="stop" style="margin-left:8px;border-color:#cf222e;color:#cf222e" '
            'onclick="return confirm(\'Stop now? A training step keeps its last epoch checkpoint; a task step reruns on Resume.\')">Stop now</button>')
    btn = {"running": '<button name="a" value="pause">Pause (after this epoch / task)</button>' + stop,
           "pausing": '<button name="a" value="unpause">Cancel pause</button>' + stop}.get(i["state"], "")
    if i["state"] in ("paused", "stopped", "failed"):
        btn = '<button name="a" value="resume">Resume pipeline</button>'
    return f"""<div class="card"><h2>pipeline (back-to-back)
<span class="pill" style="background:{COLORS[i['state']]}">{i['state']}</span></h2>
<p><b>{len([s for s in steps if s['name'] in done])}</b> of <b>{len(steps)}</b> steps done &middot; state since {i['time']}</p>
<table><tr><th>step</th><th>kind</th><th>status</th></tr>{''.join(rows)}</table>
<form method="post"><input type="hidden" name="run" value="pipeline">{btn}</form>
<details><summary>log</summary><pre>{html.escape(tail(r / 'train.log'))}</pre></details></div>"""


def page():
    cards = []
    for r in runs():
        i = run_info(r)
        if r.name == "pipeline":
            cards.insert(0, pipeline_card(r, i))
            continue
        btn = ""
        stop = ('<button name="a" value="stop" style="margin-left:8px;border-color:#cf222e;color:#cf222e" '
                'onclick="return confirm(\'Stop now? The epoch in progress is lost; you can resume from the last checkpoint.\')">Stop now</button>')
        if i["state"] == "running":
            btn = '<button name="a" value="pause">Pause after this epoch</button>' + stop
        elif i["state"] == "pausing":
            btn = '<button name="a" value="unpause">Cancel pause</button>' + stop
        elif i["state"] in ("paused", "stopped", "failed") and i["resumable"]:
            btn = '<button name="a" value="resume">Resume</button>'
        cards.append(f"""<div class="card"><h2>{html.escape(i['name'])}
<span class="pill" style="background:{COLORS[i['state']]}">{i['state']}</span></h2>
<p>epoch <b>{i['epoch']}</b> &middot; mAP50 {i['mAP50']} &middot; mAP50-95 {i['mAP50-95']} &middot; state since {i['time']}</p>
{epoch_log(i)}
<form method="post"><input type="hidden" name="run" value="{html.escape(i['name'])}">{btn}</form>
<details><summary>log</summary><pre>{html.escape(tail(r / 'train.log'))}</pre></details></div>""")
    body = "".join(cards) or "<p>No runs yet. Start a stage with <code>python -m anveshak.train ...</code></p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="refresh" content="30"><title>Anveshak training</title><style>
body{{font:15px system-ui;margin:16px;max-width:820px;background:#f6f8fa;color:#1f2328}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:12px 16px;margin:12px 0}}
h2{{font-size:18px;margin:4px 0}} .pill{{color:#fff;border-radius:12px;padding:2px 10px;font-size:13px;margin-left:8px}}
button{{font-size:16px;padding:8px 16px;border-radius:6px;border:1px solid #888;cursor:pointer}}
pre{{font-size:12px;overflow-x:auto;background:#f6f8fa;padding:8px}}
.bar{{height:8px;background:#d0d7de;border-radius:4px;overflow:hidden;margin:6px 0}} .bar div{{height:100%;background:#1a7f37}}
table{{border-collapse:collapse;font-size:13px;margin:6px 0 10px;width:100%}} th,td{{text-align:left;padding:3px 8px;border-bottom:1px solid #d0d7de}}
tr.now td{{color:#1a7f37;font-weight:600}} tr.todo td{{color:#6e7781}} details{{max-height:none}}
@media(prefers-color-scheme:dark){{body{{background:#0d1117;color:#e6edf3}}.card{{background:#161b22;border-color:#30363d}}pre{{background:#0d1117}}}}
</style></head><body><h1>Anveshak training</h1><p>GPU: {html.escape(gpu())}</p>{body}
<p style="color:#6e7781">Pause takes effect when the current epoch finishes and its checkpoint is saved. The page refreshes every 30 s.</p></body></html>"""


def act(name, action):
    run = next((r for r in runs() if r.name == name), None)  # only existing runs; no paths from the request
    if run is None:
        return
    pipe = RUNS / "pipeline"
    if action == "resume" and run != pipe and (run / "PIPELINE").exists() and (pipe / "state.json").exists():
        run = pipe  # a stage owned by the pipeline resumes through it, so the following steps still run
    state = run_info(run)["state"]
    if run == pipe:
        if action == "pause" and state == "running":
            (run / "PAUSE").touch()
        elif action == "stop" and state in ("running", "pausing"):
            (run / "STOP").touch()
        elif action == "unpause":
            (run / "PAUSE").unlink(missing_ok=True)
        elif action == "resume" and state in ("paused", "stopped", "failed"):
            for f in ("PAUSE", "STOP"):
                (run / f).unlink(missing_ok=True)
            spawn([sys.executable, "-m", "anveshak.pipeline"], run)
        return
    if action == "pause" and state == "running":
        (run / "PAUSE").touch()
    elif action == "stop" and state in ("running", "pausing"):
        (run / "STOP").touch()  # train.py checks it every batch and stops like Ctrl+C
    elif action == "unpause":
        (run / "PAUSE").unlink(missing_ok=True)
    elif action == "resume" and state in ("paused", "stopped", "failed"):
        for f in ("PAUSE", "STOP"):
            (run / f).unlink(missing_ok=True)
        spawn([sys.executable, "-m", "anveshak.train", "--resume", str(run / "weights" / "last.pt")], run)


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(page())

    def do_POST(self):
        q = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
        act(q.get("run", [""])[0], q.get("a", [""])[0])
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def _send(self, text):
        b = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("ANVESHAK_PORT", 8790)))
    a = ap.parse_args()
    print(f"Control panel on http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
