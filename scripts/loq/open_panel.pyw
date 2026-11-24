"""'Anveshak Training' desktop shortcut target, run with pythonw so no console window ever appears.

Starts the panel server if it isn't already running, then opens it in the browser (unless --no-browser).
Port 8790 by default (8765 is taken by PhoneBoxAgent on the LOQ); override with ANVESHAK_PORT.
"""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PORT = int(os.environ.get("ANVESHAK_PORT", 8790))
URL = f"http://localhost:{PORT}"


def panel_up():  # check it is really our panel, not another app answering on the port
    try:
        return b"Anveshak training" in urllib.request.urlopen(URL, timeout=2).read()
    except OSError:
        return False


if not panel_up():
    subprocess.Popen([str(REPO / ".venv" / "Scripts" / "pythonw.exe"), "-m", "anveshak.control", "--port", str(PORT)],
                     cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0))
    for _ in range(30):
        if panel_up():
            break
        time.sleep(1)
if "--no-browser" not in sys.argv:
    webbrowser.open(URL)
