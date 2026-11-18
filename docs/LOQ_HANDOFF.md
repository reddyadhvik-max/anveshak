# Handoff to the Claude session on the LOQ

You are the Claude Code session on the **Lenovo LOQ 15AHP10 (RTX 5050 Laptop, 8 GB)**. Another Claude session on the
user's Mac wrote this repo and this note. The two of you coordinate **through this git repo**: the Mac session
writes code and plans, you run things on the GPU and commit the results back. Read `PLAN.md` first (§16 is
the status log). This file is your task list for **Phase 1 setup**.

The user wants: training they can pause and resume with a desktop button, and accuracy over speed. They are a
student with no credit card. Never suggest paid or card-gated services.

## Hard rules (competition, from the brief)
- **No pretrained weights, ever.** Build models only via `python -m anveshak.train` (it imports `anveshak/patch.py`,
  which disables Ultralytics' AMP check that downloads `yolo11n.pt`). Never `YOLO("yolo11s.pt")` or similar. Keep `YOLO_OFFLINE=1` set while training.
- The repo, weights, predictions and the Drive link stay **private** (rules forbid sharing between teams).
- Test images are used for inference only. Never label or inspect them for labels.
- Commit as `S_Adhvik_Reddy <228771342+reddyadhvik-max@users.noreply.github.com>` on branch `phase/1-baseline`.
  End commit messages with `Co-Authored-By: Claude <noreply@anthropic.com>`. Never commit data or `.pt` files (they're gitignored).

## Tasks

### 1. WSL2 + repo
- If WSL2 isn't installed: in an **admin PowerShell** run `wsl --install -d Ubuntu-24.04`, reboot, and create the Linux user (the user has to do the reboot and password themselves).
- Clone **inside the Linux filesystem**, not `/mnt/c`: `cd ~ && git clone https://github.com/reddyadhvik-max/anveshak.git && cd anveshak && git switch phase/1-baseline`
  - The repo is private, so git needs auth. Easiest if Git for Windows is installed: `git config --global credential.helper "/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"`. Otherwise `gh auth login` (device code; the user approves in the browser).
- If Ubuntu isn't the **default** WSL distro, make it the default (`wsl --set-default Ubuntu-24.04`). The desktop shortcut launches the default distro.

### 2. One-shot setup
```bash
bash scripts/loq/setup_wsl.sh
```
This does: GPU check → uv + Python 3.11 venv → **torch cu128** (required for the RTX 50 series, sm_120) → tests →
downloads `round1.zip` (20 GB, needs ~45 GB free) with gdown and extracts it to `~/data/round1` → `tools/convert.py` →
the Windows desktop shortcut **"Anveshak Training"**, a Startup entry that starts the panel at login, power settings
(no sleep or hibernate on AC, lid close on AC does nothing), and `~/.wslconfig` (RAM for WSL).
- If gdown hits a Drive quota: ask the user to download `round1.zip` in the browser from
  https://drive.google.com/file/d/1A24PTWWji509iDwf50EkJuTcwyb13RDH/view, then `cp /mnt/c/Users/<them>/Downloads/round1.zip ~/data/` and re-run.
- After `convert.py`, `git status splits/` must show **no changes** (the val split was frozen on the Mac from the same labels). If it changed, stop and report.
- `.wslconfig` only takes effect after `wsl --shutdown`, which kills this session too. Tell the user to run it from PowerShell, then reopen.

### 3. Verify the button works (with the user watching)
1. Double-click **Anveshak Training** on the desktop. The browser should open http://localhost:8765.
2. Start a tiny run:
   `mkdir -p runs/panel_test && YOLO_OFFLINE=1 nohup .venv/bin/python -m anveshak.train --name panel_test --data ~/data/round1/yolo/data_A.yaml --scale n epochs=3 fraction=0.02 imgsz=640 batch=8 workers=4 >> runs/panel_test/train.log 2>&1 &`
3. In the panel: **Pause after this epoch** → it should show *paused* after epoch 1 → **Resume** → it continues at epoch 2 → **Stop now** → *stopped* → **Resume** → it finishes.
4. Close every terminal window and check that the run keeps going (the hidden panel process keeps WSL alive). Then `rm -rf runs/panel_test`.

### 4. Scale benchmark (decides s vs m, PLAN §5.1)
Run each, one after the other (≈10% of Set A, one epoch, full 1024 resolution):
```bash
for cfg in "s 8" "m 4"; do set -- $cfg; mkdir -p runs/bench_$1
  YOLO_OFFLINE=1 .venv/bin/python -m anveshak.train --name bench_$1 --data ~/data/round1/yolo/data_A.yaml --scale $1 \
    epochs=1 fraction=0.1 imgsz=1024 batch=$2 nbs=64 workers=6 optimizer=SGD plots=False >> runs/bench_$1/train.log 2>&1
done
```
Record for each: train seconds for the 10% (×10 gives the full epoch), peak `GPU_mem` from the progress bar,
any OOM or "TaskAlignedAssigner ... CPU" fallback warnings, val time, and the GPU temperature under load (`nvidia-smi`).
If m OOMs at batch 4, try batch 2.
Then write `docs/phases/PHASE_1_bench.md` with a small table and the rule from PLAN §5.1: **the largest scale that
fits ≥ 150 epochs in ~125 GPU-hours**. Commit it and push.

### 5. Report back
- Commit + push the bench file and any fixes you had to make to the setup scripts (say why in the commit message).
- Add a line to the PLAN.md §16 status log (date, what was verified, the bench numbers).
- **Do not start the main Stage A training yet.** The Mac session is still building `infer.py` and the insurance baseline (Phase 1). Starting is a decision for the user.
- If the user has Remote Control on, the Mac session (`saxmachine-a3`) may message you. Otherwise git is the channel.

## Daily operation (for the user, after setup)
- **Desktop "Anveshak Training"** → panel: *Pause after this epoch* / *Cancel pause* / *Stop now* (loses the current epoch) / *Resume*.
- A paused or stopped run resumes from `runs/<name>/weights/last.pt` with the optimizer state, the schedule and the lineage intact. Every resume is logged in `provenance.json`.
- The panel is bound to localhost on purpose. Don't expose it on hostel or college Wi-Fi.
