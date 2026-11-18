# Handoff to the Claude session on the LOQ

You are the Claude Code session on the **Lenovo LOQ 15AHP10 (Windows 11, native, RTX 5050 Laptop 8 GB)**.
The Claude session on the user's Mac (`saxmachine-a3`) wrote this repo. You coordinate two ways:
- **Remote Control messages** for quick back-and-forth (reply with SendMessage to `saxmachine-a3`);
- **git** for anything that should last: code, results, reports.

Read `PLAN.md` first (§16 is the status log). This file is your **Phase 1 setup** task list.

The user wants training they can pause and resume with a desktop button, and accuracy over speed. They are a
student with no credit card. Never suggest paid or card-gated services.

**Decision: native Windows, no WSL.** PyTorch cu128 and Ultralytics run natively, and this avoids an admin install, a reboot, a VM RAM cap and keeping WSL alive.

## Hard rules (competition)
- **No pretrained weights, ever.** Train only via `python -m anveshak.train` (it applies `anveshak/patch.py`, which disables Ultralytics' AMP check that downloads `yolo11n.pt`). Never `YOLO("yolo11s.pt")` or similar.
- The repo, weights, predictions and the Drive link stay **private**.
- Test images are used for inference only.
- Commit on branch `phase/1-baseline` as `S_Adhvik_Reddy <228771342+reddyadhvik-max@users.noreply.github.com>`. Never commit data or `.pt` files (gitignored).

## Tasks

### 1. Clone (Git Bash or PowerShell)
```bash
cd ~ && git clone https://github.com/reddyadhvik-max/anveshak.git && cd anveshak && git switch phase/1-baseline
```
The repo is private. Git for Windows' Credential Manager will ask the user to sign in to GitHub in the browser once.
Keep the repo on a local NTFS drive (not OneDrive-synced folders: syncing 40 GB of data and checkpoints would be a disaster).
If `C:` has less than ~60 GB free and another drive exists, put the data there with `$env:ANVESHAK_DATA="D:\anveshak-data"`.

### 2. One-shot setup (PowerShell, no admin)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\loq\setup_windows.ps1
```
This does: GPU check → uv + Python 3.11 `.venv` → **torch cu128** (required for the RTX 50 series, sm_120) →
downloads `round1.zip` (20 GB) with gdown and extracts it to `~\data\round1` → tests → `tools\convert.py` →
the **"Anveshak Training"** desktop shortcut + a Startup entry (panel server at login) → power settings
(on AC: no sleep or hibernate, lid close does nothing).
- If gdown hits a Drive quota: ask the user to download `round1.zip` in the browser
  (https://drive.google.com/file/d/1A24PTWWji509iDwf50EkJuTcwyb13RDH/view) into `~\data\`, then re-run.
- After convert, `git status splits` must show **no changes** (the val split was frozen on the Mac). If it changed, stop and report.
- **Optional, needs admin:** add a Windows Defender exclusion for the data folder and the repo. Real-time scanning of 10k TIFFs every epoch slows the dataloader. Ask the user first.
- Scripts were written on a Mac and never executed on Windows. If something breaks, fix it, commit with the reason, and tell the Mac session.

### 3. Verify the buttons (with the user watching)
1. Double-click **Anveshak Training** on the desktop. The browser opens http://localhost:8765.
2. Start a tiny run in the background:
   `.venv\Scripts\python -m anveshak.train --detach --name panel_test --data ~/data/round1/yolo/data_A.yaml --scale n epochs=3 fraction=0.02 imgsz=640 batch=8 workers=4`
3. In the panel: **Pause after this epoch** → *paused* after epoch 1 → **Resume** → continues at epoch 2 → **Stop now** (stops within ~1 batch) → *stopped* → **Resume** → finishes.
4. Close the terminal / end the Claude shell that launched it, and check that training keeps going (`spawn()` breaks away from the shell's job object). If closing the shell kills training, report it. That's a bug to fix before any long run.
5. Check that no console windows pop up from dataloader workers. Then delete `runs\panel_test`.

### 4. Scale benchmark (decides s vs m, PLAN §5.1)
Run one after the other in the foreground (≈10% of Set A, one epoch, full 1024 resolution):
```bash
.venv/Scripts/python -m anveshak.train --name bench_s --data ~/data/round1/yolo/data_A.yaml --scale s epochs=1 fraction=0.1 imgsz=1024 batch=8 nbs=64 workers=6 optimizer=SGD plots=False
.venv/Scripts/python -m anveshak.train --name bench_m --data ~/data/round1/yolo/data_A.yaml --scale m epochs=1 fraction=0.1 imgsz=1024 batch=4 nbs=64 workers=6 optimizer=SGD plots=False
```
Record for each: train time for the 10% (×10 gives the full epoch), peak `GPU_mem`, any OOM or "TaskAlignedAssigner … CPU" fallback warnings,
val time, and GPU temperature and clocks under load (`nvidia-smi -l 5`). If m OOMs at batch 4, try batch 2. Also try `workers=8` vs `4`,
because Windows dataloaders differ from Linux. Write `docs/phases/PHASE_1_bench.md` (table + recommendation using
"the largest scale that fits ≥ 150 epochs in ~125 GPU-hours"), commit, push, and message the Mac session the numbers.

### 5. Then stop
- Add a dated line to the PLAN.md §16 status log, commit, push.
- **Do not start the main Stage A training.** The Mac session is building `infer.py` and the insurance baseline. Starting is the user's decision.

## Daily operation (for the user)
- **Desktop "Anveshak Training"** → panel: *Pause after this epoch* / *Cancel pause* / *Stop now* (loses only the current epoch) / *Resume*.
- A paused or stopped run resumes from `runs\<name>\weights\last.pt` with the optimizer state, the schedule and the lineage intact. Resumes are logged in `provenance.json`.
- Start every stage with `--detach`, so it isn't tied to a terminal.
- The panel listens on localhost only. Don't expose it on hostel or college Wi-Fi.
