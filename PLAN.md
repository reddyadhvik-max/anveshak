# Anveshak (Techfest 2026-27, IITB-HRTI): master plan

Solo entry. Primary project (Shastra-Setu is secondary). **Accuracy over speed**: spend every GPU hour
up to the deadline on the model, but never miss a deadline.
Everything here is free and needs no credit card. Brief: `../HRT_Problem_statement__compressed.pdf`.

Last updated 2026-11-18. The status log is §16. Read it first when resuming.

---

## 0. TL;DR

- **Task:** one remote-sensing detector, trained **from random init**, that detects 40 known classes, flags everything else as `unknown`, learns 10 few-shot classes (Set B), then 10 more on site (Round 2), without forgetting, as **one model lineage** (A → B → R2).
- **Score:** `OE = 100 × HM(mAP_K, U-F1)`. Zero unknown handling means OE = 0.
- **Stack:** Ultralytics YOLO11 with a **P2 head** (tiny objects), built from yaml (random init). The head is fixed at **52 slots** from day 1: `0-39` R1 classes, `40-49` reserved for R2, `50 unknown`, `51 objectness`. Training runs on the **LOQ (RTX 5050 8 GB)**. Experiments that aren't part of the lineage run on free Kaggle/Colab.
- **Repo:** `~/Desktop/iitb anveshak/anveshak`, private GitHub `reddyadhvik-max/anveshak`. One branch per phase, merged `--no-ff`, an annotated tag per phase, and a report per phase in `docs/phases/`, so judges can see how it developed (§10).
- **Deadlines:** registration ✅ done. **R1 submission 30 Nov 23:59** (target 29 Nov). R1 results 3 Dec. **R2 on site 16 Dec**. Finale 17 Dec.

---

## 1. Rules that shape every decision

| Rule (brief) | Consequence |
|---|---|
| No pretrained weights of any kind | Always build from `configs/yolo11-p2.yaml`. `anveshak/patch.py` disables Ultralytics' AMP check (which downloads `yolo11n.pt`). The init hash goes in `runs/<stage>/provenance.json`. A smoke test verifies that no `.pt` is ever downloaded. |
| Open-vocabulary detectors not allowed | Closed-set YOLO plus our own unknown logic. |
| One lineage A → B → R2, same architecture, never restarted | 52 slots fixed from the start. Every stage after the first uses `--from <prev ckpt>`. **Variants are fine**: several B runs from the same A checkpoint. The submitted one is a single chain. |
| Checkpoint re-run must reproduce the CSV within tolerance | FP32 deterministic inference, pinned versions, thresholds in a config shipped with the weights (§8). |
| R2 starts from the verified R1 checkpoint; R2 may use all earlier-round data | Replay A+B in R2. Freeze `r1_final.pt` + sha256. |
| Training tiles contain **unlabelled** unknowns (R1) and unlabelled base/few-shot/unknown objects (R2) | Pseudo-labels for old classes and mined unknowns (§5.4, §5.5). |
| Prohibited: labelling or recovering test labels, undeclared data, reverse-engineering the scorer or ids, matching tiles to source imagery | Test images are used for inference only. No self-training on test. No attempt to identify the source dataset. **No external data** (no approved list was shipped, so stay inside the given data). |
| CSV rules: header, codes, score in [0,1], boxes inside the tile, ≤ 1000 rows per tile, no duplicates | `tools/make_csv.py` enforces all of them, then runs the official validator. It's tested against the real `validate_predictions.py`. |
| Conflict rule: known vs unknown boxes at IoU ≥ 0.5 → only the higher score counts | We resolve conflicts ourselves before writing the CSV (§7). |
| U-Rec / U-Prec are unranked counts | Every unknown row costs precision. One tuned threshold. |

Doubt-portal questions: **none needed.** The data answered the class split (below). For everything else we take the conservative reading in the table above: no external data, no test-time self-training, pre-allocated slots, replay only of released data.

---

## 2. Data facts (audited 2026-11-18 from `round1.zip`, 20 GB)

The labels and metadata were pulled remotely with HTTP range requests into `../data_meta/round1/` (113 MB). Images are **not** on the Mac.

- `class_list.csv`: ids **0–39** + `unknown` = **50**. So Round 1 = 40 classes, and ids 40–49 are evidently reserved for the R2 classes. Our slots map 1:1 to these ids.
- **Set A:** 9,659 tiles (1024², 8-bit RGB GeoTIFF), **30 base classes (ids 0–29)**, 772,752 boxes. 3,532 tiles (37%) have no labels.
  - **Extreme imbalance:** class 29 = 432,749 boxes and class 28 = 265,677 (together 90%). The rarest class (5) has 710 boxes in only 52 tiles.
  - **Tiny objects:** sqrt-area p50 = 24 px; **35% under 16 px, 61% under 32 px**. That's why we use the P2 head at native 1024.
  - **Dense tiles:** up to 2,102 boxes per tile (p90 275), so `max_det=1000`, and CSV row budgeting matters (§7).
- **Set B:** 677 tiles, **10 few-shot classes (ids 30–39)**, only 428 boxes (15–79 per class). The median tile has **0** labels.
  - **Critical:** Set B tiles certainly contain *unlabelled* base objects (cars, buildings…). Fine-tuning on them naively teaches "base objects = background", which is catastrophic forgetting. Pseudo-labelling base classes in Set B is mandatory (§5.5).
  - Set B objects are larger (p50 35 px).
- **Test (R1):** 1,637 tiles, 1024². Unknown objects there = anything outside the 40 classes (likely including the future R2 classes).
- Ground resolution looks like ~0.3 m (aerial / VHR satellite).
- Validator: `python validate_predictions.py preds.csv --tiles test/tiles.csv --classes class_list.csv`. R2 passes both class lists.

Full per-class table: `docs/phases/PHASE_0.md`.

---

## 3. Hardware and free toolchain

| Need | Tool | Notes |
|---|---|---|
| Training lineage + inference | **LOQ 15AHP10, RTX 5050 Laptop 8 GB** (Blackwell, sm_120) | **Needs PyTorch built for CUDA 12.8** (`cu128` wheels). Plugged in, Lenovo Vantage Performance mode, cooling pad, sleep and hibernate off, Windows Update paused for 2 weeks. |
| OS on LOQ | **Native Windows 11** (decided 2026-11-18; WSL wasn't installed, and native avoids admin, a reboot, the VM RAM cap and keeping WSL alive) | Data in `~\data\round1` on local NTFS, not OneDrive. Optional Defender exclusion for the data and repo folders. |
| Env | `uv` + Python 3.11 | `requirements.txt` pins `ultralytics==8.4.215`. torch is installed per machine. |
| Dev machine | Mac (8 GB) | Code, tests, docs, CPU smoke tests. No real training. |
| Mac ↔ LOQ | **Claude Remote Control** (Mac session `saxmachine-a3` ↔ LOQ session) + git | Messages for coordination, git for anything lasting. |
| Experiments outside the lineage | Kaggle (P100/T4, ~30 GPU-h/week, phone verification only), Colab free | Dev-sim proxy runs (§6). Data uploaded as a **private** dataset subset. |
| Alerts | `ntfy.sh` (free) | Phone ping on epoch milestones or a crash. |
| Code | GitHub **private** repo | Kept private until after 17 Dec (rules: no sharing). |
| Submission | Google Drive 15 GB free | Link only in the submission form. |

---

## 4. Repo layout

```
anveshak/
  PLAN.md  README.md  DEVLOG.md  requirements.txt
  configs/yolo11-p2.yaml        # nc=52, P2 head; random init only
  anveshak/patch.py             # no-AMP-download patch + objectness-slot target   [P0]
  anveshak/train.py             # stage runner + provenance.json                   [P0]
  anveshak/infer.py             # raw per-anchor scores -> known + unknown boxes   [P1/P3]
  tools/convert.py              # JSON -> YOLO txt, stratified frozen split        [P0]
  tools/eda.py                  # data audit                                       [P0]
  tools/oe_score.py             # local OE scorer from the published rules         [P0]
  tools/make_csv.py             # always-valid CSV + official validator            [P0]
  tools/rfs.py                  # repeat-factor sampling train lists               [P1]
  tools/mine_unknowns.py        # pseudo-unknowns on training tiles                [P3]
  tools/pseudo_label_old.py     # old-class pseudo-labels in new-stage tiles       [P4]
  tools/copypaste.py            # few-shot instance paste onto replay tiles        [P4]
  tools/tune_postproc.py        # grid search of postproc thresholds on OE         [P3]
  tools/repro_check.py          # rerun inference, diff vs submitted CSV           [P5]
  r2/run_r2.sh                  # one-command Round 2                              [P6]
  splits/                       # frozen val tile lists (committed)
  docs/phases/PHASE_<n>.md      # per-phase report: goal, built, results, decisions
  tests/                        # pytest; must pass before every phase merge
```

---

## 5. Method

### 5.1 Detector
- YOLO11 backbone + P2/P3/P4/P5 Detect head (`configs/yolo11-p2.yaml`), nc = 52.
- **Scale choice (accuracy first):** on day 1 of P1, benchmark one epoch-equivalent (~500 iters) of `s` and `m` at imgsz 1024 on the LOQ. Pick the **largest scale that fits ≥ 150 epochs into the Stage A budget (~125 GPU-h)** and fits in 8 GB (AMP, batch ≥ 4 with `nbs=64` accumulation).
  - Rough expectation: s-P2 ≈ 12–18 min/epoch, m-P2 ≈ 30–40 min/epoch, so **m-P2 at ~170–200 epochs is the likely pick**. The benchmark decides.
- **From-scratch recipe:** a long cosine schedule, 3 warmup epochs, EMA, SGD lr0 0.01 (Ultralytics' `auto` optimizer picks AdamW for short runs; force `optimizer=SGD` for long scratch runs), weight decay 5e-4.
  - Augmentation: `mosaic=1.0` with `close_mosaic=15`, `mixup=0.1`, `fliplr=0.5`, **`flipud=0.5`** (overhead imagery has no "up"), `degrees=0`, `scale=0.5`, HSV defaults.
  - imgsz 1024 (native, **no downscaling**: 35% of objects are < 16 px).
- **Imbalance:** repeat-factor sampling (LVIS, t ≈ 0.02 on tile frequency), written as a train list with duplicated tiles (`tools/rfs.py`). Classes 28/29 are never oversampled. Keep about half of the empty tiles as negatives.
- `cls=1.0` (the objectness slot doubles `target_scores_sum`; see patch.py), `max_det=1000`.
- Known risk: on dense tiles (2,000 boxes) the task-aligned assigner can OOM at batch 8. Ultralytics falls back to CPU for that batch. If that gets frequent, use batch 4 + `nbs=64`.

### 5.2 Fixed slots (architecture never changes)
`0-39` = R1 class ids, `40-49` = R2 (filled on 16 Dec, matching the ids the organisers reserved), `50` = unknown, `51` = objectness. Unused slots train as negatives and stay ~0. The tensor shapes are identical from random init to the R2 submission. Slot 51 is never written to a CSV.

### 5.3 Objectness slot (built and tested in P0)
For every positive anchor, target[51] = max(target[0..50]). It learns "any object" from all 773k labelled boxes. At inference: `obj = s51`, `k = max(s0..49 of known classes)`, `u = s50`. Unknown evidence ≈ `max(u, obj·(1−k))` (final form tuned in P3).

### 5.4 Unknown handling (P3)
Problem: unlabelled unknowns in training tiles are trained as background.
1. Stage A trains with the objectness slot.
2. **Mine** on Set A train tiles with the A checkpoint. Keep a box if `obj ≥ τ_hi`, `k ≤ τ_k`, IoU < 0.3 with every GT, its size is within the GT size range, it isn't > 25% of the tile, and it survives class-agnostic NMS. Write it as class `50` (unknown) in a mined label set.
3. **Stage A_mine:** continue on Set A + mined unknowns (~25% of A's epochs). Optionally a second mining pass.
4. Visual QA: 50 random overlay images of mined boxes in `docs/phases/img/`.

### 5.5 Few-shot classes and retention (P4; the same recipe in R2)
- Start from the A_mine checkpoint, `freeze=10` (backbone), lr0 ≈ 0.1×, 40–80 epochs, mosaic on.
- **Pseudo-label old classes in every Set B tile** with the A_mine model: score ≥ τ_hi becomes a label (classes 0–29 and 50). Between 0.1 and τ_hi the region is blanked (mean-fill), so it's neither positive nor background.
- **Data mix:** Set B train ×k (few-shot instances at ~30–50% of positives) + Set A replay (RFS-balanced) at about 1:1–1:2 + `copypaste.py` tiles (few-shot crops with 10% context, random flip/rot90/scale 0.8–1.25, pasted onto A tiles in free space).
- **Accuracy first:** run 2–3 variants from the same A_mine checkpoint (mix ratio, freeze depth, epochs). Pick by **retention-aware val**: OE on the dev sim plus mAP on A-val and B-val.
- Retention checklist: replay, pseudo-labels, frozen backbone, low LR, EMA. Logit distillation only if A-val mAP drops > 3 points.
- The final B stage can use **all** Set B tiles (train + val) with the chosen hyper-parameters (still the same lineage, from the same A_mine checkpoint).

---

## 6. Evaluation (no test labels, so we build our own truth)

- `tools/oe_score.py` implements the published metric exactly (conflict rule → VOC all-point mAP_K → U metrics → OE). Its assumptions are in the file header. Tested.
- **Val splits** (frozen in `splits/`): A-val ~10% stratified (each class ≥ 10% of its tiles), B-val 25%.
- **Dev-sim for unknowns (Kaggle/Colab, parallel to the LOQ run):** a proxy model trained on Set A with **6 held-out classes** (mixed size and frequency, e.g. ids 3, 9, 15, 19, 24, 5). Their labels are removed from training (so they're unlabelled objects, exactly like real unknowns) and they become `unknown` GT on val (`--as-unknown`). This gives real U-F1 and OE numbers. It's used to (a) choose the unknown method variant, (b) set mining thresholds, (c) set the shape of the postproc thresholds. The proxy is never part of the submitted lineage.
- The proxy also rehearses B and R2: hold out 10 more classes with K images each.
- Every phase report lists: A-val mAP_K (30 base), B-val mAP (10 few-shot), dev-sim U-F1 and OE.

---

## 7. Post-processing for OE (P3)

1. One forward pass gives raw scores for all 52 slots per anchor (custom `infer.py`, not Ultralytics' single-label predict).
2. **Known:** class-wise NMS (IoU ~0.6), conf ≥ 0.001.
3. **Unknown candidates:** evidence score (§5.3) ≥ τ_u, class-agnostic NMS.
4. **Self-applied conflict policy:** drop an unknown if an overlapping known (IoU ≥ 0.5) has score ≥ τ_ks. Otherwise keep the unknown and drop the overlapping weak known boxes. The unknown score is set so the scorer's conflict rule reaches the same decision.
5. **Row budget:** ≤ 1000 per tile. Unknown rows (precision-critical, few) first, then known rows by score.
6. **Accuracy extras, kept only if val improves and output stays deterministic:** flip TTA (h/v/both, merged by WBF or NMS), and imgsz 1280 inference for tiny objects.
7. All thresholds live in `configs/postproc.yaml` and are grid-searched for **OE** on the dev sim plus A/B val (`tools/tune_postproc.py`).

---

## 8. Reproducibility

- `infer.py --weights ckpt/r1_final.pt --postproc configs/postproc.yaml --tiles test/tiles.csv --out preds.csv`
- `cudnn.deterministic=True`, `benchmark=False`, FP32, sorted tile order, fixed batch, no random TTA.
- `repro_check.py`: rerun in a fresh venv on the LOQ GPU and on CPU, then diff (same rows after rounding, Δscore < 1e-3, Δbox < 1 px).
- Ship `weights + yolo11-p2.yaml + postproc.yaml + SHA256SUMS + provenance.json` of every stage.

---

## 9. Phases (each one = a git branch + merge + tag + report)

### Round 1 (online, due 30 Nov 23:59)
| Phase | Tag | Goal | Done when | Window |
|---|---|---|---|---|
| **P0 Foundations** | `p0-foundations` | Data audit, scorer, CSV writer, from-scratch guards, fixed-slot model, stage runner | Tests pass. Two chained stages run offline with no `.pt` download | **18 Nov ✅** |
| **P1 Pipeline + baseline** | `p1-baseline` | LOQ setup, full data, convert, RFS, `infer.py` v1, scale benchmark, **insurance CSV** (short A → short B → CSV passes validator) | A validator-passing CSV exists; s/m timing measured | 19 Nov |
| **P2 Base detector** | `p2-base-detector` | Main Stage A from scratch (chosen scale, full schedule) | A-val mAP plateaus or the budget is used (~125 GPU-h) | 19 Nov night → 25 Nov |
| **P3 Open world** | `p3-open-world` | Mining + A_mine, postproc + conflict policy, dev-sim proxy tuning | Dev-sim OE measured; τ in postproc.yaml | Code: 20–25 Nov (parallel). GPU: 25–26 Nov |
| **P4 Few-shot + retention** | `p4-few-shot` | Pseudo-labels, copy-paste, B variants, final B → `r1_final.pt` | Retention table A → B; best variant chosen | 27 Nov |
| **P5 R1 submission** | `r1-submission` | Final inference, repro check, README, 5–7 slide PPT, Drive, form | Submitted ✅ | 28–29 Nov (30th = buffer) |

### Round 2 (on site, 16 Dec)
| **P6 R2 readiness** | `p6-r2-ready` | `run_r2.sh`, 3 timed rehearsals, offline kit, tech-note template | One command, ≤ 3 h, rehearsed | 1–15 Dec |
| **P7 R2 on site** | `r2-submission` | Run, submit early, refine, resubmit | CSV in before close | 16 Dec |

### Finale
| **P8 Finale** | `finale` | 2-page tech note, presentation, Q&A prep | Presented | 17 Dec |

Phases can overlap in *time* (P3 code is written while P2 trains), but they merge to `main` in order.

---

## 10. Git workflow (judges can see the development)

- `main` only ever receives `--no-ff` merges of `phase/<n>-<slug>` branches, so `git log --graph` shows one bubble per phase.
- Each merge gets an annotated tag (`git tag -a p2-base-detector -m "A-val mAP_K 0.41, ..."`) with the phase's headline numbers.
- Each phase adds `docs/phases/PHASE_<n>.md`: goal, what was built, results table, decisions + why, what's next. `DEVLOG.md` gets one row per phase with the metric progression.
- Weights and data are never committed. Only `provenance.json`, sha256s, `results.csv` summaries and plots (small PNGs).
- Remote: private `github.com/reddyadhvik-max/anveshak`. For judges: the Drive code zip **includes `.git`** (or a `git bundle`), and the README shows `git log --graph --oneline --decorate`. The repo can go public after 17 Dec.
- Commit identity: `S_Adhvik_Reddy <228771342+reddyadhvik-max@users.noreply.github.com>`.

---

## 11. GPU budget (LOQ, 19 Nov evening → 28 Nov morning ≈ 200 h)

| Use | Hours |
|---|---|
| P1 benchmark + insurance baseline | 6 |
| P2 Stage A | 125 |
| P3 mining + A_mine | 35 |
| P4 B variants + final B | 20 |
| Inference, TTA eval, repro | 6 |
| Slack (crash/resume) | 8 |

The rule: if Stage A val mAP is flat or broken by 22 Nov, stop, fix, and relaunch on a shortened schedule. Never start a second architecture.

---

## 12. Round 2 playbook (16 Dec)
Prep (P6): `r2/run_r2.sh <r2.zip>` does unzip → convert (R2 ids → slots 40–49) → pseudo-label old classes + unknowns in R2 tiles with `r1_final.pt` → mix (R2 ×k + replay A/B + copy-paste) → time-capped fine-tune (`time=` hours, a checkpoint every epoch) → inference → postproc → CSV → validator.
- Three rehearsals: proxy split, a fake R2 zip built from Set B val in round1 format, and a full timing run on the LOQ.
- Offline kit: wheel cache, data already on disk, fonts pre-cached, `YOLO_OFFLINE=1` tested. Physical kit: charger, extension board, cooling pad, phone hotspot, USB with repo + `r1_final.pt`.
- On the day: submit the first CSV as early as possible (ties go to the earlier submission), then refine and resubmit before close.

## 13. Round 1 deliverables checklist (P5)
- [ ] `predictions_round1.csv` (validator OK)
- [ ] `r1_final.pt` + configs + SHA256SUMS + provenance chain
- [ ] Code zip with `.git` + README (setup, data declaration, from-scratch statement, per-stage commands, reproduction)
- [ ] PPT, 5–7 slides: problem understanding / method / why it fits / first implementation + numbers / R2 plan / team (solo: all roles)
- [ ] Drive folder set to "anyone with link can view", link not posted anywhere, form submitted by 29 Nov

## 14. Risks
| Risk | Mitigation |
|---|---|
| From scratch in ~9 GPU-days | Fixed budget per phase, insurance CSV on 19 Nov, no architecture hopping |
| Set B teaches base objects = background | Old-class pseudo-labels + replay (§5.5) |
| Tiny objects | P2 head, native 1024, no downscaling |
| Dense tiles OOM in the assigner | Batch 4 + nbs 64. CPU fallback is built in |
| Laptop thermals or crash over days, or the laptop needed for other work | Pause button (control panel) → resume from `last.pt`; `resume=True`, cooling pad, ntfy alerts, `nvidia-smi` temp log |
| RTX 5050 needs CUDA 12.8 | cu128 wheels. Verify `torch.cuda.get_device_capability()` = (12, 0) |
| Unknown flood lowers mAP via the conflict rule | Self-applied policy tuned on OE |
| Reproduction mismatch | FP32 deterministic, pinned, config shipped, GPU+CPU repro check |
| Accidental pretrained weights | patch.py + offline smoke test + provenance hashes |

## 15. Not doing (scope guard)
No DETRs (slow from scratch). No self-supervised pre-stage (breaks "A first"). No ensembles or WBF across *models* (it's one model). No OBB. No web demo. No external data.

## 16. Status log
- **2026-11-18:** Team registered. GPU confirmed (RTX 5050 8 GB). Data audited remotely (no 20 GB download needed on the Mac).
  - **P0 built:** `patch.py` (no-AMP-download + objectness slot), `yolo11-p2.yaml` (nc 52), `train.py` (provenance, lineage `--from`), `convert.py`, `eda.py`, `oe_score.py`, `make_csv.py`. 12 tests pass, including the official validator.
  - CPU smoke test: random-init stage → checkpoint stage, offline, no `.pt` downloaded.
  - **P1 started:** pause/resume control panel (`anveshak/control.py`, localhost:8765). Pause = clean exit right after the epoch's `last.pt` save; Resume = `--resume last.pt`. Tested live: paused after epoch 1/4, resumed 2→4, finished; a double-click spawns only one process.
  - Added a **Stop now** button (SIGTERM → clean stop, resumable). The val split is frozen in `splits/` (A 952 tiles, B 172), so it's identical on every machine.
  - LOQ setup is packaged: `scripts/loq/setup_wsl.sh` (env, cu128, data, convert, Windows desktop shortcut + startup panel + power settings) and `docs/LOQ_HANDOFF.md`, the task list for the Claude session on the LOQ. Coordination between the Mac and LOQ sessions goes through this repo.
  - Remote Control is connected between the Mac and LOQ sessions. The LOQ turned out to be **native Windows 11 with no WSL**, so the setup switched to native Windows: `scripts/loq/setup_windows.ps1` plus the desktop "Anveshak Training" shortcut (`open_panel.ps1`).
  - The panel is now cross-platform: Stop = a `STOP` flag checked every batch (no signals), the alive check uses psutil, and `train.py --detach` uses `spawn()` (Windows: hidden console + breakaway from the shell's job). Tested on the Mac: detach → stop → resume → pause.
  - **LOQ set up** (torch 2.11 cu128 on sm_120, data at `C:\Users\reddy\data\round1`, 16/16 tests, splits identical). The button test passed on Windows (pause → resume → stop in 3 s → resume → finished; a detached run survives its shell). LOQ fixes: panel port 8765 → **8790** (PhoneBoxAgent owns 8765), an epoch log + ETA on the panel, no console flashes (`open_panel.pyw`).
  - **`anveshak/infer.py`:** our own post-processing over the raw 52-slot head (native 1024, FP32): per-class multi-label NMS for active known slots; unknown = max(s_unk, s_obj·(1 − max known)); the conflict rule applied by us; thresholds in `configs/postproc.yaml`. **`tools/rfs.py`:** repeat-factor sampling as an oversampled train list. On real Set A, thr 0.1 → epoch ×1.30, rarest class 46 → 204 tiles, imbalance 102:1 → 34:1. **Stage A trains on `data_A_rfs.yaml`.**
  - **Next:** s/m benchmark (LOQ, running) → choose the scale → the user starts Stage A. Mac: insurance CSV, A_mine design.
  - **Scale benchmark (LOQ), `docs/phases/PHASE_1_bench.md`:** every config spills past the 8 GB of VRAM into Windows shared memory (sysmem fallback). The peaks come from the assigner on dense tiles, not from batch size. Power is about 30 W while spilling, and RAM hit 0 GB free. Epochs in 125 GPU-h with RFS: s b8 ≈ 93, m b4 ≈ 79, s b4 ≈ 232, m b2 ≈ 317 (noisy because of the spill). `rfs.py` on the LOQ: ×1.30 confirmed. `infer.py` 20-tile check passes the validator. **Next:** the user sets NVIDIA "CUDA Sysmem Fallback Policy = Prefer No Sysmem Fallback", then re-bench m b2 vs s b4 → choose the scale. Stage A not started.
  - **Re-bench after the fix** (sysmem fallback off + `patch.ASSIGN_LIMIT`): s b8 and m b4 hit a real OOM in epoch 1, and Ultralytics halved them to **s b4 / m b2**, which then ran clean with no spill (power 62–68 W, shared < 1 GB). Epochs in 125 GPU-h with RFS: **s b4 ≈ 570, m b2 ≈ 325**. Recommendation: **m, batch=2**, workers=4. Set the batch explicitly: auto-halving only happens in the first epoch. RAM is still tight (≈ 2 GB free on average). Stage A not started (the user's call).
  - **2026-11-21:** Stage A at 147/250 (mAP50 0.253, mAP50-95 0.137 on A-val; 21.6 min/epoch; 9 logged resumes). ETA 23 Nov ~04:00.
  - **`tools/mine.py` (P3):** mines class-50 unknowns with the A checkpoint (obj ≥ 0.5, max known ≤ 0.2, NMS, IoU < 0.3 to GT, GT size range, ≤ 25% of tile) into `dataset_A_mine/` (hard-linked images) + `data_A_mine.yaml`; QA overlays in `runs/mine_qa`. `infer.forward()` is shared. Smoke-tested end to end, including a training epoch on the mined set.
  - **2026-11-23: Stage A finished on the LOQ** (m, batch 2, 250 epochs, `data_A_rfs.yaml`, random init; 11 user pauses/resumes, all in `provenance.json`; no OOM or batch halving after 73476d2). Final val (952 tiles): **mAP50 0.285, mAP50-95 0.155**, P 0.496, R 0.331 (epoch 150: 0.253 / 0.138; the LR decay and close_mosaic added +0.03 mAP50). `best.pt` = epoch 250, sha256 `59b5ea14…d5d6`. Snapshots every 25 epochs. Next: unknown mining → A_mine (each run on the user's OK).
  - **2026-11-23: back-to-back pipeline** (`anveshak/pipeline.py`, `configs/pipeline_r1.yaml`, panel card). First results: A-val **mAP_K 0.2497** (30 base). B-val unknown proxy (classes 30–39 as unknown): best **U-F1 0.023** at unknown_conf 0.10 (U-Prec ≈ 1%, pessimistic because B tiles lack base labels). Insurance CSV from A validated (612,810 rows).
  - **Mining dropped (P3):** probes on 600 A tiles at obj 0.3–0.6 (0.68→0.003 boxes per tile, 85–95% real objects on visual QA), but the boxes are mostly **missed base-class labels** (a red car in a lot of labelled cars). Mining them as class 50 would teach "car = unknown". Unknowns stay objectness-based at inference. The pipeline goes straight to **P4**: `tools/pseudo.py` (base classes pseudo-labelled ≥ 0.4 in Set B tiles, 0.1–0.4 greyed out) + mix (B train ×6 + 3000 A RFS replay tiles), B from A best.pt (freeze 10, lr0 0.001, 60 epochs), evals, `r1_B.csv`. Unknown quality is now the OE bottleneck (HM).
  - **Bug caught on the first B epoch (fixed in eeafd7d by the LOQ session):** `train.py` always passed `pretrained=False`, and Ultralytics' `Model.train()` then discards the loaded checkpoint, so `--from` silently started from random init (B epoch 1: box loss 3.81, mAP 0, no "Transferred" line). Stage A is unaffected (it is a random init by design). Now `--from` passes our own checkpoint path as `pretrained` (Ultralytics' log says "pretrained weights", but the source is runs/A/weights/best.pt, nothing downloaded), and an `on_train_start` check fails the stage unless all 803 tensors equal the checkpoint. The bad run is kept as `runs/B_scratch_bug`; B restarted from A at 07:20 (580 s/epoch, ETA ~17:00).
