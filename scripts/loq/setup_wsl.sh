#!/usr/bin/env bash
# One-shot LOQ setup, run inside WSL2 Ubuntu from the repo root:  bash scripts/loq/setup_wsl.sh
# Safe to re-run. Steps: GPU check -> uv + venv + CUDA 12.8 torch -> tests -> data -> convert -> Windows shortcuts.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${ANVESHAK_DATA:-$HOME/data}"
cd "$REPO"
say() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

say "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "nvidia-smi failed: update the Windows NVIDIA driver"; exit 1; }
case "$REPO" in /mnt/*) echo "WARNING: repo is on the Windows drive ($REPO); move it under ~ for 5-10x faster IO";; esac

say "Python env"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d .venv ] || uv venv -q -p 3.11 .venv
uv pip install -q --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python .venv/bin/python -r requirements.txt gdown
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available in torch"
print(torch.__version__, torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda"); print("matmul ok", float((x @ x).abs().mean()) > 0)
PY

say "Tests"
.venv/bin/python -m pytest -q tests

say "Data -> $DATA/round1"
mkdir -p "$DATA"
if [ ! -f "$DATA/round1/class_list.csv" ]; then
  if [ ! -f "$DATA/round1.zip" ]; then
    .venv/bin/gdown 1A24PTWWji509iDwf50EkJuTcwyb13RDH -O "$DATA/round1.zip" || {
      echo "gdown failed (Drive quota?). Download round1.zip in the Windows browser, then:"
      echo "  cp /mnt/c/Users/<you>/Downloads/round1.zip $DATA/ && bash scripts/loq/setup_wsl.sh"; exit 1; }
  fi
  [ "$(stat -c %s "$DATA/round1.zip")" = 21624964974 ] || { echo "round1.zip size mismatch (partial download?)"; exit 1; }
  .venv/bin/python -m zipfile -e "$DATA/round1.zip" "$DATA/"
fi
ls "$DATA/round1/dataset_A/images" | wc -l | xargs echo "Set A images:"   # expect 9659

say "Convert (uses the frozen splits/ committed in git)"
.venv/bin/python tools/convert.py --root "$DATA/round1"
git status --short splits || true   # must be clean: the split is identical on every machine

say "Windows: desktop shortcut, panel at login, power settings"
WIN_DIR="$(wslpath "$(cmd.exe /c 'echo %LOCALAPPDATA%' 2>/dev/null | tr -d '\r')")/Anveshak"
mkdir -p "$WIN_DIR"
sed "s#__REPO__#$REPO#" scripts/loq/open_panel.ps1 > "$WIN_DIR/open_panel.ps1"
cp scripts/loq/setup_windows.ps1 "$WIN_DIR/"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$WIN_DIR/setup_windows.ps1")"

say "Done. Double-click 'Anveshak Training' on the desktop to open the panel."
