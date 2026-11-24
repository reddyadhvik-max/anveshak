# One-shot LOQ setup on native Windows. From the repo root, in PowerShell (no admin needed):
#   powershell -ExecutionPolicy Bypass -File scripts\loq\setup_windows.ps1
# Safe to re-run. GPU check -> uv + venv + CUDA 12.8 torch -> tests -> data -> convert -> shortcuts + power.
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$data = if ($env:ANVESHAK_DATA) { $env:ANVESHAK_DATA } else { Join-Path $HOME "data" }
$py = Join-Path $repo ".venv\Scripts\python.exe"
Set-Location $repo
function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Run { & $args[0] @($args | Select-Object -Skip 1); if ($LASTEXITCODE) { throw "failed ($LASTEXITCODE): $args" } }

Say "GPU"
Run nvidia-smi '--query-gpu=name,driver_version,memory.total' '--format=csv,noheader'

Say "Python env"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$HOME\.local\bin;$env:Path"
}
if (-not (Test-Path $py)) { Run uv venv -q -p 3.11 .venv }
Run uv pip install -q --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128
Run uv pip install -q --python $py -r requirements.txt gdown
Run $py -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.__version__, torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0)); x = torch.randn(4096, 4096, device='cuda'); print('matmul ok', float((x @ x).abs().mean()) > 0)"

Say "Data -> $data\round1"
New-Item -ItemType Directory -Force $data | Out-Null
$env:ANVESHAK_DATA = "$data\round1"   # lets the CSV test use the official validator
if (-not (Test-Path "$data\round1\class_list.csv")) {
    $zip = "$data\round1.zip"
    $free = (Get-PSDrive ((Resolve-Path $data).Drive.Name)).Free / 1GB
    if ($free -lt 45) { throw "need ~45 GB free on the data drive, have $([math]::Floor($free)) GB (set ANVESHAK_DATA to another drive)" }
    if (-not (Test-Path $zip)) {
        & "$repo\.venv\Scripts\gdown.exe" 1A24PTWWji509iDwf50EkJuTcwyb13RDH -O $zip
        if ($LASTEXITCODE) { throw "gdown failed (Drive quota?). Download round1.zip in the browser into $data and re-run." }
    }
    if ((Get-Item $zip).Length -ne 21624964974) { throw "round1.zip has the wrong size (partial download?)" }
    Run $py -m zipfile -e $zip $data
}
Write-Host "Set A images: $((Get-ChildItem "$data\round1\dataset_A\images").Count) (expect 9659)"

Say "Tests"
Run $py -m pytest -q tests

Say "Convert (uses the frozen splits\ committed in git)"
Run $py tools\convert.py --root "$data\round1"
git status --short splits

Say "Shortcuts"
$shell = New-Object -ComObject WScript.Shell
function New-Shortcut($path, $extra, $desc) {
    $s = $shell.CreateShortcut($path)
    $s.TargetPath = "$repo\.venv\Scripts\pythonw.exe"   # pythonw: no console window flashes on double-click
    $s.Arguments = "`"$repo\scripts\loq\open_panel.pyw`" $extra"
    $s.WorkingDirectory = $repo
    $s.IconLocation = "$env:SystemRoot\System32\shell32.dll,137"
    $s.Description = $desc
    $s.WindowStyle = 7
    $s.Save()
}
$desktop = [Environment]::GetFolderPath("Desktop")
New-Shortcut (Join-Path $desktop "Anveshak Training.lnk") "" "Pause / resume / stop Anveshak training"
New-Shortcut (Join-Path ([Environment]::GetFolderPath("Startup")) "Anveshak Panel.lnk") "--no-browser" "Start the Anveshak panel at login"
Write-Host "Desktop: $desktop\Anveshak Training.lnk ; panel also starts at login"

Say "Power (training runs for days)"
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
Write-Host "On AC: never sleep or hibernate; closing the lid does nothing."

Say "Done. Double-click 'Anveshak Training' on the desktop."
