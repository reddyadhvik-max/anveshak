# Opens the Anveshak training panel. Starts the panel server inside WSL first if it isn't running.
# The hidden wsl.exe that runs the server also keeps the WSL VM alive, so training survives closed terminals.
param([switch]$NoBrowser)
$url = "http://localhost:8765"
$repo = "__REPO__"   # filled in by setup_wsl.sh
function Test-Panel { try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; $true } catch { $false } }
if (-not (Test-Panel)) {
    Start-Process wsl.exe -WindowStyle Hidden -ArgumentList @('-e', 'bash', '-lc', "cd '$repo' && exec .venv/bin/python -m anveshak.control")
    for ($i = 0; $i -lt 30 -and -not (Test-Panel); $i++) { Start-Sleep -Seconds 1 }
}
if (-not $NoBrowser) { Start-Process $url }
