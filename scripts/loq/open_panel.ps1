# "Anveshak Training" desktop shortcut target: starts the panel server if needed, then opens it.
param([switch]$NoBrowser)
$url = "http://localhost:8765"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
function Test-Panel { try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; $true } catch { $false } }
if (-not (Test-Panel)) {
    Start-Process -FilePath "$repo\.venv\Scripts\pythonw.exe" -ArgumentList "-m", "anveshak.control" -WorkingDirectory $repo -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Test-Panel); $i++) { Start-Sleep -Seconds 1 }
}
if (-not $NoBrowser) { Start-Process $url }
