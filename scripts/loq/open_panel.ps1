# "Anveshak Training" desktop shortcut target: starts the panel server if needed, then opens it.
# Port 8790 by default (8765 is taken by PhoneBoxAgent on the LOQ); override with $env:ANVESHAK_PORT.
param([switch]$NoBrowser)
$port = if ($env:ANVESHAK_PORT) { $env:ANVESHAK_PORT } else { 8790 }
$url = "http://localhost:$port"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
# Check it is really our panel, not some other app that happens to answer on the port.
function Test-Panel { try { (Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2).Content -match "Anveshak training" } catch { $false } }
if (-not (Test-Panel)) {
    Start-Process -FilePath "$repo\.venv\Scripts\pythonw.exe" -ArgumentList "-m", "anveshak.control", "--port", "$port" -WorkingDirectory $repo -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Test-Panel); $i++) { Start-Sleep -Seconds 1 }
}
if (-not $NoBrowser) { Start-Process $url }
