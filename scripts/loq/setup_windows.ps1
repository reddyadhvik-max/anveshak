# Windows side of the LOQ setup (setup_wsl.sh runs this). No admin needed for the shortcuts.
$ErrorActionPreference = "Continue"
$dir = Join-Path $env:LOCALAPPDATA "Anveshak"
$ps1 = Join-Path $dir "open_panel.ps1"
$shell = New-Object -ComObject WScript.Shell

function New-Shortcut($path, $extra, $desc) {
    $s = $shell.CreateShortcut($path)
    $s.TargetPath = "powershell.exe"
    $s.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ps1`" $extra"
    $s.IconLocation = "$env:SystemRoot\System32\shell32.dll,137"
    $s.Description = $desc
    $s.WindowStyle = 7
    $s.Save()
}

# 1. Desktop button: opens the Pause / Resume / Stop panel
$desktop = [Environment]::GetFolderPath("Desktop")
New-Shortcut (Join-Path $desktop "Anveshak Training.lnk") "" "Pause / resume / stop Anveshak training"
Write-Host "Desktop shortcut: $desktop\Anveshak Training.lnk"

# 2. Start the panel server at login (no browser), so it's always ready
$startup = [Environment]::GetFolderPath("Startup")
New-Shortcut (Join-Path $startup "Anveshak Panel.lnk") "-NoBrowser" "Start Anveshak panel server"
Write-Host "Startup entry:    $startup\Anveshak Panel.lnk"

# 3. Never sleep or hibernate on AC power; closing the lid on AC does nothing (training runs for days)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
Write-Host "Power: no sleep/hibernate on AC, lid close on AC = do nothing"

# 4. WSL memory: default is 50% of RAM, too little for 1024px dataloaders. Applied after `wsl --shutdown`.
$cfg = Join-Path $env:USERPROFILE ".wslconfig"
if (-not (Test-Path $cfg)) {
    $ramGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $mem = [math]::Max(8, $ramGB - 4)
    "[wsl2]`nmemory=${mem}GB`nswap=24GB`n" | Set-Content -Encoding ascii $cfg
    Write-Host ".wslconfig written (memory=${mem}GB of ${ramGB}GB, swap=24GB). Run 'wsl --shutdown' once before training to apply it."
} else {
    Write-Host ".wslconfig already exists, left unchanged:"; Get-Content $cfg
}
