# Checks for a built UVR.exe, used by .github/workflows/build-windows.yml.
# Run from the repository root after the "Prepare smoke test input" and
# "Smoke test from source" steps (they create smoke\ and smoke\source\).
. $PSScriptRoot\ci.ps1

function Write-Notice([string]$message) {
    Write-Host "::notice::$message"
}

# Window titles of every process that has one (a crash shows up as an error dialog)
function Get-WindowTitles {
    Get-Process | Where-Object { $_.MainWindowTitle } | ForEach-Object { "$($_.ProcessName): $($_.MainWindowTitle)" }
}

function Get-LogTail([string]$path) {
    if (Test-Path $path) { Get-Content $path -Tail 40 } else { @("(no $(Split-Path $path -Leaf))") }
}

# Separates the smoke clip with UVR.exe --headless and checks the stems match the source run
function Test-Separation([string]$exe, [string]$label) {
    $crashLog = Join-Path (Split-Path $exe) 'UVR_crash.log'
    $cases = [ordered]@{
        mdx    = @('--arch', 'mdx', '--model', 'smoke\UVR-MDX-NET-Inst_HQ_3.onnx', '--set', 'is_denoise_model=True', 'denoiser_model=models/VR_Models/UVR-DeNoise-Lite.pth')
        demucs = @('--arch', 'demucs', '--model', 'smoke\demucs\htdemucs.yaml')
    }
    foreach ($name in $cases.Keys) {
        $out = "smoke\$label\$name"
        $exeArgs = @('--headless', '--input', 'smoke\clip.wav', '--out', $out) + $cases[$name]
        Remove-Item headless_error.log, $crashLog -ErrorAction SilentlyContinue
        $start = Get-Date
        $p = Start-Process $exe -ArgumentList $exeArgs -WorkingDirectory $pwd -PassThru
        $null = $p.Handle  # needed for ExitCode to be available later
        if (-not $p.WaitForExit(600000)) {
            $windows = Get-WindowTitles
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            Write-Annotation "$label $name hung" (@('UVR.exe did not exit within 10 minutes. Open windows:') + $windows + (Get-LogTail $crashLog) + (Get-LogTail headless_error.log))
            exit 1
        }
        Write-Host "${label} ${name}: exit code $($p.ExitCode) after $([int]((Get-Date) - $start).TotalSeconds) s"
        if ($p.ExitCode -ne 0) {
            Write-Annotation "$label $name exit $($p.ExitCode)" ((Get-LogTail headless_error.log) + (Get-LogTail $crashLog))
            exit 1
        }
        Invoke-Logged "compare $label $name" { python tools/compare_outputs.py "smoke\source\$name" $out --min-db 100 }
        Write-Notice "$label ${name}: separation OK, output matches the source run"
    }
}

# Starts the GUI the way a double-click does and waits for the main window
function Test-Gui([string]$exe, [string]$label) {
    $dir = Split-Path $exe
    $crashLog = Join-Path $dir 'UVR_crash.log'
    Remove-Item $crashLog -ErrorAction SilentlyContinue
    $p = Start-Process $exe -WorkingDirectory $dir -PassThru
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 3
        $p.Refresh()
        if ($p.HasExited) { break }
        if ($p.MainWindowTitle -like '*Ultimate Vocal Remover*') { $ok = $true; break }
    }
    $status = "exited: $($p.HasExited)  window title: '$($p.MainWindowTitle)'"
    $windows = Get-WindowTitles
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    if (-not $ok) {
        Write-Annotation "$label GUI check" (@($status) + $windows + (Get-LogTail $crashLog))
        exit 1
    }
    Write-Notice "$label GUI: main window opened"
}

# Every file under $dir (including hidden ones), relative to it
function Get-FileList([string]$dir) {
    $root = (Resolve-Path $dir).Path.TrimEnd('\') + '\'
    Get-ChildItem $dir -Recurse -File -Force | ForEach-Object { $_.FullName.Substring($root.Length) } | Sort-Object
}

# Extracts the downloaded zip with Windows' built-in tar into $dest and checks no built file is missing
function Expand-Zip([string]$zip, [string]$dest, [string]$label) {
    New-Item -ItemType Directory -Force $dest | Out-Null
    $start = Get-Date
    tar.exe -xf $zip -C $dest
    if ($LASTEXITCODE -ne 0) { Write-Annotation "$label extract" @("tar exit $LASTEXITCODE"); exit 1 }
    $extracted = Get-FileList $dest
    Write-Notice "${label}: extracted $($extracted.Count) files to '$dest' in $([int]((Get-Date) - $start).TotalSeconds) s"
    $built = Get-Content (Join-Path $env:RUNNER_TEMP 'dist_files.txt')
    $present = [System.Collections.Generic.HashSet[string]]::new([string[]]$extracted, [System.StringComparer]::OrdinalIgnoreCase)
    $missing = $built | Where-Object { -not $present.Contains($_) }
    if ($missing) {
        Write-Annotation "$label is missing $(@($missing).Count) of $($built.Count) built files" (@($missing) | Select-Object -First 30)
        exit 1
    }
}

# Runs the separation and GUI checks on the app in $dir
function Test-App([string]$dir, [string]$label) {
    $exe = Join-Path $dir 'UVR.exe'
    Test-Separation $exe $label
    Test-Gui $exe $label
}
