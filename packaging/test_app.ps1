# Checks for a built UVR.exe, used by .github/workflows/build-windows.yml.
# Run from the repository root after the "Prepare smoke test input" and
# "Smoke test from source" steps (they create smoke\ and smoke\source\).
. $PSScriptRoot\ci.ps1

# Separates the smoke clip with UVR.exe --headless and checks the stems match the source run
function Test-Separation([string]$exe, [string]$label) {
    $cases = [ordered]@{
        mdx    = @('--arch', 'mdx', '--model', 'smoke\UVR-MDX-NET-Inst_HQ_3.onnx', '--set', 'is_denoise_model=True', 'denoiser_model=models/VR_Models/UVR-DeNoise-Lite.pth')
        demucs = @('--arch', 'demucs', '--model', 'smoke\demucs\htdemucs.yaml')
    }
    foreach ($name in $cases.Keys) {
        $out = "smoke\$label\$name"
        $exeArgs = @('--headless', '--input', 'smoke\clip.wav', '--out', $out) + $cases[$name]
        Remove-Item headless_error.log -ErrorAction SilentlyContinue
        $p = Start-Process $exe -ArgumentList $exeArgs -WorkingDirectory $pwd -Wait -PassThru
        Write-Host "${label} ${name}: UVR.exe exit code $($p.ExitCode)"
        if ($p.ExitCode -ne 0) {
            $lines = if (Test-Path headless_error.log) { Get-Content headless_error.log -Tail 40 } else { @('no headless_error.log') }
            Write-Annotation "$label $name exit $($p.ExitCode)" $lines
            exit 1
        }
        Invoke-Logged "compare $label $name" { python tools/compare_outputs.py "smoke\source\$name" $out --min-db 100 }
    }
}

# Starts the GUI the way a double-click does and waits for the main window
function Test-Gui([string]$exe) {
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
    Write-Host $status
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    if (-not $ok) {
        $lines = @($status) + $(if (Test-Path $crashLog) { Get-Content $crashLog -Tail 40 } else { @('no UVR_crash.log') })
        Write-Annotation 'GUI check' $lines
        exit 1
    }
}
