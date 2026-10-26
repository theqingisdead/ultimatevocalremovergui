# Helpers for .github/workflows/build-windows.yml. A failing command's last lines
# are also reported as an annotation on the run, which is quicker to read than the
# full job log.

function Write-Annotation([string]$title, [string[]]$lines) {
    $message = ($lines | ForEach-Object { ($_ -replace '%', '%25') -replace "`r", '' }) -join '%0A'
    Write-Host "::error title=$title::$message"
}

function Invoke-Logged([string]$title, [scriptblock]$command) {
    $log = Join-Path $env:RUNNER_TEMP (($title -replace '[^\w]', '_') + '.log')
    & $command 2>&1 | Tee-Object -FilePath $log | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Annotation $title (Get-Content $log -Tail 40)
        exit 1
    }
}
