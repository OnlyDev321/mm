param(
    [string]$Python = 'C:\Users\brain\Documents\ChatGPT\MMDL\work\train_B_20261031\.venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'D1 Judge - secure key entry'
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
$env:PYTHONDONTWRITEBYTECODE = '1'
$TaskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $TaskRoot
& $Python -B -X utf8 (Join-Path $PSScriptRoot 'judge_runner.py') --scoring-root (Join-Path $TaskRoot 'scoring')
if ($LASTEXITCODE -eq 0) {
    $Host.UI.RawUI.WindowTitle = 'D1 Judge - COMPLETE'
    Write-Host 'COMPLETE. The key has been removed from the launcher environment.' -ForegroundColor Green
} else {
    $Host.UI.RawUI.WindowTitle = 'D1 Judge - FAILED / PENDING'
    Write-Host 'FAILED / PENDING. Inspect scoring/judge_launcher logs before resuming.' -ForegroundColor Red
}
