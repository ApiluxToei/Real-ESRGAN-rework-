# Real-ESRGAN Training Monitor
$expDir = "experiments/improved_RealESRGAN_x4plus"
if (-not (Test-Path $expDir)) {
    $expDir = "experiments/proposed_RealESRGAN_x4plus"
}

$log = (Get-ChildItem "$expDir/*.log" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
$models = Get-ChildItem "$expDir/models/net_g_*.pth" -ErrorAction SilentlyContinue

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Monitoring Real-ESRGAN Training: $expDir" -ForegroundColor Green
if ($models) {
    Write-Host " Saved Checkpoints: $(($models | ForEach-Object { $_.Name }) -join ', ')" -ForegroundColor Magenta
}
Write-Host " Log File: $log" -ForegroundColor Gray
Write-Host " Press Ctrl + C to exit monitor at any time" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

if ($log) {
    Get-Content $log -Wait -Tail 20
} else {
    Write-Host "No log file found in $expDir/" -ForegroundColor Red
}
