# Real-ESRGAN Training Monitor
$log = (Get-ChildItem experiments/proposed_RealESRGAN_x4plus/*.log | Select-Object -Last 1).FullName
if ($log) {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "Monitoring Real-ESRGAN Training Log: $log" -ForegroundColor Green
    Write-Host "Press Ctrl + C to exit monitor at any time" -ForegroundColor Yellow
    Write-Host "==========================================================" -ForegroundColor Cyan
    Get-Content $log -Wait -Tail 20
} else {
    Write-Host "No log file found in experiments/proposed_RealESRGAN_x4plus/" -ForegroundColor Red
}
