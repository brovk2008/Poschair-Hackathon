# Starts the PosChair Heavy Pose Estimation Server (FastAPI + YOLOv8m-pose)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " 🚀 Starting PosChair Heavy Pose Server (YOLOv8m-pose)" -ForegroundColor Green
Write-Host " Port: 8000" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

python server/heavy_pose_server.py
