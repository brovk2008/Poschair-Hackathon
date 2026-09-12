# Exposes the local Heavy Pose server publicly via Cloudflare Tunnel
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " 🌐 Starting Cloudflare Tunnel for PosChair Heavy Server" -ForegroundColor Green
Write-Host " Tunneling: http://localhost:8000" -ForegroundColor Yellow
Write-Host " Look for the https://*.trycloudflare.com URL below!" -ForegroundColor Magenta
Write-Host " Copy that URL to HEAVY_MODEL_URL in .env.local or Vercel" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

cloudflared tunnel --url http://localhost:8000
