# PosChair Heavy Model Server (YOLOv8m-Pose + Cloudflare Tunnel)

This directory provides a **desktop-grade heavy pose estimation server** that complements PosChair's in-browser MediaPipe BlazePose engine.

## Why Use the Heavy Model?
- **Extreme Precision**: YOLOv8m-pose delivers higher Mean Average Precision (mAP) than lightweight browser WASM models, especially under partial occlusion, slouching, loose clothing, and low ambient light.
- **Deep Consensus**: Cross-verifies the 252-angle Layer 2 consensus against ground-truth convolutional feature maps.
- **Anywhere Access**: By tunneling your local GPU/CPU machine through a **Cloudflare Tunnel**, your deployed app on **Vercel** can call your own laptop's heavy model without paying expensive cloud GPU hosting fees!

## Quick Start (2 Terminals)

### Terminal 1: Start the Python Server
```powershell
python server/heavy_pose_server.py
# Server starts on http://localhost:8000
```
Or run:
```powershell
.\server\start_server.ps1
```

### Terminal 2: Start the Cloudflare Tunnel
```powershell
cloudflared tunnel --url http://localhost:8000
```
Or run:
```powershell
.\server\start_tunnel.ps1
```

Cloudflare will give you a temporary public HTTPS URL like:
```
https://random-words-here.trycloudflare.com
```

### Connect to PosChair (Local or Vercel)
In your `.env.local` or in Vercel Environment Variables:
```env
HEAVY_MODEL_URL=https://random-words-here.trycloudflare.com
```

PosChair's API route `/api/heavy-pose` will automatically route verification requests to your laptop!
