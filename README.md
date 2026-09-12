<div align="center">

<img src="./logo.png" alt="PosChair Logo" width="520" />

# POSCHAIR — Real-Time AI Posture Monitor

**Real-time posture monitoring for desk workers using computer vision and voice AI.**  
No wearable. No app. Just your webcam, your browser, and an AI coach that talks to you.

![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)
![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose_Full-blue?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-3.8_Flash-orange?style=flat-square&logo=google)
![ElevenLabs](https://img.shields.io/badge/ElevenLabs-Turbo_v2-purple?style=flat-square)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square&logo=typescript)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?style=flat-square&logo=vercel)

</div>

---

## The Problem

Millions of desk workers develop chronic neck and back pain from sustained poor posture. Existing solutions rely on:

- **Screen banners / pop-ups** → ignored, disruptive, require looking at a screen
- **Wearables** → expensive, require charging, people forget to wear them
- **Timer-based reminders** → tell you *when* to check, not *what* is wrong

None of them work for visually impaired users. None of them tell you exactly *what* to fix.

---

## The Solution

PosChair uses a standard webcam to continuously track 33 body keypoints via **MediaPipe BlazePose** running entirely in the browser. When it detects that you've been in a bad position for more than 30 seconds, it:

1. Sends your posture metrics to **Gemini 3.8 Flash**, which generates a personalized, context-aware correction
2. Passes the text to **ElevenLabs**, which speaks it aloud naturally — *"Draw your chin back, your ears are 18° in front of your shoulders"*
3. Shows the correction as a popup with real-time data

Zero screen attention required. It works in the background while you work.

---

## Demo

```
[SPLASH SCREEN] → ▶ INITIALIZE SYSTEM
       ↓
[LOADING] MediaPipe BlazePose Full downloads (~2MB, cached)
       ↓
[CALIBRATION] Sit straight for 3 seconds → captures your personal neutral baseline
       ↓
[ACTIVE DASHBOARD]
  ┌──────────────────────────────┬──────────────────────────┐
  │                              │ ┌──────────────────────┐ │
  │   WEBCAM FEED                │ │  SKELETON MINIMAP    │ │
  │   + Green skeleton overlay   │ │  (wireframe only)    │ │
  │   + Amber joint dots         │ └──────────────────────┘ │
  │   + Red alert box on         │                          │
  │     bad posture zones        │ ┌──────────────────────┐ │
  │                              │ │ > HEAD_ANGLE: 161.3° │ │
  │   [Corner brackets, CRT      │ │ > TILT:      +2.1°   │ │
  │    scanlines, pixel art UI]  │ │ > SHRUG:     CLEAR   │ │
  │                              │ │ > ASYMMETRY: 1.4%    │ │
  │                              │ │ > LEAN:      +0.03   │ │
  │                              │ │ > FHP_DEPTH: -0.021  │ │
  │                              │ │ > ISSUES:    NONE    │ │
  │                              │ │ POSTURE_SCORE: 94    │ │
  │                              │ └──────────────────────┘ │
  │                              │                          │
  │                              │ ┌──────────────────────┐ │
  │                              │ │ 💡 TIP // GEMINI     │ │
  │                              │ │ "Pull your chin back │ │
  │                              │ │  and roll shoulders  │ │
  │                              │ │  down and back."     │ │
  │                              │ │ ▶ ELEVENLABS PLAYING │ │
  │                              │ └──────────────────────┘ │
  └──────────────────────────────┴──────────────────────────┘
```

---

## How It Works — Full Technical Breakdown

### Stage 1: Pose Detection (MediaPipe BlazePose Full)

MediaPipe BlazePose runs **entirely in the browser** as a WebAssembly binary — no data ever leaves your device for pose detection.

**Pipeline:**
1. Webcam stream feeds into a `<video>` element at 1280×720
2. Each animation frame (~30fps), the video frame is passed to `PoseLandmarker.detectForVideo()`
3. MediaPipe returns **33 normalized landmarks** (x, y, z, visibility) for the detected person

**Key landmarks used for posture:**

| ID | Landmark | Used For |
|----|----------|----------|
| 0 | Nose | Head-neck-shoulder angle (point A) |
| 7 | Left Ear | Lateral tilt, FHP detection |
| 8 | Right Ear | Lateral tilt, FHP detection |
| 11 | Left Shoulder | All metrics |
| 12 | Right Shoulder | All metrics |
| 23 | Left Hip | Trunk lean |
| 24 | Right Hip | Trunk lean |

**Why BlazePose Full and not others:**
- Only browser-native model with **Z-depth** (relative depth from camera)
- **33 keypoints** including ears — MoveNet only has 17, no ears → can't detect lateral tilt or FHP
- Runs on GPU via WebGL delegate, falls back to CPU WASM — no server needed

---

### Stage 2: Posture Analysis Engine (`lib/posture.ts`) — Dual-Layer v3.0 Architecture

#### 2a. 5-Second Neutral Baseline Calibration

Before monitoring starts, the user sits in their best posture for **5 seconds** (~150 frames). PosChair averages both 1D biomechanical metrics and an exhaustive 252-angle geometric manifold:

```typescript
calibration = {
  // Layer 1 Biomechanical Baseline
  headNeckShoulderAngle: 161.4°,  // YOUR natural good posture angle
  lateralTiltDelta: 0.8°,         // YOUR natural head-to-shoulder tilt
  earToShoulderRatio: 0.52,       // YOUR neck length ratio
  shoulderAsymmetry: 0.021,       // YOUR natural shoulder height variance
  trunkLean: 0.04,                // YOUR baseline sitting lean
  zFhpDelta: -0.012,              // YOUR baseline ear-to-shoulder depth
  // Layer 2 High-Dimensional Geometric Manifold
  featureAngles: [161.4, 88.2, 45.1, ...], // 252-angle triplet baseline
  meanVisibility: 0.94,
  capturedAt: 1726123456789
}
```

All alert thresholds are evaluated **relative to your baseline**, compensating for webcam elevation, body morphology, and chair ergonomics.

> **Research validation**: The ALIGN Framework (2026) demonstrated that static fixed thresholds achieve only ~82% accuracy. Personalized calibration achieved **98.74%**. Capturing a 252-angle baseline ensures Layer 2 consensus voting operates against an individualized geometric truth.

---

#### 2b. Metric Calculations

**Metric 1 — Head-Neck-Shoulder Angle (Forward Head Posture)**

The core FHP metric. Measures the angle at the ear, formed by the nose→ear→shoulder path:

```
            NOSE (A)
              |
              |  ← angle measured here
             EAR (B) ← vertex
              |
          SHOULDER (C)

Formula: angle = arccos( (BA · BC) / (|BA| × |BC|) )

Good posture: 160–180°
Mild FHP:     < 148° (or > 12° below your calibrated baseline)
Moderate FHP: < 140°
Severe FHP:   < 130°
```

*Why this formula?* The dot-product 3-point formula (used in PosePilot 2025 and ALIGN 2026) is **scale-invariant, position-invariant, and camera-distance-invariant**. The v1 `atan2` approach only measured 2D line angles and completely missed forward head pitch.

---

**Metric 2 — Z-Depth FHP Confirmation**

MediaPipe provides a relative `z` coordinate (more negative = closer to camera). For frontal camera FHP detection:

```
If ear.z << shoulder.z:
  → Head is physically in FRONT of the shoulder plane
  → Forward Head Posture confirmed via depth

zFhpDelta = earMid.z - shoulderMid.z
Alert if: zFhpDelta < (calibrated_baseline - 0.06)
```

Two independent FHP signals (angle + Z-depth) means far fewer false negatives.

---

**Metric 3 — Lateral Head Tilt**

```
earLineAngle = atan2(rEar.y - lEar.y, rEar.x - lEar.x)
shLineAngle  = atan2(rSh.y - lSh.y, rSh.x - lSh.x)
lateralTilt  = earLineAngle - shLineAngle

Alert thresholds (calibration-relative):
  Mild:   > ±8° from your baseline
  Moderate: > ±15°
  Severe:   > ±25°
```

---

**Metric 4 — Shoulder Shrug (Stress Elevation)**

```
earToShoulderRatio = (shoulderMid.y - earMid.y) / shoulderWidth

Lower value = shoulders raised toward ears (stress shrug)
Alert if < (calibrated_baseline - 0.08)
```

---

**Metric 5 — Shoulder Asymmetry**

```
asymmetry = abs(leftShoulder.y - rightShoulder.y) / shoulderWidth

Alert if > (calibrated_baseline + 0.06)
```

---

**Metric 6 — Trunk Lateral Lean**

```
trunkLean = (shoulderMid.x - hipMid.x) / shoulderWidth

Alert if abs(lean) > (calibrated_baseline + 0.10)
```

---

#### 2c. Temporal Smoothing

Raw per-frame landmark data is noisy. Without smoothing, a single frame of motion blur triggers false alerts.

```
v1: Raw per-frame values
  Frame 2: 14.1° → ALERT! (just noise)
  Frame 3: 4.0°  → OK
  Frame 4: 12.9° → ALERT! (noise again)

v2: 12-frame rolling average
  Frame 2: raw=14.1°, smoothed=4.8° → still OK
  ...after 12 genuine bad frames:
  Smoothed=13.1° → REAL alert ✅
```

Different metrics use different window sizes:
- Head angle: 12-frame window (slower, more stable)
- Z-depth FHP: 15-frame window (very noisy, needs more smoothing)
- Shoulder shrug/asymmetry: 8-frame window (faster response)

---

#### 2d. Layer 2: 252-Angle Feature Consensus Voting (PosePilot Methodology)

To reach and exceed **90% classification accuracy** without requiring cumbersome user-labeled datasets, PosChair implements an exhaustive geometric consensus voting layer:

1. **Key Landmark Subset**: 9 upper-body landmarks (`Nose, Left Ear, Right Ear, Left Shoulder, Right Shoulder, Left Elbow, Right Elbow, Left Hip, Right Hip`).
2. **Angle Triplet Computation**: $C(9,3) \times 3 = 252$ unique angle triplets $(A, V, C)$ computed at module initialization.
3. **Anatomical Sensitivity Mapping**: Each triplet is pre-indexed to the anatomical defects it is physically sensitive to (e.g., Ear-Shoulder-Hip angles indicate Forward Head; shoulder-elbow angles indicate shrug/asymmetry).
4. **Deviation Voting**:
   - Every frame, all 252 angles are compared against the user's calibrated baseline vector.
   - An angle triplet votes as "deviated" if $|\theta - \theta_{\text{cal}}| > 9.0^\circ$.
   - **Dual-Layer Gate**:
     - $\ge 25\%$ deviated $\rightarrow$ **CONFIRMED** (both Layer 1 and Layer 2 agree, confidence $\ge 0.85$).
     - $< 10\%$ deviated $\rightarrow$ **SUPPRESSED** (flagged as Layer 1 false positive, e.g. momentary head scratch or glance).
     - $10\% - 25\% \rightarrow$ Deferred to Layer 1.

---

#### 2e. Landmark Visibility Weighting & Hysteresis State Machine

- **Landmark Visibility Weighting**: When lighting is poor or keypoints are partially occluded by hair or desk edge, MediaPipe's keypoint `visibility` drops. The posture score dynamically incorporates visibility:
  $$\text{effective\_score} = \text{score} \times (0.7 + 0.3 \times \text{visibility})$$
  Ambiguous, noisy frames cannot trigger false penalties.
- **Dual-Threshold Hysteresis**:
  - `SCORE_ENTER_BAD = 68`: PosChair only transitions into the bad posture state when the score drops below 68.
  - `SCORE_EXIT_BAD = 76`: PosChair only returns to good posture when the score climbs above 76.
  - This 8-point deadband eliminates threshold boundary flickering.

---

#### 2f. Posture Score

Weighted penalty system, 0–100:

```
score = 100
       - FHP angle penalty    (max 35pts) — most important
       - Z-depth FHP penalty  (max 10pts) — confirmation signal
       - Lateral tilt penalty (max 25pts)
       - Shrug penalty        (max 15pts)
       - Asymmetry penalty    (max 15pts)
       - Lean penalty         (max 10pts)

Good posture:  score ≥ 76 (exit bad), or score ≥ 68 (enter bad threshold)
```

---

### Stage 3: Alert State Machine

```
                    ┌─────────────────┐
                    │   GOOD_POSTURE  │
                    └────────┬────────┘
                             │ bad posture detected
                             ▼
                    ┌─────────────────┐
                    │  WARN_STATE     │ ← timer starts
                    │  (counting up)  │
                    └────────┬────────┘
                             │ T_warn = 30 seconds
                             ▼
                    ┌─────────────────┐
              ┌────►│  ALERT_TRIGGER  │──► Gemini API → ElevenLabs
              │     └────────┬────────┘
              │              │ alert sent, T_cooldown = 60s
              │              ▼
T_cooldown    │     ┌─────────────────┐
(60s between) └─────┤  BAD_POSTURE    │
                    └────────┬────────┘
                             │ good posture detected
                             ▼
                    ┌─────────────────┐
                    │  T_RESET TIMER  │ ← must hold good posture
                    │  (5 seconds)    │    for 5s — no false resets
                    └────────┬────────┘
                             │ confirmed good for 5s
                             ▼
                    ┌─────────────────┐
                    │   GOOD_POSTURE  │ ← timer cleared
                    └─────────────────┘
```

**Why T_reset = 5 seconds?** In v1, a single frame of good posture (natural head movement) would reset the 28-second bad posture timer to zero — so alerts almost never fired. The T_reset state machine requires **sustained** good posture before clearing.

---

### Stage 4: AI Correction Generation (Gemini 3.8 Flash)

When the alert triggers, a structured prompt is sent to the `/api/analyze` route:

```
You are PosChair, an AI posture coach for desk workers.
The user has maintained poor posture for 47 seconds.

Detected issues: FORWARD_HEAD: Head pitched forward (138.2° / target: 149°+) [MODERATE]
Primary correction hint: Draw chin back — bring your ears over your shoulders.
Posture score: 61/100

Tone: Be clear and specific. They need to correct this now.

Write ONE specific, actionable voice correction in 1-2 short sentences (max 25 words).
- Address the #1 issue directly by body part
- Give the exact corrective movement
- No filler words ("I notice", "It seems", "Try to")
- Sound human, not robotic
```

The prompt adapts urgency with duration:
- **30–60s** → `GENTLE` — encouraging, first reminder
- **60–90s** → `FIRM` — clear, direct
- **90s+** → `URGENT` — firm, two sentences

---

### Stage 5: Voice Output (ElevenLabs)

The correction text is sent to `/api/speak`, which calls the ElevenLabs API:

```typescript
POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
{
  text: "Pull your chin back and drop your shoulders away from your ears.",
  model_id: "eleven_turbo_v2",   // lowest latency (~300ms vs 800ms standard)
  voice_settings: {
    stability: 0.5,              // consistent, not robotic
    similarity_boost: 0.75,      // clear and natural
    style: 0.0,                  // neutral style
    use_speaker_boost: true      // enhanced clarity
  }
}
```

**Voice**: Rachel (`21m00Tcm4TlvDq8ikWAM`) — calm, clear, professional. Chosen for desk environment intelligibility.

**Why `eleven_turbo_v2`?** For a real-time coaching app, a 300ms TTS response vs 800ms makes the alert feel immediate, not delayed. The quality difference is imperceptible for short corrections.

The audio bytes stream back as `audio/mpeg`, get wrapped in a Blob URL, and are played directly by a hidden `<Audio>` element.

---

## Architecture Overview

`
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Client)                         │
│                                                                 │
│  Webcam → <video> → MediaPipe WASM → 33 Landmarks              │
│                                ↓                                │
│                   Dual-Layer Engine (lib/posture.ts)            │
│                   ├── Layer 1: 6 Biomechanical Metrics         │
│                   ├── Layer 2: 252-Angle Consensus Voting       │
│                   ├── Visibility-Weighted Scoring               │
│                   ├── Rolling Temporal Smoothers                │
│                   └── Hysteresis State Machine (68 / 76)        │
│                                ↓                                │
│              score < 68 for > 30s → API calls                  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┴────────────────────────┐
        ▼                                                ▼
┌──────────────────────────────────────┐ ┌──────────────────────────────────────┐
│       NEXT.JS API ROUTES (Cloud)     │ │   HEAVY LOCAL MODEL VIA CLOUDFLARE   │
│                                      │ │                                      │
│  /api/analyze → Gemini 3.8 Flash     │ │  /api/heavy-pose → Cloudflare Tunnel │
│  /api/speak   → ElevenLabs Turbo v2  │ │        ↓                             │
│                                      │ │  Python FastAPI (Local Laptop GPU)   │
│                                      │ │  YOLOv8m-Pose (Heavy Precision)      │
└──────────────────────────────────────┘ └──────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                           BROWSER                               │
│                                                                 │
│  Audio plays → popup appears → user corrects posture           │
└─────────────────────────────────────────────────────────────────┘
`

**No pose data ever leaves your control.** High-speed inference runs locally in the browser or on your private laptop.

---

## Heavy Pose Engine via Cloudflare Tunnel (Desktop-Grade Accuracy)

For users who want the absolute maximum accuracy (e.g. overcoming low light, baggy clothing, or partial occlusion), PosChair includes a **Heavy Pose Engine** powered by **YOLOv8m-Pose**:

- **Local Model**: Runs on your laptop using PyTorch / CUDA.
- **Cloudflare Tunnel**: cloudflared tunnel --url http://localhost:8000 assigns a secure HTTPS address.
- **Vercel-Compatible**: Even when PosChair is deployed on Vercel, it connects back to your local laptop through the tunnel URL specified in HEAVY_MODEL_URL.
- **Zero Cloud GPU Costs**: You get server-grade deep learning without paying for GPU servers!

### 1-Click Windows Launchers:
- **`run.bat`**: Double-click to launch the **full local stack** (starts the heavy pose engine on port 8000, starts the Next.js web app on port 3000, and opens your browser automatically).
- **`run_heavy_vercel.bat`**: Double-click to launch the **Heavy Model + Cloudflare Tunnel** to connect your laptop GPU to your live **Vercel deployment**.

### Or via Terminal:
```powershell
# 1. Start Python Heavy Model Server (port 8000)
npm run server:heavy

# 2. In a second terminal, start Cloudflare Tunnel
npm run tunnel

# 3. Add the generated https://*.trycloudflare.com URL to .env.local or Vercel:
# HEAVY_MODEL_URL=https://<tunnel-id>.trycloudflare.com
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Framework | Next.js 14 (App Router) | API routes + Vercel deployment |
| Pose Detection | MediaPipe BlazePose Full (WASM) | Only browser-native model with Z-depth + ear landmarks |
| AI Analysis | Gemini 3.8 Flash (Google AI Studio) | Fastest available on account, sub-500ms |
| Voice TTS | ElevenLabs Rachel, eleven_turbo_v2 | Lowest latency, highest clarity |
| Language | TypeScript | Type safety for landmark data |
| Styling | Vanilla CSS | 8-bit arcade pixel art design system |
| Font | Press Start 2P + VT323 | Pixel aesthetic, terminal readouts |
| Deployment | Vercel | Zero-config Next.js deploy |

---

## File Structure

```
poschair/
├── app/
│   ├── api/
│   │   ├── analyze/route.ts       # Gemini API → correction text
│   │   ├── speak/route.ts         # ElevenLabs API → audio stream
│   │   └── heavy-pose/route.ts    # Proxies frames to Cloudflare Tunnel / Python server
│   ├── globals.css                # Complete 8-bit design system
│   ├── layout.tsx                 # Root layout + Google Fonts
│   └── page.tsx                   # Main dashboard (v3.0 dual-layer UI & state machine)
├── lib/
│   └── posture.ts                 # Dual-layer posture engine (v3.0)
│       ├── Layer 1: 6 biomechanical smoothed metrics
│       ├── Layer 2: 252-angle consensus voting
│       ├── Visibility-weighted score calculation
│       ├── Hysteresis thresholds (68 enter / 76 exit)
│       └── 5-second calibration with 252-angle baseline
├── server/
│   ├── heavy_pose_server.py       # FastAPI YOLOv8m-pose server
│   ├── start_server.ps1           # Startup script for Python server
│   ├── start_tunnel.ps1           # Startup script for Cloudflare Tunnel
│   └── README.md                  # Heavy model & tunnel instructions
├── .env.local                     # API keys (gitignored)
├── .gitignore
├── next.config.js
├── package.json
├── tsconfig.json
├── run.bat                     # 1-click full stack launcher (Next.js + heavy model)
└── run_heavy_vercel.bat        # 1-click launcher for Heavy Model + Cloudflare Tunnel
```

---

## Design System — 8-Bit Arcade

Inspired by retro CRT terminal aesthetics and arcade game UIs (Nothing's dot-matrix design, Habbo Hotel's pixelated community interfaces).

**Color Palette:**
```
--bg-void:      #020804   // Deep black-green
--green-neon:   #00ff41   // Matrix terminal green
--amber:        #ffd700   // Joint dots, calibration, alerts
--orange:       #ff6b35   // Warning accents
--red-alert:    #ff0040   // Bad posture, critical alerts
--cyan:         #00e5ff   // Secondary data
```

**Effects:**
- `repeating-linear-gradient` scanlines over camera (CRT effect)
- `text-shadow` neon glow on all green text
- `box-shadow` pulsing glow animation on status badges
- `animation: blink` on status dots (classic arcade heartbeat)
- Pixel corner brackets around camera frame
- `Press Start 2P` font for headings/labels
- `VT323` monospace for terminal-style live data

---

## Setup & Local Development

### Prerequisites
- Node.js 18+
- Webcam
- API keys (see below)

### 1. Clone

```bash
git clone https://github.com/brovk2008/Poschair-Hackathon.git
cd Poschair-Hackathon
npm install
```

### 2. Environment Variables

Create `.env.local`:

```env
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
GOOGLE_GEMINI_API_KEY=your_gemini_key
```

| Variable | Where to get |
|---|---|
| `ELEVENLABS_API_KEY` | [elevenlabs.io](https://elevenlabs.io) → API Keys |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` = Rachel (default), or any ElevenLabs voice ID |
| `GOOGLE_GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |

### 3. Run

```bash
npm run dev
# → http://localhost:3000
```

### 4. Usage

1. Open `localhost:3000` → click **▶ INITIALIZE SYSTEM**
2. Allow camera access
3. **Calibration**: Sit in your best posture, look straight at camera, hold for 3 seconds
4. The dashboard activates — skeleton overlay appears in amber/green
5. Deliberately slouch → watch the metrics turn red
6. Hold bad posture for 30 seconds → AI correction fires + ElevenLabs speaks it

---

## Deployment (Vercel)

```bash
# Push to GitHub (already done)
# Go to vercel.com → Import repository
# Add environment variables in Vercel dashboard:
#   ELEVENLABS_API_KEY
#   ELEVENLABS_VOICE_ID
#   GOOGLE_GEMINI_API_KEY
# Deploy → done
```

---

## Posture Metrics Reference

| Metric | What It Detects | Good Range | Alert Threshold |
|---|---|---|---|
| `HEAD_ANGLE` | Forward head posture | 160–180° | < 148° (or -12° from calibrated) |
| `TILT` | Lateral head tilt | ± 0–5° | > ±8° from calibrated |
| `SHRUG` | Elevated shoulders | 0.40–0.65 ratio | < -0.08 from calibrated |
| `ASYMMETRY` | Uneven shoulder height | < 3% | > +6% from calibrated |
| `LEAN` | Trunk lean left/right | < 0.12 | > +0.10 from calibrated |
| `FHP_DEPTH` | Forward head Z-depth | > -0.06 | < -0.06 from calibrated |

---

## Research & References

This project was built on a deep research compilation (September 2026) covering the state-of-the-art in posture detection systems. Key sources that directly influenced the implementation:

### Pose Estimation Models

| Paper / System | Year | How It Influenced PosChair |
|---|---|---|
| **BlazePose: On-device Real-time Body Pose Tracking** (Google) | 2020 | Core model choice — only browser model with Z-depth and ear landmarks |
| **RTMPose** (Shanghai AI Lab / OpenMMLab) | 2023 | Informed model benchmarking; RTMPose used as accuracy reference (75.8 AP at 90+ FPS CPU) |
| **ViTPose++** (Microsoft) | 2023 | Established SOTA accuracy ceiling (81.1 AP COCO); confirmed BlazePose is the right tradeoff for browser |
| **YOLOv11-Pose** (Ultralytics) | 2024 | Evaluated, rejected — 17 keypoints, no ear landmark |
| **LSP-YOLO** | 2025 | Studied for edge-AI posture classification architecture (<2MB, 91% precision) |

### Posture Classification & Systems

| Paper / System | Year | Key Insight Applied |
|---|---|---|
| **ALIGN Framework** | 2026 | Calibration-relative thresholds → 98.74% KNN accuracy. Directly implemented calibration system. |
| **PosePilot** | 2025 | 3-point dot-product angle formula for 680 angle features; 97.52% accuracy. Implemented the core angle3pt() function. |
| **SitPose** | 2025 | 3D skeleton + auxiliary virtual geometry points for spinal curvature. Informed Z-depth usage. |
| **Depth Camera PostureAx** | 2023 | Validated against OptiTrack; showed fixed thresholds fail near boundaries → personalized calibration non-negotiable |
| **FHP GCN Study** | 2024 | 78.27% F1 for frontal FHP detection — confirmed Z-depth needed as second signal to compensate |
| **HRNet + DTW** | 2025 | 92.3% PCK@0.5 for sports correction — informed T_reset state machine design |

### Biomechanics & Clinical Standards

| Source | Applied Insight |
|---|---|
| PMC 2026 — Neck posture study | Correct neck angle: ~135.57°; incorrect: ~126.52° → 9° delta as detection threshold |
| Ergonomic research consensus | Camera at eye level ±5cm, 0.8–1.5m distance, subject filling 50–80% of frame |
| Ophthalmology trainee pilot (2025) | Confirmed closed-loop real-time feedback + voice coaching improves posture scores |
| One Euro Filter (Géry Casiez et al.) | Informed rolling-window smoother design for landmark noise |

### Alert Timing Research

| Finding | Implementation |
|---|---|
| T_warn: 15–30 seconds (ALIGN, real-world systems) | Set to 30 seconds |
| T_reset: 5 seconds to confirm good posture | Implemented in state machine |
| T_cooldown: 60 seconds minimum between alerts | Set to 60 seconds to prevent alarm fatigue |
| Urgency escalation at 60s and 90s | GENTLE → FIRM → URGENT tone progression |

### Tools & APIs

| Tool | Version | Purpose |
|---|---|---|
| [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) | 0.10.14 | Browser WASM pose detection |
| [Google Generative AI SDK](https://ai.google.dev) | 0.21.0 | Gemini 3.8 Flash API client |
| [ElevenLabs TTS API](https://elevenlabs.io/docs) | v1 | Text-to-speech voice alerts |
| [Next.js](https://nextjs.org) | 14.2.5 | Full-stack React framework |
| [Press Start 2P](https://fonts.google.com/specimen/Press+Start+2P) | — | Pixel art typography |
| [VT323](https://fonts.google.com/specimen/VT323) | — | Terminal-style data readouts |

---

## Accuracy: v1 → v2

| Problem | v1 | v2 | Fix Applied |
|---|---|---|---|
| Angle formula | `atan2` line difference | **Dot-product 3-point** | PosePilot / ALIGN methodology |
| FHP detection | Nose-Y proxy (missed horizontal push) | **Z-depth + head-neck-shoulder angle** | MediaPipe z-coordinate + 3pt formula |
| Thresholds | Fixed global values | **Calibrated to your body** | ALIGN 2026 (+17% accuracy) |
| Noise | Raw per-frame | **12-frame rolling average** | Temporal smoothing |
| State machine | Single-frame reset | **T_reset = 5s sustained good posture** | Prevents false timer resets |
| Model | BlazePose Lite | **BlazePose Full** | Higher landmark accuracy |
| Alert cooldown | 45s | **60s** | Research recommendation |

---

## Accessibility Note

PosChair was specifically designed to work **without screen attention**:
- All corrections are delivered via **voice** (ElevenLabs)
- The system runs in the background while you work
- Fully compatible with screen readers (semantic HTML throughout)
- No reliance on color alone for status indication (text labels accompany all colors)

This makes it the only posture monitoring system that is **accessible to visually impaired desk workers**.

---

## License

MIT — built for the PosChair Hackathon, September 2026.

---

<div align="center">
Built with 🎮 8-bit love · MediaPipe · Gemini · ElevenLabs
</div>
