'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import {
  analyzePosture,
  buildAnalysisPrompt,
  getUrgencyLevel,
  sampleCalibrationFrame,
  captureCalibration,
  resetSmoothers,
  TIMING,
  SCORE_ENTER_BAD,
  SCORE_EXIT_BAD,
  type PostureMetrics,
  type PoseLandmark,
  type CalibrationBaseline,
  LM,
} from '@/lib/posture';

// MediaPipe skeleton connections (upper body focused)
const POSE_CONNECTIONS: [number, number][] = [
  [7, 8],   // ear–ear
  [7, 11],  [8, 12],  // ears–shoulders
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // shoulders–hips
  [23, 24], // hips
];

type AppState = 'SPLASH' | 'LOADING' | 'CALIBRATING' | 'ACTIVE';

interface AlertData {
  text: string;
  urgency: 'GENTLE' | 'FIRM' | 'URGENT';
  isPlaying: boolean;
  timestamp: number;
}

// ── Posture timer state machine ───────────────────────────────────────────────
// T_WARN: 30s bad → alert
// T_RESET: 5s good → clear bad timer (prevents false resets on minor frame)
// T_COOLDOWN: 60s between alerts

export default function PosChair() {
  const videoRef    = useRef<HTMLVideoElement>(null);
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const minimapRef  = useRef<HTMLCanvasElement>(null);
  const streamRef   = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<any>(null);
  const rafRef      = useRef<number>(0);
  const audioRef    = useRef<HTMLAudioElement | null>(null);

  // Timer refs (avoid stale closure issues in rAF loop)
  const badStartRef      = useRef<number | null>(null);
  const goodStartRef     = useRef<number | null>(null);
  const lastAlertRef     = useRef<number>(0);
  const isAnalyzingRef   = useRef(false);
  const calibSamplesRef  = useRef<Partial<CalibrationBaseline>[]>([]);
  const calibrationRef   = useRef<CalibrationBaseline | null>(null);
  const metricsRef       = useRef<PostureMetrics | null>(null);

  const [appState, setState] = useState<AppState>('SPLASH');
  const [metrics, setMetrics] = useState<PostureMetrics | null>(null);
  const [alert, setAlert] = useState<AlertData | null>(null);
  const [badMs, setBadMs] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [calibProgress, setCalibProgress] = useState(0); // 0-100
  const [calibration, setCalibration] = useState<CalibrationBaseline | null>(null);
  const [fps, setFps] = useState(0);
  const [noPerson, setNoPerson] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [loadingMsg, setLoadingMsg] = useState('Downloading MediaPipe BlazePose vision model...');

  const appStateRef = useRef<AppState>('SPLASH');
  appStateRef.current = appState;

  // ── Sync Video Stream Whenever Mounted ─────────────────────────────────────
  useEffect(() => {
    if (streamRef.current && videoRef.current && videoRef.current.srcObject !== streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.muted = true;
      videoRef.current.play().catch(err => console.warn('Video sync play warning:', err));
    }
  });

  // ── Load MediaPipe ─────────────────────────────────────────────────────────
  const loadMediaPipe = useCallback(async () => {
    setState('LOADING');
    setLoadingMsg('Downloading BlazePose Full vision model...');
    try {
      const { PoseLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      try {
        const landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputSegmentationMasks: false,
        });
        landmarkerRef.current = landmarker;
        setState('CALIBRATING');
      } catch (gpuErr) {
        console.warn('GPU delegate failed, switching to CPU Lite model...', gpuErr);
        setLoadingMsg('GPU busy/unsupported. Loading CPU vision model...');
        const landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputSegmentationMasks: false,
        });
        landmarkerRef.current = landmarker;
        setState('CALIBRATING');
      }
    } catch (e: any) {
      console.error('MediaPipe failed to load:', e);
      setCameraError('AI Engine failed to load: ' + (e?.message || String(e)));
      setState('SPLASH');
    }
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setState('LOADING');
    setLoadingMsg('Requesting webcam access...');
    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
          },
          audio: false,
        });
      } catch (constraintErr) {
        console.warn('Constrained camera request failed, falling back to default:', constraintErr);
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        try {
          await videoRef.current.play();
        } catch (playErr) {
          console.warn('Initial video.play() warning:', playErr);
        }
      }

      await loadMediaPipe();
    } catch (e: any) {
      console.error('Camera error:', e);
      let msg = 'Could not access webcam. ';
      if (e?.name === 'NotAllowedError' || e?.name === 'PermissionDeniedError') {
        msg = 'Camera permission was denied. Please click the camera icon in your browser address bar, select Allow, and try again.';
      } else if (e?.name === 'NotFoundError' || e?.name === 'DevicesNotFoundError') {
        msg = 'No camera device was found on your computer.';
      } else if (e?.name === 'NotReadableError' || e?.name === 'TrackStartError') {
        msg = 'Webcam is in use by another program (Zoom, Teams, etc.). Please close it and try again.';
      } else {
        msg += e?.message || String(e);
      }
      setCameraError(msg);
      setState('SPLASH');
    }
  }, [loadMediaPipe]);

  // ── AI Analysis ────────────────────────────────────────────────────────────
  const triggerAnalysis = useCallback(async (m: PostureMetrics, badDurationMs: number) => {
    if (isAnalyzingRef.current) return;
    isAnalyzingRef.current = true;
    setIsAnalyzing(true);

    try {
      const urgency = getUrgencyLevel(badDurationMs);
      const prompt  = buildAnalysisPrompt(m, urgency, badDurationMs);

      const [analyzeRes] = await Promise.all([
        fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt }),
        }),
      ]);

      if (!analyzeRes.ok) throw new Error('Analysis API failed');
      const { correction } = await analyzeRes.json();
      const alertData: AlertData = { text: correction, urgency, isPlaying: true, timestamp: Date.now() };
      setAlert(alertData);

      // Speak it
      const speakRes = await fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: correction }),
      });

      if (speakRes.ok) {
        const blob    = await speakRes.blob();
        const url     = URL.createObjectURL(blob);
        const audio   = audioRef.current!;
        audio.src     = url;
        await audio.play();
        audio.onended = () => {
          setAlert(prev => prev ? { ...prev, isPlaying: false } : null);
          URL.revokeObjectURL(url);
          setTimeout(() => setAlert(null), 8000);
        };
      }

      lastAlertRef.current = Date.now();
    } catch (e) {
      console.error('Analysis error:', e);
    } finally {
      isAnalyzingRef.current = false;
      setIsAnalyzing(false);
    }
  }, []);

  // ── Skeleton Drawing ───────────────────────────────────────────────────────
  const drawSkeleton = useCallback((
    ctx: CanvasRenderingContext2D,
    lms: PoseLandmark[],
    W: number, H: number,
    isGood: boolean,
    mini: boolean
  ) => {
    const lineCol = isGood ? '#00ff41' : '#ff0040';
    const dotCol  = '#ffd700';
    const lw      = mini ? 1.5 : 2.5;
    const dr      = mini ? 3   : 5;

    ctx.shadowColor = lineCol;
    ctx.shadowBlur  = mini ? 4 : 10;
    ctx.strokeStyle = lineCol;
    ctx.lineWidth   = lw;

    for (const [a, b] of POSE_CONNECTIONS) {
      const lmA = lms[a]; const lmB = lms[b];
      if (!lmA || !lmB) continue;
      if ((lmA.visibility ?? 1) < 0.3 || (lmB.visibility ?? 1) < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(lmA.x * W, lmA.y * H);
      ctx.lineTo(lmB.x * W, lmB.y * H);
      ctx.stroke();
    }

    ctx.fillStyle   = dotCol;
    ctx.shadowColor = dotCol;
    ctx.shadowBlur  = mini ? 6 : 14;

    const keyIds = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24];
    for (const id of keyIds) {
      const lm = lms[id];
      if (!lm || (lm.visibility ?? 1) < 0.3) continue;
      ctx.beginPath();
      ctx.arc(lm.x * W, lm.y * H, dr, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }, []);

  // ── Main rAF Loop ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (appState !== 'CALIBRATING' && appState !== 'ACTIVE') return;

    const video   = videoRef.current;
    const canvas  = canvasRef.current;
    const minimap = minimapRef.current;
    const lander  = landmarkerRef.current;
    if (!video || !canvas || !minimap || !lander) return;

    const ctx  = canvas.getContext('2d')!;
    const mCtx = minimap.getContext('2d')!;

    let calibStart      = appState === 'CALIBRATING' ? Date.now() : 0;
    const CALIB_DURATION = 5000; // 5-second calibration (more stable baseline)
    let frameCount = 0;
    let fpsTimer   = 0;

    let lastTs = -1;
    const loop = (ts: number) => {
      if (video.readyState < 2) {
        if (video.paused && video.srcObject) {
          video.play().catch(() => {});
        }
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      if (ts <= lastTs) {
        ts = lastTs + 1;
      }
      lastTs = ts;

      // FPS
      frameCount++;
      if (ts - fpsTimer > 1000) { setFps(frameCount); frameCount = 0; fpsTimer = ts; }

      // Detect
      const results = lander.detectForVideo(video, ts);
      const lms: PoseLandmark[] = results.landmarks?.[0] ?? [];

      // Resize canvases
      canvas.width  = canvas.offsetWidth || video.videoWidth || 640;
      canvas.height = canvas.offsetHeight || video.videoHeight || 480;
      minimap.width  = minimap.offsetWidth || 300;
      minimap.height = minimap.offsetHeight || 140;
      const [W, H, mW, mH] = [canvas.width, canvas.height, minimap.width, minimap.height];

      ctx.clearRect(0, 0, W, H);
      mCtx.fillStyle = '#000';
      mCtx.fillRect(0, 0, mW, mH);

      if (lms.length === 0) {
        setNoPerson(true);
        // Draw no-signal grid on minimap
        mCtx.strokeStyle = '#1a5c22'; mCtx.lineWidth = 0.5;
        for (let x = 0; x < mW; x += 20) { mCtx.beginPath(); mCtx.moveTo(x,0); mCtx.lineTo(x,mH); mCtx.stroke(); }
        for (let y = 0; y < mH; y += 20) { mCtx.beginPath(); mCtx.moveTo(0,y); mCtx.lineTo(mW,y); mCtx.stroke(); }
        mCtx.fillStyle = '#1a5c22'; mCtx.font = '10px monospace'; mCtx.textAlign = 'center';
        mCtx.fillText('NO SIGNAL', mW/2, mH/2);
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      setNoPerson(false);

      // ── CALIBRATION PHASE ──────────────────────────────────────────────────
      if (appStateRef.current === 'CALIBRATING') {
        const elapsed = Date.now() - calibStart;
        const progress = Math.min(100, (elapsed / CALIB_DURATION) * 100);
        setCalibProgress(progress);

        // Sample this frame
        const sample = sampleCalibrationFrame(lms);
        if (sample) calibSamplesRef.current.push(sample);

        // Draw skeleton (green during calibration)
        drawSkeleton(ctx, lms, W, H, true, false);
        drawSkeleton(mCtx, lms, mW, mH, true, true);

        // Draw calibration ring on main canvas
        const earMid = { x: (lms[7].x + lms[8].x) / 2, y: (lms[7].y + lms[8].y) / 2 };
        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth   = 3;
        ctx.shadowColor = '#ffd700';
        ctx.shadowBlur  = 15;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.arc(earMid.x * W, earMid.y * H, 40, -Math.PI/2, (-Math.PI/2) + (2 * Math.PI * progress / 100));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;

        if (elapsed >= CALIB_DURATION) {
          // Finalize calibration
          const baseline = captureCalibration(calibSamplesRef.current);
          calibrationRef.current = baseline;
          setCalibration(baseline);
          resetSmoothers();
          setState('ACTIVE');
          appStateRef.current = 'ACTIVE';
          badStartRef.current = null;
          goodStartRef.current = null;
          lastAlertRef.current = 0;
          calibSamplesRef.current = [];
        }

        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      // ── ACTIVE PHASE ───────────────────────────────────────────────────────
      const m = analyzePosture(lms, calibrationRef.current);
      if (m) {
        setMetrics(m);
        metricsRef.current = m;

        const isGood = m.isGoodPosture;
        const now    = Date.now();

        drawSkeleton(ctx, lms, W, H, isGood, false);
        drawSkeleton(mCtx, lms, mW, mH, isGood, true);

        // Draw issue highlight box
        if (!isGood && lms[LM.LEFT_SHOULDER] && lms[LM.RIGHT_SHOULDER]) {
          const lsh = lms[LM.LEFT_SHOULDER];
          const rsh = lms[LM.RIGHT_SHOULDER];
          ctx.strokeStyle = 'rgba(255,0,64,0.35)';
          ctx.lineWidth   = 1;
          ctx.setLineDash([5, 5]);
          ctx.strokeRect(
            rsh.x * W - 12, (lms[LM.NOSE].y * H) - 8,
            (lsh.x - rsh.x) * W + 24,
            (lsh.y - lms[LM.NOSE].y) * H + 40
          );
          ctx.setLineDash([]);
        }

        // ── Hysteresis state machine ──────────────────────────────────────────
        // Enter bad state: score drops BELOW SCORE_ENTER_BAD (68)
        // Exit bad state: score must EXCEED SCORE_EXIT_BAD (76) for T_RESET_MS
        const isBadScore = m.postureScore < SCORE_ENTER_BAD;
        const isGoodScore = m.postureScore > SCORE_EXIT_BAD && m.issues.length === 0;

        if (isBadScore) {
          goodStartRef.current = null;
          if (!badStartRef.current) badStartRef.current = now;
          const duration = now - badStartRef.current;
          setBadMs(duration);

          const shouldAlert =
            duration >= TIMING.T_WARN_MS &&
            !isAnalyzingRef.current &&
            now - lastAlertRef.current > TIMING.T_COOLDOWN_MS;

          if (shouldAlert) triggerAnalysis(m, duration);

        } else if (isGoodScore) {
          // Must sustain good score for T_RESET_MS before clearing
          if (!goodStartRef.current) goodStartRef.current = now;
          const goodDuration = now - goodStartRef.current;
          if (goodDuration >= TIMING.T_RESET_MS) {
            badStartRef.current  = null;
            goodStartRef.current = null;
            setBadMs(0);
          }
        }
        // Between 68–76: hold state (hysteresis zone — no change)
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [appState, drawSkeleton, triggerAnalysis]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  // ── Helpers ────────────────────────────────────────────────────────────────
  const fmt = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  };

  const recalibrate = () => {
    calibSamplesRef.current = [];
    badStartRef.current     = null;
    goodStartRef.current    = null;
    setBadMs(0);
    setAlert(null);
    setCalibProgress(0);
    resetSmoothers();
    setState('CALIBRATING');
  };

  const score    = metrics?.postureScore ?? 100;
  const isGood   = metrics?.isGoodPosture ?? true;
  const isBad    = badMs > 0;

  // ── JSX ────────────────────────────────────────────────────────────────────
  return (
    <div className="dashboard">
      <audio ref={audioRef} style={{ display: 'none' }} />

      {/* ── SPLASH OVERLAY ── */}
      {appState === 'SPLASH' && (
        <div className="splash-screen" style={{ position: 'fixed', inset: 0, zIndex: 100, background: 'var(--bg-void)' }}>
          <div className="ascii-logo">{`
 ██████╗  ██████╗ ███████╗ ██████╗██╗  ██╗ █████╗ ██╗██████╗ 
 ██╔══██╗██╔═══██╗██╔════╝██╔════╝██║  ██║██╔══██╗██║██╔══██╗
 ██████╔╝██║   ██║███████╗██║     ███████║███████║██║██████╔╝
 ██╔═══╝ ██║   ██║╚════██║██║     ██╔══██║██╔══██║██║██╔══██╗
 ██║     ╚██████╔╝███████║╚██████╗██║  ██║██║  ██║██║██║  ██║
 ╚═╝      ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝`}
          </div>
          <div className="splash-title">POSTURE MONITORING SYSTEM</div>
          <div className="splash-subtitle">
            MediaPipe BlazePose → Gemini 3.8 Flash → ElevenLabs Voice
          </div>
          {cameraError && (
            <div style={{
              background: 'rgba(255, 0, 64, 0.15)',
              border: '2px solid var(--red-alert)',
              color: 'var(--red-alert)',
              padding: '12px 20px',
              fontFamily: 'var(--font-mono)',
              fontSize: '18px',
              maxWidth: '560px',
              textAlign: 'center',
              boxShadow: 'var(--glow-red)',
              lineHeight: 1.4,
            }}>
              ⚠ CAMERA ERROR: {cameraError}
            </div>
          )}
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', justifyContent: 'center' }}>
            <div className="spec-badge">📡 33 KEYPOINTS</div>
            <div className="spec-badge">🧠 DUAL-LAYER 252-ANGLE</div>
            <div className="spec-badge">🎙️ VOICE ALERTS</div>
            <div className="spec-badge">🎯 CALIBRATED</div>
          </div>
          <button className="pixel-btn" onClick={startCamera} id="start-btn">
            ▶ INITIALIZE SYSTEM
          </button>
          <div className="splash-subtitle" style={{ fontSize: '13px', color: 'var(--green-dim)' }}>
            Camera access required · Sit in good posture for calibration
          </div>
        </div>
      )}

      {/* ── LOADING OVERLAY ── */}
      {appState === 'LOADING' && (
        <div className="splash-screen" style={{ position: 'fixed', inset: 0, zIndex: 100, background: 'var(--bg-void)' }}>
          <div className="splash-title">LOADING AI ENGINE</div>
          <div className="splash-subtitle">{loadingMsg}</div>
          <div className="loading-bar-container">
            <div className="loading-bar-fill" />
          </div>
          <div className="splash-subtitle" style={{ fontSize: '14px', color: 'var(--green-dim)' }}>
            ~3-5 seconds on first load (cached after)
          </div>
        </div>
      )}

      {/* ── PERSISTENT DASHBOARD & CAMERA ── */}
      <header className="header-bar">
        <div className="header-logo">
          🎮 POS<span>CHAIR</span>
          <span style={{ color: 'var(--green-dim)', marginLeft: 8 }}>v3.0 DUAL-LAYER</span>
          {appState === 'CALIBRATING' && (
            <span style={{ color: 'var(--amber)', marginLeft: 8 }}>// CALIBRATING</span>
          )}
        </div>
        <div className="header-status">
          {appState === 'ACTIVE' && (
            <>
              <span className="score-badge">SCORE: {score}/100</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', color: 'var(--green-dim)' }}>{fps}fps</span>
              {calibration && (
                <span style={{ fontFamily: 'var(--font-pixel)', fontSize: '7px', color: 'var(--amber)' }}>
                  🎯 CAL
                </span>
              )}
              <button
                className="pixel-btn"
                style={{ fontSize: '7px', padding: '4px 8px' }}
                onClick={recalibrate}
                id="recalibrate-btn"
              >↺ RECAL</button>
              <div className={`status-pill ${noPerson ? 'idle' : isGood ? 'good' : 'bad'}`}>
                <div className="status-dot" />
                {noPerson ? 'NO SIGNAL' : isGood ? 'GOOD' : 'BAD POSTURE'}
              </div>
            </>
          )}
          {appState === 'CALIBRATING' && (
            <div className="status-pill idle">
              <div className="status-dot" />
              CALIBRATING {calibProgress.toFixed(0)}%
            </div>
          )}
        </div>
      </header>

      <div className="main-content">
        {/* Camera Section - NEVER UNMOUNTS */}
        <section className="camera-section">
          <video ref={videoRef} className="camera-video" playsInline muted autoPlay />
          <canvas ref={canvasRef} className="camera-canvas" />
          <div className="camera-label">
            {appState === 'CALIBRATING' ? 'CAM_01 // CALIBRATING BASELINE' : 'CAM_01 // MEDIAPIPE BLAZEPOSE FULL'}
          </div>
          <div className="camera-corner tl" /><div className="camera-corner tr" />
          <div className="camera-corner bl" /><div className="camera-corner br" />

          {/* Calibration overlay */}
          {appState === 'CALIBRATING' && (
            <div className="calib-overlay">
              <div className="calib-title">SIT IN YOUR BEST POSTURE</div>
              <div className="calib-sub">Head up · Shoulders level · Spine tall</div>
              <div className="calib-bar-wrap">
                <div className="calib-bar" style={{ width: `${calibProgress}%` }} />
              </div>
              <div className="calib-pct">{calibProgress.toFixed(0)}%</div>
            </div>
          )}
        </section>

        {/* Right Panel */}
        <aside className="right-panel">
          {/* Skeleton Minimap */}
          <div className="minimap-section">
            <div className="section-header">SKELETON_WIREFRAME</div>
            <div className="minimap-canvas-wrapper">
              <canvas ref={minimapRef} className="minimap-canvas" />
              <div className="minimap-scanline" />
            </div>
          </div>

          {/* Dynamic Content based on Calibration vs Active */}
          {appState === 'CALIBRATING' ? (
            <>
              <div className="metrics-section">
                <div className="section-header">CALIBRATION_GUIDE</div>
                {[
                  { icon: '👀', text: 'Look straight at camera' },
                  { icon: '📐', text: 'Ears aligned over shoulders' },
                  { icon: '💪', text: 'Relax shoulders — don\'t shrug' },
                  { icon: '🪑', text: 'Sit tall — back straight' },
                  { icon: '⚖️', text: 'Weight even on both hips' },
                ].map(({ icon, text }) => (
                  <div key={text} className="metric-row" style={{ gap: '12px', padding: '4px 0' }}>
                    <span style={{ fontSize: '20px' }}>{icon}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', color: 'var(--green-mid)' }}>{text}</span>
                  </div>
                ))}
              </div>
              <div className="alert-section">
                <div className="alert-idle" style={{ borderColor: 'var(--amber)', color: 'var(--amber)' }}>
                  ⚡ CAPTURING 252-ANGLE BASELINE<br/>
                  <span style={{ fontSize: '14px' }}>Dual-layer consensus voting for 90%+ accuracy</span>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Live Metrics */}
              <div className="metrics-section">
                <div className="section-header">LIVE_POSTURE_DATA {calibration ? '// CALIBRATED' : '// UNCALIBRATED'}</div>

                {/* Score bar */}
                <div className="metric-bar-row">
                  <span className="metric-bar-label">POSTURE_SCORE</span>
                  <div className="metric-bar-track">
                    <div
                      className={`metric-bar-fill score-fill${score < 50 ? ' low' : score < 72 ? ' mid' : ''}`}
                      style={{ width: `${score}%` }}
                    />
                  </div>
                  <span className="score-number">{score}</span>
                </div>

                {metrics ? (
                  <>
                    <div className="metric-row">
                      <span className="metric-key">&gt; HEAD_ANGLE:</span>
                      <span className={`metric-value ${metrics.headNeckShoulderAngle < 140 ? 'bad' : metrics.headNeckShoulderAngle < 152 ? 'warn' : 'ok'}`}>
                        {metrics.headNeckShoulderAngle.toFixed(1)}°
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; TILT:</span>
                      <span className={`metric-value ${Math.abs(metrics.lateralTiltDeg) > 15 ? 'bad' : Math.abs(metrics.lateralTiltDeg) > 8 ? 'warn' : 'ok'}`}>
                        {metrics.lateralTiltDeg > 0 ? '+' : ''}{metrics.lateralTiltDeg.toFixed(1)}°
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; SHRUG:</span>
                      <span className={`metric-value ${metrics.shoulderShrug < 0.30 ? 'bad' : metrics.shoulderShrug < 0.38 ? 'warn' : 'ok'}`}>
                        {metrics.shoulderShrug < 0.38 ? `${metrics.shoulderShrug.toFixed(2)} ⚠` : 'CLEAR'}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; ASYMMETRY:</span>
                      <span className={`metric-value ${metrics.shoulderAsymmetry > 0.14 ? 'bad' : metrics.shoulderAsymmetry > 0.07 ? 'warn' : 'ok'}`}>
                        {(metrics.shoulderAsymmetry * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; LEAN:</span>
                      <span className={`metric-value ${Math.abs(metrics.trunkLean) > 0.20 ? 'bad' : Math.abs(metrics.trunkLean) > 0.12 ? 'warn' : 'ok'}`}>
                        {metrics.trunkLean > 0 ? '+' : ''}{metrics.trunkLean.toFixed(2)}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; FHP_DEPTH:</span>
                      <span className={`metric-value ${metrics.zFhpDelta < -0.10 ? 'bad' : metrics.zFhpDelta < -0.06 ? 'warn' : 'ok'}`}>
                        {metrics.zFhpDelta.toFixed(3)}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; VISIBILITY:</span>
                      <span className={`metric-value ${metrics.visibilityScore > 0.75 ? 'ok' : metrics.visibilityScore > 0.50 ? 'warn' : 'bad'}`}>
                        {(metrics.visibilityScore * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; L2_CONSENSUS:</span>
                      <span className={`metric-value ${calibration ? 'ok' : 'warn'}`}>
                        {calibration ? '252-ANGLE ACTIVE' : 'UNCALIBRATED'}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; ISSUES:</span>
                      <span className={`metric-value ${metrics.issues.length > 0 ? 'bad' : 'ok'}`}>
                        {metrics.issues.length === 0
                          ? 'NONE (L1+L2 AGREED)'
                          : metrics.issues.map(i => `${i.label} [${(i.layer2Confidence * 100).toFixed(0)}%]`).join(', ')}
                      </span>
                    </div>
                    {calibration && Object.keys(metrics.deviations).length > 0 && (
                      <div className="metric-row" style={{ marginTop: '4px' }}>
                        <span className="metric-key" style={{ color: 'var(--amber)' }}>&gt; CAL_DELTA:</span>
                        <span className={`metric-value ${(metrics.deviations.headNeckShoulder ?? 0) > 15 ? 'bad' : 'warn'}`}>
                          {(metrics.deviations.headNeckShoulder ?? 0) > 0
                            ? `-${(metrics.deviations.headNeckShoulder ?? 0).toFixed(1)}°`
                            : 'ON BASELINE'}
                        </span>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="metric-row" style={{ color: 'var(--green-dim)' }}>
                    <span>&gt; AWAITING SIGNAL...<span className="cursor-blink" /></span>
                  </div>
                )}

                {/* Bad posture timer */}
                {isBad && badMs > 0 && (
                  <div className="timer-row">
                    ⚠ BAD: {fmt(badMs)}
                    {isAnalyzing && <span style={{ color: 'var(--amber)', fontSize: '16px' }}> [QUERYING AI...]</span>}
                  </div>
                )}

                {/* Alert cooldown indicator */}
                {!isBad && !isGood && (
                  <div className="metric-row" style={{ color: 'var(--green-dim)', fontSize: '16px' }}>
                    <span>&gt; CONFIRMING CORRECTION...<span className="cursor-blink" /></span>
                  </div>
                )}
              </div>

              {/* AI Alert */}
              <div className="alert-section">
                <div className="section-header">AI_COACH_OUTPUT</div>
                {alert ? (
                  <div className={`alert-box ${alert.urgency === 'URGENT' ? 'critical' : ''}`}>
                    <div className="alert-header">
                      <span>
                        {alert.urgency === 'URGENT' ? '🚨 URGENT' :
                         alert.urgency === 'FIRM'   ? '⚠ CORRECTION' : '💡 TIP'}
                        {' // GEMINI 3.8 FLASH'}
                      </span>
                      <span style={{ color: 'var(--green-dim)', fontSize: '6px' }}>
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="alert-text">{alert.text}</div>
                    {alert.isPlaying && (
                      <div className="alert-playing">
                        <div className="playing-bars">
                          {[0.2, 0.4, 0.3, 0.5, 0.2, 0.4].map((d, i) => (
                            <div key={i} className="playing-bar"
                              style={{ height: `${6 + Math.random() * 6}px`, '--delay': `${d}s` } as React.CSSProperties}
                            />
                          ))}
                        </div>
                        ▶ ELEVENLABS SPEAKING...
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="alert-idle">
                    {isAnalyzing ? (
                      <span style={{ color: 'var(--amber)' }}>⏳ AI ANALYZING...<span className="cursor-blink" /></span>
                    ) : (
                      <>
                        <span>MONITORING ACTIVE</span>
                        <br/>
                        <span style={{ fontSize: '13px', color: 'var(--green-dim)' }}>
                          Alert after {TIMING.T_WARN_MS / 1000}s bad posture
                        </span>
                      </>
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
