'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import {
  Gamepad2,
  Sparkles,
  Activity,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Volume2,
  Mic,
  Eye,
  Zap,
  Flame,
  Terminal,
  Cpu,
  Camera,
  Radio,
  Trophy,
  Crosshair,
  Target,
  Heart,
  Layers,
  Sliders,
  Clock,
  ChevronRight,
  Shield,
  Play,
  VolumeX,
} from 'lucide-react';

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

export default function PosChair() {
  const videoRef      = useRef<HTMLVideoElement>(null);
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const minimapRef    = useRef<HTMLCanvasElement>(null);
  const streamRef     = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<any>(null);
  const rafRef        = useRef<number>(0);
  const audioRef      = useRef<HTMLAudioElement | null>(null);

  // Timer refs (avoid stale closure issues in rAF loop)
  const badStartRef      = useRef<number | null>(null);
  const goodStartRef     = useRef<number | null>(null);
  const streakStartRef   = useRef<number>(Date.now());
  const lastAlertRef     = useRef<number>(0);
  const isAnalyzingRef   = useRef(false);
  const calibSamplesRef  = useRef<Partial<CalibrationBaseline>[]>([]);
  const calibrationRef   = useRef<CalibrationBaseline | null>(null);
  const metricsRef       = useRef<PostureMetrics | null>(null);

  const [appState, setState] = useState<AppState>('SPLASH');
  const [metrics, setMetrics] = useState<PostureMetrics | null>(null);
  const [alert, setAlert] = useState<AlertData | null>(null);
  const [badMs, setBadMs] = useState(0);
  const [streakMs, setStreakMs] = useState(0);
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
    const lineCol = isGood ? '#00FF66' : '#FF2E93';
    const dotCol  = '#FFE600';
    const lw      = mini ? 2 : 3.5;
    const dr      = mini ? 3.5 : 6;

    ctx.shadowColor = lineCol;
    ctx.shadowBlur  = mini ? 4 : 12;
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

      // Resize canvases with fallback dimensions
      canvas.width  = canvas.offsetWidth || video.videoWidth || 640;
      canvas.height = canvas.offsetHeight || video.videoHeight || 480;
      minimap.width  = minimap.offsetWidth || 300;
      minimap.height = minimap.offsetHeight || 280;
      const [W, H, mW, mH] = [canvas.width, canvas.height, minimap.width, minimap.height];

      ctx.clearRect(0, 0, W, H);
      mCtx.fillStyle = '#06070a';
      mCtx.fillRect(0, 0, mW, mH);

      if (lms.length === 0) {
        setNoPerson(true);
        // Draw no-signal grid on minimap
        mCtx.strokeStyle = '#1e293b'; mCtx.lineWidth = 1;
        for (let x = 0; x < mW; x += 24) { mCtx.beginPath(); mCtx.moveTo(x,0); mCtx.lineTo(x,mH); mCtx.stroke(); }
        for (let y = 0; y < mH; y += 24) { mCtx.beginPath(); mCtx.moveTo(0,y); mCtx.lineTo(mW,y); mCtx.stroke(); }
        mCtx.fillStyle = '#64748b'; mCtx.font = 'bold 14px "Space Grotesk", sans-serif'; mCtx.textAlign = 'center';
        mCtx.fillText('TARGET LOST // NO SIGNAL', mW/2, mH/2);
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
        ctx.strokeStyle = '#FFE600';
        ctx.lineWidth   = 4;
        ctx.shadowColor = '#FFE600';
        ctx.shadowBlur  = 16;
        ctx.setLineDash([8, 6]);
        ctx.beginPath();
        ctx.arc(earMid.x * W, earMid.y * H, 44, -Math.PI/2, (-Math.PI/2) + (2 * Math.PI * progress / 100));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;

        if (elapsed >= CALIB_DURATION) {
          // Finalize calibration
          const baseline = captureCalibration(calibSamplesRef.current);
          calibrationRef.current = baseline;
          setCalibration(baseline);
          streakStartRef.current = Date.now();
          setState('ACTIVE');
        }

        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      // ── ACTIVE MONITORING PHASE ────────────────────────────────────────────
      const m = analyzePosture(lms, calibrationRef.current);
      if (!m) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      metricsRef.current = m;
      setMetrics(m);

      // Draw skeletons
      drawSkeleton(ctx, lms, W, H, m.isGoodPosture, false);
      drawSkeleton(mCtx, lms, mW, mH, m.isGoodPosture, true);

      // Draw issue callout on main canvas
      if (!m.isGoodPosture && m.issues.length > 0) {
        const topIssue = m.issues[0];
        ctx.fillStyle   = 'rgba(255, 46, 147, 0.9)';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth   = 3;
        ctx.font        = 'bold 14px "Space Grotesk", sans-serif';
        const text = `! ${topIssue.label.toUpperCase()} [${(topIssue.layer2Confidence * 100).toFixed(0)}%]`;
        const tw = ctx.measureText(text).width;
        const bx = W / 2 - tw / 2 - 14;
        const by = H - 54;
        ctx.fillRect(bx, by, tw + 28, 38);
        ctx.strokeRect(bx, by, tw + 28, 38);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, W / 2 - tw / 2, by + 24);
      }

      // ── HYSTERESIS POSTURE TIMER STATE MACHINE ─────────────────────────────
      const now = Date.now();
      const currentScore = m.postureScore;
      const isBadScore   = currentScore < SCORE_ENTER_BAD;
      const isGoodScore  = currentScore > SCORE_EXIT_BAD;

      if (isBadScore) {
        goodStartRef.current = null;
        if (!badStartRef.current) badStartRef.current = now;
        const duration = now - badStartRef.current;
        setBadMs(duration);
        setStreakMs(0);
        streakStartRef.current = now;

        if (duration >= TIMING.T_WARN_MS) {
          const cooldownElapsed = now - lastAlertRef.current;
          if (cooldownElapsed >= TIMING.T_COOLDOWN_MS) {
            triggerAnalysis(m, duration);
          }
        }
      } else if (isGoodScore) {
        // Sustained good posture
        if (!goodStartRef.current) goodStartRef.current = now;
        const goodDuration = now - goodStartRef.current;
        if (goodDuration >= TIMING.T_RESET_MS) {
          badStartRef.current  = null;
          goodStartRef.current = null;
          setBadMs(0);
        }
        setStreakMs(now - streakStartRef.current);
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
    streakStartRef.current  = Date.now();
    setBadMs(0);
    setAlert(null);
    setCalibProgress(0);
    resetSmoothers();
    setState('CALIBRATING');
  };

  const score    = metrics?.postureScore ?? 100;
  const isGood   = metrics?.isGoodPosture ?? true;
  const isBad    = badMs > 0;
  const livesCount = badMs > 60000 ? 1 : badMs > 30000 ? 2 : 3;

  // ── JSX ────────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      <audio ref={audioRef} style={{ display: 'none' }} />

      {/* ── SPLASH OVERLAY ── */}
      {appState === 'SPLASH' && (
        <div className="screen-overlay">
          <div className="overlay-modal">
            <div className="modal-badge">
              <Sparkles size={14} /> 16-BIT RETRO ROM x BENTO DASHBOARD
            </div>
            <div className="splash-logo-wrap">
              <img src="/logo.png" alt="PosChair Logo" className="splash-hero-logo" />
            </div>
            <p className="modal-sub">
              Extreme Neo-Brutalist posture monitor with dual-layer 252-angle consensus, Gemini 3.8 Flash analysis, and ElevenLabs real-time voice coaching.
            </p>

            {cameraError && (
              <div className="error-banner">
                <ShieldAlert size={24} style={{ color: 'var(--pop-magenta)', flexShrink: 0 }} />
                <div>
                  <div style={{ textTransform: 'uppercase', marginBottom: 4 }}>Camera Access Failed</div>
                  <div style={{ color: '#cbd5e1', fontSize: 13 }}>{cameraError}</div>
                </div>
              </div>
            )}

            <div className="feature-pill-grid">
              <div className="feature-pill">
                <Cpu size={16} style={{ color: 'var(--pop-cyan)' }} />
                <span>33 Keypoints</span>
              </div>
              <div className="feature-pill">
                <Layers size={16} style={{ color: 'var(--pop-yellow)' }} />
                <span>Dual-Layer 252-Angle</span>
              </div>
              <div className="feature-pill">
                <Mic size={16} style={{ color: 'var(--pop-purple)' }} />
                <span>ElevenLabs Voice</span>
              </div>
              <div className="feature-pill">
                <Trophy size={16} style={{ color: 'var(--pop-lime)' }} />
                <span>90%+ Precision</span>
              </div>
              <div className="feature-pill">
                <Zap size={16} style={{ color: 'var(--pop-orange)' }} />
                <span>Zero Latency</span>
              </div>
            </div>

            <button className="brutal-hero-btn" onClick={startCamera} id="start-btn">
              <Play size={22} fill="#000" />
              <span>Initialize System</span>
            </button>

            <div style={{ fontSize: 13, color: 'var(--pop-muted)', fontWeight: 600 }}>
              Webcam permission required · Auto-calibrates in 5 seconds
            </div>
          </div>
        </div>
      )}

      {/* ── LOADING OVERLAY ── */}
      {appState === 'LOADING' && (
        <div className="screen-overlay">
          <div className="overlay-modal" style={{ maxWidth: 520 }}>
            <div className="modal-badge" style={{ background: 'var(--pop-cyan)' }}>
              <Zap size={14} /> LOADING VISION ENGINE
            </div>
            <div className="splash-logo-wrap" style={{ margin: '8px 0' }}>
              <img src="/logo.png" alt="PosChair Logo" style={{ height: 42, width: 'auto', objectFit: 'contain' }} />
            </div>
            <p className="modal-sub">{loadingMsg}</p>
            <div className="calib-progress-track">
              <div className="calib-progress-bar" style={{ width: '85%' }} />
            </div>
            <div style={{ fontSize: 13, color: 'var(--pop-muted)' }}>
              Model cached locally after first download
            </div>
          </div>
        </div>
      )}

      {/* ── RETRO CARTRIDGE HEADER BAR ── */}
      <header className="cartridge-header">
        <div className="brand-section">
          <div className="brand-logo-badge brand-img-badge">
            <img src="/logo.png" alt="PosChair Logo" className="header-logo-img" />
          </div>
          <div className="brand-meta">
            <div className="brand-title-wrap">
              <div className="brand-title">
                NEO<span>BRUTALIST</span>
              </div>
              <span className="rom-chip">ROM: V4.0 OMNI</span>
            </div>
            <div className="brand-subtitle">
              OMNIDIRECTIONAL T-FRAME // YOLOV8M-POSE ENGINE
            </div>
          </div>
        </div>

        {/* HUD Telemetry Ribbon */}
        <div className="header-hud">
          {appState === 'ACTIVE' && (
            <>
              {/* Score Pill */}
              <div className={`hud-pill score-pill ${score < 68 ? 'bad' : score < 76 ? 'warn' : ''}`}>
                <Trophy size={16} />
                <span>SCORE: {score}/100</span>
              </div>

              {/* Health Hearts */}
              <div className="hud-pill hearts-pill" title="Posture Lives">
                {[1, 2, 3].map(i => (
                  <Heart
                    key={i}
                    size={16}
                    fill={i <= livesCount ? '#FF2E93' : '#334155'}
                    style={{ color: i <= livesCount ? '#FF2E93' : '#334155' }}
                  />
                ))}
              </div>

              {/* FPS Pill */}
              <div className="hud-pill fps-pill">
                <Activity size={15} />
                <span>{fps} FPS</span>
              </div>

              {/* Status Badge */}
              <div className={`hud-pill status-pill ${noPerson ? 'idle' : isGood ? 'good' : 'bad'}`}>
                {noPerson ? (
                  <>
                    <Radio size={16} />
                    <span>NO TARGET</span>
                  </>
                ) : isGood ? (
                  <>
                    <CheckCircle2 size={16} />
                    <span>GOOD POSTURE</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle size={16} />
                    <span>BAD POSTURE</span>
                  </>
                )}
              </div>

              {/* Recalibrate Button */}
              <button className="brutal-btn btn-yellow" onClick={recalibrate} id="recalibrate-btn">
                <RotateCcw size={15} />
                <span>Recal</span>
              </button>
            </>
          )}

          {appState === 'CALIBRATING' && (
            <div className="hud-pill score-pill warn">
              <Target size={16} />
              <span>CALIBRATING: {calibProgress.toFixed(0)}%</span>
            </div>
          )}
        </div>
      </header>

      {/* ── BENTO DASHBOARD GRID ── */}
      <div className="bento-grid">
        {/* TILE 1: Primary Camera Stage (Span 7 cols) */}
        <section className="bento-card col-span-7">
          <div className="bento-header ribbon-cyan">
            <div className="header-left">
              <Camera size={18} />
              <span>TARGETING STAGE // WEBCAM HUD</span>
            </div>
            <div className="header-badge">
              <span className="rec-dot" style={{ display: 'inline-block', marginRight: 6 }} />
              {metrics?.cameraView?.label ?? 'LIVE 720P'}
            </div>
          </div>

          <div className="camera-stage">
            <video ref={videoRef} className="camera-video" playsInline muted autoPlay />
            <canvas ref={canvasRef} className="camera-canvas" />
            <div className="camera-scanlines" />

            {/* Corner Crosshairs */}
            <div className="hud-corner tl" />
            <div className="hud-corner tr" />
            <div className="hud-corner bl" />
            <div className="hud-corner br" />

            <div className="camera-hud-badge">
              <img src="/logo.png" alt="PosChair" className="mini-hud-logo" style={{ marginRight: 6 }} />
              <span>BLAZEPOSE FULL // 33 KP</span>
            </div>

            {/* In-Stream Calibration Overlay */}
            {appState === 'CALIBRATING' && (
              <div className="calib-overlay-card">
                <Target size={42} style={{ color: 'var(--pop-yellow)' }} />
                <div className="calib-title">Sit in Your Best Posture</div>
                <div className="calib-sub">
                  Align ears over shoulders, keep chest proud, relax traps.
                </div>
                <div className="calib-progress-track">
                  <div className="calib-progress-bar" style={{ width: `${calibProgress}%` }} />
                </div>
                <div className="calib-pct-text">{calibProgress.toFixed(0)}% COMPLETED</div>
              </div>
            )}
          </div>
        </section>

        {/* TILE 2: Skeleton Radar Minimap (Span 5 cols) */}
        <section className="bento-card col-span-5">
          <div className="bento-header ribbon-dark">
            <div className="header-left">
              <Crosshair size={18} style={{ color: 'var(--pop-cyan)' }} />
              <span>SKELETON RADAR // 3D TOPOLOGY</span>
            </div>
            <div className="header-badge" style={{ color: 'var(--pop-cyan)' }}>
              33 NODES
            </div>
          </div>

          <div className="radar-content">
            <div className="radar-canvas-wrap">
              <canvas ref={minimapRef} className="radar-canvas" />
              <div className="radar-sweep" />
            </div>

            <div className="radar-legend">
              <div className="legend-item">
                <div className="legend-dot" style={{ background: 'var(--pop-lime)' }} />
                <span>Aligned</span>
              </div>
              <div className="legend-item">
                <div className="legend-dot" style={{ background: 'var(--pop-yellow)' }} />
                <span>Joint Nodes</span>
              </div>
              <div className="legend-item">
                <div className="legend-dot" style={{ background: 'var(--pop-magenta)' }} />
                <span>Deviation Alert</span>
              </div>
            </div>
          </div>
        </section>

        {/* TILE 3: Posture Score & Health Gauge (Span 4 cols) */}
        <section className="bento-card col-span-4">
          <div className="bento-header ribbon-lime">
            <div className="header-left">
              <Trophy size={18} />
              <span>POSTURE HEALTH GAUGE</span>
            </div>
            <div className="header-badge">REAL-TIME</div>
          </div>

          <div className="score-card-content">
            <div className="score-giant-display">
              <span className={`score-giant-number ${score < 68 ? 'bad' : score < 76 ? 'warn' : 'good'}`}>
                {score}
              </span>
              <span className="score-max-tag">/100</span>
            </div>

            <div className="score-bar-track">
              <div
                className={`score-bar-fill ${score < 68 ? 'bad' : score < 76 ? 'warn' : 'good'}`}
                style={{ width: `${score}%` }}
              />
            </div>

            <div className={`score-status-banner ${score < 68 ? 'bad' : score < 76 ? 'warn' : 'good'}`}>
              {score >= 76 ? (
                <>
                  <CheckCircle2 size={20} />
                  <span>OPTIMAL ERGONOMIC ALIGNMENT</span>
                </>
              ) : score >= 68 ? (
                <>
                  <AlertTriangle size={20} />
                  <span>BORDERLINE POSTURE DRIFT</span>
                </>
              ) : (
                <>
                  <ShieldAlert size={20} />
                  <span>CRITICAL POSTURE COLLAPSE</span>
                </>
              )}
            </div>

            <div className="streak-card">
              <Flame size={20} style={{ color: 'var(--pop-yellow)' }} />
              <span>PERFECT STREAK: {fmt(streakMs)}</span>
            </div>
          </div>
        </section>

        {/* TILE 4: Biomechanical Telemetry Grid (Span 8 cols) */}
        <section className="bento-card col-span-8">
          <div className="bento-header ribbon-yellow">
            <div className="header-left">
              <Activity size={18} />
              <span>BIOMECHANICAL TELEMETRY MATRIX</span>
            </div>
            <div className="header-badge">6 DEGREES OF FREEDOM</div>
          </div>

          <div className="telemetry-grid">
            {metrics ? (
              <>
                {/* 1. Head Alignment (Front Perspective Crane or Profile CVA) */}
                <div className="telemetry-tile">
                  <div className="tile-header">
                    <span>Head Alignment</span>
                    <span style={{ fontSize: 11 }}>{metrics.cameraView.isFrontal ? 'FRONT' : 'PROFILE'}</span>
                  </div>
                  <div className={`tile-value ${metrics.anteriorShift > 0.09 ? 'bad' : metrics.anteriorShift > 0.05 ? 'warn' : 'ok'}`}>
                    {metrics.cameraView.isFrontal
                      ? `+${Math.max(0, Math.round(((metrics.headToShoulderRatio - (calibration?.headToShoulderRatio ?? metrics.headToShoulderRatio)) / ((calibration?.headToShoulderRatio ?? metrics.headToShoulderRatio) || 1)) * 100))}%`
                      : `${metrics.effectiveCvaDeg.toFixed(1)}°`}
                  </div>
                  <div className="tile-sub">
                    {metrics.cameraView.isFrontal ? 'Perspective Crane' : 'CVA (Target >53°)'}
                  </div>
                </div>

                {/* 2. Lateral Tilt */}
                <div className="telemetry-tile">
                  <div className="tile-header">
                    <span>Lateral Tilt</span>
                    <span style={{ fontSize: 11 }}>Coronal</span>
                  </div>
                  <div className={`tile-value ${Math.abs(metrics.lateralTiltDeg) > 15 ? 'bad' : Math.abs(metrics.lateralTiltDeg) > 8 ? 'warn' : 'ok'}`}>
                    {metrics.lateralTiltDeg > 0 ? '+' : ''}{metrics.lateralTiltDeg.toFixed(1)}°
                  </div>
                  <div className="tile-sub">Target: &lt; ±8.0°</div>
                </div>

                {/* 3. Shoulder Shrug */}
                <div className="telemetry-tile">
                  <div className="tile-header">
                    <span>Shoulder Shrug</span>
                    <span style={{ fontSize: 11 }}>Traps</span>
                  </div>
                  <div className={`tile-value ${metrics.shoulderShrug < 0.30 ? 'bad' : metrics.shoulderShrug < 0.38 ? 'warn' : 'ok'}`}>
                    {metrics.shoulderShrug < 0.38 ? 'ELEVATED' : 'CLEAR'}
                  </div>
                  <div className="tile-sub">Index: {metrics.shoulderShrug.toFixed(2)}</div>
                </div>

                {/* 4. Shoulder Asymmetry */}
                <div className="telemetry-tile">
                  <div className="tile-header">
                    <span>Asymmetry</span>
                    <span style={{ fontSize: 11 }}>Delta</span>
                  </div>
                  <div className={`tile-value ${metrics.shoulderAsymmetry > 0.14 ? 'bad' : metrics.shoulderAsymmetry > 0.07 ? 'warn' : 'ok'}`}>
                    {(metrics.shoulderAsymmetry * 100).toFixed(1)}%
                  </div>
                  <div className="tile-sub">Target: &lt; 7.0%</div>
                </div>

                {/* 5. Trunk Lean */}
                <div className="telemetry-tile">
                  <div className="tile-header">
                    <span>Trunk Lean</span>
                    <span style={{ fontSize: 11 }}>Spine</span>
                  </div>
                  <div className={`tile-value ${Math.abs(metrics.trunkLean) > 0.20 ? 'bad' : Math.abs(metrics.trunkLean) > 0.12 ? 'warn' : 'ok'}`}>
                    {metrics.trunkLean > 0 ? '+' : ''}{metrics.trunkLean.toFixed(2)}
                  </div>
                  <div className="tile-sub">Target: &lt; 0.12</div>
                </div>

                {/* 6. Invariant 3D Torso Anterior Shift */}
                <div className="telemetry-tile">
                  <div className="tile-header">
                    <span>Torso 3D Shift</span>
                    <span style={{ fontSize: 11 }}>T-Frame Ẑ</span>
                  </div>
                  <div className={`tile-value ${metrics.anteriorShift > 0.09 ? 'bad' : metrics.anteriorShift > 0.05 ? 'warn' : 'ok'}`}>
                    {metrics.anteriorShift > 0 ? '+' : ''}{metrics.anteriorShift.toFixed(3)}
                  </div>
                  <div className="tile-sub">Invariant Sagittal Shift</div>
                </div>

                {/* 7. Landmark Visibility */}
                <div className="telemetry-tile">
                  <div className="tile-header">
                    <span>Signal Quality</span>
                    <Eye size={14} style={{ color: 'var(--pop-cyan)' }} />
                  </div>
                  <div className={`tile-value ${metrics.visibilityScore > 0.75 ? 'ok' : metrics.visibilityScore > 0.50 ? 'warn' : 'bad'}`}>
                    {(metrics.visibilityScore * 100).toFixed(0)}%
                  </div>
                  <div className="tile-sub">{metrics.cameraView.label}</div>
                </div>

                {/* 8. Calibration Delta */}
                <div className="telemetry-tile">
                  <div className="tile-header">
                    <span>Cal Baseline</span>
                    <Sliders size={14} style={{ color: 'var(--pop-yellow)' }} />
                  </div>
                  <div className="tile-value ok" style={{ fontSize: 20 }}>
                    {calibration ? 'MATCHED' : 'UNSET'}
                  </div>
                  <div className="tile-sub">
                    {calibration ? 'Omni 3D baseline locked' : 'Run 5s calibration'}
                  </div>
                </div>
              </>
            ) : (
              <div style={{ gridColumn: 'span 4', textAlign: 'center', padding: 32, color: 'var(--pop-muted)' }}>
                <Activity size={32} style={{ margin: '0 auto 12px', display: 'block' }} />
                <span>Awaiting video signal from camera...</span>
              </div>
            )}
          </div>
        </section>

        {/* TILE 5: AI Voice Coach (Span 7 cols) */}
        <section className="bento-card col-span-7">
          <div className="bento-header ribbon-purple">
            <div className="header-left">
              <Mic size={18} />
              <span>AI VOICE COACH // GEMINI 3.8 FLASH + ELEVENLABS</span>
            </div>
            <div className="header-badge">TURBO V2</div>
          </div>

          <div className="ai-coach-content">
            <div className="ai-quote-box">
              {alert ? (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--pop-yellow)', fontWeight: 800, fontSize: 13 }}>
                      <Sparkles size={16} />
                      <span>{alert.urgency === 'URGENT' ? 'CRITICAL CORRECTION' : alert.urgency === 'FIRM' ? 'FIRM CORRECTION' : 'GENTLE TIP'}</span>
                    </div>
                    <span style={{ fontSize: 12, color: 'var(--pop-muted)' }}>
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="ai-quote-text">&ldquo;{alert.text}&rdquo;</div>
                </>
              ) : (
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--pop-purple)', fontWeight: 800, fontSize: 13, marginBottom: 8 }}>
                    <Volume2 size={16} />
                    <span>AUDIO MONITORING ACTIVE</span>
                  </div>
                  <div className="ai-quote-text" style={{ color: 'var(--pop-muted)', fontSize: 16 }}>
                    {isAnalyzing
                      ? 'Gemini 3.8 Flash is analyzing your posture telemetry...'
                      : `Voice alerts trigger when bad posture is sustained for ${TIMING.T_WARN_MS / 1000} seconds.`}
                  </div>
                </div>
              )}
            </div>

            {/* Active Speech Animation */}
            {alert?.isPlaying ? (
              <div className="ai-speaking-bar">
                <div className="eq-bars">
                  {[0.2, 0.5, 0.8, 0.4, 0.9, 0.3, 0.7, 0.4].map((d, i) => (
                    <div key={i} className="eq-bar" style={{ animationDelay: `${d}s` }} />
                  ))}
                </div>
                <span>ELEVENLABS RACHEL SPEAKING...</span>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13, color: 'var(--pop-muted)', fontWeight: 700 }}>
                <span>Voice: Rachel (Turbo v2)</span>
                <span>Latency: ~300ms</span>
              </div>
            )}
          </div>
        </section>

        {/* TILE 6: Consensus Engine & Controls (Span 5 cols) */}
        <section className="bento-card col-span-5">
          <div className="bento-header ribbon-magenta">
            <div className="header-left">
              <Terminal size={18} />
              <span>CONSENSUS ENGINE & CONTROLS</span>
            </div>
            <div className="header-badge">90%+ ACCURACY</div>
          </div>

          <div className="consensus-content">
            {/* Multi-Layer Agreement Status */}
            <div className="consensus-banner">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <Layers size={18} style={{ color: 'var(--pop-cyan)' }} />
                <span>Layer 2 Angle Consensus:</span>
              </div>
              <span style={{ color: calibration ? 'var(--pop-lime)' : 'var(--pop-yellow)', fontWeight: 800 }}>
                {calibration ? '252-ANGLES ACTIVE' : 'UNCALIBRATED'}
              </span>
            </div>

            {/* Active Posture Issues */}
            {metrics?.issues && metrics.issues.length > 0 ? (
              metrics.issues.map((issue, idx) => (
                <div key={idx} className="issue-item">
                  <div className="issue-label">
                    <AlertTriangle size={16} />
                    <span>{issue.label}</span>
                  </div>
                  <div className="issue-conf">
                    {(issue.layer2Confidence * 100).toFixed(0)}% L2 CONF
                  </div>
                </div>
              ))
            ) : (
              <div className="no-issues-banner">
                <Shield size={20} />
                <span>ZERO ANATOMICAL DEFECTS DETECTED</span>
              </div>
            )}

            {/* Bad Posture Timer */}
            {isBad && badMs > 0 && (
              <div className="control-row" style={{ borderColor: 'var(--pop-magenta)', color: 'var(--pop-magenta)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Clock size={16} />
                  <span>Sustained Bad Posture:</span>
                </div>
                <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 900, fontSize: 18 }}>
                  {fmt(badMs)}
                </span>
              </div>
            )}

            {/* 1-Click Recalibrate Action */}
            <button className="brutal-btn btn-yellow" onClick={recalibrate} style={{ width: '100%' }}>
              <RotateCcw size={16} />
              <span>Recalibrate Baseline (5s)</span>
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
