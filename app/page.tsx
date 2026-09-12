'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import {
  analyzePosture,
  buildAnalysisPrompt,
  getUrgencyLevel,
  type PostureMetrics,
  type PoseLandmark,
} from '@/lib/posture';

// MediaPipe connection pairs for skeleton drawing
const POSE_CONNECTIONS = [
  [7, 8],   // ear-ear
  [7, 11],  [8, 12],  // ear-shoulder
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // shoulders-hips
  [23, 24], // hips
];

type AppState = 'SPLASH' | 'LOADING' | 'ACTIVE' | 'NO_PERSON';

interface AlertData {
  text: string;
  urgency: 'GENTLE' | 'FIRM' | 'URGENT';
  isPlaying: boolean;
}

export default function PosChair() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);       // main camera
  const minimapRef = useRef<HTMLCanvasElement>(null);       // skeleton minimap
  const streamRef = useRef<MediaStream | null>(null);
  const poseLandmarkerRef = useRef<any>(null);
  const rafRef = useRef<number>(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const [appState, setAppState] = useState<AppState>('SPLASH');
  const [metrics, setMetrics] = useState<PostureMetrics | null>(null);
  const [alert, setAlert] = useState<AlertData | null>(null);
  const [badPostureStart, setBadPostureStart] = useState<number | null>(null);
  const [lastAnalysisTime, setLastAnalysisTime] = useState<number>(0);
  const [elapsedBadMs, setElapsedBadMs] = useState<number>(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [fps, setFps] = useState(0);

  const metricsRef = useRef<PostureMetrics | null>(null);
  const badPostureStartRef = useRef<number | null>(null);
  const lastAnalysisTimeRef = useRef<number>(0);
  const isAnalyzingRef = useRef(false);

  // Keep refs in sync with state
  metricsRef.current = metrics;
  badPostureStartRef.current = badPostureStart;
  lastAnalysisTimeRef.current = lastAnalysisTime;
  isAnalyzingRef.current = isAnalyzing;

  // ── Load MediaPipe ────────────────────────────────────────────────
  const loadMediaPipe = useCallback(async () => {
    setAppState('LOADING');
    try {
      const vision = await import('@mediapipe/tasks-vision');
      const { PoseLandmarker, FilesetResolver } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );

      const poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false,
      });

      poseLandmarkerRef.current = poseLandmarker;
      setAppState('ACTIVE');
    } catch (err) {
      console.error('MediaPipe load error:', err);
      // Retry with CPU
      try {
        const vision = await import('@mediapipe/tasks-vision');
        const { PoseLandmarker, FilesetResolver } = vision;
        const filesetResolver = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        const poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        poseLandmarkerRef.current = poseLandmarker;
        setAppState('ACTIVE');
      } catch (err2) {
        console.error('MediaPipe CPU fallback error:', err2);
      }
    }
  }, []);

  // ── Start Camera ──────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      await loadMediaPipe();
    } catch (err) {
      console.error('Camera error:', err);
    }
  }, [loadMediaPipe]);

  // ── AI Analysis ───────────────────────────────────────────────────
  const triggerAnalysis = useCallback(async (currentMetrics: PostureMetrics, badMs: number) => {
    if (isAnalyzingRef.current) return;
    isAnalyzingRef.current = true;
    setIsAnalyzing(true);

    try {
      const urgency = getUrgencyLevel(badMs);
      const prompt = buildAnalysisPrompt(currentMetrics, urgency);

      const analysisRes = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      if (!analysisRes.ok) throw new Error('Analysis failed');
      const { correction } = await analysisRes.json();

      setAlert({ text: correction, urgency, isPlaying: true });

      // TTS via ElevenLabs
      const speakRes = await fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: correction }),
      });

      if (speakRes.ok) {
        const audioBlob = await speakRes.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        if (audioRef.current) {
          audioRef.current.src = audioUrl;
          await audioRef.current.play();
          audioRef.current.onended = () => {
            setAlert(prev => prev ? { ...prev, isPlaying: false } : null);
            URL.revokeObjectURL(audioUrl);
            // Clear alert after 8 seconds of silence
            setTimeout(() => setAlert(null), 8000);
          };
        }
      }

      setLastAnalysisTime(Date.now());
      lastAnalysisTimeRef.current = Date.now();
    } catch (err) {
      console.error('Analysis error:', err);
    } finally {
      isAnalyzingRef.current = false;
      setIsAnalyzing(false);
    }
  }, []);

  // ── Draw Skeleton on Canvas ───────────────────────────────────────
  const drawSkeleton = useCallback((
    ctx: CanvasRenderingContext2D,
    landmarks: PoseLandmark[],
    w: number,
    h: number,
    isGood: boolean,
    isMinimap: boolean
  ) => {
    const lineColor = isGood ? '#00ff41' : '#ff0040';
    const dotColor = '#ffd700';
    const lineWidth = isMinimap ? 1.5 : 2.5;
    const dotRadius = isMinimap ? 3 : 5;

    // Draw connections
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = lineWidth;
    ctx.shadowColor = lineColor;
    ctx.shadowBlur = isMinimap ? 4 : 8;

    for (const [a, b] of POSE_CONNECTIONS) {
      const lmA = landmarks[a];
      const lmB = landmarks[b];
      if (!lmA || !lmB) continue;
      if ((lmA.visibility ?? 1) < 0.3 || (lmB.visibility ?? 1) < 0.3) continue;

      ctx.beginPath();
      ctx.moveTo(lmA.x * w, lmA.y * h);
      ctx.lineTo(lmB.x * w, lmB.y * h);
      ctx.stroke();
    }

    // Draw joints
    ctx.fillStyle = dotColor;
    ctx.shadowColor = dotColor;
    ctx.shadowBlur = isMinimap ? 6 : 12;

    const keyLandmarks = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24];
    for (const idx of keyLandmarks) {
      const lm = landmarks[idx];
      if (!lm || (lm.visibility ?? 1) < 0.3) continue;
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, dotRadius, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.shadowBlur = 0;
  }, []);

  // ── Main Render Loop ──────────────────────────────────────────────
  useEffect(() => {
    if (appState !== 'ACTIVE') return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const minimap = minimapRef.current;
    const landmarker = poseLandmarkerRef.current;

    if (!video || !canvas || !minimap || !landmarker) return;

    const ctx = canvas.getContext('2d')!;
    const mCtx = minimap.getContext('2d')!;

    let lastTs = 0;
    let frameCount = 0;
    let fpsTimer = 0;

    const render = (timestamp: number) => {
      if (video.readyState < 2) {
        rafRef.current = requestAnimationFrame(render);
        return;
      }

      // FPS counter
      frameCount++;
      if (timestamp - fpsTimer > 1000) {
        setFps(frameCount);
        frameCount = 0;
        fpsTimer = timestamp;
      }

      // Run pose detection
      const results = landmarker.detectForVideo(video, timestamp);
      const lms: PoseLandmark[] = results.landmarks?.[0] ?? [];

      // ── Main canvas ──
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      const W = canvas.width;
      const H = canvas.height;

      ctx.clearRect(0, 0, W, H);

      // ── Minimap canvas ──
      minimap.width = minimap.offsetWidth;
      minimap.height = minimap.offsetHeight;
      const mW = minimap.width;
      const mH = minimap.height;

      mCtx.fillStyle = '#000';
      mCtx.fillRect(0, 0, mW, mH);

      if (lms.length > 0) {
        setAppState('ACTIVE');

        // Analyze posture
        const newMetrics = analyzePosture(lms);
        if (newMetrics) {
          setMetrics(newMetrics);

          const isGood = newMetrics.isGoodPosture;

          // Draw on main camera
          drawSkeleton(ctx, lms, W, H, isGood, false);

          // Draw on minimap
          drawSkeleton(mCtx, lms, mW, mH, isGood, true);

          // Draw posture issue highlights on main canvas
          if (!isGood && newMetrics.issues.length > 0) {
            const shoulder11 = lms[11];
            const shoulder12 = lms[12];
            if (shoulder11 && shoulder12) {
              ctx.strokeStyle = 'rgba(255,0,64,0.4)';
              ctx.lineWidth = 1;
              ctx.setLineDash([4, 4]);
              ctx.strokeRect(
                shoulder12.x * W - 10,
                shoulder11.y * H - 30,
                (shoulder11.x - shoulder12.x) * W + 20,
                80
              );
              ctx.setLineDash([]);
            }
          }

          // ── Bad posture timer + trigger AI ──
          const now = Date.now();
          if (!isGood) {
            if (!badPostureStartRef.current) {
              setBadPostureStart(now);
              badPostureStartRef.current = now;
            }
            const badMs = now - badPostureStartRef.current;
            setElapsedBadMs(badMs);

            const cooldown = 45000; // 45s between alerts
            const shouldTrigger =
              badMs >= 30000 &&
              !isAnalyzingRef.current &&
              now - lastAnalysisTimeRef.current > cooldown;

            if (shouldTrigger) {
              triggerAnalysis(newMetrics, badMs);
            }
          } else {
            setBadPostureStart(null);
            badPostureStartRef.current = null;
            setElapsedBadMs(0);
          }
        }
      } else {
        // No person detected — draw grid on minimap
        mCtx.strokeStyle = '#1a5c22';
        mCtx.lineWidth = 0.5;
        for (let x = 0; x < mW; x += 20) {
          mCtx.beginPath();
          mCtx.moveTo(x, 0); mCtx.lineTo(x, mH); mCtx.stroke();
        }
        for (let y = 0; y < mH; y += 20) {
          mCtx.beginPath();
          mCtx.moveTo(0, y); mCtx.lineTo(mW, y); mCtx.stroke();
        }
        mCtx.fillStyle = '#1a5c22';
        mCtx.font = '10px monospace';
        mCtx.textAlign = 'center';
        mCtx.fillText('NO SIGNAL', mW / 2, mH / 2);
      }

      lastTs = timestamp;
      rafRef.current = requestAnimationFrame(render);
    };

    rafRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(rafRef.current);
  }, [appState, drawSkeleton, triggerAnalysis]);

  // ── Cleanup ───────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  // ── Score color helper ────────────────────────────────────────────
  const scoreColor = (score: number) => score >= 80 ? 'ok' : score >= 60 ? 'warn' : 'bad';
  const metricStatus = (val: number, good: boolean) => good ? 'ok' : val > 0 ? 'warn' : 'bad';

  const formatTimer = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    return `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  };

  const postureOk = metrics?.isGoodPosture ?? true;

  // ── Render ────────────────────────────────────────────────────────
  return (
    <div className="dashboard">
      {/* Hidden audio element */}
      <audio ref={audioRef} style={{ display: 'none' }} />

      {/* ── Splash Screen ── */}
      {appState === 'SPLASH' && (
        <div className="splash-screen">
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
            Real-time AI posture coach. Sit straight. Feel great.
            Voice corrections via ElevenLabs.
          </div>
          <button className="pixel-btn" onClick={startCamera} id="start-btn">
            ▶ INITIALIZE CAMERA
          </button>
          <div className="splash-subtitle" style={{ fontSize: '14px', color: 'var(--green-dim)' }}>
            Camera access required for pose detection
          </div>
        </div>
      )}

      {/* ── Loading Screen ── */}
      {appState === 'LOADING' && (
        <div className="splash-screen">
          <div className="splash-title">LOADING AI ENGINE</div>
          <div className="splash-subtitle">Initializing MediaPipe BlazePose...</div>
          <div className="loading-bar-container">
            <div className="loading-bar-fill" />
          </div>
          <div className="splash-subtitle" style={{ fontSize: '14px', color: 'var(--green-dim)' }}>
            Loading pose landmarker model...
          </div>
        </div>
      )}

      {/* ── Header Bar ── */}
      {appState === 'ACTIVE' && (
        <>
          <header className="header-bar">
            <div className="header-logo">
              🎮 POS<span>CHAIR</span>
              <span style={{ color: 'var(--green-dim)', marginLeft: 8 }}>v1.0</span>
            </div>
            <div className="header-status">
              <span className="score-badge">
                SCORE: {metrics?.postureScore ?? '--'}/100
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '16px', color: 'var(--green-dim)' }}>
                {fps}fps
              </span>
              <div className={`status-pill ${postureOk ? 'good' : badPostureStart ? 'bad' : 'idle'}`}>
                <div className="status-dot" />
                {postureOk ? 'GOOD' : badPostureStart ? 'BAD POSTURE' : 'IDLE'}
              </div>
            </div>
          </header>

          {/* ── Main Content ── */}
          <div className="main-content">
            {/* ── Camera (left) ── */}
            <section className="camera-section">
              <video ref={videoRef} className="camera-video" playsInline muted />
              <canvas ref={canvasRef} className="camera-canvas" />
              <div className="camera-label">CAM_01 // POSE TRACKING</div>
              <div className="camera-corner tl" />
              <div className="camera-corner tr" />
              <div className="camera-corner bl" />
              <div className="camera-corner br" />
            </section>

            {/* ── Right Panel ── */}
            <aside className="right-panel">

              {/* ── Skeleton Minimap ── */}
              <div className="minimap-section">
                <div className="section-header">SKELETON_WIREFRAME</div>
                <div className="minimap-canvas-wrapper">
                  <canvas ref={minimapRef} className="minimap-canvas" />
                  <div className="minimap-scanline" />
                </div>
              </div>

              {/* ── Live Metrics ── */}
              <div className="metrics-section">
                <div className="section-header">LIVE_POSTURE_DATA</div>

                {/* Score bar */}
                <div className="metric-bar-row">
                  <span className="metric-bar-label">POSTURE_SCORE</span>
                  <div className="metric-bar-track">
                    <div
                      className={`metric-bar-fill score-fill ${
                        (metrics?.postureScore ?? 100) < 50 ? 'low' :
                        (metrics?.postureScore ?? 100) < 70 ? 'mid' : ''
                      }`}
                      style={{ width: `${metrics?.postureScore ?? 0}%` }}
                    />
                  </div>
                  <span className="score-number">{metrics?.postureScore ?? '--'}</span>
                </div>

                {/* Metrics */}
                {metrics ? (
                  <>
                    <div className="metric-row">
                      <span className="metric-key">&gt; HEAD_TILT:</span>
                      <span className={`metric-value ${Math.abs(metrics.headTiltAngle) > 18 ? 'bad' : Math.abs(metrics.headTiltAngle) > 10 ? 'warn' : 'ok'}`}>
                        {metrics.headTiltAngle > 0 ? '+' : ''}{metrics.headTiltAngle.toFixed(1)}°
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; CHIN_RATIO:</span>
                      <span className={`metric-value ${metrics.chinNodRatio < 0.5 ? 'bad' : metrics.chinNodRatio < 0.7 ? 'warn' : 'ok'}`}>
                        {metrics.chinNodRatio.toFixed(2)}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; SHRUG:</span>
                      <span className={`metric-value ${metrics.shoulderShrug < 0.3 ? 'bad' : metrics.shoulderShrug < 0.4 ? 'warn' : 'ok'}`}>
                        {metrics.shoulderShrug < 0.4 ? `${(metrics.shoulderShrug).toFixed(2)} ⚠` : 'CLEAR'}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; ASYMMETRY:</span>
                      <span className={`metric-value ${metrics.shoulderAsymmetry > 0.15 ? 'bad' : metrics.shoulderAsymmetry > 0.08 ? 'warn' : 'ok'}`}>
                        {(metrics.shoulderAsymmetry * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; TRUNK_LEAN:</span>
                      <span className={`metric-value ${Math.abs(metrics.trunkLean) > 0.20 ? 'bad' : Math.abs(metrics.trunkLean) > 0.12 ? 'warn' : 'ok'}`}>
                        {metrics.trunkLean > 0 ? '+' : ''}{metrics.trunkLean.toFixed(2)}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-key">&gt; ISSUES:</span>
                      <span className={`metric-value ${metrics.issues.length > 0 ? 'bad' : 'ok'}`}>
                        {metrics.issues.length === 0 ? 'NONE' : metrics.issues.length}
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="metric-row" style={{ color: 'var(--green-dim)' }}>
                    <span>&gt; AWAITING SIGNAL...</span>
                    <span className="cursor-blink" />
                  </div>
                )}

                {/* Bad posture timer */}
                {badPostureStart && elapsedBadMs > 0 && (
                  <div className="timer-row">
                    ⚠ BAD_POSTURE: {formatTimer(elapsedBadMs)}
                    {isAnalyzing && <span style={{ color: 'var(--amber)' }}> [ANALYZING...]</span>}
                  </div>
                )}
              </div>

              {/* ── AI Alert Section ── */}
              <div className="alert-section">
                <div className="section-header">AI_COACH_OUTPUT</div>

                {alert ? (
                  <div className={`alert-box ${alert.urgency === 'URGENT' ? 'critical' : ''}`}>
                    <div className="alert-header">
                      <span>
                        {alert.urgency === 'URGENT' ? '🚨 URGENT' :
                         alert.urgency === 'FIRM' ? '⚠ CORRECTION' :
                         '💡 TIP'} // GEMINI 3.8 FLASH
                      </span>
                      <span style={{ color: 'var(--green-dim)', fontSize: '6px' }}>
                        {new Date().toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="alert-text">{alert.text}</div>
                    {alert.isPlaying && (
                      <div className="alert-playing">
                        <div className="playing-bars">
                          {[0.2, 0.4, 0.3, 0.5, 0.2, 0.4].map((d, i) => (
                            <div
                              key={i}
                              className="playing-bar"
                              style={{
                                height: `${6 + Math.random() * 6}px`,
                                '--delay': `${d}s`,
                              } as React.CSSProperties}
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
                      <span style={{ color: 'var(--amber)' }}>
                        ⏳ ANALYZING POSTURE...<span className="cursor-blink" />
                      </span>
                    ) : (
                      <span>
                        MONITORING ACTIVE
                        <br />
                        <span style={{ fontSize: '14px', color: 'var(--green-dim)' }}>
                          Alerts fire after 30s bad posture
                        </span>
                      </span>
                    )}
                  </div>
                )}
              </div>

            </aside>
          </div>
        </>
      )}
    </div>
  );
}
