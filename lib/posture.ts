/**
 * PosChair Posture Engine — Research-Backed v2
 * ─────────────────────────────────────────────
 * Based on:
 *  - ALIGN Framework (2026): 98.74% KNN accuracy, ear/shoulder/hip landmarks
 *  - PosePilot (2025): 3-point dot-product angle features, 97.52% accuracy
 *  - SitPose (2025): auxiliary geometry points for spinal curvature
 *  - Depth Camera PostureAx (2023): personalized calibration > fixed thresholds
 *
 * Key design decisions:
 *  1. 3-point dot-product angle calculation (NOT atan2 diff) — biomechanically correct
 *  2. MediaPipe z-coordinate for frontal FHP detection
 *  3. Calibration baseline — all thresholds relative to user's neutral
 *  4. Rolling-window temporal smoothing (One Euro Filter approximation)
 *  5. State machine with T_reset (5s good posture before clearing alert)
 */

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

// ── Calibration baseline (captured during 3-second neutral pose) ───────────
export interface CalibrationBaseline {
  headNeckShoulderAngle: number;  // nose→ear→shoulder, ~160-180° good
  lateralTiltDelta: number;       // earL.y - earR.y, should be ~0
  earToShoulderRatio: number;     // vertical distance normalized
  shoulderAsymmetry: number;      // abs(shL.y - shR.y) / width
  trunkLean: number;              // shoulder_mid.x - hip_mid.x
  zFhpDelta: number;              // ear.z - shoulder.z (frontal FHP)
  capturedAt: number;
}

export interface PostureMetrics {
  // Raw computed values
  headNeckShoulderAngle: number;  // degrees, 160-180° = good (3-point formula)
  lateralTiltDeg: number;         // head tilt in degrees, 0 = level
  shoulderShrug: number;          // ear-to-shoulder normalized ratio
  shoulderAsymmetry: number;      // 0-1, > 0.08 = uneven
  trunkLean: number;              // normalized horizontal offset
  zFhpDelta: number;              // z-depth FHP proxy (positive = head forward)

  // Deviation from calibration (if calibrated)
  deviations: Record<string, number>;

  // Output
  postureScore: number;           // 0-100
  issues: PostureIssue[];
  isGoodPosture: boolean;
  isCalibrated: boolean;
}

export interface PostureIssue {
  type: 'FORWARD_HEAD' | 'LATERAL_TILT' | 'SHOULDER_SHRUG' | 'SHOULDER_ASYMMETRY' | 'TRUNK_LEAN';
  severity: 'MILD' | 'MODERATE' | 'SEVERE';
  value: number;
  label: string;
  description: string;
  correctionHint: string;
}

// MediaPipe landmark indices
export const LM = {
  NOSE: 0,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
} as const;

// ── Core Math ────────────────────────────────────────────────────────────────

/**
 * 3-point angle at joint B, formed by segments B→A and B→C.
 * Uses dot product formula — matches PosePilot / ALIGN methodology.
 * Returns angle in degrees [0, 180].
 */
export function angle3pt(
  a: PoseLandmark,
  b: PoseLandmark,
  c: PoseLandmark,
  use3D = false
): number {
  if (use3D) {
    const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
    const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
    const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
    const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
    const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);
    if (magBA < 1e-8 || magBC < 1e-8) return 180;
    return Math.acos(Math.max(-1, Math.min(1, dot / (magBA * magBC)))) * (180 / Math.PI);
  } else {
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };
    const dot = ba.x * bc.x + ba.y * bc.y;
    const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2);
    const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2);
    if (magBA < 1e-8 || magBC < 1e-8) return 180;
    return Math.acos(Math.max(-1, Math.min(1, dot / (magBA * magBC)))) * (180 / Math.PI);
  }
}

function midpoint(a: PoseLandmark, b: PoseLandmark): PoseLandmark {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

function dist2D(a: PoseLandmark, b: PoseLandmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function angleDeg2D(a: PoseLandmark, b: PoseLandmark): number {
  return Math.atan2(b.y - a.y, b.x - a.x) * (180 / Math.PI);
}

// ── Rolling Window Smoother ──────────────────────────────────────────────────
export class RollingAverage {
  private buf: number[];
  private size: number;
  private idx = 0;
  private filled = false;

  constructor(windowSize = 10) {
    this.size = windowSize;
    this.buf = new Array(windowSize).fill(0);
  }

  update(val: number): number {
    this.buf[this.idx] = val;
    this.idx = (this.idx + 1) % this.size;
    if (this.idx === 0) this.filled = true;
    const count = this.filled ? this.size : this.idx;
    return this.buf.slice(0, count).reduce((a, b) => a + b, 0) / count;
  }

  get value(): number {
    const count = this.filled ? this.size : (this.idx || this.size);
    return this.buf.slice(0, count).reduce((a, b) => a + b, 0) / count;
  }

  reset() {
    this.buf = new Array(this.size).fill(0);
    this.idx = 0;
    this.filled = false;
  }
}

// Shared smoothers (singleton per session)
const smoothers = {
  headNeckShoulder: new RollingAverage(12),
  lateralTilt: new RollingAverage(12),
  shoulderShrug: new RollingAverage(8),
  shoulderAsym: new RollingAverage(8),
  trunkLean: new RollingAverage(8),
  zFhp: new RollingAverage(15),
};

// ── Calibration ──────────────────────────────────────────────────────────────

/**
 * Capture a calibration frame from the current landmark set.
 * Call this 30+ times over 3 seconds and average with captureCalibration().
 */
export function sampleCalibrationFrame(
  landmarks: PoseLandmark[]
): Partial<CalibrationBaseline> | null {
  const raw = extractRawMetrics(landmarks);
  if (!raw) return null;
  return {
    headNeckShoulderAngle: raw.headNeckShoulderAngle,
    lateralTiltDelta: raw.lateralTiltDelta,
    earToShoulderRatio: raw.earToShoulderRatio,
    shoulderAsymmetry: raw.shoulderAsymmetry,
    trunkLean: raw.trunkLean,
    zFhpDelta: raw.zFhpDelta,
  };
}

export function captureCalibration(
  samples: Partial<CalibrationBaseline>[]
): CalibrationBaseline {
  const avg = (key: keyof CalibrationBaseline) => {
    const vals = samples.map(s => s[key] as number).filter(v => !isNaN(v));
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  };
  return {
    headNeckShoulderAngle: avg('headNeckShoulderAngle'),
    lateralTiltDelta: avg('lateralTiltDelta'),
    earToShoulderRatio: avg('earToShoulderRatio'),
    shoulderAsymmetry: avg('shoulderAsymmetry'),
    trunkLean: avg('trunkLean'),
    zFhpDelta: avg('zFhpDelta'),
    capturedAt: Date.now(),
  };
}

// ── Raw Metric Extraction (unsmoothed) ───────────────────────────────────────
interface RawMetrics {
  headNeckShoulderAngle: number;
  lateralTiltDelta: number;
  earToShoulderRatio: number;
  shoulderAsymmetry: number;
  trunkLean: number;
  zFhpDelta: number;
  shoulderWidth: number;
}

function extractRawMetrics(landmarks: PoseLandmark[]): RawMetrics | null {
  if (!landmarks || landmarks.length < 25) return null;

  const nose = landmarks[LM.NOSE];
  const lEar = landmarks[LM.LEFT_EAR];
  const rEar = landmarks[LM.RIGHT_EAR];
  const lSh  = landmarks[LM.LEFT_SHOULDER];
  const rSh  = landmarks[LM.RIGHT_SHOULDER];
  const lHip = landmarks[LM.LEFT_HIP];
  const rHip = landmarks[LM.RIGHT_HIP];

  // Visibility gate — discard if key landmarks not visible enough
  const minVis = 0.4;
  if (
    (lEar.visibility ?? 1) < minVis ||
    (rEar.visibility ?? 1) < minVis ||
    (lSh.visibility  ?? 1) < minVis ||
    (rSh.visibility  ?? 1) < minVis
  ) return null;

  const earMid = midpoint(lEar, rEar);
  const shMid  = midpoint(lSh, rSh);
  const hipMid = midpoint(lHip, rHip);
  const sw     = dist2D(lSh, rSh);
  if (sw < 0.04) return null;

  // ── 1. Head-neck-shoulder angle (3-point dot product) ────────────────────
  // Angle at ear midpoint: nose → earMid → shoulderMid
  // Research: correct ~160-180°, forward head < 145°
  const headNeckShoulderAngle = angle3pt(nose, earMid, shMid, false);

  // ── 2. Lateral head tilt (ear line vs shoulder line angle delta) ─────────
  const earLineAngle = angleDeg2D(lEar, rEar);
  const shLineAngle  = angleDeg2D(lSh, rSh);
  const lateralTiltDelta = earLineAngle - shLineAngle;

  // ── 3. Ear-to-shoulder vertical distance ratio (shrug detection) ─────────
  // Research: normal ratio ~0.40-0.60 normalized by shoulder width
  const earToShoulderRatio = (shMid.y - earMid.y) / sw;

  // ── 4. Shoulder height asymmetry ─────────────────────────────────────────
  const shoulderAsymmetry = Math.abs(lSh.y - rSh.y) / sw;

  // ── 5. Trunk lateral lean ────────────────────────────────────────────────
  const trunkLean = (shMid.x - hipMid.x) / sw;

  // ── 6. Z-depth FHP proxy (MediaPipe relative depth) ─────────────────────
  // Research: If ear.z < shoulder.z, head is in front of shoulders (FHP)
  // z in MediaPipe: more negative = closer to camera
  const zFhpDelta = earMid.z - shMid.z; // negative = head forward of shoulders

  return {
    headNeckShoulderAngle,
    lateralTiltDelta,
    earToShoulderRatio,
    shoulderAsymmetry,
    trunkLean,
    zFhpDelta,
    shoulderWidth: sw,
  };
}

// ── Main Analysis Function ───────────────────────────────────────────────────

export function analyzePosture(
  landmarks: PoseLandmark[],
  calibration: CalibrationBaseline | null = null
): PostureMetrics | null {
  const raw = extractRawMetrics(landmarks);
  if (!raw) return null;

  // Apply temporal smoothing
  const hnsa  = smoothers.headNeckShoulder.update(raw.headNeckShoulderAngle);
  const tilt  = smoothers.lateralTilt.update(raw.lateralTiltDelta);
  const shrug = smoothers.shoulderShrug.update(raw.earToShoulderRatio);
  const asym  = smoothers.shoulderAsym.update(raw.shoulderAsymmetry);
  const lean  = smoothers.trunkLean.update(raw.trunkLean);
  const zFhp  = smoothers.zFhp.update(raw.zFhpDelta);

  const isCalibrated = !!calibration;

  // ── Compute deviations from calibrated baseline ──────────────────────────
  const deviations: Record<string, number> = {};
  if (calibration) {
    deviations.headNeckShoulder = calibration.headNeckShoulderAngle - hnsa; // positive = worse
    deviations.lateralTilt = Math.abs(tilt) - Math.abs(calibration.lateralTiltDelta);
    deviations.shoulderShrug = calibration.earToShoulderRatio - shrug;     // positive = worse
    deviations.shoulderAsym = asym - calibration.shoulderAsymmetry;
    deviations.trunkLean = Math.abs(lean) - Math.abs(calibration.trunkLean);
    deviations.zFhp = zFhp - calibration.zFhpDelta;                        // more negative = worse FHP
  }

  // ── Detect Issues ─────────────────────────────────────────────────────────
  const issues: PostureIssue[] = [];

  // Thresholds: research-backed with calibration-relative adjustments
  // If calibrated, use deviation thresholds; else use absolute thresholds

  // 1. Forward head posture (head-neck-shoulder angle)
  // Research: correct 160-180°, alert < 145°. PMC: correct 135.57° vs incorrect 126.52°
  const hnsaThreshMild   = calibration ? calibration.headNeckShoulderAngle - 12 : 148;
  const hnsaThreshMod    = calibration ? calibration.headNeckShoulderAngle - 20 : 140;
  const hnsaThreshSevere = calibration ? calibration.headNeckShoulderAngle - 30 : 130;

  if (hnsa < hnsaThreshMild) {
    issues.push({
      type: 'FORWARD_HEAD',
      severity: hnsa < hnsaThreshSevere ? 'SEVERE' : hnsa < hnsaThreshMod ? 'MODERATE' : 'MILD',
      value: hnsa,
      label: `HEAD_ANGLE: ${hnsa.toFixed(1)}°`,
      description: `Head pitched forward (${hnsa.toFixed(1)}° / target: ${hnsaThreshMild.toFixed(0)}°+)`,
      correctionHint: 'Draw chin back — bring your ears over your shoulders.',
    });
  }

  // Z-depth FHP confirmation (frontal camera — MediaPipe z heuristic)
  // If no angle issue but z strongly suggests FHP, add it
  const zFhpThresh = calibration ? calibration.zFhpDelta - 0.06 : -0.08;
  if (zFhp < zFhpThresh && !issues.find(i => i.type === 'FORWARD_HEAD')) {
    issues.push({
      type: 'FORWARD_HEAD',
      severity: zFhp < zFhpThresh - 0.04 ? 'MODERATE' : 'MILD',
      value: zFhp,
      label: `FHP_DEPTH: ${zFhp.toFixed(3)}`,
      description: 'Head detected in front of shoulder plane (depth sensor)',
      correctionHint: 'Sit back and align your ears over your shoulders.',
    });
  }

  // 2. Lateral head tilt
  // Research: < 2% frame height = ok, > 5% = alert → ~4-10° in degrees
  const tiltThreshMild   = calibration ? Math.abs(calibration.lateralTiltDelta) + 8  : 8;
  const tiltThreshMod    = calibration ? Math.abs(calibration.lateralTiltDelta) + 15 : 15;
  const tiltThreshSevere = calibration ? Math.abs(calibration.lateralTiltDelta) + 25 : 25;
  const absTilt = Math.abs(tilt);

  if (absTilt > tiltThreshMild) {
    issues.push({
      type: 'LATERAL_TILT',
      severity: absTilt > tiltThreshSevere ? 'SEVERE' : absTilt > tiltThreshMod ? 'MODERATE' : 'MILD',
      value: tilt,
      label: `TILT: ${tilt > 0 ? '+' : ''}${tilt.toFixed(1)}°`,
      description: `Head tilted ${tilt > 0 ? 'right' : 'left'} by ${absTilt.toFixed(1)}°`,
      correctionHint: `Level your head — your ${tilt > 0 ? 'right' : 'left'} ear is lower.`,
    });
  }

  // 3. Shoulder shrug (elevated shoulders)
  const shrugThreshMild   = calibration ? calibration.earToShoulderRatio - 0.08 : 0.38;
  const shrugThreshMod    = calibration ? calibration.earToShoulderRatio - 0.14 : 0.30;
  const shrugThreshSevere = calibration ? calibration.earToShoulderRatio - 0.22 : 0.22;

  if (shrug < shrugThreshMild) {
    issues.push({
      type: 'SHOULDER_SHRUG',
      severity: shrug < shrugThreshSevere ? 'SEVERE' : shrug < shrugThreshMod ? 'MODERATE' : 'MILD',
      value: shrug,
      label: `SHRUG: ${shrug.toFixed(2)}`,
      description: 'Shoulders elevated toward ears (stress shrug)',
      correctionHint: 'Drop your shoulders away from your ears. Relax your traps.',
    });
  }

  // 4. Shoulder asymmetry
  // Research: < 2% frame = ok, > 5% = alert
  const asymThreshMild   = calibration ? calibration.shoulderAsymmetry + 0.06 : 0.07;
  const asymThreshMod    = calibration ? calibration.shoulderAsymmetry + 0.12 : 0.14;
  const asymThreshSevere = calibration ? calibration.shoulderAsymmetry + 0.22 : 0.24;

  if (asym > asymThreshMild) {
    issues.push({
      type: 'SHOULDER_ASYMMETRY',
      severity: asym > asymThreshSevere ? 'SEVERE' : asym > asymThreshMod ? 'MODERATE' : 'MILD',
      value: asym,
      label: `ASYMMETRY: ${(asym * 100).toFixed(1)}%`,
      description: `${landmarks[LM.LEFT_SHOULDER].y > landmarks[LM.RIGHT_SHOULDER].y ? 'Left' : 'Right'} shoulder is lower`,
      correctionHint: 'Level your shoulders — keep them at the same height.',
    });
  }

  // 5. Trunk lateral lean
  const leanThreshMild   = calibration ? Math.abs(calibration.trunkLean) + 0.10 : 0.12;
  const leanThreshMod    = calibration ? Math.abs(calibration.trunkLean) + 0.18 : 0.20;
  const leanThreshSevere = calibration ? Math.abs(calibration.trunkLean) + 0.28 : 0.30;
  const absLean = Math.abs(lean);

  if (absLean > leanThreshMild) {
    issues.push({
      type: 'TRUNK_LEAN',
      severity: absLean > leanThreshSevere ? 'SEVERE' : absLean > leanThreshMod ? 'MODERATE' : 'MILD',
      value: lean,
      label: `LEAN: ${lean > 0 ? '+' : ''}${lean.toFixed(2)}`,
      description: `Trunk leaning ${lean > 0 ? 'right' : 'left'}`,
      correctionHint: 'Sit tall and centered. Shift your weight evenly on both sit bones.',
    });
  }

  // ── Posture Score ─────────────────────────────────────────────────────────
  // Weighted penalty system
  let score = 100;

  // FHP penalty (most important — 35% weight)
  const hnsaMin = calibration ? calibration.headNeckShoulderAngle - 35 : 125;
  const hnsaTarget = calibration ? calibration.headNeckShoulderAngle : 165;
  if (hnsa < hnsaTarget) {
    score -= Math.min(35, ((hnsaTarget - hnsa) / (hnsaTarget - hnsaMin)) * 35);
  }
  // Z-FHP extra penalty
  if (zFhp < zFhpThresh) {
    score -= Math.min(10, Math.abs(zFhp - zFhpThresh) * 100);
  }
  // Lateral tilt penalty (25% weight)
  if (absTilt > tiltThreshMild) {
    score -= Math.min(25, ((absTilt - tiltThreshMild) / 20) * 25);
  }
  // Shrug penalty (15%)
  if (shrug < shrugThreshMild) {
    score -= Math.min(15, ((shrugThreshMild - shrug) / 0.20) * 15);
  }
  // Asymmetry penalty (15%)
  if (asym > asymThreshMild) {
    score -= Math.min(15, ((asym - asymThreshMild) / 0.20) * 15);
  }
  // Lean penalty (10%)
  if (absLean > leanThreshMild) {
    score -= Math.min(10, ((absLean - leanThreshMild) / 0.20) * 10);
  }

  const postureScore = Math.max(0, Math.round(score));

  return {
    headNeckShoulderAngle: hnsa,
    lateralTiltDeg: tilt,
    shoulderShrug: shrug,
    shoulderAsymmetry: asym,
    trunkLean: lean,
    zFhpDelta: zFhp,
    deviations,
    postureScore,
    issues,
    isGoodPosture: postureScore >= 72 && issues.length === 0,
    isCalibrated,
  };
}

// ── State Machine ─────────────────────────────────────────────────────────────
/**
 * T_warn = 30s bad posture before first alert
 * T_reset = 5s good posture before clearing bad state (prevents false resets)
 * T_cooldown = 60s between alerts (prevents alarm fatigue — research recommendation)
 */
export const TIMING = {
  T_WARN_MS:     30_000,  // 30s bad posture → first alert
  T_RESET_MS:     5_000,  // 5s good posture → reset bad posture timer
  T_COOLDOWN_MS: 60_000,  // 60s minimum between alerts
} as const;

export function getUrgencyLevel(badPostureDurationMs: number): 'GENTLE' | 'FIRM' | 'URGENT' {
  if (badPostureDurationMs >= 90_000) return 'URGENT';
  if (badPostureDurationMs >= 60_000) return 'FIRM';
  return 'GENTLE';
}

export function buildAnalysisPrompt(
  metrics: PostureMetrics,
  urgency: string,
  badDurationMs: number
): string {
  const topIssue = metrics.issues[0];
  const allIssues = metrics.issues.map(i =>
    `${i.type}: ${i.description} [${i.severity}]`
  ).join('; ');

  const durationStr = badDurationMs >= 60_000
    ? `${Math.floor(badDurationMs / 60_000)} min ${Math.floor((badDurationMs % 60_000) / 1000)}s`
    : `${Math.floor(badDurationMs / 1000)} seconds`;

  const toneGuide = urgency === 'URGENT'
    ? 'Be direct and firm — the person has been ignoring this for a long time.'
    : urgency === 'FIRM'
    ? 'Be clear and specific. They need to correct this now.'
    : 'Be gentle and encouraging. First reminder.';

  return `You are PosChair, an AI posture coach for desk workers. The user has maintained poor posture for ${durationStr}.

Detected issues: ${allIssues || 'general poor posture'}
Primary issue: ${topIssue?.correctionHint || 'Sit up straight and align your head over your shoulders.'}
Posture score: ${metrics.postureScore}/100

Tone: ${toneGuide}

Write ONE specific, actionable voice correction in 1-2 short sentences (max 25 words). 
- Address the #1 issue directly by body part
- Give the exact corrective movement
- No filler words ("I notice", "It seems", "Try to")
- Sound human, not robotic
- Do not repeat from a previous correction — vary the phrasing`;
}

export function resetSmoothers() {
  Object.values(smoothers).forEach(s => s.reset());
}
