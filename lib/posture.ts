/**
 * PosChair Posture Engine — v4 (Omnidirectional Biomechanical Engine)
 * ────────────────────────────────────────────────────────────────────
 * Features:
 * 1. Camera Viewpoint Estimator: continuously classifies camera angle
 *    (Front-Level, Front-High, Front-Low, Diagonal-Left, Diagonal-Right, Profile-Left, Profile-Right)
 * 2. Camera-Invariant 3D Torso Reference Frame (T-Frame):
 *    Gram-Schmidt Orthonormal Basis (Lateral X̂, Spine Ŷ, Sagittal Ẑ).
 *    Forward head displacement is projected along Ẑ (anterior to user's chest)
 *    which is 100% invariant to camera yaw, pitch, and roll!
 * 3. Frontal Perspective Foreshortening & Cranio-Clavicular Fusion:
 *    Detects subtle forward craning from pure front & high-front angles via
 *    apparent head-to-shoulder ratio (R_head/sh), chin-clavicle clearance, and cervical pitch.
 * 4. Viewpoint-Adaptive Keypoint Synthesis:
 *    Never drops frames when viewed from side/diagonal angles where one ear is occluded.
 *    Dynamically switches to dominant-side sagittal tracking (Craniovertebral Angle - CVA).
 * 5. Layer 2: 252-angle consensus voting with multi-angle calibration.
 */

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export type CameraViewType =
  | 'FRONT_LEVEL'
  | 'FRONT_HIGH'
  | 'FRONT_LOW'
  | 'DIAGONAL_LEFT'
  | 'DIAGONAL_RIGHT'
  | 'PROFILE_LEFT'
  | 'PROFILE_RIGHT';

export interface CameraViewInfo {
  type: CameraViewType;
  yawDeg: number;       // approx horizontal angle relative to user's chest (-90 to +90)
  pitchDeg: number;     // approx camera elevation angle (-45 to +60)
  label: string;        // e.g. "FRONT [HIGH 28°]" or "DIAGONAL [45°]"
  dominantSide: 'LEFT' | 'RIGHT' | 'BILATERAL';
  isFrontal: boolean;
  isHighAngle: boolean;
}

export interface CalibrationBaseline {
  // 3D Torso Frame metrics
  anteriorShift: number;       // baseline head anterior offset relative to chest
  lateralShift: number;        // baseline lateral offset
  cranialHeight: number;       // baseline vertical clearance
  // Frontal perspective metrics
  headToShoulderRatio: number; // eye distance / shoulder width
  chinClavicleClearance: number;
  cervicalPitchDeg: number;
  // Sagittal / Profile metric
  effectiveCvaDeg: number;     // Craniovertebral Angle in degrees
  // Classical biomechanical metrics
  lateralTiltDelta: number;
  earToShoulderRatio: number;
  shoulderAsymmetry: number;
  trunkLean: number;
  // Layer 2: 252-angle feature vector
  featureAngles: number[];
  // Viewpoint metadata
  cameraView: CameraViewInfo;
  meanVisibility: number;
  capturedAt: number;
}

export interface PostureMetrics {
  // Viewpoint detection
  cameraView: CameraViewInfo;
  // Invariant 3D Torso Frame telemetry
  anteriorShift: number;       // head anterior offset (FHP in torso frame)
  lateralShift: number;
  cranialHeight: number;
  // Frontal perspective telemetry
  headToShoulderRatio: number;
  chinClavicleClearance: number;
  cervicalPitchDeg: number;
  // Profile telemetry
  effectiveCvaDeg: number;
  // Classical telemetry
  lateralTiltDeg: number;
  shoulderShrug: number;
  shoulderAsymmetry: number;
  trunkLean: number;
  // Layer 2 output
  consensusVotes: Record<string, number>;
  // Score & issues
  postureScore: number;
  visibilityScore: number;
  issues: PostureIssue[];
  isGoodPosture: boolean;
  isCalibrated: boolean;
  deviations: Record<string, number>;
}

export interface PostureIssue {
  type: 'FORWARD_HEAD' | 'LATERAL_TILT' | 'SHOULDER_SHRUG' | 'SHOULDER_ASYMMETRY' | 'TRUNK_LEAN';
  severity: 'MILD' | 'MODERATE' | 'SEVERE';
  layer2Confidence: number; // 0–1, from consensus vote
  value: number;
  label: string;
  problemName: string;
  fixAction: string;
  description: string;
  correctionHint: string;
}

// MediaPipe landmark indices used
export const LM = {
  NOSE: 0,
  LEFT_EYE: 2, RIGHT_EYE: 5,
  LEFT_EAR: 7, RIGHT_EAR: 8,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_HIP: 23, RIGHT_HIP: 24,
} as const;

const KEY_LM_IDS = [0, 7, 8, 11, 12, 13, 14, 23, 24]; // 9 key landmarks

// ── Layer 2: 252-angle Feature Specs ─────────────────────────────────────────
const ANGLE_SPECS: [number, number, number][] = (() => {
  const specs: [number, number, number][] = [];
  for (let vi = 0; vi < KEY_LM_IDS.length; vi++) {
    for (let ai = 0; ai < KEY_LM_IDS.length; ai++) {
      if (ai === vi) continue;
      for (let ci = ai + 1; ci < KEY_LM_IDS.length; ci++) {
        if (ci === vi) continue;
        specs.push([KEY_LM_IDS[ai], KEY_LM_IDS[vi], KEY_LM_IDS[ci]]);
      }
    }
  }
  return specs; // 9 * C(8,2) = 252 angles
})();

type IssueType = PostureIssue['type'];
const FHP_SET = new Set([0, 7, 8]);
const SH_SET  = new Set([11, 12]);
const HIP_SET = new Set([23, 24]);

function specSensitivity(a: number, v: number, c: number): IssueType[] {
  const pts = [a, v, c];
  const has = (s: Set<number>) => pts.some(p => s.has(p));
  const issues: IssueType[] = [];
  if (has(FHP_SET) && has(SH_SET)) issues.push('FORWARD_HEAD');
  if (pts.includes(7) && pts.includes(8)) issues.push('LATERAL_TILT');
  if ((pts.includes(7) !== pts.includes(8)) && has(SH_SET)) issues.push('LATERAL_TILT');
  if (has(FHP_SET) && has(SH_SET) && !has(HIP_SET)) issues.push('SHOULDER_SHRUG');
  if (pts.includes(11) && pts.includes(12)) issues.push('SHOULDER_ASYMMETRY');
  if (has(SH_SET) && has(HIP_SET)) issues.push('TRUNK_LEAN');
  return Array.from(new Set(issues));
}

const ISSUE_SPEC_MAP: Record<IssueType, number[]> = {
  FORWARD_HEAD: [], LATERAL_TILT: [], SHOULDER_SHRUG: [],
  SHOULDER_ASYMMETRY: [], TRUNK_LEAN: [],
};
ANGLE_SPECS.forEach(([a, v, c], idx) => {
  specSensitivity(a, v, c).forEach(issue => ISSUE_SPEC_MAP[issue].push(idx));
});

// ── Math & Geometry Helpers ──────────────────────────────────────────────────

export function angle3pt(
  a: PoseLandmark, b: PoseLandmark, c: PoseLandmark, use3D = true
): number {
  if (use3D) {
    const ba = [a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0)];
    const bc = [c.x - b.x, c.y - b.y, (c.z ?? 0) - (b.z ?? 0)];
    const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    const m = Math.sqrt((ba[0]**2 + ba[1]**2 + ba[2]**2) * (bc[0]**2 + bc[1]**2 + bc[2]**2));
    return m < 1e-8 ? 180 : Math.acos(Math.max(-1, Math.min(1, dot / m))) * (180 / Math.PI);
  }
  const ba = [a.x - b.x, a.y - b.y];
  const bc = [c.x - b.x, c.y - b.y];
  const dot = ba[0] * bc[0] + ba[1] * bc[1];
  const m = Math.sqrt((ba[0]**2 + ba[1]**2) * (bc[0]**2 + bc[1]**2));
  return m < 1e-8 ? 180 : Math.acos(Math.max(-1, Math.min(1, dot / m))) * (180 / Math.PI);
}

function mid(a: PoseLandmark, b: PoseLandmark): PoseLandmark {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z ?? 0) + (b.z ?? 0)) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
  };
}

function dist2D(a: PoseLandmark, b: PoseLandmark): number {
  return Math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2);
}

function dist3D(a: PoseLandmark, b: PoseLandmark): number {
  return Math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + ((a.z ?? 0) - (b.z ?? 0))**2);
}

// ── 1. Camera Viewpoint Estimator ─────────────────────────────────────────────

export function estimateCameraViewpoint(lms: PoseLandmark[]): CameraViewInfo {
  const nose = lms[0];
  const lEar = lms[7];
  const rEar = lms[8];
  const lSh  = lms[11];
  const rSh  = lms[12];
  const lHip = lms[23];
  const rHip = lms[24];

  const lEarVis = lEar?.visibility ?? 0;
  const rEarVis = rEar?.visibility ?? 0;

  // Determine dominant side
  let dominantSide: 'LEFT' | 'RIGHT' | 'BILATERAL' = 'BILATERAL';
  if (lEarVis < 0.35 && rEarVis >= 0.35) {
    dominantSide = 'RIGHT';
  } else if (rEarVis < 0.35 && lEarVis >= 0.35) {
    dominantSide = 'LEFT';
  }

  // Horizontal Yaw estimation
  let yawDeg = 0;
  if (lEar && rEar && nose && lEarVis >= 0.3 && rEarVis >= 0.3) {
    const distL = Math.hypot(nose.x - lEar.x, nose.y - lEar.y);
    const distR = Math.hypot(nose.x - rEar.x, nose.y - rEar.y);
    const ratio = (distR - distL) / Math.max(distR + distL, 1e-4);
    yawDeg = ratio * 70; // -70° to +70°
  } else if (dominantSide === 'RIGHT') {
    yawDeg = 65; // User turned left or camera on right
  } else if (dominantSide === 'LEFT') {
    yawDeg = -65;
  }

  // Pitch / Elevation estimation
  let pitchDeg = 0;
  if (lSh && rSh && lHip && rHip && nose) {
    const shMid = mid(lSh, rSh);
    const hipMid = mid(lHip, rHip);
    const dySpine = hipMid.y - shMid.y;
    const dzSpine = (hipMid.z ?? 0) - (shMid.z ?? 0);
    // When camera is high looking down, hips are further in z or dySpine foreshortens
    pitchDeg = Math.atan2(dzSpine, Math.max(dySpine, 0.05)) * (180 / Math.PI);
    // Refine with nose height above shoulders
    const noseShDelta = shMid.y - nose.y;
    if (noseShDelta < 0.18) {
      pitchDeg += 15; // Camera looking downward compresses vertical nose-shoulder distance
    }
  }

  const isFrontal = Math.abs(yawDeg) < 25;
  const isHighAngle = pitchDeg > 15;

  let type: CameraViewType = 'FRONT_LEVEL';
  let label = 'FRONT [EYE-LINE]';

  if (isFrontal) {
    if (pitchDeg > 18) {
      type = 'FRONT_HIGH';
      label = `FRONT-HIGH [${Math.min(60, Math.round(pitchDeg))}°]`;
    } else if (pitchDeg < -18) {
      type = 'FRONT_LOW';
      label = `FRONT-LOW [${Math.abs(Math.round(pitchDeg))}°]`;
    } else {
      type = 'FRONT_LEVEL';
      label = `FRONT [${Math.round(yawDeg)}°]`;
    }
  } else if (Math.abs(yawDeg) < 65) {
    if (yawDeg < 0) {
      type = 'DIAGONAL_LEFT';
      label = `DIAG-L [${Math.abs(Math.round(yawDeg))}°]`;
    } else {
      type = 'DIAGONAL_RIGHT';
      label = `DIAG-R [${Math.abs(Math.round(yawDeg))}°]`;
    }
  } else {
    if (yawDeg < 0) {
      type = 'PROFILE_LEFT';
      label = `PROFILE-L [${Math.abs(Math.round(yawDeg))}°]`;
    } else {
      type = 'PROFILE_RIGHT';
      label = `PROFILE-R [${Math.abs(Math.round(yawDeg))}°]`;
    }
  }

  return {
    type,
    yawDeg: Math.round(yawDeg),
    pitchDeg: Math.round(pitchDeg),
    label,
    dominantSide,
    isFrontal,
    isHighAngle,
  };
}

// ── 2. Camera-Invariant 3D Torso Coordinate Reference Frame (T-Frame) ─────────

interface RawBiometrics {
  cameraView: CameraViewInfo;
  // Invariant 3D Torso Frame
  anteriorShift: number;
  lateralShift: number;
  cranialHeight: number;
  // Frontal Perspective Metrics
  headToShoulderRatio: number;
  chinClavicleClearance: number;
  cervicalPitchDeg: number;
  // Profile Metric
  effectiveCvaDeg: number;
  // Classical Normalized Metrics
  lateralTiltDelta: number;
  earToShoulderRatio: number;
  shoulderAsymmetry: number;
  trunkLean: number;
  // Layer 2 Features
  featureAngles: number[];
  meanVisibility: number;
}

function extractRawBiometrics(lms: PoseLandmark[]): RawBiometrics | null {
  if (!lms || lms.length < 25) return null;

  const nose = lms[0];
  const lEye = lms[2], rEye = lms[5];
  const lEar = lms[7], rEar = lms[8];
  const lSh  = lms[11], rSh = lms[12];
  const lHip = lms[23], rHip = lms[24];

  // Robust visibility gate: requires at least ONE ear and at least ONE shoulder
  const earSeen = (lEar && (lEar.visibility ?? 1) >= 0.25) || (rEar && (rEar.visibility ?? 1) >= 0.25);
  const shSeen  = (lSh && (lSh.visibility ?? 1) >= 0.25) || (rSh && (rSh.visibility ?? 1) >= 0.25);
  if (!earSeen || !shSeen || !nose) return null;

  const cameraView = estimateCameraViewpoint(lms);

  // Determine effective head landmark (bilateral midpoint or dominant ear)
  let headPoint: PoseLandmark;
  if ((lEar?.visibility ?? 0) >= 0.35 && (rEar?.visibility ?? 0) >= 0.35) {
    headPoint = mid(lEar, rEar);
  } else if ((lEar?.visibility ?? 0) >= (rEar?.visibility ?? 0)) {
    headPoint = lEar;
  } else {
    headPoint = rEar;
  }

  // Determine effective shoulder center & width
  let shMid: PoseLandmark;
  let sw: number;
  if ((lSh?.visibility ?? 0) >= 0.3 && (rSh?.visibility ?? 0) >= 0.3) {
    shMid = mid(lSh, rSh);
    sw = Math.max(0.04, dist2D(lSh, rSh));
  } else if ((lSh?.visibility ?? 0) >= 0.3) {
    shMid = lSh;
    sw = 0.18; // Default fallback shoulder scale
  } else {
    shMid = rSh;
    sw = 0.18;
  }

  // Determine hip center
  const hipMid = (lHip && rHip) ? mid(lHip, rHip) : { x: shMid.x, y: shMid.y + 0.35, z: shMid.z };

  // ── Construct Orthonormal Torso Basis (X̂, Ŷ, Ẑ) ──────────────────────────
  // X̂: Coronal Lateral Axis (Right Shoulder → Left Shoulder)
  let vx = [1, 0, 0];
  if (lSh && rSh && (lSh.visibility ?? 0) >= 0.3 && (rSh.visibility ?? 0) >= 0.3) {
    vx = [lSh.x - rSh.x, lSh.y - rSh.y, (lSh.z ?? 0) - (rSh.z ?? 0)];
  }
  const normX = Math.hypot(vx[0], vx[1], vx[2]) || 1;
  const X_hat = [vx[0] / normX, vx[1] / normX, vx[2] / normX];

  // Ŷ: Longitudinal Spine Axis (Mid-Hips → Mid-Shoulders)
  const vy_raw = [shMid.x - hipMid.x, shMid.y - hipMid.y, (shMid.z ?? 0) - (hipMid.z ?? 0)];
  // Gram-Schmidt orthogonalize Ŷ against X̂
  const dot_yx = vy_raw[0] * X_hat[0] + vy_raw[1] * X_hat[1] + vy_raw[2] * X_hat[2];
  const vy = [
    vy_raw[0] - dot_yx * X_hat[0],
    vy_raw[1] - dot_yx * X_hat[1],
    vy_raw[2] - dot_yx * X_hat[2],
  ];
  const normY = Math.hypot(vy[0], vy[1], vy[2]) || 1;
  const Y_hat = [vy[0] / normY, vy[1] / normY, vy[2] / normY];

  // Ẑ: Sagittal Axis (X̂ × Ŷ) — points strictly anteriorly (forward from chest)
  let Z_hat = [
    X_hat[1] * Y_hat[2] - X_hat[2] * Y_hat[1],
    X_hat[2] * Y_hat[0] - X_hat[0] * Y_hat[2],
    X_hat[0] * Y_hat[1] - X_hat[1] * Y_hat[0],
  ];
  // Ensure Ẑ points anteriorly toward nose
  const noseDisp = [nose.x - shMid.x, nose.y - shMid.y, (nose.z ?? 0) - (shMid.z ?? 0)];
  const noseDotZ = noseDisp[0] * Z_hat[0] + noseDisp[1] * Z_hat[1] + noseDisp[2] * Z_hat[2];
  if (noseDotZ < 0) {
    Z_hat = [-Z_hat[0], -Z_hat[1], -Z_hat[2]];
  }

  // ── Project Head Displacement onto Torso Basis ───────────────────────────
  const headDisp = [headPoint.x - shMid.x, headPoint.y - shMid.y, (headPoint.z ?? 0) - (shMid.z ?? 0)];
  // Invariant Anterior Shift (FHP along chest normal, normalized by shoulder width)
  const anteriorShift = (headDisp[0] * Z_hat[0] + headDisp[1] * Z_hat[1] + headDisp[2] * Z_hat[2]) / sw;
  // Invariant Lateral Shift
  const lateralShift  = (headDisp[0] * X_hat[0] + headDisp[1] * X_hat[1] + headDisp[2] * X_hat[2]) / sw;
  // Invariant Cranial Height
  const cranialHeight = (headDisp[0] * Y_hat[0] + headDisp[1] * Y_hat[1] + headDisp[2] * Y_hat[2]) / sw;

  // ── Frontal Perspective Foreshortening Metrics ───────────────────────────
  let headToShoulderRatio = 0.28;
  if (lEye && rEye && (lEye.visibility ?? 0) >= 0.3 && (rEye.visibility ?? 0) >= 0.3) {
    const eyeDist = Math.hypot(lEye.x - rEye.x, lEye.y - rEye.y);
    headToShoulderRatio = eyeDist / sw;
  }

  const chinClavicleClearance = (shMid.y - nose.y) / sw;

  let cervicalPitchDeg = 0;
  if (lEar && lEye && (lEar.visibility ?? 0) >= 0.3) {
    cervicalPitchDeg = Math.atan2(lEar.y - lEye.y, Math.abs(lEar.x - lEye.x)) * (180 / Math.PI);
  } else if (rEar && rEye && (rEar.visibility ?? 0) >= 0.3) {
    cervicalPitchDeg = Math.atan2(rEar.y - rEye.y, Math.abs(rEar.x - rEye.x)) * (180 / Math.PI);
  }

  // ── Profile / Diagonal Sagittal Metric (CVA) ────────────────────────────
  let effectiveCvaDeg = 55.0; // Normal upright resting CVA is ~55°
  const activeEar = cameraView.dominantSide === 'RIGHT' ? rEar : lEar;
  const activeSh  = cameraView.dominantSide === 'RIGHT' ? rSh  : lSh;
  if (activeEar && activeSh) {
    const dx = activeEar.x - activeSh.x;
    const dy = activeSh.y - activeEar.y; // positive upward
    effectiveCvaDeg = Math.atan2(dy, Math.abs(dx) || 1e-4) * (180 / Math.PI);
  }

  // ── Classical Metrics ───────────────────────────────────────────────────
  let lateralTiltDelta = 0;
  if (lEar && rEar && lSh && rSh) {
    lateralTiltDelta = (Math.atan2(rEar.y - lEar.y, rEar.x - lEar.x) * 180 / Math.PI)
                     - (Math.atan2(rSh.y - lSh.y, rSh.x - lSh.x) * 180 / Math.PI);
  } else {
    lateralTiltDelta = lateralShift * 40; // Fallback to 3D lateral shift
  }

  const earToShoulderRatio = (shMid.y - headPoint.y) / sw;
  const shoulderAsymmetry  = (lSh && rSh) ? Math.abs(lSh.y - rSh.y) / sw : 0;
  const trunkLean          = (shMid.x - hipMid.x) / sw;

  // Feature vector for Layer 2 consensus
  const featureAngles = ANGLE_SPECS.map(([a, v, c]) =>
    angle3pt(lms[a] ?? nose, lms[v] ?? nose, lms[c] ?? nose, true)
  );

  const meanVis = KEY_LM_IDS
    .map(id => lms[id]?.visibility ?? 0.5)
    .reduce((a, b) => a + b, 0) / KEY_LM_IDS.length;

  return {
    cameraView,
    anteriorShift,
    lateralShift,
    cranialHeight,
    headToShoulderRatio,
    chinClavicleClearance,
    cervicalPitchDeg,
    effectiveCvaDeg,
    lateralTiltDelta,
    earToShoulderRatio,
    shoulderAsymmetry,
    trunkLean,
    featureAngles,
    meanVisibility: meanVis,
  };
}

// ── 3. Rolling Temporal Smoothers ─────────────────────────────────────────────

export class RollingAverage {
  private buf: number[];
  private size: number;
  private idx = 0;
  private filled = false;

  constructor(w = 10) {
    this.size = w;
    this.buf = new Array(w).fill(0);
  }

  update(v: number): number {
    this.buf[this.idx] = v;
    this.idx = (this.idx + 1) % this.size;
    if (this.idx === 0) this.filled = true;
    const n = this.filled ? this.size : this.idx;
    return this.buf.slice(0, n).reduce((a, b) => a + b, 0) / n;
  }

  reset() {
    this.buf = new Array(this.size).fill(0);
    this.idx = 0;
    this.filled = false;
  }
}

const smoothers = {
  anteriorShift:       new RollingAverage(10),
  lateralShift:        new RollingAverage(10),
  cranialHeight:       new RollingAverage(10),
  headToShoulderRatio: new RollingAverage(12),
  chinClavicle:        new RollingAverage(12),
  cvaDeg:              new RollingAverage(10),
  lateralTilt:         new RollingAverage(10),
  shoulderShrug:       new RollingAverage(8),
  shoulderAsym:        new RollingAverage(8),
  trunkLean:           new RollingAverage(8),
};

// ── 4. Multi-Angle Calibration Sampling ───────────────────────────────────────

export function sampleCalibrationFrame(lms: PoseLandmark[]): Partial<CalibrationBaseline> | null {
  const r = extractRawBiometrics(lms);
  if (!r) return null;
  return {
    anteriorShift: r.anteriorShift,
    lateralShift: r.lateralShift,
    cranialHeight: r.cranialHeight,
    headToShoulderRatio: r.headToShoulderRatio,
    chinClavicleClearance: r.chinClavicleClearance,
    cervicalPitchDeg: r.cervicalPitchDeg,
    effectiveCvaDeg: r.effectiveCvaDeg,
    lateralTiltDelta: r.lateralTiltDelta,
    earToShoulderRatio: r.earToShoulderRatio,
    shoulderAsymmetry: r.shoulderAsymmetry,
    trunkLean: r.trunkLean,
    featureAngles: r.featureAngles,
    cameraView: r.cameraView,
    meanVisibility: r.meanVisibility,
  };
}

export function captureCalibration(samples: Partial<CalibrationBaseline>[]): CalibrationBaseline {
  const valid = samples.filter(s => s.featureAngles && s.featureAngles.length === ANGLE_SPECS.length);
  const avgScalar = (key: keyof CalibrationBaseline) => {
    const vals = valid.map(s => s[key] as number).filter(v => typeof v === 'number' && !isNaN(v));
    return vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
  };

  const featureAngles = Array.from({ length: ANGLE_SPECS.length }, (_, i) => {
    const vals = valid.map(s => (s.featureAngles as number[])[i]).filter(v => !isNaN(v));
    return vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
  });

  const lastView = valid[valid.length - 1]?.cameraView ?? {
    type: 'FRONT_LEVEL',
    yawDeg: 0,
    pitchDeg: 0,
    label: 'FRONT [EYE-LINE]',
    dominantSide: 'BILATERAL',
    isFrontal: true,
    isHighAngle: false,
  };

  return {
    anteriorShift: avgScalar('anteriorShift'),
    lateralShift: avgScalar('lateralShift'),
    cranialHeight: avgScalar('cranialHeight'),
    headToShoulderRatio: avgScalar('headToShoulderRatio'),
    chinClavicleClearance: avgScalar('chinClavicleClearance'),
    cervicalPitchDeg: avgScalar('cervicalPitchDeg'),
    effectiveCvaDeg: avgScalar('effectiveCvaDeg'),
    lateralTiltDelta: avgScalar('lateralTiltDelta'),
    earToShoulderRatio: avgScalar('earToShoulderRatio'),
    shoulderAsymmetry: avgScalar('shoulderAsymmetry'),
    trunkLean: avgScalar('trunkLean'),
    featureAngles,
    cameraView: lastView,
    meanVisibility: avgScalar('meanVisibility'),
    capturedAt: Date.now(),
  };
}

// ── 5. Consensus Voting ───────────────────────────────────────────────────────

function consensusVote(
  current: number[],
  baseline: number[],
  issue: IssueType,
  thresholdDeg = 8.5
): number {
  const indices = ISSUE_SPEC_MAP[issue];
  if (indices.length === 0) return 0;
  let deviated = 0;
  for (const idx of indices) {
    if (Math.abs(current[idx] - baseline[idx]) > thresholdDeg) deviated++;
  }
  return deviated / indices.length;
}

// ── 6. Main Omnidirectional Posture Analysis ──────────────────────────────────

export const SCORE_ENTER_BAD = 68;
export const SCORE_EXIT_BAD  = 76;
const L2_CONFIRM_THRESHOLD   = 0.22;
const L2_SUPPRESS_THRESHOLD  = 0.08;

export function analyzePosture(
  lms: PoseLandmark[],
  calibration: CalibrationBaseline | null = null
): PostureMetrics | null {
  const raw = extractRawBiometrics(lms);
  if (!raw) return null;

  // Temporal smoothing
  const antShift = smoothers.anteriorShift.update(raw.anteriorShift);
  const latShift = smoothers.lateralShift.update(raw.lateralShift);
  const cranH    = smoothers.cranialHeight.update(raw.cranialHeight);
  const h2sRatio = smoothers.headToShoulderRatio.update(raw.headToShoulderRatio);
  const chinClav = smoothers.chinClavicle.update(raw.chinClavicleClearance);
  const cva      = smoothers.cvaDeg.update(raw.effectiveCvaDeg);
  const tilt     = smoothers.lateralTilt.update(raw.lateralTiltDelta);
  const shrug    = smoothers.shoulderShrug.update(raw.earToShoulderRatio);
  const asym     = smoothers.shoulderAsym.update(raw.shoulderAsymmetry);
  const lean     = smoothers.trunkLean.update(raw.trunkLean);

  // Layer 2: consensus votes
  const ISSUE_TYPES: IssueType[] = [
    'FORWARD_HEAD', 'LATERAL_TILT', 'SHOULDER_SHRUG', 'SHOULDER_ASYMMETRY', 'TRUNK_LEAN',
  ];
  const consensusVotes: Record<string, number> = {};
  if (calibration?.featureAngles?.length === ANGLE_SPECS.length) {
    for (const issue of ISSUE_TYPES) {
      consensusVotes[issue] = consensusVote(raw.featureAngles, calibration.featureAngles, issue);
    }
  } else {
    ISSUE_TYPES.forEach(i => { consensusVotes[i] = 0.5; });
  }

  // Deviations from calibrated baseline
  const deviations: Record<string, number> = {};
  if (calibration) {
    deviations.anteriorShift = antShift - calibration.anteriorShift;
    deviations.lateralShift = Math.abs(latShift) - Math.abs(calibration.lateralShift);
    deviations.headToShoulderRatio = h2sRatio - calibration.headToShoulderRatio;
    deviations.chinClavicle = calibration.chinClavicleClearance - chinClav;
    deviations.cva = calibration.effectiveCvaDeg - cva;
    deviations.lateralTilt = Math.abs(tilt) - Math.abs(calibration.lateralTiltDelta);
    deviations.shoulderShrug = calibration.earToShoulderRatio - shrug;
    deviations.shoulderAsym = asym - calibration.shoulderAsymmetry;
    deviations.trunkLean = Math.abs(lean) - Math.abs(calibration.trunkLean);
  }

  const visibilityScore = raw.meanVisibility;
  const issues: PostureIssue[] = [];

  const shouldFlag = (issue: IssueType): boolean => {
    const l2 = consensusVotes[issue] ?? 0.5;
    if (l2 < L2_SUPPRESS_THRESHOLD) return false;
    return true;
  };

  // ── Omnidirectional Issue Evaluation ───────────────────────────────────────

  // 1. Forward Head Posture (Evaluated via Torso-Frame Ẑ + Front Perspective + Profile CVA)
  let fhpDetected = false;
  let fhpSeverity: 'MILD' | 'MODERATE' | 'SEVERE' = 'MILD';
  let fhpLabel = '';
  let fhpDesc = '';

  // Signal A: Invariant 3D Torso Frame Anterior Shift
  const antDeltaThreshMild = calibration ? 0.045 : 0.06;
  const antDelta = calibration ? (antShift - calibration.anteriorShift) : antShift;

  // Signal B: Frontal Perspective Foreshortening Expansion (Active in Front & Front-High views)
  const h2sExpansion = calibration ? (h2sRatio - calibration.headToShoulderRatio) / calibration.headToShoulderRatio : 0;
  const isFrontalCrane = raw.cameraView.isFrontal && h2sExpansion > 0.12;

  // Signal C: Profile / Diagonal Craniovertebral Flexion
  const cvaDrop = calibration ? (calibration.effectiveCvaDeg - cva) : (52 - cva);
  const isProfileDrop = !raw.cameraView.isFrontal && cvaDrop > 7.0;

  if ((antDelta > antDeltaThreshMild || isFrontalCrane || isProfileDrop) && shouldFlag('FORWARD_HEAD')) {
    fhpDetected = true;
    if (antDelta > antDeltaThreshMild * 2.0 || h2sExpansion > 0.25 || cvaDrop > 14) {
      fhpSeverity = 'SEVERE';
    } else if (antDelta > antDeltaThreshMild * 1.4 || h2sExpansion > 0.18 || cvaDrop > 10) {
      fhpSeverity = 'MODERATE';
    }

    if (raw.cameraView.isFrontal) {
      fhpLabel = `FHP CRANE: +${(h2sExpansion * 100).toFixed(0)}%`;
      fhpDesc = `Head craning forward toward screen (perspective expansion +${(h2sExpansion * 100).toFixed(0)}%)`;
    } else {
      fhpLabel = `CVA DROP: -${cvaDrop.toFixed(1)}°`;
      fhpDesc = `Craniovertebral angle flexed downward (${cva.toFixed(1)}° vs baseline ${calibration?.effectiveCvaDeg.toFixed(1) ?? '55'}°)`;
    }

    issues.push({
      type: 'FORWARD_HEAD',
      severity: fhpSeverity,
      layer2Confidence: consensusVotes['FORWARD_HEAD'] ?? 0.5,
      value: antDelta,
      label: fhpLabel,
      problemName: 'Forward head crane',
      fixAction: 'Draw your chin back and align your ears over your shoulders.',
      description: fhpDesc,
      correctionHint: 'Draw your chin back and align your ears over your shoulders.',
    });
  }

  // 2. Lateral Head Tilt
  const tiltThreshMild = calibration ? Math.abs(calibration.lateralTiltDelta) + 8 : 8;
  const absTilt = Math.abs(tilt);
  if (absTilt > tiltThreshMild && shouldFlag('LATERAL_TILT')) {
    const isRight = tilt > 0;
    issues.push({
      type: 'LATERAL_TILT',
      severity: absTilt > tiltThreshMild + 15 ? 'SEVERE' : absTilt > tiltThreshMild + 7 ? 'MODERATE' : 'MILD',
      layer2Confidence: consensusVotes['LATERAL_TILT'] ?? 0.5,
      value: tilt,
      label: `TILT: ${tilt > 0 ? '+' : ''}${tilt.toFixed(1)}°`,
      problemName: `Head tilt to the ${isRight ? 'right' : 'left'}`,
      fixAction: 'Keep your neck straight and level your head.',
      description: `Head tilted ${isRight ? 'right' : 'left'} by ${absTilt.toFixed(1)}°`,
      correctionHint: `Level your head — ${isRight ? 'right' : 'left'} ear is tilted downward.`,
    });
  }

  // 3. Shoulder Shrug / Elevation
  const shrugThreshMild = calibration ? calibration.earToShoulderRatio - 0.08 : 0.40;
  if (shrug < shrugThreshMild && shouldFlag('SHOULDER_SHRUG')) {
    issues.push({
      type: 'SHOULDER_SHRUG',
      severity: shrug < shrugThreshMild - 0.10 ? 'SEVERE' : 'MODERATE',
      layer2Confidence: consensusVotes['SHOULDER_SHRUG'] ?? 0.5,
      value: shrug,
      label: `SHRUG: ${shrug.toFixed(2)}`,
      problemName: 'Elevated shoulders',
      fixAction: 'Drop your shoulders away from your ears and relax your traps.',
      description: 'Shoulders elevated toward ears (stress shrug)',
      correctionHint: 'Drop your shoulders away from your ears. Relax your traps.',
    });
  }

  // 4. Shoulder Asymmetry
  const asymThreshMild = calibration ? calibration.shoulderAsymmetry + 0.06 : 0.08;
  if (asym > asymThreshMild && shouldFlag('SHOULDER_ASYMMETRY')) {
    issues.push({
      type: 'SHOULDER_ASYMMETRY',
      severity: asym > asymThreshMild + 0.08 ? 'SEVERE' : 'MODERATE',
      layer2Confidence: consensusVotes['SHOULDER_ASYMMETRY'] ?? 0.5,
      value: asym,
      label: `ASYM: ${(asym * 100).toFixed(1)}%`,
      problemName: 'Uneven shoulders',
      fixAction: 'Level your shoulders to equal height.',
      description: 'Shoulder line tilted unevenly',
      correctionHint: 'Level your shoulders — keep them at equal height.',
    });
  }

  // 5. Trunk Lean
  const absLean = Math.abs(lean);
  const leanThreshMild = calibration ? Math.abs(calibration.trunkLean) + 0.10 : 0.12;
  if (absLean > leanThreshMild && shouldFlag('TRUNK_LEAN')) {
    const isRight = lean > 0;
    issues.push({
      type: 'TRUNK_LEAN',
      severity: absLean > leanThreshMild + 0.08 ? 'SEVERE' : 'MODERATE',
      layer2Confidence: consensusVotes['TRUNK_LEAN'] ?? 0.5,
      value: lean,
      label: `LEAN: ${lean > 0 ? '+' : ''}${lean.toFixed(2)}`,
      problemName: `Torso leaning to the ${isRight ? 'right' : 'left'}`,
      fixAction: 'Sit tall and center your weight evenly on both hips.',
      description: `Torso leaning ${isRight ? 'right' : 'left'}`,
      correctionHint: 'Sit tall and centered with even weight on both hips.',
    });
  }

  // ── Visibility-Weighted Posture Score (0–100) ──────────────────────────────
  let score = 100;

  // FHP deduction (calibrated to viewpoint)
  if (fhpDetected) {
    const penalty = fhpSeverity === 'SEVERE' ? 38 : fhpSeverity === 'MODERATE' ? 24 : 14;
    score -= penalty * Math.min(1.0, visibilityScore + 0.2);
  }

  if (absTilt > tiltThreshMild) {
    const penalty = Math.min(22, ((absTilt - tiltThreshMild) / 18) * 22);
    score -= penalty;
  }

  if (shrug < shrugThreshMild) {
    const penalty = Math.min(18, ((shrugThreshMild - shrug) / 0.15) * 18);
    score -= penalty;
  }

  if (asym > asymThreshMild) {
    const penalty = Math.min(16, ((asym - asymThreshMild) / 0.14) * 16);
    score -= penalty;
  }

  if (absLean > leanThreshMild) {
    const penalty = Math.min(14, ((absLean - leanThreshMild) / 0.15) * 14);
    score -= penalty;
  }

  // Layer 2 bonus: if consensus votes strongly confirm good posture
  const avgGoodVote = ISSUE_TYPES.reduce((s, t) => s + (1 - (consensusVotes[t] ?? 0.5)), 0) / ISSUE_TYPES.length;
  if (avgGoodVote > 0.65) {
    score = Math.min(100, score + (avgGoodVote - 0.65) * 14);
  }

  const postureScore = Math.max(0, Math.round(score));

  return {
    cameraView: raw.cameraView,
    anteriorShift: antShift,
    lateralShift: latShift,
    cranialHeight: cranH,
    headToShoulderRatio: h2sRatio,
    chinClavicleClearance: chinClav,
    cervicalPitchDeg: raw.cervicalPitchDeg,
    effectiveCvaDeg: cva,
    lateralTiltDeg: tilt,
    shoulderShrug: shrug,
    shoulderAsymmetry: asym,
    trunkLean: lean,
    consensusVotes,
    postureScore,
    visibilityScore,
    issues,
    isGoodPosture: postureScore > SCORE_EXIT_BAD && issues.length === 0,
    isCalibrated: !!calibration,
    deviations,
  };
}

// ── Timing Constants (state machine) ─────────────────────────────────────────
export const TIMING = {
  T_WARN_MS:      30_000,  // 30s bad → alert
  T_RESET_MS:      5_000,  // 5s good posture → reset bad timer
  T_COOLDOWN_MS:  60_000,  // 60s between alerts
} as const;

export function getUrgencyLevel(ms: number): 'GENTLE' | 'FIRM' | 'URGENT' {
  if (ms >= 90_000) return 'URGENT';
  if (ms >= 60_000) return 'FIRM';
  return 'GENTLE';
}

export function buildAnalysisPrompt(
  m: PostureMetrics, urgency: string, badMs: number
): string {
  const topIssue = m.issues[0];
  const problemName = topIssue?.problemName || 'Poor posture detected';
  const fixAction = topIssue?.fixAction || 'Sit upright and align your spine.';
  const durStr = badMs >= 60_000
    ? `${Math.floor(badMs / 60000)}m ${Math.floor((badMs % 60000) / 1000)}s`
    : `${Math.floor(badMs / 1000)}s`;

  return `You are PosChair, an AI posture voice coach. Poor posture detected for ${durStr}.

DIAGNOSED PROBLEM: ${problemName}
PHYSICAL FIX ACTION: ${fixAction}

MANDATORY RESPONSE FORMAT:
You MUST speak in exactly two parts:
Part 1 (The Problem): State the exact diagnosed problem (e.g. "${problemName}.").
Part 2 (The Fix): Tell the user how to fix it immediately (e.g. "${fixAction}").

Total length must be under 16 words. Never include filler words like "Hey", "Oops", "I noticed", or "Please".
Exact Output Format: "${problemName}. ${fixAction}"`;
}

export function resetSmoothers() {
  Object.values(smoothers).forEach(s => s.reset());
}
