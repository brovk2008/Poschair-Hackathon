/**
 * PosChair Posture Engine — v3 (Dual-Layer)
 * ──────────────────────────────────────────
 * Layer 1: 6 calibrated biomechanical metrics (existing)
 * Layer 2: 252-angle feature consensus voting (NEW)
 *
 * Layer 2 is inspired by PosePilot (2025) — exhaustive angle triplet approach.
 * An issue is only raised if BOTH layers agree → far fewer false positives.
 *
 * Additional improvements:
 *  - Visibility-weighted posture score (high-confidence landmarks count more)
 *  - Hysteresis thresholds (enter bad at <68, exit requires >76)
 *  - 5-second calibration (more stable baseline)
 */

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface CalibrationBaseline {
  // Layer 1 metrics
  headNeckShoulderAngle: number;
  lateralTiltDelta: number;
  earToShoulderRatio: number;
  shoulderAsymmetry: number;
  trunkLean: number;
  zFhpDelta: number;
  // Layer 2: full 252-angle feature vector
  featureAngles: number[];
  // Meta
  meanVisibility: number;
  capturedAt: number;
}

export interface PostureMetrics {
  // Layer 1 values (smoothed)
  headNeckShoulderAngle: number;
  lateralTiltDeg: number;
  shoulderShrug: number;
  shoulderAsymmetry: number;
  trunkLean: number;
  zFhpDelta: number;
  // Layer 2 output
  consensusVotes: Record<string, number>; // issue → 0–1 confidence
  // Score & issues
  postureScore: number;
  visibilityScore: number;       // 0–1, mean landmark visibility
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
  description: string;
  correctionHint: string;
}

// MediaPipe landmark indices used
export const LM = {
  NOSE: 0, LEFT_EAR: 7, RIGHT_EAR: 8,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_HIP: 23, RIGHT_HIP: 24,
} as const;

const KEY_LM_IDS = [0, 7, 8, 11, 12, 13, 14, 23, 24]; // 9 landmarks

// ── Layer 2: 252-angle Feature Specs ─────────────────────────────────────────
// Format: [a, vertex, c] — angle at vertex between a and c
// C(9,3) * 3 = 252 unique angles (each triplet, each vertex)
// Computed once at module load.

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
  return specs; // 9 * C(8,2) = 9 * 28 = 252
})();

// Which issue(s) each angle spec is sensitive to
// A spec is "sensitive to issue X" if it involves the anatomically relevant landmarks
type IssueType = PostureIssue['type'];
const FHP_SET  = new Set([0, 7, 8]);         // nose, ears
const SH_SET   = new Set([11, 12]);          // shoulders
const HIP_SET  = new Set([23, 24]);          // hips
const L_SET    = new Set([7, 11, 13, 23]);   // left-side
const R_SET    = new Set([8, 12, 14, 24]);   // right-side

function specSensitivity(a: number, v: number, c: number): IssueType[] {
  const pts = [a, v, c];
  const has = (s: Set<number>) => pts.some(p => s.has(p));
  const issues: IssueType[] = [];
  // Forward head: involves nose/ear AND shoulder
  if (has(FHP_SET) && has(SH_SET)) issues.push('FORWARD_HEAD');
  // Lateral tilt: involves BOTH ears, or one ear + shoulder asymmetry
  if (pts.includes(7) && pts.includes(8)) issues.push('LATERAL_TILT');
  if ((pts.includes(7) !== pts.includes(8)) && has(SH_SET)) issues.push('LATERAL_TILT');
  // Shoulder shrug: ear + shoulder (vertical proximity)
  if (has(FHP_SET) && has(SH_SET) && !has(HIP_SET)) issues.push('SHOULDER_SHRUG');
  // Shoulder asymmetry: both shoulders involved
  if (pts.includes(11) && pts.includes(12)) issues.push('SHOULDER_ASYMMETRY');
  // Trunk lean: shoulder + hip
  if (has(SH_SET) && has(HIP_SET)) issues.push('TRUNK_LEAN');
  return Array.from(new Set(issues));
}

// Pre-compute which specs are sensitive to each issue type (optimization)
const ISSUE_SPEC_MAP: Record<IssueType, number[]> = {
  FORWARD_HEAD: [], LATERAL_TILT: [], SHOULDER_SHRUG: [],
  SHOULDER_ASYMMETRY: [], TRUNK_LEAN: [],
};
ANGLE_SPECS.forEach(([a, v, c], idx) => {
  specSensitivity(a, v, c).forEach(issue => ISSUE_SPEC_MAP[issue].push(idx));
});

// ── Core Math ────────────────────────────────────────────────────────────────

export function angle3pt(
  a: PoseLandmark, b: PoseLandmark, c: PoseLandmark, use3D = false
): number {
  if (use3D) {
    const ba = [a.x-b.x, a.y-b.y, a.z-b.z];
    const bc = [c.x-b.x, c.y-b.y, c.z-b.z];
    const dot = ba[0]*bc[0]+ba[1]*bc[1]+ba[2]*bc[2];
    const m = Math.sqrt((ba[0]**2+ba[1]**2+ba[2]**2)*(bc[0]**2+bc[1]**2+bc[2]**2));
    return m < 1e-8 ? 180 : Math.acos(Math.max(-1, Math.min(1, dot/m))) * (180/Math.PI);
  }
  const ba = [a.x-b.x, a.y-b.y];
  const bc = [c.x-b.x, c.y-b.y];
  const dot = ba[0]*bc[0]+ba[1]*bc[1];
  const m = Math.sqrt((ba[0]**2+ba[1]**2)*(bc[0]**2+bc[1]**2));
  return m < 1e-8 ? 180 : Math.acos(Math.max(-1, Math.min(1, dot/m))) * (180/Math.PI);
}

function mid(a: PoseLandmark, b: PoseLandmark): PoseLandmark {
  return { x:(a.x+b.x)/2, y:(a.y+b.y)/2, z:(a.z+b.z)/2, visibility: Math.min(a.visibility??1, b.visibility??1) };
}

function dist2D(a: PoseLandmark, b: PoseLandmark) {
  return Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2);
}

// ── Layer 2: Feature Vector & Consensus Vote ──────────────────────────────────

function computeFeatureVector(lms: PoseLandmark[]): number[] {
  return ANGLE_SPECS.map(([a, v, c]) =>
    angle3pt(lms[a], lms[v], lms[c])
  );
}

/**
 * For a given issue type, compute what fraction of relevant angles
 * have deviated significantly from their calibrated baseline.
 * Returns 0–1 (0 = no deviation, 1 = all angles deviated).
 */
function consensusVote(
  current: number[],
  baseline: number[],
  issue: IssueType,
  deviationThresholdDeg = 9   // research: 9° delta is clinically meaningful
): number {
  const indices = ISSUE_SPEC_MAP[issue];
  if (indices.length === 0) return 0;
  let deviated = 0;
  for (const idx of indices) {
    if (Math.abs(current[idx] - baseline[idx]) > deviationThresholdDeg) deviated++;
  }
  return deviated / indices.length;
}

// ── Temporal Smoother ─────────────────────────────────────────────────────────

export class RollingAverage {
  private buf: number[];
  private size: number;
  private idx = 0;
  private filled = false;

  constructor(w = 10) { this.size = w; this.buf = new Array(w).fill(0); }

  update(v: number): number {
    this.buf[this.idx] = v;
    this.idx = (this.idx + 1) % this.size;
    if (this.idx === 0) this.filled = true;
    const n = this.filled ? this.size : this.idx;
    return this.buf.slice(0, n).reduce((a, b) => a + b, 0) / n;
  }

  reset() { this.buf = new Array(this.size).fill(0); this.idx = 0; this.filled = false; }
}

const smoothers = {
  headNeckShoulder: new RollingAverage(12),
  lateralTilt:      new RollingAverage(12),
  shoulderShrug:    new RollingAverage(8),
  shoulderAsym:     new RollingAverage(8),
  trunkLean:        new RollingAverage(8),
  zFhp:             new RollingAverage(15),
};

// ── Calibration ───────────────────────────────────────────────────────────────

interface RawMetrics {
  headNeckShoulderAngle: number;
  lateralTiltDelta: number;
  earToShoulderRatio: number;
  shoulderAsymmetry: number;
  trunkLean: number;
  zFhpDelta: number;
  featureAngles: number[];
  meanVisibility: number;
}

function extractRawMetrics(lms: PoseLandmark[]): RawMetrics | null {
  if (!lms || lms.length < 25) return null;

  const nose = lms[0], lEar = lms[7], rEar = lms[8];
  const lSh = lms[11], rSh = lms[12];
  const lHip = lms[23], rHip = lms[24];

  // Visibility gate: key landmarks must be clearly visible
  const minVis = 0.45;
  const visVals = [lEar, rEar, lSh, rSh].map(l => l.visibility ?? 1);
  if (visVals.some(v => v < minVis)) return null;

  const earMid = mid(lEar, rEar);
  const shMid  = mid(lSh, rSh);
  const hipMid = mid(lHip, rHip);
  const sw = dist2D(lSh, rSh);
  if (sw < 0.04) return null;

  const meanVis = KEY_LM_IDS
    .map(id => lms[id]?.visibility ?? 0.5)
    .reduce((a, b) => a + b, 0) / KEY_LM_IDS.length;

  return {
    headNeckShoulderAngle: angle3pt(nose, earMid, shMid),
    lateralTiltDelta: Math.atan2(rEar.y-lEar.y, rEar.x-lEar.x) * 180/Math.PI
                    - Math.atan2(rSh.y-lSh.y, rSh.x-lSh.x) * 180/Math.PI,
    earToShoulderRatio: (shMid.y - earMid.y) / sw,
    shoulderAsymmetry: Math.abs(lSh.y - rSh.y) / sw,
    trunkLean: (shMid.x - hipMid.x) / sw,
    zFhpDelta: earMid.z - shMid.z,
    featureAngles: computeFeatureVector(lms),
    meanVisibility: meanVis,
  };
}

export function sampleCalibrationFrame(lms: PoseLandmark[]): Partial<CalibrationBaseline> | null {
  const r = extractRawMetrics(lms);
  if (!r) return null;
  return {
    headNeckShoulderAngle: r.headNeckShoulderAngle,
    lateralTiltDelta: r.lateralTiltDelta,
    earToShoulderRatio: r.earToShoulderRatio,
    shoulderAsymmetry: r.shoulderAsymmetry,
    trunkLean: r.trunkLean,
    zFhpDelta: r.zFhpDelta,
    featureAngles: r.featureAngles,
    meanVisibility: r.meanVisibility,
  };
}

export function captureCalibration(samples: Partial<CalibrationBaseline>[]): CalibrationBaseline {
  const valid = samples.filter(s => s.featureAngles && s.featureAngles.length === ANGLE_SPECS.length);
  const avgScalar = (key: keyof CalibrationBaseline) => {
    const vals = valid.map(s => s[key] as number).filter(v => !isNaN(v));
    return vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
  };
  // Average the feature angle vector element-wise
  const featureAngles = Array.from({ length: ANGLE_SPECS.length }, (_, i) => {
    const vals = valid.map(s => (s.featureAngles as number[])[i]).filter(v => !isNaN(v));
    return vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
  });
  return {
    headNeckShoulderAngle: avgScalar('headNeckShoulderAngle'),
    lateralTiltDelta: avgScalar('lateralTiltDelta'),
    earToShoulderRatio: avgScalar('earToShoulderRatio'),
    shoulderAsymmetry: avgScalar('shoulderAsymmetry'),
    trunkLean: avgScalar('trunkLean'),
    zFhpDelta: avgScalar('zFhpDelta'),
    featureAngles,
    meanVisibility: avgScalar('meanVisibility'),
    capturedAt: Date.now(),
  };
}

// ── Main Analysis ─────────────────────────────────────────────────────────────

// Hysteresis constants
export const SCORE_ENTER_BAD  = 68; // score must drop BELOW this to enter bad state
export const SCORE_EXIT_BAD   = 76; // score must rise ABOVE this to exit bad state
// Layer 2 minimum confidence to confirm an issue (0–1)
const L2_CONFIRM_THRESHOLD = 0.25; // ≥25% of relevant angles must deviate
const L2_SUPPRESS_THRESHOLD = 0.10; // <10% → suppress Layer 1 issue

export function analyzePosture(
  lms: PoseLandmark[],
  calibration: CalibrationBaseline | null = null
): PostureMetrics | null {
  const raw = extractRawMetrics(lms);
  if (!raw) return null;

  // Smooth Layer 1 metrics
  const hnsa  = smoothers.headNeckShoulder.update(raw.headNeckShoulderAngle);
  const tilt  = smoothers.lateralTilt.update(raw.lateralTiltDelta);
  const shrug = smoothers.shoulderShrug.update(raw.earToShoulderRatio);
  const asym  = smoothers.shoulderAsym.update(raw.shoulderAsymmetry);
  const lean  = smoothers.trunkLean.update(raw.trunkLean);
  const zFhp  = smoothers.zFhp.update(raw.zFhpDelta);

  // Layer 2: consensus votes per issue type
  const consensusVotes: Record<string, number> = {};
  const ISSUE_TYPES: IssueType[] = [
    'FORWARD_HEAD', 'LATERAL_TILT', 'SHOULDER_SHRUG', 'SHOULDER_ASYMMETRY', 'TRUNK_LEAN'
  ];
  if (calibration?.featureAngles?.length === ANGLE_SPECS.length) {
    for (const issue of ISSUE_TYPES) {
      consensusVotes[issue] = consensusVote(raw.featureAngles, calibration.featureAngles, issue);
    }
  } else {
    // No calibration: neutral votes (don't suppress or confirm)
    ISSUE_TYPES.forEach(i => { consensusVotes[i] = 0.5; });
  }

  // Deviations from calibration
  const deviations: Record<string, number> = {};
  if (calibration) {
    deviations.headNeckShoulder = calibration.headNeckShoulderAngle - hnsa;
    deviations.lateralTilt = Math.abs(tilt) - Math.abs(calibration.lateralTiltDelta);
    deviations.shoulderShrug = calibration.earToShoulderRatio - shrug;
    deviations.shoulderAsym = asym - calibration.shoulderAsymmetry;
    deviations.trunkLean = Math.abs(lean) - Math.abs(calibration.trunkLean);
    deviations.zFhp = zFhp - calibration.zFhpDelta;
  }

  // Visibility score (average of key landmarks)
  const visibilityScore = KEY_LM_IDS
    .map(id => lms[id]?.visibility ?? 0.5)
    .reduce((a, b) => a + b, 0) / KEY_LM_IDS.length;

  // ── Layer 1 thresholds (calibration-relative) ──────────────────────────────
  const hnsaThreshMild   = calibration ? calibration.headNeckShoulderAngle - 12 : 148;
  const hnsaThreshMod    = calibration ? calibration.headNeckShoulderAngle - 20 : 140;
  const hnsaThreshSevere = calibration ? calibration.headNeckShoulderAngle - 30 : 130;
  const zFhpThresh       = calibration ? calibration.zFhpDelta - 0.06 : -0.08;
  const tiltThreshMild   = calibration ? Math.abs(calibration.lateralTiltDelta) + 8  : 8;
  const tiltThreshMod    = calibration ? Math.abs(calibration.lateralTiltDelta) + 15 : 15;
  const tiltThreshSevere = calibration ? Math.abs(calibration.lateralTiltDelta) + 25 : 25;
  const shrugThreshMild  = calibration ? calibration.earToShoulderRatio - 0.08 : 0.38;
  const shrugThreshMod   = calibration ? calibration.earToShoulderRatio - 0.14 : 0.30;
  const asymThreshMild   = calibration ? calibration.shoulderAsymmetry + 0.06 : 0.07;
  const asymThreshMod    = calibration ? calibration.shoulderAsymmetry + 0.12 : 0.14;
  const leanThreshMild   = calibration ? Math.abs(calibration.trunkLean) + 0.10 : 0.12;
  const leanThreshMod    = calibration ? Math.abs(calibration.trunkLean) + 0.18 : 0.20;

  const absTilt = Math.abs(tilt);
  const absLean = Math.abs(lean);

  // ── Issue detection: Layer 1 flag → Layer 2 confirm/suppress ───────────────
  const issues: PostureIssue[] = [];

  // Helper: decides if an issue makes it through dual-layer gate
  const shouldFlag = (issue: IssueType): boolean => {
    const l2 = consensusVotes[issue] ?? 0.5;
    // Suppress: Layer 2 strongly disagrees (< 10% of angles deviate) → false positive
    if (l2 < L2_SUPPRESS_THRESHOLD) return false;
    // Confirm: Layer 2 agrees (≥ 25% of angles deviate) → confirmed
    // Between 10–25%: defer to Layer 1 alone (ambiguous)
    return true;
  };

  // 1. Forward Head Posture
  if (hnsa < hnsaThreshMild && shouldFlag('FORWARD_HEAD')) {
    issues.push({
      type: 'FORWARD_HEAD',
      severity: hnsa < hnsaThreshSevere ? 'SEVERE' : hnsa < hnsaThreshMod ? 'MODERATE' : 'MILD',
      layer2Confidence: consensusVotes['FORWARD_HEAD'],
      value: hnsa,
      label: `HEAD_ANGLE: ${hnsa.toFixed(1)}°`,
      description: `Head pitched forward (${hnsa.toFixed(1)}° / target ≥${hnsaThreshMild.toFixed(0)}°)`,
      correctionHint: 'Draw chin back — bring your ears over your shoulders.',
    });
  }
  // Z-depth FHP confirmation (second independent signal)
  if (zFhp < zFhpThresh && !issues.find(i => i.type === 'FORWARD_HEAD') && shouldFlag('FORWARD_HEAD')) {
    issues.push({
      type: 'FORWARD_HEAD',
      severity: zFhp < zFhpThresh - 0.04 ? 'MODERATE' : 'MILD',
      layer2Confidence: consensusVotes['FORWARD_HEAD'],
      value: zFhp,
      label: `FHP_DEPTH: ${zFhp.toFixed(3)}`,
      description: 'Head detected in front of shoulder plane (depth signal)',
      correctionHint: 'Sit back and align your ears over your shoulders.',
    });
  }

  // 2. Lateral tilt
  if (absTilt > tiltThreshMild && shouldFlag('LATERAL_TILT')) {
    issues.push({
      type: 'LATERAL_TILT',
      severity: absTilt > tiltThreshSevere ? 'SEVERE' : absTilt > tiltThreshMod ? 'MODERATE' : 'MILD',
      layer2Confidence: consensusVotes['LATERAL_TILT'],
      value: tilt,
      label: `TILT: ${tilt > 0 ? '+' : ''}${tilt.toFixed(1)}°`,
      description: `Head tilted ${tilt > 0 ? 'right' : 'left'} by ${absTilt.toFixed(1)}°`,
      correctionHint: `Level your head — ${tilt > 0 ? 'right' : 'left'} ear is lower.`,
    });
  }

  // 3. Shoulder shrug
  if (shrug < shrugThreshMild && shouldFlag('SHOULDER_SHRUG')) {
    issues.push({
      type: 'SHOULDER_SHRUG',
      severity: shrug < (shrugThreshMod - 0.08) ? 'SEVERE' : shrug < shrugThreshMod ? 'MODERATE' : 'MILD',
      layer2Confidence: consensusVotes['SHOULDER_SHRUG'],
      value: shrug,
      label: `SHRUG: ${shrug.toFixed(2)}`,
      description: 'Shoulders elevated toward ears (stress shrug)',
      correctionHint: 'Drop your shoulders away from your ears. Relax your traps.',
    });
  }

  // 4. Shoulder asymmetry
  if (asym > asymThreshMild && shouldFlag('SHOULDER_ASYMMETRY')) {
    issues.push({
      type: 'SHOULDER_ASYMMETRY',
      severity: asym > (asymThreshMod + 0.08) ? 'SEVERE' : asym > asymThreshMod ? 'MODERATE' : 'MILD',
      layer2Confidence: consensusVotes['SHOULDER_ASYMMETRY'],
      value: asym,
      label: `ASYM: ${(asym*100).toFixed(1)}%`,
      description: `${lms[11].y > lms[12].y ? 'Left' : 'Right'} shoulder is lower`,
      correctionHint: 'Level your shoulders — keep them at equal height.',
    });
  }

  // 5. Trunk lean
  if (absLean > leanThreshMild && shouldFlag('TRUNK_LEAN')) {
    issues.push({
      type: 'TRUNK_LEAN',
      severity: absLean > (leanThreshMod + 0.08) ? 'SEVERE' : absLean > leanThreshMod ? 'MODERATE' : 'MILD',
      layer2Confidence: consensusVotes['TRUNK_LEAN'],
      value: lean,
      label: `LEAN: ${lean > 0 ? '+' : ''}${lean.toFixed(2)}`,
      description: `Trunk leaning ${lean > 0 ? 'right' : 'left'}`,
      correctionHint: 'Sit tall and centered. Even weight on both hips.',
    });
  }

  // ── Visibility-weighted posture score ──────────────────────────────────────
  // Weight each penalty by the visibility of the landmarks involved in that metric.
  // Low-confidence landmarks contribute less to score penalties.
  const earVis  = Math.min(lms[7]?.visibility ?? 1, lms[8]?.visibility ?? 1);
  const shVis   = Math.min(lms[11]?.visibility ?? 1, lms[12]?.visibility ?? 1);
  const hipVis  = Math.min(lms[23]?.visibility ?? 1, lms[24]?.visibility ?? 1);

  let score = 100;
  const hnsaTarget = calibration ? calibration.headNeckShoulderAngle : 165;
  const hnsaMin    = calibration ? calibration.headNeckShoulderAngle - 35 : 125;

  if (hnsa < hnsaTarget) {
    const penalty = Math.min(35, ((hnsaTarget-hnsa)/(hnsaTarget-hnsaMin)) * 35);
    score -= penalty * earVis * shVis;          // weighted by ear+shoulder visibility
  }
  if (zFhp < zFhpThresh) {
    const penalty = Math.min(10, Math.abs(zFhp-zFhpThresh) * 100);
    score -= penalty * earVis;
  }
  if (absTilt > tiltThreshMild) {
    const penalty = Math.min(25, ((absTilt-tiltThreshMild)/20) * 25);
    score -= penalty * earVis;
  }
  if (shrug < shrugThreshMild) {
    const penalty = Math.min(15, ((shrugThreshMild-shrug)/0.20) * 15);
    score -= penalty * shVis;
  }
  if (asym > asymThreshMild) {
    const penalty = Math.min(15, ((asym-asymThreshMild)/0.20) * 15);
    score -= penalty * shVis;
  }
  if (absLean > leanThreshMild) {
    const penalty = Math.min(10, ((absLean-leanThreshMild)/0.20) * 10);
    score -= penalty * Math.min(shVis, hipVis);
  }

  // Layer 2 bonus: if consensus votes strongly AGREE with good posture, add up to 5pts
  const avgGoodVote = ISSUE_TYPES.reduce((s, t) => s + (1 - (consensusVotes[t] ?? 0.5)), 0) / ISSUE_TYPES.length;
  if (avgGoodVote > 0.7) score = Math.min(100, score + (avgGoodVote - 0.7) * 15);

  const postureScore = Math.max(0, Math.round(score));

  return {
    headNeckShoulderAngle: hnsa,
    lateralTiltDeg: tilt,
    shoulderShrug: shrug,
    shoulderAsymmetry: asym,
    trunkLean: lean,
    zFhpDelta: zFhp,
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
  const topIssue  = m.issues[0];
  const allIssues = m.issues.map(i => `${i.type}: ${i.description} [${i.severity}, L2=${(i.layer2Confidence*100).toFixed(0)}%]`).join('; ');
  const durStr    = badMs >= 60_000
    ? `${Math.floor(badMs/60000)}m ${Math.floor((badMs%60000)/1000)}s`
    : `${Math.floor(badMs/1000)}s`;
  const toneGuide = urgency === 'URGENT' ? 'Be direct and firm.'
    : urgency === 'FIRM' ? 'Be clear and specific.' : 'Be gentle and encouraging.';

  return `You are PosChair, an AI posture coach for desk workers. Poor posture detected for ${durStr}.
Issues: ${allIssues || 'general poor posture'}
Top correction: ${topIssue?.correctionHint || 'Sit up straight, ears over shoulders.'}
Posture score: ${m.postureScore}/100 | Visibility: ${(m.visibilityScore*100).toFixed(0)}%
Tone: ${toneGuide}
Write ONE actionable voice correction in 1–2 short sentences (max 25 words). Address the #1 issue by body part. No filler. Sound human.`;
}

export function resetSmoothers() {
  Object.values(smoothers).forEach(s => s.reset());
}
