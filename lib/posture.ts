// Core posture analysis engine
// Uses MediaPipe pose landmarks to calculate 5 key postural metrics

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface PostureMetrics {
  headTiltAngle: number;       // degrees, 0 = level, + = right tilt, - = left tilt
  chinNodRatio: number;        // normalized 0-1, < 0.55 = chin forward/down
  shoulderShrug: number;       // normalized ear-to-shoulder distance, < 0.35 = shrugging
  shoulderAsymmetry: number;   // 0-1, > 0.08 = uneven shoulders
  trunkLean: number;           // normalized, > 0.15 = lateral lean
  postureScore: number;        // 0-100, composite score
  issues: PostureIssue[];
  isGoodPosture: boolean;
}

export interface PostureIssue {
  type: 'HEAD_TILT' | 'CHIN_NOD' | 'SHOULDER_SHRUG' | 'SHOULDER_ASYMMETRY' | 'TRUNK_LEAN';
  severity: 'MILD' | 'MODERATE' | 'SEVERE';
  value: number;
  label: string;
  description: string;
}

// MediaPipe landmark indices
const LM = {
  NOSE: 0,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
};

function angleDeg(a: PoseLandmark, b: PoseLandmark): number {
  return Math.atan2(b.y - a.y, b.x - a.x) * (180 / Math.PI);
}

function midpoint(a: PoseLandmark, b: PoseLandmark): PoseLandmark {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

function dist(a: PoseLandmark, b: PoseLandmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

export function analyzePosture(landmarks: PoseLandmark[]): PostureMetrics | null {
  if (!landmarks || landmarks.length < 25) return null;

  const nose = landmarks[LM.NOSE];
  const leftEar = landmarks[LM.LEFT_EAR];
  const rightEar = landmarks[LM.RIGHT_EAR];
  const leftShoulder = landmarks[LM.LEFT_SHOULDER];
  const rightShoulder = landmarks[LM.RIGHT_SHOULDER];
  const leftHip = landmarks[LM.LEFT_HIP];
  const rightHip = landmarks[LM.RIGHT_HIP];

  // Require minimum visibility on key landmarks
  const minVis = 0.3;
  if (
    (leftEar.visibility ?? 1) < minVis ||
    (rightEar.visibility ?? 1) < minVis ||
    (leftShoulder.visibility ?? 1) < minVis ||
    (rightShoulder.visibility ?? 1) < minVis
  ) return null;

  const earMid = midpoint(leftEar, rightEar);
  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  const hipMid = midpoint(leftHip, rightHip);
  const shoulderWidth = dist(leftShoulder, rightShoulder);

  if (shoulderWidth < 0.05) return null; // person not visible enough

  // ── Metric 1: Lateral Head Tilt ──────────────────────────────────────────
  // Difference between ear line angle and shoulder line angle
  const earAngle = angleDeg(leftEar, rightEar);
  const shoulderAngle = angleDeg(leftShoulder, rightShoulder);
  const headTiltAngle = earAngle - shoulderAngle;

  // ── Metric 2: Chin Nod / Forward Head ────────────────────────────────────
  // How far nose is below shoulder midpoint, normalized by shoulder width
  // Lower value = chin tucked down / forward head posture
  const chinNodRatio = (shoulderMid.y - nose.y) / shoulderWidth;

  // ── Metric 3: Shoulder Shrug ─────────────────────────────────────────────
  // Ear-to-shoulder vertical distance, normalized by shoulder width
  // Low value = shoulders raised toward ears (stress shrug)
  const shoulderShrug = (shoulderMid.y - earMid.y) / shoulderWidth;

  // ── Metric 4: Shoulder Asymmetry ─────────────────────────────────────────
  // Normalized height difference between left and right shoulder
  const shoulderAsymmetry = Math.abs(leftShoulder.y - rightShoulder.y) / shoulderWidth;

  // ── Metric 5: Trunk Lateral Lean ─────────────────────────────────────────
  // Horizontal offset of shoulder midpoint vs hip midpoint, normalized
  const trunkLean = (shoulderMid.x - hipMid.x) / shoulderWidth;

  // ── Detect Issues ─────────────────────────────────────────────────────────
  const issues: PostureIssue[] = [];

  // Head tilt thresholds: >10° mild, >18° moderate, >28° severe
  const absTilt = Math.abs(headTiltAngle);
  if (absTilt > 10) {
    issues.push({
      type: 'HEAD_TILT',
      severity: absTilt > 28 ? 'SEVERE' : absTilt > 18 ? 'MODERATE' : 'MILD',
      value: headTiltAngle,
      label: `HEAD_TILT: ${headTiltAngle > 0 ? '+' : ''}${headTiltAngle.toFixed(1)}°`,
      description: `Head tilted ${headTiltAngle > 0 ? 'right' : 'left'} by ${absTilt.toFixed(1)}°`,
    });
  }

  // Chin nod: <0.7 mild, <0.5 moderate, <0.3 severe
  if (chinNodRatio < 0.7) {
    issues.push({
      type: 'CHIN_NOD',
      severity: chinNodRatio < 0.3 ? 'SEVERE' : chinNodRatio < 0.5 ? 'MODERATE' : 'MILD',
      value: chinNodRatio,
      label: `CHIN_NOD: ${chinNodRatio.toFixed(2)}`,
      description: 'Head pitched forward / chin tucked down',
    });
  }

  // Shoulder shrug: <0.4 mild, <0.3 moderate, <0.2 severe
  if (shoulderShrug < 0.4) {
    issues.push({
      type: 'SHOULDER_SHRUG',
      severity: shoulderShrug < 0.2 ? 'SEVERE' : shoulderShrug < 0.3 ? 'MODERATE' : 'MILD',
      value: shoulderShrug,
      label: `SHRUG: ${shoulderShrug.toFixed(2)}`,
      description: 'Shoulders elevated / stress shrug detected',
    });
  }

  // Shoulder asymmetry: >0.08 mild, >0.15 moderate, >0.25 severe
  if (shoulderAsymmetry > 0.08) {
    issues.push({
      type: 'SHOULDER_ASYMMETRY',
      severity: shoulderAsymmetry > 0.25 ? 'SEVERE' : shoulderAsymmetry > 0.15 ? 'MODERATE' : 'MILD',
      value: shoulderAsymmetry,
      label: `ASYMMETRY: ${(shoulderAsymmetry * 100).toFixed(1)}%`,
      description: 'Uneven shoulder heights detected',
    });
  }

  // Trunk lean: >0.12 mild, >0.20 moderate, >0.30 severe
  const absTrunk = Math.abs(trunkLean);
  if (absTrunk > 0.12) {
    issues.push({
      type: 'TRUNK_LEAN',
      severity: absTrunk > 0.30 ? 'SEVERE' : absTrunk > 0.20 ? 'MODERATE' : 'MILD',
      value: trunkLean,
      label: `LEAN: ${trunkLean > 0 ? '+' : ''}${trunkLean.toFixed(2)}`,
      description: `Trunk leaning ${trunkLean > 0 ? 'right' : 'left'}`,
    });
  }

  // ── Compute Posture Score ─────────────────────────────────────────────────
  // Each metric contributes a penalty. Max score = 100.
  const tiltPenalty = Math.min(30, (absTilt / 28) * 30);
  const chinPenalty = Math.min(30, chinNodRatio < 0.7 ? ((0.7 - chinNodRatio) / 0.4) * 30 : 0);
  const shrugPenalty = Math.min(15, shoulderShrug < 0.4 ? ((0.4 - shoulderShrug) / 0.2) * 15 : 0);
  const asymPenalty = Math.min(15, (shoulderAsymmetry / 0.25) * 15);
  const leanPenalty = Math.min(10, (absTrunk / 0.30) * 10);

  const postureScore = Math.max(0, Math.round(100 - tiltPenalty - chinPenalty - shrugPenalty - asymPenalty - leanPenalty));

  return {
    headTiltAngle,
    chinNodRatio,
    shoulderShrug,
    shoulderAsymmetry,
    trunkLean,
    postureScore,
    issues,
    isGoodPosture: postureScore >= 70 && issues.length === 0,
  };
}

export function getUrgencyLevel(badPostureDurationMs: number): 'GENTLE' | 'FIRM' | 'URGENT' {
  if (badPostureDurationMs >= 90000) return 'URGENT';
  if (badPostureDurationMs >= 60000) return 'FIRM';
  return 'GENTLE';
}

export function buildAnalysisPrompt(metrics: PostureMetrics, urgency: string): string {
  const issueDescriptions = metrics.issues.map(i => `- ${i.description} (${i.severity})`).join('\n');
  
  return `You are PosChair, an AI posture coach for desk workers. A user has had poor posture for ${
    urgency === 'URGENT' ? 'over 90 seconds' : urgency === 'FIRM' ? 'over 60 seconds' : 'about 30 seconds'
  }.

Current posture issues detected:
${issueDescriptions}

Posture score: ${metrics.postureScore}/100

Respond with a single, natural, conversational correction in ${
    urgency === 'URGENT' ? '2 firm sentences' : urgency === 'FIRM' ? '1-2 clear sentences' : '1 gentle sentence'
  }. Be specific about which body part to adjust and exactly how. Sound like a caring coach, not a robot. No intro phrases like "I notice" or "It seems". Just the correction directly. Max 30 words.`;
}
