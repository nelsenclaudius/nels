/**
 * REBA Training Studio — v3
 * ==========================
 * Sistem ML berbasis K-Nearest Neighbors (k-NN):
 *   1. User tandai titik manual → simpan sudut + skor REBA + stickman
 *   2. Foto baru → hitung sudut → cari pose tersimpan yang paling mirip
 *   3. Pakai skor pose terdekat → hasil konsisten & akurat
 *
 * Mengapa k-NN bukan neural network?
 *   - Bekerja dengan 5–50 sampel (NN butuh 100+)
 *   - Tidak perlu training — langsung pakai setelah simpan
 *   - Hasil bisa dijelaskan: "mirip dengan pose X yang skornya Y"
 *   - Tidak ada risiko gagal training / error TF.js
 *
 * Storage: IndexedDB (persisten, offline, privat)
 * Integrasi: import REBATrainingStudio from "./REBATrainingStudio"
 */

import { useState, useEffect, useRef, useCallback } from "react";

// ─────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────
declare global {
  interface Window { Pose: any; tf: any; }
}

interface ManualPoint { x: number; y: number; label: string; }

interface AngleSet {
  neck: number; trunk: number; knee: number;
  upperArm: number; lowerArm: number; wrist: number;
}

interface REBAScoreSet {
  neck: number; trunk: number; legs: number;
  upperArm: number; lowerArm: number; wrist: number;
  scoreA: number; scoreB: number; scoreC: number; final: number;
}

interface PoseSample {
  id: string;
  timestamp: string;
  poseLabel: string;
  notes: string;
  // Sudut anatomi (vektor fitur untuk k-NN)
  angles: AngleSet;
  // Skor REBA hasil manual
  rebaScores: REBAScoreSet;
  // Titik landmark untuk render stickman
  landmarks: ManualPoint[];
  // Dimensi kanvas saat anotasi (untuk normalisasi stickman)
  canvasW: number;
  canvasH: number;
  // Thumbnail (base64 kecil, max 200px)
  thumbnail: string;
  verified: boolean;
}

interface KNNResult {
  sample: PoseSample;
  distance: number;
  similarity: number; // 0–100%
}

// ─────────────────────────────────────────────
// STORAGE LAYER — IndexedDB primary, localStorage fallback
// ─────────────────────────────────────────────
const DB_NAME    = "reba_knn_db_v3";
const DB_VERSION = 1;
const STORE      = "poses";
const LS_KEY     = "reba_poses_backup"; // localStorage fallback key

// ── IndexedDB helpers ──────────────────────────
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") { reject(new Error("IndexedDB not available")); return; }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE))
        db.createObjectStore(STORE, { keyPath: "id" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
    req.onblocked = () => reject(new Error("IndexedDB blocked"));
  });
}

async function idbSave(v: PoseSample): Promise<void> {
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, "readwrite");
    const req = tx.objectStore(STORE).put(v);
    req.onerror = () => rej(req.error);
    tx.oncomplete = () => res();
    tx.onerror    = () => rej(tx.error);
  });
}

async function idbLoadAll(): Promise<PoseSample[]> {
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx  = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => res(req.result as PoseSample[]);
    req.onerror   = () => rej(req.error);
  });
}

async function idbRemove(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).delete(id);
    tx.oncomplete = () => res();
    tx.onerror    = () => rej(tx.error);
  });
}

// ── localStorage fallback helpers ─────────────
function lsLoadAll(): PoseSample[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as PoseSample[];
  } catch { return []; }
}

function lsSave(poses: PoseSample[]): void {
  try {
    // Simpan tanpa thumbnail agar tidak melebihi 5MB quota localStorage
    const slim = poses.map(p => ({ ...p, thumbnail: "" }));
    localStorage.setItem(LS_KEY, JSON.stringify(slim));
  } catch (e) {
    console.warn("localStorage quota exceeded, skipping backup:", e);
  }
}

function lsRemove(id: string): void {
  const all = lsLoadAll().filter(p => p.id !== id);
  lsSave(all);
}

// ── Unified public API ─────────────────────────
let _useIDB = true; // diset saat load pertama

async function dbSave(v: PoseSample): Promise<void> {
  if (_useIDB) {
    try {
      await idbSave(v);
      // Sync ke localStorage sebagai backup (tanpa thumbnail)
      const all = await idbLoadAll();
      lsSave(all);
      return;
    } catch (e) {
      console.warn("IndexedDB save failed, switching to localStorage:", e);
      _useIDB = false;
    }
  }
  // Fallback: localStorage
  const all = lsLoadAll();
  const idx = all.findIndex(p => p.id === v.id);
  if (idx >= 0) all[idx] = v; else all.push(v);
  lsSave(all);
}

async function dbLoadAll(): Promise<PoseSample[]> {
  if (_useIDB) {
    try {
      const data = await idbLoadAll();
      if (data.length > 0) return data;
      // IDB kosong tapi LS punya data → migrate
      const fallback = lsLoadAll();
      if (fallback.length > 0) {
        console.info(`Migrating ${fallback.length} poses from localStorage → IndexedDB`);
        for (const p of fallback) await idbSave(p).catch(() => {});
        return idbLoadAll();
      }
      return [];
    } catch (e) {
      console.warn("IndexedDB load failed, switching to localStorage:", e);
      _useIDB = false;
    }
  }
  return lsLoadAll();
}

export async function loadPoseSamples(): Promise<PoseSample[]> {
  const raw = await dbLoadAll();
  const sanitized = raw
    .map(item => sanitizePoseSample(item))
    .filter((item): item is PoseSample => item !== null);
  return migratePoseSamples(sanitized);
}

export function predictFromTrainingDB(query: AngleSet, database: PoseSample[], k = 3) {
  return knnPredict(query, database, k);
}

async function dbRemove(id: string): Promise<void> {
  if (_useIDB) {
    try {
      await idbRemove(id);
      lsRemove(id);
      return;
    } catch (e) {
      console.warn("IndexedDB remove failed:", e);
      _useIDB = false;
    }
  }
  lsRemove(id);
}

// ── Sanitasi PoseSample dari JSON eksternal ────
function sanitizePoseSample(raw: any): PoseSample | null {
  try {
    if (typeof raw !== "object" || raw === null) return null;
    // Field wajib
    const angles: AngleSet = {
      neck:     Number(raw.angles?.neck     ?? 0),
      trunk:    Number(raw.angles?.trunk    ?? 0),
      knee:     Number(raw.angles?.knee     ?? 0),
      upperArm: Number(raw.angles?.upperArm ?? 0),
      lowerArm: Number(raw.angles?.lowerArm ?? 0),
      wrist:    Number(raw.angles?.wrist    ?? 0),
    };
    const rs = raw.rebaScores ?? {};
    const rebaScores: REBAScoreSet = {
      neck:     Number(rs.neck     ?? 1),
      trunk:    Number(rs.trunk    ?? 1),
      legs:     Number(rs.legs     ?? 1),
      upperArm: Number(rs.upperArm ?? 1),
      lowerArm: Number(rs.lowerArm ?? 1),
      wrist:    Number(rs.wrist    ?? 1),
      scoreA:   Number(rs.scoreA   ?? 1),
      scoreB:   Number(rs.scoreB   ?? 1),
      scoreC:   Number(rs.scoreC   ?? 1),
      final:    Number(rs.final    ?? rs.scoreC ?? 1),
    };
    const landmarks: ManualPoint[] = Array.isArray(raw.landmarks)
      ? raw.landmarks.map((l: any) => ({ x: Number(l.x??0), y: Number(l.y??0), label: String(l.label??"") }))
      : [];
    return {
      id:        String(raw.id ?? `pose_import_${Date.now()}_${Math.random().toString(36).slice(2)}`),
      timestamp: String(raw.timestamp ?? new Date().toISOString()),
      poseLabel: String(raw.poseLabel ?? "Pose Impor"),
      notes:     String(raw.notes ?? ""),
      angles,
      rebaScores,
      landmarks,
      canvasW:   Number(raw.canvasW ?? 800),
      canvasH:   Number(raw.canvasH ?? 600),
      thumbnail: typeof raw.thumbnail === "string" ? raw.thumbnail : "",
      verified:  Boolean(raw.verified ?? true),
    };
  } catch { return null; }
}

function anglesEqual(a: AngleSet, b: AngleSet) {
  return a.neck === b.neck && a.trunk === b.trunk && a.knee === b.knee
      && a.upperArm === b.upperArm && a.lowerArm === b.lowerArm && a.wrist === b.wrist;
}

function rebaScoresEqual(a: REBAScoreSet, b: REBAScoreSet) {
  return a.neck === b.neck && a.trunk === b.trunk && a.legs === b.legs
      && a.upperArm === b.upperArm && a.lowerArm === b.lowerArm && a.wrist === b.wrist
      && a.scoreA === b.scoreA && a.scoreB === b.scoreB && a.scoreC === b.scoreC && a.final === b.final;
}

function needsRebaMigration(raw: any): boolean {
  if (typeof raw !== "object" || raw === null) return true;
  const keys: Array<keyof REBAScoreSet> = [
    "neck", "trunk", "legs", "upperArm", "lowerArm", "wrist",
    "scoreA", "scoreB", "scoreC", "final",
  ];
  return keys.some(key => typeof raw[key] !== "number" || Number.isNaN(raw[key]) || raw[key] <= 0);
}

function migratePoseSample(sample: PoseSample): PoseSample {
  let updatedAngles = sample.angles;
  if (Array.isArray(sample.landmarks) && sample.landmarks.length > 0) {
    try {
      const recomputed = computeAnglesFromPoints(sample.landmarks);
      if (!anglesEqual(sample.angles, recomputed)) {
        updatedAngles = recomputed;
      }
    } catch {
      updatedAngles = sample.angles;
    }
  }

  const shouldFixReba = needsRebaMigration(sample.rebaScores);
  if (!anglesEqual(sample.angles, updatedAngles) || shouldFixReba) {
    return {
      ...sample,
      angles: updatedAngles,
      rebaScores: shouldFixReba ? computeREBA(updatedAngles) : sample.rebaScores,
    };
  }

  return sample;
}

async function migratePoseSamples(samples: PoseSample[]): Promise<PoseSample[]> {
  const migrated = samples.map(migratePoseSample);
  const updates = migrated.filter((sample, index) =>
    !anglesEqual(sample.angles, samples[index].angles)
    || (needsRebaMigration(samples[index].rebaScores) && !rebaScoresEqual(sample.rebaScores, samples[index].rebaScores))
  );

  if (updates.length > 0) {
    await Promise.all(updates.map(sample => dbSave(sample).catch(() => {})));
  }

  return migrated;
}

// ─────────────────────────────────────────────
// ANGLE MATH
// ─────────────────────────────────────────────
function angleBetween(a: ManualPoint, b: ManualPoint, c: ManualPoint): number {
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const dot = ba.x * bc.x + ba.y * bc.y;
  const mag = Math.sqrt(ba.x**2 + ba.y**2) * Math.sqrt(bc.x**2 + bc.y**2);
  if (mag < 0.001) return 0;
  return Math.acos(Math.min(1, Math.max(-1, dot / mag))) * (180 / Math.PI);
}

function angleWithVertical(a: ManualPoint, b: ManualPoint): number {
  const dx = b.x - a.x, dy = b.y - a.y;
  if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) return 0;
  return Math.atan2(Math.abs(dx), dy) * (180 / Math.PI);
}

function computeAnglesFromPoints(pts: ManualPoint[]): AngleSet {
  const g = (l: string) => pts.find(p => p.label === l) ?? null;
  const ear=g("Ear"), sh=g("Shoulder"), hi=g("Hip"),
        kn=g("Knee"), an=g("Ankle"),    el=g("Elbow"),
        wr=g("Wrist"), ix=g("Index");

  // ── Neck: deviasi leher dari alignment trunk (bukan vertikal absolut) ──
  // REBA mengukur neck RELATIF terhadap trunk. Sudut antara vektor trunk (shoulder→hip
  // dibalik menjadi shoulder→atas) dan vektor neck (shoulder→ear).
  // Jika kepala sejajar dengan badan → 0°. Menunduk/mendongak dari badan → bertambah.
  const neckAngle = (): number => {
    if (!ear || !sh || !hi) return 0;
    const trunk = { x: hi.x - sh.x, y: hi.y - sh.y };
    const neck  = { x: ear.x - sh.x, y: ear.y - sh.y };
    const dot = trunk.x * neck.x + trunk.y * neck.y;
    const mag = Math.sqrt(trunk.x**2 + trunk.y**2) * Math.sqrt(neck.x**2 + neck.y**2);
    if (mag < 0.001) return 0;
    const angleBetweenTrunkNeck = Math.acos(Math.min(1, Math.max(-1, dot/mag))) * (180/Math.PI);
    return Math.round(Math.abs(180 - angleBetweenTrunkNeck));
  };

  // ── Upper arm: dari vertikal GRAVITASI (bukan trunk) ──
  // REBA mengukur upper arm dari vertikal absolut gravitasi.
  const upperArmAngle = (): number => {
    if (!sh || !el) return 0;
    return Math.round(angleWithVertical(sh, el));
  };

  return {
    neck:     neckAngle(),
    trunk:    sh&&hi     ? Math.round(angleWithVertical(sh, hi))             : 0,
    knee:     hi&&kn&&an ? Math.round(180 - angleBetween(hi, kn, an))       : 0,
    upperArm: upperArmAngle(),
    lowerArm: sh&&el&&wr ? Math.round(angleBetween(sh, el, wr))             : 0,
    wrist:    el&&wr&&ix ? Math.round(angleBetween(el, wr, ix))             : 0,
  };
}

// ─────────────────────────────────────────────
// REBA SCORING TABLES
// ─────────────────────────────────────────────
const TABLE_A: number[][][] = [
  [[1,2,3,4],[1,2,3,4],[3,3,5,6]],[[2,3,4,5],[3,4,5,6],[4,5,6,7]],
  [[2,4,5,6],[4,5,6,7],[5,6,7,8]],[[3,5,6,7],[5,6,7,8],[6,7,8,9]],
  [[4,6,7,8],[6,7,8,9],[7,8,9,9]],
];
const TABLE_B: number[][][] = [
  [[1,2,2],[1,2,3]],[[1,2,3],[2,3,4]],[[3,4,5],[4,5,5]],
  [[4,5,5],[5,6,7]],[[7,8,8],[8,9,9]],
];
const TABLE_C: number[][] = [
  [1,1,1,2,3,3,4,5,6,7,7,7],[1,2,2,3,4,4,5,6,6,7,7,8],
  [2,3,3,3,4,5,6,7,7,8,8,8],[3,4,4,4,5,6,7,8,8,9,9,9],
  [4,4,4,5,6,7,8,8,9,9,9,9],[6,6,6,7,8,8,9,9,10,10,10,10],
  [7,7,7,8,9,9,9,10,10,11,11,11],[8,8,8,9,10,10,10,10,10,11,11,11],
  [9,9,9,10,10,10,11,11,11,12,12,12],[10,10,10,11,11,11,11,12,12,12,12,12],
  [11,11,11,11,12,12,12,12,12,12,12,12],[12,12,12,12,12,12,12,12,12,12,12,12],
];

const NOISE = 3;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function deriveRebaFromPostures(value: Partial<REBAScoreSet>): REBAScoreSet {
  const neck     = clamp(Number(value.neck ?? 1), 1, 3);
  const trunk    = clamp(Number(value.trunk ?? 1), 1, 5);
  const legs     = clamp(Number(value.legs ?? 1), 1, 3);
  const upperArm = clamp(Number(value.upperArm ?? 1), 1, 5);
  const lowerArm = clamp(Number(value.lowerArm ?? 1), 1, 2);
  const wrist    = clamp(Number(value.wrist ?? 1), 1, 3);
  const scoreA   = TABLE_A[trunk-1]?.[neck-1]?.[legs-1] ?? 1;
  const scoreB   = TABLE_B[upperArm-1]?.[lowerArm-1]?.[wrist-1] ?? 1;
  const ai       = Math.min(Math.max(scoreA, 1), 12) - 1;
  const bi       = Math.min(Math.max(scoreB, 1), 12) - 1;
  const scoreC   = TABLE_C[ai]?.[bi] ?? 1;
  return { neck, trunk, legs, upperArm, lowerArm, wrist, scoreA, scoreB, scoreC, final: scoreC };
}

function computeREBA(a: AngleSet): REBAScoreSet {
  // Neck: deviasi dari trunk. ≤20° = skor 1, >20° = skor 2.
  const neck     = a.neck     <= (20+NOISE) ? 1 : 2;
  // Trunk: dari vertikal gravitasi. 0–5°=1, 6–20°=2, 21–60°=3, >60°=4.
  const trunk    = a.trunk    <= (5+NOISE)  ? 1 : a.trunk<=(20+NOISE)?2:a.trunk<=(60+NOISE)?3:4;
  // Legs: knee flexion. <30°=1, 30–60°=2, >60°=3. MAX 3 (Table A hanya 3 kolom).
  const legs     = a.knee     <= (30+NOISE) ? 1 : a.knee <=(60+NOISE)?2:3;
  // Upper arm: dari vertikal gravitasi. 0–20°=1, 21–45°=2, 46–90°=3, >90°=4.
  const upperArm = a.upperArm <= (20+NOISE) ? 1 : a.upperArm<=(45+NOISE)?2:a.upperArm<=(90+NOISE)?3:4;
  // Lower arm: 60–100°=1, lainnya=2.
  const lowerArm = (a.lowerArm>=(60-NOISE) && a.lowerArm<=(100+NOISE)) ? 1 : 2;
  // Wrist: ≤15°=1, >15°=2.
  const wrist    = a.wrist    <= (15+NOISE) ? 1 : 2;
  const n = Math.min(Math.max(neck,1),3),    t = Math.min(Math.max(trunk,1),5);
  const u = Math.min(Math.max(upperArm,1),5), w = Math.min(Math.max(wrist,1),3);
  const legsIdx = Math.min(Math.max(legs,1),3); // max index = 3 (kolom 0,1,2)
  const scoreA = TABLE_A[t-1]?.[n-1]?.[legsIdx-1] ?? 1;
  const scoreB = TABLE_B[u-1]?.[lowerArm-1]?.[w-1] ?? 1;
  const ai = Math.min(Math.max(scoreA,1),12)-1, bi = Math.min(Math.max(scoreB,1),12)-1;
  const scoreC = TABLE_C[ai]?.[bi] ?? 1;
  return { neck, trunk, legs, upperArm, lowerArm, wrist, scoreA, scoreB, scoreC, final: scoreC };
}

// ─────────────────────────────────────────────
// K-NN ENGINE
// ─────────────────────────────────────────────

/** Normalisasi AngleSet ke vektor 0–1 */
function toVector(a: AngleSet): number[] {
  return [
    Math.min(a.neck    / 60,  1),
    Math.min(a.trunk   / 90,  1),
    Math.min(a.knee    / 120, 1),
    Math.min(a.upperArm/ 120, 1),
    Math.min(a.lowerArm/ 180, 1),
    Math.min(a.wrist   / 60,  1),
  ];
}

/**
 * Bobot per fitur — trunk & neck lebih penting dalam REBA
 */
const WEIGHTS = [1.5, 2.0, 1.2, 1.2, 0.8, 0.8];

/** Jarak Euclidean berbobot antara dua vektor pose */
function weightedDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < 6; i++) {
    sum += WEIGHTS[i] * (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

/**
 * Cari k pose terdekat dari database.
 * Kembalikan prediksi skor via weighted voting.
 */
function knnPredict(
  query: AngleSet,
  database: PoseSample[],
  k = 3
): { results: KNNResult[]; predicted: REBAScoreSet | null } {
  if (database.length === 0) return { results: [], predicted: null };

  const qVec = toVector(query);

  // Hitung jarak ke semua sampel
  const distances: KNNResult[] = database.map(s => {
    const dist = weightedDistance(qVec, toVector(s.angles));
    // Konversi jarak ke similarity % (max jarak theoritis = sqrt(sum weights) ≈ 3)
    const maxDist = Math.sqrt(WEIGHTS.reduce((a, w) => a + w, 0));
    const sim     = Math.max(0, Math.round((1 - dist / maxDist) * 100));
    return { sample: s, distance: dist, similarity: sim };
  });

  // Urutkan dari terdekat
  distances.sort((a, b) => a.distance - b.distance);
  const topK = distances.slice(0, Math.min(k, distances.length));

  if (topK.length === 0) return { results: [], predicted: null };

  // Weighted voting: pose yang lebih dekat punya bobot lebih besar
  const weights = topK.map(r => 1 / (r.distance + 0.0001));
  const totalW  = weights.reduce((a, b) => a + b, 0);

  // Rata-rata berbobot tiap skor
  const avg = (key: keyof REBAScoreSet) => {
    const sum = topK.reduce((acc, r, i) => acc + r.sample.rebaScores[key] * weights[i], 0);
    return Math.round(sum / totalW);
  };

  const neck     = avg("neck")     as 1|2;
  const trunk    = avg("trunk")    as 1|2|3|4;
  const legs     = avg("legs")     as 1|2|3|4;
  const upperArm = avg("upperArm") as 1|2|3|4;
  const lowerArm = avg("lowerArm") as 1|2;
  const wrist    = avg("wrist")    as 1|2;

  const n  = Math.min(Math.max(neck,1),3), t = Math.min(Math.max(trunk,1),5);
  const u  = Math.min(Math.max(upperArm,1),5), w = Math.min(Math.max(wrist,1),3);
  const scoreA = TABLE_A[t-1]?.[n-1]?.[legs-1] ?? 1;
  const scoreB = TABLE_B[u-1]?.[lowerArm-1]?.[w-1] ?? 1;
  const ai = Math.min(Math.max(scoreA,1),12)-1, bi = Math.min(Math.max(scoreB,1),12)-1;
  const scoreC = TABLE_C[ai]?.[bi] ?? 1;

  const predicted: REBAScoreSet = {
    neck, trunk, legs, upperArm, lowerArm, wrist,
    scoreA, scoreB, scoreC, final: scoreC
  };

  return { results: topK, predicted };
}

// ─────────────────────────────────────────────
// STICKMAN RENDERER (SVG)
// ─────────────────────────────────────────────
function StickmanSVG({
  landmarks,
  cw,
  ch,
  size = 120,
  highlight = false,
}: {
  landmarks: ManualPoint[];
  cw: number;
  ch: number;
  size?: number;
  highlight?: boolean;
}) {
  if (!landmarks.length) return (
    <div style={{ width:size, height:size, background:"#f1f5f9", borderRadius:"8px",
      display:"flex", alignItems:"center", justifyContent:"center",
      fontSize:"10px", color:"#94a3b8" }}>No data</div>
  );

  // Normalisasi ke kotak size×size
  const norm = (p: ManualPoint) => ({
    x: (p.x / cw) * size,
    y: (p.y / ch) * size,
    label: p.label,
  });
  const pts = landmarks.map(norm);
  const g   = (l: string) => pts.find(p => p.label === l);

  const lines: [string, string, string][] = [
    ["Ear","Shoulder","#f97316"],
    ["Shoulder","Hip","#2563eb"],
    ["Hip","Knee","#7c3aed"],
    ["Knee","Ankle","#059669"],
    ["Shoulder","Elbow","#d97706"],
    ["Elbow","Wrist","#0891b2"],
    ["Wrist","Index","#65a30d"],
  ];

  const DOT_COLORS: Record<string,string> = {
    Ear:"#f97316",Shoulder:"#2563eb",Hip:"#7c3aed",Knee:"#059669",
    Ankle:"#db2777",Elbow:"#d97706",Wrist:"#0891b2",Index:"#65a30d",
  };

  return (
    <svg
      width={size} height={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{
        borderRadius:"8px",
        background: highlight ? "#eff6ff" : "#f8fafc",
        border: `1px solid ${highlight?"#bfdbfe":"#e2e8f0"}`,
        flexShrink: 0,
      }}
    >
      {lines.map(([l1, l2, color]) => {
        const p1 = g(l1), p2 = g(l2);
        if (!p1 || !p2) return null;
        return <line key={`${l1}-${l2}`} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
          stroke={color} strokeWidth="2.5" strokeLinecap="round" />;
      })}
      {pts.map(p => (
        <circle key={p.label} cx={p.x} cy={p.y} r={4}
          fill={DOT_COLORS[p.label]||"#64748b"}
          stroke="white" strokeWidth="1.5" />
      ))}
    </svg>
  );
}

// ─────────────────────────────────────────────
// LANDMARK ORDER & COLORS
// ─────────────────────────────────────────────
const LM_ORDER = ["Ear","Shoulder","Hip","Knee","Ankle","Elbow","Wrist","Index"];
const LM_COLORS: Record<string,string> = {
  Ear:"#f97316",Shoulder:"#2563eb",Hip:"#7c3aed",Knee:"#059669",
  Ankle:"#db2777",Elbow:"#d97706",Wrist:"#0891b2",Index:"#65a30d",
};
const LM_HINTS: Record<string,string> = {
  Ear:"Titik telinga", Shoulder:"Sendi bahu", Hip:"Sendi pinggul",
  Knee:"Sendi lutut", Ankle:"Pergelangan kaki",
  Elbow:"Sendi siku", Wrist:"Pergelangan tangan", Index:"Ujung jari (opsional)",
};

// ─────────────────────────────────────────────
// CANVAS ANNOTATOR
// ─────────────────────────────────────────────
interface AnnotatorProps {
  imageBase64: string;
  onComplete: (pts: ManualPoint[], cw: number, ch: number) => void;
}

function CanvasAnnotator({ imageBase64, onComplete }: AnnotatorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img,    setImg]    = useState<HTMLImageElement | null>(null);
  const [points, setPoints] = useState<ManualPoint[]>([]);
  const [step,   setStep]   = useState(0);
  const [done,   setDone]   = useState(false);

  useEffect(() => {
    setPoints([]); setStep(0); setDone(false);
    const image = new window.Image();
    image.crossOrigin = "anonymous";
    image.src = imageBase64;
    image.onload = () => setImg(image);
  }, [imageBase64]);

  const draw = useCallback((pts: ManualPoint[], imgEl: HTMLImageElement | null) => {
    const cv = canvasRef.current;
    if (!cv || !imgEl) return;
    const ctx = cv.getContext("2d")!;
    cv.width = imgEl.naturalWidth; cv.height = imgEl.naturalHeight;
    ctx.drawImage(imgEl, 0, 0);

    const lw = Math.max(2, cv.width * 0.004);
    const drawLine = (l1: string, l2: string, c: string) => {
      const p1 = pts.find(p=>p.label===l1), p2 = pts.find(p=>p.label===l2);
      if (!p1||!p2) return;
      ctx.beginPath(); ctx.moveTo(p1.x,p1.y); ctx.lineTo(p2.x,p2.y);
      ctx.strokeStyle=c; ctx.lineWidth=lw; ctx.stroke();
    };
    drawLine("Ear","Shoulder","#f97316");     drawLine("Shoulder","Hip","#2563eb");
    drawLine("Hip","Knee","#7c3aed");         drawLine("Knee","Ankle","#059669");
    drawLine("Shoulder","Elbow","#d97706");   drawLine("Elbow","Wrist","#0891b2");
    drawLine("Wrist","Index","#65a30d");

    const r  = Math.max(8, cv.width * 0.014);
    const fs = Math.max(12, cv.width * 0.022);
    pts.forEach(p => {
      const color = LM_COLORS[p.label] || "#ef4444";
      ctx.beginPath(); ctx.arc(p.x,p.y,r+3,0,2*Math.PI);
      ctx.strokeStyle="rgba(255,255,255,0.85)"; ctx.lineWidth=3; ctx.stroke();
      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,2*Math.PI);
      ctx.fillStyle=color; ctx.fill();
      ctx.font=`bold ${fs}px sans-serif`;
      const tw=ctx.measureText(p.label).width;
      ctx.fillStyle="rgba(0,0,0,0.65)";
      (ctx as any).roundRect?.(p.x+r+5,p.y-fs,tw+8,fs+6,4);
      ctx.fill();
      ctx.fillStyle="#fff"; ctx.fillText(p.label, p.x+r+5, p.y);
    });
  }, []);

  useEffect(() => { draw(points, img); }, [points, img, draw]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (done || !canvasRef.current) return;
    const cv = canvasRef.current;
    const rect = cv.getBoundingClientRect();
    const x = (e.clientX-rect.left)*(cv.width/rect.width);
    const y = (e.clientY-rect.top )*(cv.height/rect.height);
    const label = LM_ORDER[step];
    const updated = [...points, { x, y, label }];
    setPoints(updated);
    const next = step + 1;
    if (next >= LM_ORDER.length) {
      setDone(true);
      if (canvasRef.current)
        onComplete(updated, canvasRef.current.width, canvasRef.current.height);
    } else {
      setStep(next);
    }
  };

  const handleSkip = () => {
    setDone(true);
    if (canvasRef.current)
      onComplete(points, canvasRef.current.width, canvasRef.current.height);
  };

  const handleUndo = () => {
    if (points.length === 0) return;
    const updated = points.slice(0, -1);
    setPoints(updated);
    setStep(Math.max(0, step - 1));
    setDone(false);
  };

  const handleReset = () => { setPoints([]); setStep(0); setDone(false); };

  const currentLabel = LM_ORDER[step] ?? "";
  const prog = Math.round((points.length / LM_ORDER.length) * 100);

  return (
    <div>
      {/* Progress */}
      <div style={{ display:"flex",justifyContent:"space-between",fontSize:"11px",color:"#64748b",marginBottom:"4px" }}>
        <span>{points.length}/{LM_ORDER.length} titik</span>
        <span style={{ fontWeight:"700",color:"#2563eb" }}>{prog}%</span>
      </div>
      <div style={{ background:"#e2e8f0",borderRadius:"99px",height:"6px",marginBottom:"8px",overflow:"hidden" }}>
        <div style={{ background:"linear-gradient(90deg,#2563eb,#06b6d4)",height:"100%",width:`${prog}%`,transition:"width 0.25s" }}/>
      </div>

      {/* Instruction */}
      {!done ? (
        <div style={{ background:"#eff6ff",border:"1px solid #bfdbfe",borderRadius:"9px",padding:"8px 12px",marginBottom:"8px",display:"flex",alignItems:"center",gap:"8px" }}>
          <div style={{ width:"24px",height:"24px",borderRadius:"50%",background:LM_COLORS[currentLabel]||"#2563eb",display:"flex",alignItems:"center",justifyContent:"center",color:"#fff",fontSize:"11px",fontWeight:"700",flexShrink:0 }}>
            {step+1}
          </div>
          <div style={{ flex:1 }}>
            <span style={{ fontSize:"12px",fontWeight:"700",color:"#1e40af" }}>
              Klik: <span style={{ color:LM_COLORS[currentLabel] }}>{currentLabel}</span>
            </span>
            <div style={{ fontSize:"10px",color:"#64748b" }}>{LM_HINTS[currentLabel]}</div>
          </div>
          {(currentLabel==="Index"||step>=5) && (
            <button onClick={handleSkip}
              style={{ background:"#f1f5f9",border:"1px solid #e2e8f0",borderRadius:"6px",color:"#64748b",padding:"3px 9px",cursor:"pointer",fontSize:"10px",fontWeight:"600" }}>
              Skip →
            </button>
          )}
        </div>
      ) : (
        <div style={{ background:"#f0fdf4",border:"1px solid #bbf7d0",borderRadius:"9px",padding:"8px 12px",marginBottom:"8px",fontSize:"12px",color:"#15803d",fontWeight:"700" }}>
          ✅ Semua titik selesai! Simpan di bawah.
        </div>
      )}

      {/* Canvas */}
      <div style={{ background:"#f8fafc",borderRadius:"10px",border:"1px solid #e2e8f0",overflow:"hidden",cursor:done?"default":"crosshair" }}>
        {img
          ? <canvas ref={canvasRef} style={{ width:"100%",height:"auto",display:"block" }} onClick={handleClick}/>
          : <div style={{ padding:"40px",textAlign:"center",color:"#94a3b8",fontSize:"12px" }}>Memuat gambar...</div>
        }
      </div>

      <div style={{ display:"flex",gap:"8px",marginTop:"8px" }}>
        <button onClick={handleUndo} disabled={points.length===0}
          style={{ padding:"5px 12px",background:points.length===0?"#f1f5f9":"#fffbeb",border:`1px solid ${points.length===0?"#e2e8f0":"#fde68a"}`,borderRadius:"7px",color:points.length===0?"#cbd5e1":"#92400e",cursor:points.length===0?"not-allowed":"pointer",fontSize:"11px",fontWeight:"600" }}>
          ↩ Undo
        </button>
        <button onClick={handleReset}
          style={{ padding:"5px 12px",background:"#fee2e2",border:"1px solid #fecaca",borderRadius:"7px",color:"#dc2626",cursor:"pointer",fontSize:"11px",fontWeight:"600" }}>
          🔄 Reset
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// RISK HELPERS
// ─────────────────────────────────────────────
function riskOf(score: number) {
  if (score<=1)  return { label:"Negligible",color:"#16a34a",bg:"#dcfce7",bd:"#86efac" };
  if (score<=3)  return { label:"Low",       color:"#ca8a04",bg:"#fef9c3",bd:"#fde047" };
  if (score<=7)  return { label:"Medium",    color:"#ea580c",bg:"#ffedd5",bd:"#fdba74" };
  if (score<=10) return { label:"High",      color:"#dc2626",bg:"#fee2e2",bd:"#fca5a5" };
  return           { label:"Very High",      color:"#991b1b",bg:"#fee2e2",bd:"#f87171" };
}

const POSE_LABELS = [
  "Berdiri tegak","Membungkuk ringan","Membungkuk sedang","Membungkuk berat",
  "Jongkok","Duduk tegak","Duduk membungkuk","Berdiri dengan beban",
  "Mengangkat objek","Menjangkau ke atas","Postur lainnya",
];

// ─────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────
// ─────────────────────────────────────────────
// EXPORTED TYPES (untuk integrasi dengan App.tsx)
// ─────────────────────────────────────────────
export interface TrainingPoseResult {
  poseLabel: string;
  rebaScores: REBAScoreSet;
  angles: AngleSet;
}

interface REBATrainingStudioProps {
  /** Dipanggil saat pose berhasil disimpan atau diprediksi dan diterapkan ke App.tsx */
  onPoseApplied?: (result: TrainingPoseResult) => void;
  /** Dipanggil saat database pose berubah (save/delete/import) agar App.tsx dapat memuat ulang data training */
  onDatabaseUpdated?: () => void;
}

export default function REBATrainingStudio({ onPoseApplied, onDatabaseUpdated }: REBATrainingStudioProps = {}) {
  const [tab,        setTab]        = useState<"collect"|"database"|"predict">("collect");
  const [poses,      setPoses]      = useState<PoseSample[]>([]);
  const [dbReady,    setDbReady]    = useState(false);
  const [editingLabelId, setEditingLabelId] = useState<string|null>(null);
  const [editingLabelValue, setEditingLabelValue] = useState("");

  // ── Collect tab ──
  const [imageB64,   setImageB64]   = useState("");
  const [curPoints,  setCurPoints]  = useState<ManualPoint[]>([]);
  const [curCW,      setCurCW]      = useState(1);
  const [curCH,      setCurCH]      = useState(1);
  const [poseLabel,  setPoseLabel]  = useState(POSE_LABELS[0]);
  const [notes,      setNotes]      = useState("");
  const [saving,     setSaving]     = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);
  const [liveAngles, setLiveAngles] = useState<AngleSet|null>(null);
  const [liveReba,   setLiveReba]   = useState<REBAScoreSet|null>(null);

  const resetLiveReba = useCallback(() => {
    if (!liveAngles) return;
    setLiveReba(computeREBA(liveAngles));
  }, [liveAngles]);

  const updateLiveRebaPosture = useCallback((key: keyof REBAScoreSet, value: number) => {
    if (!liveReba) return;
    const next = deriveRebaFromPostures({ ...liveReba, [key]: value });
    setLiveReba(next);
  }, [liveReba]);

  // ── Predict tab ──
  const [predB64,    setPredB64]    = useState("");
  const [predPoints, setPredPoints] = useState<ManualPoint[]>([]);
  const [predCW,     setPredCW]     = useState(1);
  const [predCH,     setPredCH]     = useState(1);
  const [predResult, setPredResult] = useState<{ results:KNNResult[]; predicted:REBAScoreSet|null }|null>(null);
  const [predAngles, setPredAngles] = useState<AngleSet|null>(null);
  const [kVal,       setKVal]       = useState(3);

  const [dbError,      setDbError]      = useState<string|null>(null);
  const [importStatus, setImportStatus] = useState<{ ok:number; fail:number; msg:string }|null>(null);

  const fileCollect = useRef<HTMLInputElement>(null);
  const filePred    = useRef<HTMLInputElement>(null);
  const fileImport  = useRef<HTMLInputElement>(null);

  // Load DB — dengan error handling & fallback info
  useEffect(() => {
    dbLoadAll()
      .then(data => { setPoses(data); setDbReady(true); })
      .catch(err  => {
        console.error("dbLoadAll error:", err);
        setDbError("Gagal memuat database. Data mungkin dari localStorage.");
        setDbReady(true);
      });
  }, []);

  // ── Upload collect ──
  const handleCollectUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      setImageB64(ev.target?.result as string);
      setCurPoints([]); setLiveAngles(null); setLiveReba(null);
    };
    reader.readAsDataURL(file);
  };

  // ── Annotator complete ──
  const handleAnnotateComplete = useCallback((pts: ManualPoint[], cw: number, ch: number) => {
    setCurPoints(pts); setCurCW(cw); setCurCH(ch);
    const a = computeAnglesFromPoints(pts);
    const r = computeREBA(a);
    setLiveAngles(a); setLiveReba(r);
  }, []);

  // ── Make thumbnail ──
  const makeThumbnail = (b64: string): Promise<string> =>
    new Promise(resolve => {
      const img = new window.Image();
      img.src = b64;
      img.onload = () => {
        const MAX = 200;
        const scale = Math.min(MAX/img.width, MAX/img.height, 1);
        const oc = document.createElement("canvas");
        oc.width  = Math.round(img.width  * scale);
        oc.height = Math.round(img.height * scale);
        oc.getContext("2d")!.drawImage(img, 0, 0, oc.width, oc.height);
        resolve(oc.toDataURL("image/jpeg", 0.7));
      };
      img.onerror = () => resolve("");
    });

  // ── Save sample ──
  const handleSave = async () => {
    if (curPoints.length < 2) return alert("Tandai minimal titik Shoulder dan Hip.");
    if (!liveAngles || !liveReba) return alert("Selesaikan anotasi terlebih dahulu.");
    setSaving(true);
    const thumb = await makeThumbnail(imageB64);
    const sample: PoseSample = {
      id:        `pose_${Date.now()}`,
      timestamp: new Date().toISOString(),
      poseLabel, notes,
      angles:    liveAngles,
      rebaScores:liveReba,
      landmarks: curPoints,
      canvasW:   curCW,
      canvasH:   curCH,
      thumbnail: thumb,
      verified:  true,
    };
    await dbSave(sample);
    setPoses(prev => [...prev, sample]);
    setSaving(false); setSavedFlash(true);
    setTimeout(() => setSavedFlash(false), 2500);
    if (onDatabaseUpdated) onDatabaseUpdated();
    // ── Kirim ke App.tsx jika ada callback ──
    if (onPoseApplied) {
      onPoseApplied({ poseLabel, rebaScores: liveReba, angles: liveAngles });
    }
    setImageB64(""); setCurPoints([]); setNotes(""); setLiveAngles(null); setLiveReba(null);
    if (fileCollect.current) fileCollect.current.value = "";
  };

  // ── Delete ──
  const handleDelete = async (id: string) => {
    await dbRemove(id);
    setPoses(prev => prev.filter(p => p.id !== id));
    if (onDatabaseUpdated) onDatabaseUpdated();
  };

  // ── Rename label ──
  const handleRenameStart = (id: string, currentLabel: string) => {
    setEditingLabelId(id);
    setEditingLabelValue(currentLabel);
  };

  const handleRenameCancel = () => {
    setEditingLabelId(null);
    setEditingLabelValue("");
  };

  const handleRenameSave = async (id: string) => {
    const trimmed = editingLabelValue.trim();
    if (!trimmed) return alert("Nama pose tidak boleh kosong.");
    const target = poses.find(p => p.id === id);
    if (!target) return;
    const updated = { ...target, poseLabel: trimmed };
    await dbSave(updated);
    setPoses(prev => prev.map(p => p.id === id ? updated : p));
    setEditingLabelId(null);
    setEditingLabelValue("");
    if (onDatabaseUpdated) onDatabaseUpdated();
  };

  // ── Export / Import ──
  const handleExport = () => {
    const blob = new Blob([JSON.stringify(poses, null, 2)], { type:"application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `reba_poses_${Date.now()}.json`;
    a.click();
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    // Reset input agar file yang sama bisa diupload lagi
    e.target.value = "";
    setImportStatus(null);
    const reader = new FileReader();
    reader.onload = async ev => {
      try {
        const raw = JSON.parse(ev.target?.result as string);
        // Bisa array langsung atau wrapped dalam objek
        const items: any[] = Array.isArray(raw) ? raw : (Array.isArray(raw.poses) ? raw.poses : [raw]);
        if (items.length === 0) { setImportStatus({ ok:0, fail:0, msg:"File kosong atau format tidak dikenali." }); return; }

        let ok = 0, fail = 0;
        const newSamples: PoseSample[] = [];
        const existingIds = new Set(poses.map(p => p.id));

        for (const item of items) {
          const sample = sanitizePoseSample(item);
          if (!sample) { fail++; continue; }
          // Jika id sudah ada, buat id baru agar tidak overwrite
          if (existingIds.has(sample.id)) {
            sample.id = `pose_import_${Date.now()}_${Math.random().toString(36).slice(2)}`;
          }
          try {
            await dbSave(sample);
            newSamples.push(sample);
            existingIds.add(sample.id);
            ok++;
          } catch (saveErr) {
            console.error("dbSave error for sample:", sample.id, saveErr);
            fail++;
          }
        }

        if (newSamples.length > 0) {
          setPoses(prev => [...prev, ...newSamples]);
          if (onDatabaseUpdated) onDatabaseUpdated();
        }
        setImportStatus({
          ok, fail,
          msg: fail === 0
            ? `✅ ${ok} pose berhasil diimpor dan disimpan permanen.`
            : `⚠️ ${ok} berhasil, ${fail} gagal (data tidak valid).`,
        });
      } catch (parseErr) {
        setImportStatus({ ok:0, fail:0, msg:"❌ File bukan JSON valid. Pastikan format benar." });
      }
    };
    reader.readAsText(file);
  };

  // ── Predict ──
  const handlePredAnnotateComplete = useCallback((pts: ManualPoint[], cw: number, ch: number) => {
    setPredPoints(pts); setPredCW(cw); setPredCH(ch);
    const a = computeAnglesFromPoints(pts);
    setPredAngles(a);
    const res = knnPredict(a, poses, kVal);
    setPredResult(res);
  }, [poses, kVal]);

  // Re-run k-NN saat k berubah
  useEffect(() => {
    if (predAngles && poses.length > 0) {
      setPredResult(knnPredict(predAngles, poses, kVal));
    }
  }, [kVal, predAngles, poses]);

  // ─────────────────────────────────────────
  // UI
  // ─────────────────────────────────────────
  const TAB_ITEMS = [
    { key:"collect",  label:"📷 Tambah Pose",   badge:`${poses.length} tersimpan` },
    { key:"database", label:"🗂 Database Pose",  badge:`${poses.length} pose` },
    { key:"predict",  label:"🔮 Prediksi",       badge:poses.length>0?"k-NN aktif":"Perlu data" },
  ];

  return (
    <div style={{ minHeight:"100vh",background:"#f1f5f9",fontFamily:"system-ui,sans-serif" }}>

      {/* Header */}
      <div style={{ background:"#fff",borderBottom:"1px solid #e2e8f0",padding:"0 24px",boxShadow:"0 1px 4px rgba(0,0,0,0.07)" }}>
        <div style={{ maxWidth:"1200px",margin:"0 auto",height:"56px",display:"flex",alignItems:"center",justifyContent:"space-between" }}>
          <div style={{ display:"flex",alignItems:"center",gap:"12px" }}>
            <div style={{ width:"36px",height:"36px",borderRadius:"10px",background:"linear-gradient(135deg,#7c3aed,#6d28d9)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:"18px" }}>🧬</div>
            <div>
              <h1 style={{ fontSize:"16px",fontWeight:"800",color:"#1e293b",margin:0 }}>REBA Training Studio</h1>
              <p style={{ fontSize:"11px",color:"#94a3b8",margin:0 }}>K-NN Pose Similarity — simpan pose, prediksi otomatis</p>
            </div>
          </div>
          <div style={{ display:"flex",alignItems:"center",gap:"8px" }}>
            <div style={{ padding:"4px 12px",background:poses.length>0?"#f0fdf4":"#f8fafc",border:`1px solid ${poses.length>0?"#bbf7d0":"#e2e8f0"}`,borderRadius:"99px" }}>
              <span style={{ fontSize:"11px",fontWeight:"700",color:poses.length>0?"#16a34a":"#94a3b8" }}>
                {poses.length>0?`🟢 ${poses.length} pose tersimpan`:"⚪ Belum ada data"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background:"#fff",borderBottom:"1px solid #e2e8f0" }}>
        <div style={{ maxWidth:"1200px",margin:"0 auto",display:"flex" }}>
          {TAB_ITEMS.map(t => (
            <button key={t.key} onClick={()=>setTab(t.key as any)} style={{
              padding:"12px 20px",border:"none",background:"transparent",
              borderBottom:tab===t.key?"2px solid #7c3aed":"2px solid transparent",
              color:tab===t.key?"#7c3aed":"#64748b",
              cursor:"pointer",fontSize:"13px",fontWeight:tab===t.key?"700":"500",
              transition:"all 0.15s",
            }}>
              {t.label}
              <span style={{ marginLeft:"6px",fontSize:"10px",padding:"1px 7px",borderRadius:"99px",
                background:tab===t.key?"#f5f3ff":"#f1f5f9",
                color:tab===t.key?"#7c3aed":"#94a3b8",fontWeight:"700" }}>
                {t.badge}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth:"1200px",margin:"0 auto",padding:"24px 16px" }}>

        {/* ═══ TAB 1: COLLECT ═══ */}
        {tab === "collect" && (
          <div style={{ display:"grid",gridTemplateColumns:"1fr 320px",gap:"20px",alignItems:"start" }}>

            {/* Canvas panel */}
            <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"20px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
              <h2 style={{ fontSize:"15px",fontWeight:"700",color:"#1e293b",margin:"0 0 14px" }}>
                📷 Anotasi Pose Baru
              </h2>

              {!imageB64 ? (
                <div onClick={()=>fileCollect.current?.click()}
                  style={{ border:"2px dashed #e2e8f0",borderRadius:"12px",padding:"48px 20px",textAlign:"center",cursor:"pointer",background:"#f8fafc",transition:"all 0.2s" }}
                  onMouseEnter={e=>{e.currentTarget.style.borderColor="#7c3aed";e.currentTarget.style.background="#faf5ff"}}
                  onMouseLeave={e=>{e.currentTarget.style.borderColor="#e2e8f0";e.currentTarget.style.background="#f8fafc"}}>
                  <div style={{ fontSize:"40px",marginBottom:"12px" }}>📸</div>
                  <p style={{ color:"#374151",fontSize:"13px",fontWeight:"600",margin:"0 0 5px" }}>Klik untuk upload foto pose</p>
                  <p style={{ color:"#94a3b8",fontSize:"11px",margin:0 }}>JPG, PNG — Tampak samping direkomendasikan</p>
                </div>
              ) : (
                <div>
                  <CanvasAnnotator imageBase64={imageB64} onComplete={handleAnnotateComplete}/>
                  <button onClick={()=>{ setImageB64(""); setCurPoints([]); setLiveAngles(null); setLiveReba(null); if(fileCollect.current)fileCollect.current.value=""; }}
                    style={{ marginTop:"8px",padding:"5px 12px",background:"#f1f5f9",border:"1px solid #e2e8f0",borderRadius:"7px",color:"#64748b",cursor:"pointer",fontSize:"11px",fontWeight:"600" }}>
                    🔄 Ganti Foto
                  </button>
                </div>
              )}
            </div>

            {/* Right panel */}
            <div style={{ display:"flex",flexDirection:"column",gap:"14px" }}>

              {/* Info form */}
              <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"16px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
                <h3 style={{ fontSize:"13px",fontWeight:"700",color:"#1e293b",margin:"0 0 12px" }}>📝 Label Pose</h3>
                <select value={poseLabel} onChange={e=>setPoseLabel(e.target.value)}
                  style={{ width:"100%",padding:"8px 10px",border:"1px solid #e2e8f0",borderRadius:"8px",fontSize:"12px",color:"#374151",background:"#f8fafc",outline:"none",marginBottom:"8px" }}>
                  {POSE_LABELS.map(l=><option key={l}>{l}</option>)}
                </select>
                <textarea value={notes} onChange={e=>setNotes(e.target.value)} rows={2}
                  placeholder="Catatan (opsional)..."
                  style={{ width:"100%",padding:"7px 10px",border:"1px solid #e2e8f0",borderRadius:"8px",fontSize:"12px",color:"#374151",background:"#f8fafc",outline:"none",resize:"vertical",boxSizing:"border-box" }}/>
              </div>

              {/* Live preview stickman + angles */}
              {liveAngles && liveReba && curPoints.length >= 2 && (
                <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"14px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
                  <h3 style={{ fontSize:"12px",fontWeight:"700",color:"#1e293b",margin:"0 0 10px" }}>📐 Preview Pose</h3>
                  <div style={{ display:"flex",alignItems:"flex-start",gap:"12px",marginBottom:"10px" }}>
                    <StickmanSVG landmarks={curPoints} cw={curCW} ch={curCH} size={100}/>
                    <div style={{ flex:1,display:"flex",flexDirection:"column",gap:"4px" }}>
                      {Object.entries(liveAngles).filter(([,v])=>v>0).map(([k,v])=>(
                        <div key={k} style={{ display:"flex",justifyContent:"space-between",padding:"3px 8px",background:"#f8fafc",borderRadius:"5px",fontSize:"10px" }}>
                          <span style={{ color:"#64748b",textTransform:"capitalize" }}>{k}</span>
                          <span style={{ fontWeight:"700",color:"#2563eb" }}>{v}°</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  {(() => {
                    const risk = riskOf(liveReba.final);
                    return (
                      <div style={{ background:risk.bg,border:`1px solid ${risk.bd}`,borderRadius:"8px",padding:"8px 12px",display:"flex",justifyContent:"space-between",alignItems:"center" }}>
                        <span style={{ fontSize:"12px",fontWeight:"700",color:risk.color }}>REBA Score</span>
                        <div style={{ display:"flex",alignItems:"center",gap:"8px" }}>
                          <span style={{ fontSize:"24px",fontWeight:"900",color:risk.color }}>{liveReba.final}</span>
                          <span style={{ fontSize:"11px",fontWeight:"700",color:risk.color,padding:"2px 7px",background:"rgba(255,255,255,0.6)",borderRadius:"6px" }}>{risk.label}</span>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              )}

              {liveAngles && liveReba && curPoints.length >= 2 && (
                <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"14px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
                  <div style={{ display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"12px" }}>
                    <div>
                      <h3 style={{ fontSize:"12px",fontWeight:"700",color:"#1e293b",margin:0 }}>✏️ Edit Manual Skor</h3>
                      <p style={{ margin:"6px 0 0",fontSize:"11px",color:"#475569" }}>Tutor: derajat = score ini.</p>
                      <p style={{ margin:"4px 0 0",fontSize:"11px",color:"#64748b" }}>Contoh: sudut kecil = skor rendah, sudut besar = skor tinggi.</p>
                    </div>
                    <button onClick={resetLiveReba}
                      style={{ padding:"4px 8px",border:"1px solid #e2e8f0",borderRadius:"8px",background:"#f8fafc",color:"#475569",fontSize:"11px",cursor:"pointer" }}>
                      Reset Otomatis
                    </button>
                  </div>
                  {([
                    {
                      key:"neck", label:"Neck", options:[
                        { value:1, label:"1 — aligned/sedikit maju (0–20°)" },
                        { value:2, label:"2 — kepala jauh dari trunk (>20°)" },
                      ],
                    },
                    {
                      key:"trunk", label:"Trunk", options:[
                        { value:1, label:"1 — netral tegak" },
                        { value:2, label:"2 — fleksi ringan (0–20°)" },
                        { value:3, label:"3 — fleksi sedang (20–60°)" },
                        { value:4, label:"4 — fleksi berat/ekstensi (>60° atau >20°)" },
                      ],
                    },
                    {
                      key:"legs", label:"Legs", options:[
                        { value:1, label:"1 — kedua kaki turun / duduk stabil" },
                        { value:2, label:"2 — satu kaki terangkat atau tidak rata" },
                        { value:3, label:"3 — sudut lutut besar / tidak stabil" },
                      ],
                    },
                    {
                      key:"upperArm", label:"Upper Arm", options:[
                        { value:1, label:"1 — 0–20°" },
                        { value:2, label:"2 — 20–45°" },
                        { value:3, label:"3 — 45–90°" },
                        { value:4, label:"4 — >90°" },
                      ],
                    },
                    {
                      key:"lowerArm", label:"Lower Arm", options:[
                        { value:1, label:"1 — siku 60–100° (ergonomis)" },
                        { value:2, label:"2 — siku terlalu kecil atau terlalu besar" },
                      ],
                    },
                    {
                      key:"wrist", label:"Wrist", options:[
                        { value:1, label:"1 — pergelangan lurus (≤15°)" },
                        { value:2, label:"2 — pergelangan menekuk (>15°)" },
                      ],
                    },
                  ] as const).map(field => (
                    <label key={field.key} style={{ display:"flex",flexDirection:"column",gap:"6px",marginBottom:"10px",fontSize:"11px",color:"#475569" }}>
                      <span style={{ fontWeight:"700" }}>{field.label}</span>
                      <select value={liveReba[field.key]}
                        onChange={e => updateLiveRebaPosture(field.key, Number(e.target.value))}
                        style={{ width:"100%",padding:"8px 10px",borderRadius:"8px",border:"1px solid #e2e8f0",background:"#f8fafc",fontSize:"12px",color:"#1e293b" }}>
                        {field.options.map(opt => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </label>
                  ))}
                  <div style={{ marginTop:"8px",padding:"10px",background:"#f8fafc",borderRadius:"12px",border:"1px dashed #c7d2fe",fontSize:"11px",color:"#475569" }}>
                    Skor A/B/C/final dihitung otomatis berdasarkan nilai posture yang diedit.
                  </div>
                </div>
              )}

              {/* Save button */}
              <button onClick={handleSave}
                disabled={saving||!imageB64||curPoints.length<2||!liveReba}
                style={{
                  padding:"12px",width:"100%",
                  background:saving||!imageB64||curPoints.length<2||!liveReba?"#e2e8f0":"linear-gradient(135deg,#7c3aed,#6d28d9)",
                  border:"none",borderRadius:"10px",
                  color:saving||!imageB64||curPoints.length<2||!liveReba?"#94a3b8":"#fff",
                  cursor:saving||!imageB64||curPoints.length<2||!liveReba?"not-allowed":"pointer",
                  fontSize:"14px",fontWeight:"700",
                  boxShadow:saving||!imageB64||curPoints.length<2||!liveReba?"none":"0 4px 12px rgba(124,58,237,0.3)",
                }}>
                {saving?"⏳ Menyimpan...":savedFlash?"✅ Tersimpan!":curPoints.length<2?"Tandai minimal Shoulder+Hip":"💾 Simpan ke Database Pose"}
              </button>

              <div style={{ background:"#fffbeb",border:"1px solid #fde68a",borderRadius:"10px",padding:"12px",fontSize:"11px",color:"#78716c",lineHeight:"1.8" }}>
                <strong style={{ color:"#92400e" }}>💡 Cara kerja:</strong><br/>
                Setiap pose yang disimpan menjadi "referensi". Saat prediksi, sistem mencari pose yang paling mirip berdasarkan sudut tubuh dan menggunakan skor REBA-nya.<br/><br/>
                <strong style={{ color:"#92400e" }}>Target:</strong> simpan minimal 3 pose berbeda per kategori risiko untuk hasil terbaik.
              </div>
            </div>
          </div>
        )}

        {/* ═══ TAB 2: DATABASE ═══ */}
        {tab === "database" && (
          <div>
            {/* Toolbar */}
            <div style={{ background:"#fff",borderRadius:"12px",border:"1px solid #e2e8f0",padding:"12px 16px",marginBottom:"16px",display:"flex",alignItems:"center",gap:"10px",flexWrap:"wrap",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
              <span style={{ fontSize:"14px",fontWeight:"700",color:"#1e293b",marginRight:"auto" }}>
                🗂 {poses.length} Pose Tersimpan
              </span>
              <button onClick={handleExport} disabled={poses.length===0}
                style={{ padding:"6px 14px",background:poses.length===0?"#f1f5f9":"#eff6ff",border:`1px solid ${poses.length===0?"#e2e8f0":"#bfdbfe"}`,borderRadius:"8px",color:poses.length===0?"#94a3b8":"#2563eb",cursor:poses.length===0?"not-allowed":"pointer",fontSize:"12px",fontWeight:"600" }}>
                📤 Export JSON
              </button>
              <button onClick={()=>fileImport.current?.click()}
                style={{ padding:"6px 14px",background:"#ecfdf5",border:"1px solid #a7f3d0",borderRadius:"8px",color:"#059669",cursor:"pointer",fontSize:"12px",fontWeight:"600" }}>
                📥 Import JSON
              </button>
            </div>

            {/* Import status feedback */}
            {importStatus && (
              <div style={{
                marginBottom:"14px", padding:"10px 14px",
                background: importStatus.fail>0 ? "#fffbeb" : "#f0fdf4",
                border:`1px solid ${importStatus.fail>0?"#fde68a":"#bbf7d0"}`,
                borderRadius:"10px", fontSize:"12px",
                color: importStatus.fail>0 ? "#92400e" : "#15803d",
                fontWeight:"600", display:"flex", alignItems:"center", justifyContent:"space-between", gap:"8px",
              }}>
                <span>{importStatus.msg}</span>
                <button onClick={()=>setImportStatus(null)}
                  style={{ background:"none",border:"none",cursor:"pointer",fontSize:"14px",color:"inherit",padding:"0 2px" }}>✕</button>
              </div>
            )}

            {/* DB error banner */}
            {dbError && (
              <div style={{ marginBottom:"14px",padding:"10px 14px",background:"#fff7ed",border:"1px solid #fed7aa",borderRadius:"10px",fontSize:"12px",color:"#c2410c",fontWeight:"600" }}>
                ⚠️ {dbError}
              </div>
            )}

            {poses.length === 0 ? (
              <div style={{ background:"#fff",borderRadius:"14px",border:"2px dashed #e2e8f0",padding:"60px",textAlign:"center",color:"#94a3b8" }}>
                <div style={{ fontSize:"48px",marginBottom:"12px" }}>📭</div>
                <p style={{ fontSize:"14px",fontWeight:"600",margin:"0 0 6px" }}>Belum ada pose tersimpan</p>
                <p style={{ fontSize:"12px",margin:0 }}>Pergi ke tab "Tambah Pose" untuk mulai mengumpulkan data.</p>
              </div>
            ) : (
              <>
                {/* Stats row */}
                <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(150px,1fr))",gap:"10px",marginBottom:"16px" }}>
                  {[
                    { label:"Total Pose",  value:poses.length,                                           color:"#7c3aed",bg:"#f5f3ff" },
                    { label:"Pose Unik",   value:new Set(poses.map(p=>p.poseLabel)).size,               color:"#2563eb",bg:"#eff6ff" },
                    { label:"Avg REBA",    value:(poses.reduce((a,p)=>a+p.rebaScores.final,0)/poses.length).toFixed(1), color:"#ea580c",bg:"#fff7ed" },
                    { label:"Max REBA",    value:Math.max(...poses.map(p=>p.rebaScores.final)),         color:"#dc2626",bg:"#fef2f2" },
                  ].map(({label,value,color,bg})=>(
                    <div key={label} style={{ background:"#fff",borderLeft:`4px solid ${color}`,border:`1px solid ${bg}`,borderRadius:"10px",padding:"10px 14px",boxShadow:"0 1px 3px rgba(0,0,0,0.05)" }}>
                      <p style={{ fontSize:"9px",fontWeight:"700",color:"#64748b",textTransform:"uppercase",margin:"0 0 4px" }}>{label}</p>
                      <p style={{ fontSize:"20px",fontWeight:"800",color,margin:0 }}>{value}</p>
                    </div>
                  ))}
                </div>

                {/* Pose grid */}
                <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(240px,1fr))",gap:"12px" }}>
                  {poses.map(pose => {
                    const risk = riskOf(pose.rebaScores.final);
                    return (
                      <div key={pose.id} style={{ background:"#fff",borderRadius:"12px",border:"1px solid #e2e8f0",overflow:"hidden",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
                        {/* Thumbnail + stickman */}
                        <div style={{ position:"relative",display:"flex",gap:0 }}>
                          {pose.thumbnail ? (
                            <img src={pose.thumbnail} alt="" style={{ width:"50%",height:"100px",objectFit:"cover",display:"block" }}/>
                          ) : (
                            <div style={{ width:"50%",height:"100px",background:"#f1f5f9",display:"flex",alignItems:"center",justifyContent:"center",fontSize:"24px" }}>📷</div>
                          )}
                          <div style={{ width:"50%",height:"100px",display:"flex",alignItems:"center",justifyContent:"center",background:"#f8fafc",padding:"8px",boxSizing:"border-box" }}>
                            <StickmanSVG landmarks={pose.landmarks} cw={pose.canvasW} ch={pose.canvasH} size={84}/>
                          </div>
                          <div style={{ position:"absolute",top:"6px",right:"6px",background:risk.bg,border:`1px solid ${risk.bd}`,borderRadius:"5px",padding:"1px 7px",fontSize:"10px",fontWeight:"700",color:risk.color }}>
                            {pose.rebaScores.final} · {risk.label}
                          </div>
                        </div>

                        <div style={{ padding:"10px 12px" }}>
                          {editingLabelId === pose.id ? (
                            <div style={{ display:"flex",flexDirection:"column",gap:"8px",marginBottom:"8px" }}>
                              <input value={editingLabelValue}
                                onChange={e=>setEditingLabelValue(e.target.value)}
                                style={{ width:"100%",padding:"8px 10px",borderRadius:"8px",border:"1px solid #cbd5e1",fontSize:"12px" }}
                                placeholder="Nama pose baru..." />
                              <div style={{ display:"flex",gap:"8px" }}>
                                <button onClick={()=>handleRenameSave(pose.id)}
                                  style={{ flex:1,padding:"8px 10px",border:"none",borderRadius:"8px",background:"#2563eb",color:"#fff",fontSize:"12px",cursor:"pointer" }}>
                                  💾 Simpan
                                </button>
                                <button onClick={handleRenameCancel}
                                  style={{ flex:1,padding:"8px 10px",border:"1px solid #e2e8f0",borderRadius:"8px",background:"#f8fafc",color:"#475569",fontSize:"12px",cursor:"pointer" }}>
                                  Batal
                                </button>
                              </div>
                            </div>
                          ) : (
                            <p style={{ fontSize:"12px",fontWeight:"700",color:"#1e293b",margin:"0 0 6px",display:"flex",justifyContent:"space-between",alignItems:"center",gap:"8px" }}>
                              <span>{pose.poseLabel}</span>
                              <button onClick={()=>handleRenameStart(pose.id, pose.poseLabel)}
                                style={{ padding:"3px 8px",background:"#e0e7ff",border:"none",borderRadius:"6px",color:"#3730a3",cursor:"pointer",fontSize:"10px",fontWeight:"700" }}>
                                ✏️ Ubah
                              </button>
                            </p>
                          )}
                          {/* Angles mini grid */}
                          <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"3px",marginBottom:"8px" }}>
                            {Object.entries(pose.angles).filter(([,v])=>v>0).map(([k,v])=>(
                              <div key={k} style={{ background:"#f8fafc",borderRadius:"4px",padding:"3px 5px",display:"flex",justifyContent:"space-between" }}>
                                <span style={{ fontSize:"9px",color:"#64748b",textTransform:"capitalize" }}>{k.replace("upperArm","uArm").replace("lowerArm","lArm")}</span>
                                <span style={{ fontSize:"9px",fontWeight:"700",color:"#2563eb" }}>{v}°</span>
                              </div>
                            ))}
                          </div>
                          <div style={{ display:"flex",justifyContent:"space-between",alignItems:"center",gap:"8px" }}>
                            <span style={{ fontSize:"10px",color:"#94a3b8" }}>{new Date(pose.timestamp).toLocaleDateString("id-ID")}</span>
                            <button onClick={()=>handleDelete(pose.id)}
                              style={{ padding:"6px 10px",background:"#fee2e2",border:"none",borderRadius:"6px",color:"#dc2626",cursor:"pointer",fontSize:"10px",fontWeight:"600" }}>
                              🗑 Hapus
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        )}

        {/* ═══ TAB 3: PREDICT ═══ */}
        {tab === "predict" && (
          <div>
            {poses.length === 0 && (
              <div style={{ background:"#fef2f2",border:"1px solid #fecaca",borderRadius:"10px",padding:"14px 18px",marginBottom:"16px",fontSize:"13px",color:"#b91c1c",fontWeight:"600" }}>
                ⚠️ Belum ada pose tersimpan. Pergi ke tab "Tambah Pose" untuk mengumpulkan data referensi dulu.
              </div>
            )}

            <div style={{ display:"grid",gridTemplateColumns:"1fr 340px",gap:"20px",alignItems:"start" }}>

              {/* Canvas panel */}
              <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"20px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
                <div style={{ display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:"14px" }}>
                  <h2 style={{ fontSize:"15px",fontWeight:"700",color:"#1e293b",margin:0 }}>🔮 Prediksi Pose Baru</h2>
                  <div style={{ display:"flex",alignItems:"center",gap:"8px" }}>
                    <label style={{ fontSize:"11px",color:"#64748b" }}>K tetangga:</label>
                    <select value={kVal} onChange={e=>setKVal(Number(e.target.value))}
                      style={{ padding:"4px 8px",border:"1px solid #e2e8f0",borderRadius:"6px",fontSize:"12px",background:"#f8fafc",outline:"none" }}>
                      {[1,2,3,4,5].map(k=><option key={k} value={k}>{k}</option>)}
                    </select>
                  </div>
                </div>

                {!predB64 ? (
                  <div onClick={()=>filePred.current?.click()}
                    style={{ border:"2px dashed #e2e8f0",borderRadius:"12px",padding:"48px 20px",textAlign:"center",cursor:"pointer",background:"#f8fafc",transition:"all 0.2s" }}
                    onMouseEnter={e=>{e.currentTarget.style.borderColor="#7c3aed";e.currentTarget.style.background="#faf5ff"}}
                    onMouseLeave={e=>{e.currentTarget.style.borderColor="#e2e8f0";e.currentTarget.style.background="#f8fafc"}}>
                    <div style={{ fontSize:"40px",marginBottom:"10px" }}>📸</div>
                    <p style={{ color:"#374151",fontSize:"13px",fontWeight:"600",margin:"0 0 4px" }}>Upload foto untuk diprediksi</p>
                    <p style={{ color:"#94a3b8",fontSize:"11px",margin:0 }}>Tandai titik anatomi → cari pose mirip → dapat skor REBA</p>
                  </div>
                ) : (
                  <div>
                    <CanvasAnnotator imageBase64={predB64} onComplete={handlePredAnnotateComplete}/>
                    <button onClick={()=>{ setPredB64(""); setPredPoints([]); setPredResult(null); setPredAngles(null); if(filePred.current)filePred.current.value=""; }}
                      style={{ marginTop:"8px",padding:"5px 12px",background:"#f1f5f9",border:"1px solid #e2e8f0",borderRadius:"7px",color:"#64748b",cursor:"pointer",fontSize:"11px",fontWeight:"600" }}>
                      🔄 Ganti Foto
                    </button>
                  </div>
                )}
              </div>

              {/* Results panel */}
              <div style={{ display:"flex",flexDirection:"column",gap:"14px" }}>

                {predResult && predResult.predicted ? (
                  <>
                    {/* Apply to Assessment button */}
                    {onPoseApplied && predResult.predicted && (
                      <button
                        onClick={() => {
                          if (!predResult.predicted || !predAngles) return;
                          onPoseApplied({
                            poseLabel: predResult.results[0]?.sample.poseLabel ?? "Prediksi k-NN",
                            rebaScores: predResult.predicted,
                            angles: predAngles,
                          });
                        }}
                        style={{
                          width:"100%", padding:"10px 16px",
                          background:"linear-gradient(135deg,#7c3aed,#6d28d9)",
                          border:"none", borderRadius:"10px",
                          color:"#fff", fontSize:"13px", fontWeight:"700",
                          cursor:"pointer", boxShadow:"0 2px 8px rgba(124,58,237,0.35)",
                          display:"flex", alignItems:"center", justifyContent:"center", gap:"8px",
                        }}
                      >
                        📋 Terapkan ke Assessment
                      </button>
                    )}

                    {/* Final score */}
                    {(() => {
                      const risk = riskOf(predResult.predicted.final);
                      return (
                        <div style={{ background:risk.bg,border:`2px solid ${risk.bd}`,borderRadius:"14px",padding:"16px" }}>
                          <p style={{ fontSize:"11px",fontWeight:"700",color:risk.color,textTransform:"uppercase",margin:"0 0 6px" }}>
                            Prediksi REBA (k-NN, k={kVal})
                          </p>
                          <div style={{ display:"flex",alignItems:"center",justifyContent:"space-between" }}>
                            <span style={{ fontSize:"52px",fontWeight:"900",color:risk.color,lineHeight:1 }}>
                              {predResult.predicted.final}
                            </span>
                            <div style={{ textAlign:"right" }}>
                              <p style={{ fontSize:"16px",fontWeight:"800",color:risk.color,margin:"0 0 4px" }}>{risk.label}</p>
                              <p style={{ fontSize:"11px",color:risk.color,margin:0,opacity:0.8 }}>
                                Mirip {predResult.results[0]?.similarity ?? 0}% dengan<br/>"{predResult.results[0]?.sample.poseLabel}"
                              </p>
                            </div>
                          </div>
                        </div>
                      );
                    })()}

                    {/* Sudut terukur */}
                    {predAngles && (
                      <div style={{ background:"#fff",borderRadius:"12px",border:"1px solid #e2e8f0",padding:"12px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
                        <p style={{ fontSize:"11px",fontWeight:"700",color:"#475569",textTransform:"uppercase",margin:"0 0 8px" }}>Sudut Terukur</p>
                        <div style={{ display:"flex",flexDirection:"column",gap:"4px" }}>
                          {Object.entries(predAngles).filter(([,v])=>v>0).map(([k,v])=>(
                            <div key={k} style={{ display:"flex",justifyContent:"space-between",padding:"5px 9px",background:"#eff6ff",borderRadius:"6px" }}>
                              <span style={{ fontSize:"11px",color:"#64748b",textTransform:"capitalize" }}>{k}</span>
                              <span style={{ fontSize:"12px",fontWeight:"700",color:"#2563eb" }}>{v}°</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* K pose terdekat */}
                    <div style={{ background:"#fff",borderRadius:"12px",border:"1px solid #e2e8f0",padding:"14px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
                      <p style={{ fontSize:"11px",fontWeight:"700",color:"#475569",textTransform:"uppercase",margin:"0 0 10px" }}>
                        {predResult.results.length} Pose Terdekat
                      </p>
                      <div style={{ display:"flex",flexDirection:"column",gap:"10px" }}>
                        {predResult.results.map((r, i) => {
                          const risk = riskOf(r.sample.rebaScores.final);
                          return (
                            <div key={r.sample.id} style={{
                              display:"flex",alignItems:"center",gap:"10px",
                              padding:"10px",borderRadius:"10px",
                              background:i===0?"#f5f3ff":"#f8fafc",
                              border:`1px solid ${i===0?"#ddd6fe":"#e2e8f0"}`,
                            }}>
                              <div style={{ position:"relative",flexShrink:0 }}>
                                <StickmanSVG landmarks={r.sample.landmarks} cw={r.sample.canvasW} ch={r.sample.canvasH} size={64} highlight={i===0}/>
                                <div style={{ position:"absolute",top:-4,left:-4,width:"18px",height:"18px",borderRadius:"50%",background:i===0?"#7c3aed":"#94a3b8",color:"#fff",fontSize:"10px",fontWeight:"800",display:"flex",alignItems:"center",justifyContent:"center" }}>
                                  {i+1}
                                </div>
                              </div>
                              <div style={{ flex:1,minWidth:0 }}>
                                <p style={{ fontSize:"11px",fontWeight:"700",color:"#1e293b",margin:"0 0 3px",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis" }}>
                                  {r.sample.poseLabel}
                                </p>
                                <div style={{ display:"flex",alignItems:"center",gap:"6px",marginBottom:"3px" }}>
                                  <span style={{ fontSize:"10px",padding:"1px 6px",borderRadius:"99px",background:risk.bg,color:risk.color,fontWeight:"700" }}>
                                    REBA {r.sample.rebaScores.final}
                                  </span>
                                  <span style={{ fontSize:"10px",color:"#64748b" }}>
                                    {r.similarity}% mirip
                                  </span>
                                </div>
                                {/* Similarity bar */}
                                <div style={{ background:"#e2e8f0",borderRadius:"99px",height:"4px",overflow:"hidden" }}>
                                  <div style={{ background:i===0?"#7c3aed":"#94a3b8",height:"100%",width:`${r.similarity}%`,borderRadius:"99px" }}/>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div style={{ background:"#fffbeb",border:"1px solid #fde68a",borderRadius:"10px",padding:"11px",fontSize:"11px",color:"#78716c",lineHeight:"1.7" }}>
                      <strong style={{ color:"#92400e" }}>ℹ️ Cara baca:</strong> Skor REBA diambil dari {kVal} pose paling mirip dengan bobot jarak. Semakin tinggi % mirip, semakin yakin hasilnya. Tambah lebih banyak pose referensi untuk akurasi lebih baik.
                    </div>
                  </>
                ) : (
                  <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"40px",textAlign:"center",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
                    <div style={{ fontSize:"40px",marginBottom:"10px" }}>🔮</div>
                    <p style={{ fontSize:"13px",fontWeight:"600",color:"#64748b",margin:"0 0 6px" }}>Hasil muncul di sini</p>
                    <p style={{ fontSize:"11px",color:"#94a3b8",margin:0 }}>Upload foto, tandai minimal Shoulder+Hip, hasil prediksi otomatis tampil.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

      </div>

      {/* Hidden inputs */}
      <input ref={fileCollect} type="file" accept="image/*" onChange={handleCollectUpload} style={{ display:"none" }}/>
      <input ref={filePred}    type="file" accept="image/*" onChange={e=>{ const f=e.target.files?.[0]; if(!f) return; const r=new FileReader(); r.onload=ev=>{ setPredB64(ev.target?.result as string); setPredPoints([]); setPredResult(null); setPredAngles(null); }; r.readAsDataURL(f); }} style={{ display:"none" }}/>
      <input ref={fileImport}  type="file" accept=".json,application/json" onChange={handleImport} style={{ display:"none" }}/>
    </div>
  );
}