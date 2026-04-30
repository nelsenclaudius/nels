import { useState, useMemo, useRef, useEffect, useCallback } from "react";

// ============================================================
// POSE ESTIMATION UTILS (LOGIKA TIDAK BERUBAH)
// ============================================================
declare global {
  interface Window { Pose: any; }
}

const L = {
  LEFT_EAR: 7, RIGHT_EAR: 8,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_INDEX: 17, RIGHT_INDEX: 18,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
};

function angleBetween(a: any, b: any, c: any): number {
  const ab = { x: b.x - a.x, y: b.y - a.y };
  const cb = { x: b.x - c.x, y: b.y - c.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.sqrt(ab.x ** 2 + ab.y ** 2);
  const magCB = Math.sqrt(cb.x ** 2 + cb.y ** 2);
  if (magAB === 0 || magCB === 0) return 0;
  return Math.acos(Math.min(1, Math.max(-1, dot / (magAB * magCB)))) * (180 / Math.PI);
}

function angleWithVertical(a: any, b: any): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.abs(Math.atan2(dx, dy) * (180 / Math.PI));
}

export function computeREBAFromLandmarks(landmarks: any[]) {
  // Threshold lebih ketat (0.65) agar hanya keypoint yang benar-benar visible digunakan
  const get = (idx: number, altIdx: number) => {
    const p = landmarks[idx];
    if (p && p.visibility > 0.65) return p;
    const alt = landmarks[altIdx];
    return alt && alt.visibility > 0.65 ? alt : null;
  };

  const ear      = get(L.LEFT_EAR,      L.RIGHT_EAR);
  const shoulder = get(L.LEFT_SHOULDER, L.RIGHT_SHOULDER);
  const hip      = get(L.LEFT_HIP,      L.RIGHT_HIP);
  const elbow    = get(L.LEFT_ELBOW,    L.RIGHT_ELBOW);
  const wrist    = get(L.LEFT_WRIST,    L.RIGHT_WRIST);
  const index    = get(L.LEFT_INDEX,    L.RIGHT_INDEX);
  const knee     = get(L.LEFT_KNEE,     L.RIGHT_KNEE);
  const ankle    = get(L.LEFT_ANKLE,    L.RIGHT_ANKLE);

  if (!shoulder || !hip || !elbow || !wrist || !knee || !ankle) {
    return {
      neck: { posture: 1, twisted: false, sideBending: false },
      trunk: { posture: 1, twisted: false, sideBending: false },
      legs: 1,
      upperArm: { posture: 1, shoulderRaised: false, armAbducted: false, armSupported: false },
      lowerArm: 1,
      wrist: { posture: 1, wristTwisted: false, wristDeviated: false },
      load: 0, coupling: 0,
    };
  }

  let neckPosture = 1;
  if (ear) {
    const neckAngle = angleWithVertical(ear, shoulder);
    neckPosture = neckAngle > 20 ? 2 : 1;
  }

  const trunkAngle = angleWithVertical(shoulder, hip);
  let trunkPosture = 1;
  if (trunkAngle <= 5) trunkPosture = 1;
  else if (trunkAngle <= 20) trunkPosture = 2;
  else if (trunkAngle <= 60) trunkPosture = 3;
  else trunkPosture = 4;

  const kneeAngle = angleBetween(hip, knee, ankle);
  let legPosture = 1;
  if (kneeAngle > 60) legPosture = 4;
  else if (kneeAngle > 30) legPosture = 3;
  else if (kneeAngle > 0) legPosture = 2;

  const upperArmAngle = angleWithVertical(shoulder, elbow);
  let upperArmPosture = 1;
  if (upperArmAngle <= 20) upperArmPosture = 1;
  else if (upperArmAngle <= 45) upperArmPosture = 2;
  else if (upperArmAngle <= 90) upperArmPosture = 3;
  else upperArmPosture = 4;

  const lowerArmAngle = angleBetween(shoulder, elbow, wrist);
  const lowerArmPosture = (lowerArmAngle >= 60 && lowerArmAngle <= 100) ? 1 : 2;

  let wristPosture = 1;
  if (index) {
    const wristAngle = angleBetween(elbow, wrist, index);
    wristPosture = wristAngle > 15 ? 2 : 1;
  }

  return {
    neck:     { posture: neckPosture, twisted: false, sideBending: false },
    trunk:    { posture: trunkPosture, twisted: false, sideBending: false },
    legs:     legPosture,
    upperArm: { posture: upperArmPosture, shoulderRaised: false, armAbducted: false, armSupported: false },
    lowerArm: lowerArmPosture,
    wrist:    { posture: wristPosture, wristTwisted: false, wristDeviated: false },
    load: 0, coupling: 0,
  };
}

// ============================================================
// MANUAL LANDMARKING UTILS (LOGIKA TIDAK BERUBAH)
// ============================================================
interface LandmarkPoint { x: number; y: number; label: string; }

function computeREBAFromManualPoints(points: LandmarkPoint[]) {
  const get = (label: string) => points.find(p => p.label === label);
  const ear      = get("Ear");
  const shoulder = get("Shoulder");
  const hip      = get("Hip");
  const knee     = get("Knee");
  const ankle    = get("Ankle");
  const elbow    = get("Elbow");
  const wrist    = get("Wrist");
  const index    = get("Index");

  if (!shoulder || !hip || !elbow || !wrist || !knee || !ankle) {
    return {
      neck:     { posture: 1, twisted: false, sideBending: false },
      trunk:    { posture: 1, twisted: false, sideBending: false },
      legs:     1,
      upperArm: { posture: 1, shoulderRaised: false, armAbducted: false, armSupported: false },
      lowerArm: 1,
      wrist:    { posture: 1, wristTwisted: false, wristDeviated: false },
      load: 0, coupling: 0,
    };
  }

  let neckPosture = 1;
  if (ear) {
    const neckAngle = angleWithVertical(ear, shoulder);
    neckPosture = neckAngle > 20 ? 2 : 1;
  }
  const trunkAngle = angleWithVertical(shoulder, hip);
  let trunkPosture = 1;
  if (trunkAngle <= 5) trunkPosture = 1;
  else if (trunkAngle <= 20) trunkPosture = 2;
  else if (trunkAngle <= 60) trunkPosture = 3;
  else trunkPosture = 4;

  const kneeAngle = angleBetween(hip, knee, ankle);
  let legPosture = 1;
  if (kneeAngle > 60) legPosture = 4;
  else if (kneeAngle > 30) legPosture = 3;
  else if (kneeAngle > 0) legPosture = 2;

  const upperArmAngle = angleWithVertical(shoulder, elbow);
  let upperArmPosture = 1;
  if (upperArmAngle <= 20) upperArmPosture = 1;
  else if (upperArmAngle <= 45) upperArmPosture = 2;
  else if (upperArmAngle <= 90) upperArmPosture = 3;
  else upperArmPosture = 4;

  const lowerArmAngle = angleBetween(shoulder, elbow, wrist);
  const lowerArmPosture = (lowerArmAngle >= 60 && lowerArmAngle <= 100) ? 1 : 2;

  let wristPosture = 1;
  if (index) {
    const wristAngle = angleBetween(elbow, wrist, index);
    wristPosture = wristAngle > 15 ? 2 : 1;
  }

  return {
    neck:     { posture: neckPosture, twisted: false, sideBending: false },
    trunk:    { posture: trunkPosture, twisted: false, sideBending: false },
    legs:     legPosture,
    upperArm: { posture: upperArmPosture, shoulderRaised: false, armAbducted: false, armSupported: false },
    lowerArm: lowerArmPosture,
    wrist:    { posture: wristPosture, wristTwisted: false, wristDeviated: false },
    load: 0, coupling: 0,
  };
}

// ============================================================
// MANUAL LANDMARKER COMPONENT
// ============================================================
const LANDMARK_ORDER = ["Ear","Shoulder","Hip","Knee","Ankle","Elbow","Wrist","Index"];

const LANDMARK_COLORS: Record<string, string> = {
  Ear:"#f97316", Shoulder:"#2563eb", Hip:"#7c3aed",
  Knee:"#059669", Ankle:"#db2777", Elbow:"#d97706",
  Wrist:"#0891b2", Index:"#65a30d",
};

const LANDMARK_HINTS: Record<string, string> = {
  Ear:      "Titik telinga (samping kepala)",
  Shoulder: "Sendi bahu",
  Hip:      "Sendi pinggul / tulang panggul",
  Knee:     "Sendi lutut",
  Ankle:    "Sendi pergelangan kaki",
  Elbow:    "Sendi siku",
  Wrist:    "Sendi pergelangan tangan",
  Index:    "Ujung jari telunjuk (opsional)",
};

interface ManualLandmarkerProps {
  imageUrl: string;
  onScores: (scores: ReturnType<typeof computeREBAFromManualPoints>) => void;
  onClose: () => void;
}

function ManualLandmarker({ imageUrl, onScores, onClose }: ManualLandmarkerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [points, setPoints] = useState<LandmarkPoint[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [finished, setFinished] = useState(false);
  const [calculated, setCalculated] = useState(false);
  const [angles, setAngles] = useState<Record<string, number>>({});

  useEffect(() => {
    const image = new window.Image();
    image.crossOrigin = "anonymous";
    image.src = imageUrl;
    image.onload = () => {
      setImg(image);
      setPoints([]); setCurrentStep(0); setFinished(false); setCalculated(false); setAngles({});
    };
  }, [imageUrl]);

  const drawCanvas = useCallback((pts: LandmarkPoint[], imgEl: HTMLImageElement | null) => {
    const canvas = canvasRef.current;
    if (!canvas || !imgEl) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width  = imgEl.naturalWidth;
    canvas.height = imgEl.naturalHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgEl, 0, 0);

    const lw = Math.max(2, canvas.width * 0.004);
    const drawLine = (l1: string, l2: string, color: string) => {
      const p1 = pts.find(p => p.label === l1);
      const p2 = pts.find(p => p.label === l2);
      if (!p1 || !p2) return;
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash([]); ctx.stroke();
    };
    drawLine("Ear","Shoulder","#f97316"); drawLine("Shoulder","Hip","#2563eb");
    drawLine("Hip","Knee","#7c3aed");     drawLine("Knee","Ankle","#059669");
    drawLine("Shoulder","Elbow","#d97706"); drawLine("Elbow","Wrist","#0891b2");
    if (pts.find(p => p.label === "Index")) drawLine("Wrist","Index","#65a30d");

    const r = Math.max(9, canvas.width * 0.016);
    const fs = Math.max(13, canvas.width * 0.022);
    pts.forEach((p) => {
      const color = LANDMARK_COLORS[p.label] || "#ef4444";
      ctx.beginPath(); ctx.arc(p.x, p.y, r + 4, 0, 2 * Math.PI);
      ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 3; ctx.stroke();
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
      ctx.fillStyle = color; ctx.fill();
      ctx.font = `bold ${fs}px sans-serif`;
      const tw = ctx.measureText(p.label).width;
      const lx = p.x + r + 6, ly = p.y - 4;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      (ctx as any).roundRect?.(lx - 3, ly - fs, tw + 8, fs + 6, 4);
      ctx.fill();
      ctx.fillStyle = "#fff"; ctx.fillText(p.label, lx, ly);
    });
  }, []);

  useEffect(() => { drawCanvas(points, img); }, [points, img, drawCanvas]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || finished) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = canvasRef.current.width  / rect.width;
    const sy = canvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * sx;
    const y = (e.clientY - rect.top)  * sy;
    const label = LANDMARK_ORDER[currentStep];
    const updated = [...points, { x, y, label }];
    setPoints(updated);
    const next = currentStep + 1;
    if (next >= LANDMARK_ORDER.length) setFinished(true);
    else setCurrentStep(next);
  };

  const handleCalculate = () => {
    const scores = computeREBAFromManualPoints(points);
    const get = (l: string) => points.find(p => p.label === l);
    const ear = get("Ear"), sh = get("Shoulder"), hi = get("Hip"),
      kn = get("Knee"), an = get("Ankle"), el = get("Elbow"),
      wr = get("Wrist"), ix = get("Index");
    const computed: Record<string, number> = {};
    if (ear && sh) computed["Neck"]      = Math.round(angleWithVertical(ear, sh));
    if (sh  && hi) computed["Trunk"]     = Math.round(angleWithVertical(sh, hi));
    if (hi && kn && an) computed["Knee"] = Math.round(angleBetween(hi, kn, an));
    if (sh  && el) computed["Upper Arm"] = Math.round(angleWithVertical(sh, el));
    if (sh && el && wr) computed["Lower Arm"] = Math.round(angleBetween(sh, el, wr));
    if (el && wr && ix) computed["Wrist"] = Math.round(angleBetween(el, wr, ix));
    setAngles(computed); setCalculated(true); onScores(scores);
  };

  const handleReset = () => {
    setPoints([]); setCurrentStep(0); setFinished(false); setCalculated(false); setAngles({});
  };

  const currentLabel = LANDMARK_ORDER[currentStep];
  const progress = Math.round((points.length / LANDMARK_ORDER.length) * 100);

  return (
    <div style={{
      position:"fixed", inset:0, background:"rgba(15,23,42,0.6)",
      zIndex:50, display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"flex-start",
      overflowY:"auto", padding:"20px",
      backdropFilter:"blur(6px)",
    }}>
      <div style={{
        background:"#ffffff", borderRadius:"16px",
        width:"100%", maxWidth:"1100px",
        boxShadow:"0 24px 60px rgba(0,0,0,0.25)",
        border:"1px solid #e2e8f0", overflow:"hidden",
      }}>
        {/* Modal Header */}
        <div style={{
          background:"linear-gradient(135deg,#1d4ed8 0%,#2563eb 100%)",
          padding:"14px 22px", display:"flex", alignItems:"center", justifyContent:"space-between",
        }}>
          <div style={{ display:"flex", alignItems:"center", gap:"12px" }}>
            <div style={{ width:"34px",height:"34px",borderRadius:"9px",background:"rgba(255,255,255,0.2)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:"17px" }}>🎯</div>
            <div>
              <h2 style={{ color:"#fff",fontSize:"15px",fontWeight:"700",margin:0 }}>Manual Landmarking</h2>
              <p style={{ color:"rgba(255,255,255,0.7)",fontSize:"11px",margin:0 }}>Tandai titik anatomi satu per satu pada foto</p>
            </div>
          </div>
          <button onClick={onClose} style={{ background:"rgba(255,255,255,0.15)",border:"1px solid rgba(255,255,255,0.3)",borderRadius:"8px",color:"#fff",padding:"6px 14px",cursor:"pointer",fontSize:"13px",fontWeight:"600" }}>
            ✕ Tutup
          </button>
        </div>

        <div style={{ display:"flex", flexWrap:"wrap" }}>
          {/* Canvas area */}
          <div style={{ flex:"1 1 520px", padding:"16px", borderRight:"1px solid #f1f5f9" }}>
            {/* Progress bar */}
            <div style={{ marginBottom:"10px" }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:"5px" }}>
                <span style={{ color:"#64748b",fontSize:"12px" }}>{points.length} / {LANDMARK_ORDER.length} titik</span>
                <span style={{ color:"#2563eb",fontSize:"12px",fontWeight:"700" }}>{progress}%</span>
              </div>
              <div style={{ background:"#e2e8f0",borderRadius:"99px",height:"7px",overflow:"hidden" }}>
                <div style={{ background:"linear-gradient(90deg,#2563eb,#06b6d4)",height:"100%",width:`${progress}%`,borderRadius:"99px",transition:"width 0.3s" }} />
              </div>
            </div>

            {/* Instruction */}
            {!finished ? (
              <div style={{ background:"#eff6ff",border:"1px solid #bfdbfe",borderRadius:"10px",padding:"10px 14px",marginBottom:"10px",display:"flex",alignItems:"center",gap:"10px" }}>
                <div style={{ width:"28px",height:"28px",borderRadius:"50%",background:LANDMARK_COLORS[currentLabel]||"#2563eb",display:"flex",alignItems:"center",justifyContent:"center",color:"#fff",fontWeight:"700",fontSize:"12px",flexShrink:0 }}>
                  {currentStep+1}
                </div>
                <div style={{ flex:1 }}>
                  <div style={{ fontSize:"13px",fontWeight:"700",color:"#1e40af" }}>
                    Klik: <span style={{ color:LANDMARK_COLORS[currentLabel] }}>{currentLabel}</span>
                  </div>
                  <div style={{ fontSize:"11px",color:"#64748b" }}>{LANDMARK_HINTS[currentLabel]}</div>
                </div>
                {currentLabel === "Index" && (
                  <button onClick={() => setFinished(true)} style={{ background:"#f1f5f9",border:"1px solid #e2e8f0",borderRadius:"6px",color:"#64748b",padding:"4px 10px",cursor:"pointer",fontSize:"11px",fontWeight:"600" }}>
                    Skip →
                  </button>
                )}
              </div>
            ) : (
              <div style={{ background:"#f0fdf4",border:"1px solid #bbf7d0",borderRadius:"10px",padding:"10px 14px",marginBottom:"10px",display:"flex",alignItems:"center",gap:"10px" }}>
                <span style={{ fontSize:"20px" }}>✅</span>
                <div>
                  <div style={{ fontSize:"13px",fontWeight:"700",color:"#15803d" }}>Semua titik ditandai!</div>
                  <div style={{ fontSize:"11px",color:"#64748b" }}>Tekan "Hitung REBA" untuk kalkulasi.</div>
                </div>
              </div>
            )}

            {/* Canvas */}
            <div style={{ background:"#f8fafc",borderRadius:"10px",border:"1px solid #e2e8f0",overflow:"hidden",cursor:finished?"default":"crosshair" }}>
              {img
                ? <canvas ref={canvasRef} style={{ width:"100%",height:"auto",display:"block" }} onClick={handleCanvasClick} />
                : <div style={{ padding:"60px",textAlign:"center",color:"#94a3b8" }}>Memuat gambar...</div>
              }
            </div>

            {/* Buttons */}
            <div style={{ display:"flex",gap:"10px",marginTop:"12px",flexWrap:"wrap" }}>
              <button onClick={handleCalculate} disabled={!finished} style={{
                flex:"1 1 auto",padding:"10px 18px",
                background:finished?"linear-gradient(135deg,#2563eb,#1d4ed8)":"#e2e8f0",
                border:"none",borderRadius:"9px",
                color:finished?"#fff":"#94a3b8",
                cursor:finished?"pointer":"not-allowed",
                fontSize:"14px",fontWeight:"700",
                boxShadow:finished?"0 2px 8px rgba(37,99,235,0.3)":"none",
              }}>
                {calculated ? "✅ Hitung Ulang REBA" : "🔢 Hitung REBA"}
              </button>
              <button onClick={handleReset} style={{ padding:"10px 14px",background:"#fee2e2",border:"1px solid #fecaca",borderRadius:"9px",color:"#dc2626",cursor:"pointer",fontSize:"13px",fontWeight:"600" }}>
                🔄 Reset
              </button>
            </div>
          </div>

          {/* Sidebar */}
          <div style={{ width:"230px",flexShrink:0,padding:"16px",display:"flex",flexDirection:"column",gap:"14px" }}>
            <div>
              <h3 style={{ color:"#475569",fontSize:"11px",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.08em",margin:"0 0 8px" }}>Checklist Titik</h3>
              <div style={{ display:"flex",flexDirection:"column",gap:"5px" }}>
                {LANDMARK_ORDER.map((label, i) => {
                  const done   = points.some(p => p.label === label);
                  const active = !finished && currentStep === i;
                  return (
                    <div key={label} style={{
                      display:"flex",alignItems:"center",gap:"8px",
                      padding:"7px 10px",borderRadius:"8px",
                      background:done?"#f0fdf4":active?"#eff6ff":"#f8fafc",
                      border:`1px solid ${done?"#bbf7d0":active?"#bfdbfe":"#e2e8f0"}`,
                    }}>
                      <div style={{
                        width:"20px",height:"20px",borderRadius:"50%",flexShrink:0,
                        background:done?"#22c55e":active?LANDMARK_COLORS[label]:"#e2e8f0",
                        display:"flex",alignItems:"center",justifyContent:"center",
                        color:"#fff",fontWeight:"700",fontSize:"10px",
                      }}>{done?"✓":i+1}</div>
                      <span style={{ fontSize:"12px",fontWeight:active?"700":"500",color:done?"#16a34a":active?"#1e40af":"#64748b" }}>
                        {label}
                      </span>
                      {label==="Index" && <span style={{ marginLeft:"auto",fontSize:"9px",color:"#94a3b8" }}>opt</span>}
                    </div>
                  );
                })}
              </div>
            </div>

            {calculated && Object.keys(angles).length > 0 && (
              <div>
                <h3 style={{ color:"#475569",fontSize:"11px",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.08em",margin:"0 0 8px" }}>Sudut Terukur</h3>
                <div style={{ display:"flex",flexDirection:"column",gap:"5px" }}>
                  {Object.entries(angles).map(([k,v]) => (
                    <div key={k} style={{ display:"flex",justifyContent:"space-between",alignItems:"center",padding:"6px 10px",background:"#eff6ff",borderRadius:"7px",border:"1px solid #bfdbfe" }}>
                      <span style={{ fontSize:"10px",color:"#64748b" }}>{k}</span>
                      <span style={{ fontSize:"13px",fontWeight:"700",color:"#2563eb" }}>{v}°</span>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop:"8px",padding:"9px",background:"#f0fdf4",borderRadius:"8px",border:"1px solid #bbf7d0" }}>
                  <p style={{ margin:0,color:"#16a34a",fontSize:"12px",fontWeight:"600" }}>✅ Skor REBA diperbarui!</p>
                  <p style={{ margin:"3px 0 0",color:"#64748b",fontSize:"11px" }}>Lihat Group A & B di bawah.</p>
                </div>
              </div>
            )}

            <div style={{ padding:"12px",background:"#fffbeb",borderRadius:"10px",border:"1px solid #fde68a" }}>
              <p style={{ margin:"0 0 6px",color:"#92400e",fontSize:"12px",fontWeight:"700" }}>💡 Tips</p>
              <ul style={{ margin:0,padding:"0 0 0 14px",color:"#78716c",fontSize:"11px",lineHeight:"1.7" }}>
                <li>Foto tampak samping = hasil terbaik</li>
                <li>Titik "Index" bisa di-skip</li>
                <li>Klik Reset untuk mengulang</li>
                <li>Zoom browser jika gambar kecil</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// ASSESSMENT SECTION COMPONENT
// ============================================================
interface PostureOption { value: number; label: string; }
interface AdjustmentProps {
  twisted?: boolean; sideBending?: boolean;
  shoulderRaised?: boolean; armAbducted?: boolean; armSupported?: boolean;
  wristTwisted?: boolean; wristDeviated?: boolean;
  onAdjustmentChange?: (type: string, value: boolean) => void;
}
interface AssessmentSectionProps {
  title: string; postureOptions: PostureOption[];
  groupColor: "blue" | "emerald";
  onChange: (value: number) => void;
  selectedValue?: number; adjustments?: AdjustmentProps;
}

export function AssessmentSection({ title, postureOptions, groupColor, onChange, selectedValue = 0, adjustments }: AssessmentSectionProps) {
  const accent = groupColor === "blue" ? "#2563eb" : "#059669";
  const acBg   = groupColor === "blue" ? "#eff6ff" : "#ecfdf5";
  const acBd   = groupColor === "blue" ? "#bfdbfe" : "#a7f3d0";

  const getScore = () => {
    if (selectedValue === 0) return "";
    let s = selectedValue;
    if (adjustments) {
      if (title === "Neck" || title === "Trunk") {
        if (adjustments.twisted) s++;
        if (adjustments.sideBending) s++;
      } else if (title === "Upper Arm") {
        if (adjustments.shoulderRaised) s++;
        if (adjustments.armAbducted) s++;
        if (adjustments.armSupported) s--;
      } else if (title === "Wrist") {
        if (adjustments.wristTwisted) s++;
        if (adjustments.wristDeviated) s++;
      }
    }
    return String(Math.max(s, 0));
  };

  const renderAdj = () => {
    if (!adjustments) return null;
    const s = { fontSize:"11px", color:"#64748b" };

    if (title === "Neck" || title === "Trunk") {
      const nm = title === "Neck" ? "neck-adj" : "trunk-adj";
      return (
        <div style={{ paddingTop:"9px", borderTop:"1px solid #f1f5f9" }}>
          <p style={{ fontSize:"11px",fontWeight:"600",color:"#64748b",margin:"0 0 7px" }}>Adjustments</p>
          {[{key:"twisted",label:`${title} twisted +1`,alt:"sideBending"},{key:"sideBending",label:`${title} side bending +1`,alt:"twisted"}].map(({key,label,alt})=>(
            <label key={key} style={{ display:"flex",alignItems:"center",gap:"7px",marginBottom:"4px",cursor:"pointer" }}>
              <input type="radio" name={nm} style={{ accentColor:accent }}
                checked={key==="twisted"?(adjustments.twisted||false):(adjustments.sideBending||false)}
                onChange={e=>{ if(e.target.checked){ adjustments.onAdjustmentChange?.(key,true); adjustments.onAdjustmentChange?.(alt,false); } }}
              />
              <span style={s}>{label}</span>
            </label>
          ))}
          <label style={{ display:"flex",alignItems:"center",gap:"7px",cursor:"pointer" }}>
            <input type="radio" name={nm} style={{ accentColor:accent }}
              checked={!adjustments.twisted&&!adjustments.sideBending}
              onChange={e=>{ if(e.target.checked){ adjustments.onAdjustmentChange?.("twisted",false); adjustments.onAdjustmentChange?.("sideBending",false); } }}
            />
            <span style={s}>None</span>
          </label>
        </div>
      );
    }
    if (title === "Upper Arm") {
      return (
        <div style={{ paddingTop:"9px",borderTop:"1px solid #f1f5f9" }}>
          <p style={{ fontSize:"11px",fontWeight:"600",color:"#64748b",margin:"0 0 7px" }}>Adjustments</p>
          {[{key:"shoulderRaised",label:"Shoulder raised +1"},{key:"armAbducted",label:"Arm abducted +1"},{key:"armSupported",label:"Arm supported −1"}].map(({key,label})=>(
            <label key={key} style={{ display:"flex",alignItems:"center",gap:"7px",marginBottom:"4px",cursor:"pointer" }}>
              <input type="checkbox" style={{ accentColor:accent }}
                checked={(adjustments as any)[key]||false}
                onChange={e=>adjustments.onAdjustmentChange?.(key,e.target.checked)}
              />
              <span style={s}>{label}</span>
            </label>
          ))}
        </div>
      );
    }
    if (title === "Wrist") {
      return (
        <div style={{ paddingTop:"9px",borderTop:"1px solid #f1f5f9" }}>
          <p style={{ fontSize:"11px",fontWeight:"600",color:"#64748b",margin:"0 0 7px" }}>Adjustments</p>
          {[{key:"wristTwisted",label:"Wrist twisted +1",alt:"wristDeviated"},{key:"wristDeviated",label:"Wrist deviated +1",alt:"wristTwisted"}].map(({key,label,alt})=>(
            <label key={key} style={{ display:"flex",alignItems:"center",gap:"7px",marginBottom:"4px",cursor:"pointer" }}>
              <input type="radio" name="wrist-adj" style={{ accentColor:accent }}
                checked={(adjustments as any)[key]||false}
                onChange={e=>{ if(e.target.checked){ adjustments.onAdjustmentChange?.(key,true); adjustments.onAdjustmentChange?.(alt,false); } }}
              />
              <span style={s}>{label}</span>
            </label>
          ))}
          <label style={{ display:"flex",alignItems:"center",gap:"7px",cursor:"pointer" }}>
            <input type="radio" name="wrist-adj" style={{ accentColor:accent }}
              checked={!adjustments.wristTwisted&&!adjustments.wristDeviated}
              onChange={e=>{ if(e.target.checked){ adjustments.onAdjustmentChange?.("wristTwisted",false); adjustments.onAdjustmentChange?.("wristDeviated",false); } }}
            />
            <span style={s}>None</span>
          </label>
        </div>
      );
    }
    return null;
  };

  const score = getScore();
  return (
    <div style={{ background:"#fff",border:`1px solid ${acBd}`,borderRadius:"12px",overflow:"hidden",boxShadow:"0 1px 3px rgba(0,0,0,0.06)" }}>
      <div style={{ background:accent,padding:"8px 14px" }}>
        <h3 style={{ color:"#fff",fontSize:"12px",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.06em",margin:0 }}>{title}</h3>
      </div>
      <div style={{ padding:"13px" }}>
        <label style={{ display:"block",fontSize:"11px",fontWeight:"600",color:"#475569",marginBottom:"5px" }}>Posture Category</label>
        <select value={selectedValue} onChange={e=>onChange(Number(e.target.value))}
          style={{ width:"100%",padding:"7px 10px",border:"1px solid #e2e8f0",borderRadius:"8px",fontSize:"12px",color:"#374151",background:"#f8fafc",outline:"none",cursor:"pointer",marginBottom:"8px" }}>
          <option value={0}>Pilih postur...</option>
          {postureOptions.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        {renderAdj()}
        <div style={{ marginTop:"9px" }}>
          <label style={{ display:"block",fontSize:"11px",fontWeight:"600",color:"#475569",marginBottom:"5px" }}>Score</label>
          <div style={{ height:"36px",borderRadius:"8px",background:score?acBg:"#f8fafc",border:`1px solid ${score?acBd:"#e2e8f0"}`,display:"flex",alignItems:"center",justifyContent:"center" }}>
            <span style={{ fontSize:"17px",fontWeight:"800",color:score?accent:"#d1d5db" }}>{score||"—"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// SCORING TABLE
// ============================================================
interface ScoringTableProps { title:string; tableLabel:string; formula:string; components:string[]; groupColor:"blue"|"emerald"|"purple"; score:number; }

export function ScoringTable({ title, tableLabel, formula, components, groupColor, score }: ScoringTableProps) {
  const C = { blue:{a:"#2563eb",bg:"#eff6ff",bd:"#bfdbfe"}, emerald:{a:"#059669",bg:"#ecfdf5",bd:"#a7f3d0"}, purple:{a:"#7c3aed",bg:"#f5f3ff",bd:"#ddd6fe"} };
  const { a:ac, bg, bd } = C[groupColor];
  return (
    <div style={{ background:"#fff",border:`1px solid ${bd}`,borderRadius:"12px",overflow:"hidden",boxShadow:"0 1px 3px rgba(0,0,0,0.06)" }}>
      <div style={{ background:ac,padding:"8px 14px" }}>
        <h3 style={{ color:"#fff",fontSize:"12px",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.06em",margin:0 }}>{title}</h3>
      </div>
      <div style={{ padding:"13px" }}>
        <div style={{ background:bg,border:`1px solid ${bd}`,borderRadius:"8px",padding:"9px 11px",marginBottom:"9px" }}>
          <p style={{ fontSize:"10px",fontWeight:"700",color:"#64748b",margin:"0 0 3px" }}>FORMULA</p>
          <p style={{ fontSize:"11px",fontFamily:"monospace",color:"#1e293b",margin:0 }}>{formula}</p>
        </div>
        <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr",gap:"5px",marginBottom:"10px" }}>
          {components.map((c,i)=><div key={i} style={{ background:"#f8fafc",border:"1px solid #e2e8f0",borderRadius:"6px",padding:"5px 8px",textAlign:"center",fontSize:"11px",color:"#475569" }}>{c}</div>)}
        </div>
        <div style={{ borderTop:"1px solid #f1f5f9",paddingTop:"9px",display:"flex",alignItems:"center",justifyContent:"space-between" }}>
          <span style={{ fontSize:"11px",fontWeight:"700",color:"#475569" }}>{tableLabel}:</span>
          <div style={{ background:score>0?bg:"#f8fafc",border:`1px solid ${score>0?bd:"#e2e8f0"}`,borderRadius:"8px",padding:"4px 16px",minWidth:"50px",textAlign:"center" }}>
            <span style={{ fontSize:"16px",fontWeight:"800",color:score>0?ac:"#d1d5db" }}>{score>0?score:"—"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// COMBINATION TABLE
// ============================================================
interface CombinationTableProps { scoreA:number; scoreB:number; scoreC:number; }

export function CombinationTable({ scoreA, scoreB, scoreC }: CombinationTableProps) {
  const mx = [
    [1,1,1,2,3,3,4,5,6,7,7,7],[1,2,2,3,4,4,5,6,6,7,7,8],
    [2,3,3,3,4,5,6,7,7,8,8,8],[3,4,4,4,5,6,7,8,8,9,9,9],
    [4,4,4,5,6,7,8,8,9,9,9,9],[6,6,6,7,8,8,9,9,10,10,10,10],
    [7,7,7,8,9,9,9,10,10,11,11,11],[8,8,8,9,10,10,10,10,10,11,11,11],
    [9,9,9,10,10,10,11,11,11,12,12,12],[10,10,10,11,11,11,11,12,12,12,12,12],
    [11,11,11,11,12,12,12,12,12,12,12,12],[12,12,12,12,12,12,12,12,12,12,12,12],
  ];
  return (
    <div style={{ background:"#fff",border:"1px solid #ddd6fe",borderRadius:"12px",overflow:"hidden",boxShadow:"0 1px 3px rgba(0,0,0,0.06)" }}>
      <div style={{ background:"#7c3aed",padding:"8px 14px" }}>
        <h3 style={{ color:"#fff",fontSize:"12px",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.06em",margin:0 }}>Table C — Combination</h3>
      </div>
      <div style={{ padding:"13px" }}>
        <div style={{ background:"#f5f3ff",border:"1px solid #ddd6fe",borderRadius:"8px",padding:"9px 11px",marginBottom:"9px" }}>
          <p style={{ fontSize:"10px",fontWeight:"700",color:"#64748b",margin:"0 0 3px" }}>Score C = Table C[A, B]</p>
          <div style={{ display:"flex",gap:"14px" }}>
            <span style={{ fontSize:"12px",color:"#1e293b" }}>A: <strong style={{ color:"#2563eb" }}>{scoreA||"—"}</strong></span>
            <span style={{ fontSize:"12px",color:"#1e293b" }}>B: <strong style={{ color:"#059669" }}>{scoreB||"—"}</strong></span>
          </div>
        </div>
        <div style={{ overflowX:"auto" }}>
          <table style={{ borderCollapse:"collapse",fontSize:"10px",width:"100%" }}>
            <thead>
              <tr style={{ background:"#f1f5f9" }}>
                <th style={{ border:"1px solid #e2e8f0",padding:"4px 6px",fontWeight:"700",color:"#475569",fontSize:"9px" }}>A→<br/>B↓</th>
                {[1,2,3,4].map(n=><th key={n} style={{ border:"1px solid #e2e8f0",padding:"4px 6px",color:"#475569" }}>{n}</th>)}
                <th style={{ border:"1px solid #e2e8f0",padding:"4px 6px",color:"#94a3b8" }}>…</th>
              </tr>
            </thead>
            <tbody>
              {mx.slice(0,4).map((row,ri)=>(
                <tr key={ri}>
                  <td style={{ border:"1px solid #e2e8f0",padding:"4px 6px",fontWeight:"700",background:"#f8fafc",color:"#475569" }}>{ri+1}</td>
                  {row.slice(0,4).map((cell,ci)=>(
                    <td key={ci} style={{
                      border:"1px solid #e2e8f0",padding:"4px 6px",textAlign:"center",
                      background:scoreA>0&&scoreB>0&&ri===scoreA-1&&ci===scoreB-1?"#fef9c3":"#fff",
                      fontWeight:scoreA>0&&scoreB>0&&ri===scoreA-1&&ci===scoreB-1?"800":"400",
                      color:"#374151",
                    }}>{cell}</td>
                  ))}
                  <td style={{ border:"1px solid #e2e8f0",padding:"4px 6px",color:"#94a3b8",textAlign:"center" }}>…</td>
                </tr>
              ))}
              <tr>{[...Array(6)].map((_,i)=><td key={i} style={{ border:"1px solid #e2e8f0",padding:"4px 6px",color:"#94a3b8",textAlign:"center" }}>…</td>)}</tr>
            </tbody>
          </table>
        </div>
        <div style={{ borderTop:"1px solid #f1f5f9",paddingTop:"9px",marginTop:"9px",display:"flex",alignItems:"center",justifyContent:"space-between" }}>
          <span style={{ fontSize:"11px",fontWeight:"700",color:"#475569" }}>Score C:</span>
          <div style={{ background:scoreC>0?"#f5f3ff":"#f8fafc",border:`1px solid ${scoreC>0?"#ddd6fe":"#e2e8f0"}`,borderRadius:"8px",padding:"4px 16px",minWidth:"50px",textAlign:"center" }}>
            <span style={{ fontSize:"16px",fontWeight:"800",color:scoreC>0?"#7c3aed":"#d1d5db" }}>{scoreC>0?scoreC:"—"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// ACTIVITY SCORE
// ============================================================
interface ActivityScoreProps { activityScore:number; onActivityChange:(a:{static:boolean;repetitive:boolean;rapid:boolean})=>void; }

export function ActivityScore({ activityScore, onActivityChange }: ActivityScoreProps) {
  const [act, setAct] = useState({ static:false, repetitive:false, rapid:false });
  const handle = (k: keyof typeof act) => {
    const n = { ...act, [k]:!act[k] };
    setAct(n); onActivityChange(n);
  };
  const items = [
    { k:"static" as const,   title:"Static Posture",       desc:"Ditahan >1 menit" },
    { k:"repetitive" as const, title:"Repeated Actions",   desc:">4× per menit" },
    { k:"rapid" as const,    title:"Rapid Large Changes",  desc:"Perubahan postur mendadak" },
  ];
  return (
    <div style={{ background:"#fff",border:"1px solid #fed7aa",borderRadius:"12px",overflow:"hidden",boxShadow:"0 1px 3px rgba(0,0,0,0.06)" }}>
      <div style={{ background:"#ea580c",padding:"8px 14px" }}>
        <h3 style={{ color:"#fff",fontSize:"12px",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.06em",margin:0 }}>Activity Score</h3>
      </div>
      <div style={{ padding:"13px" }}>
        <p style={{ fontSize:"12px",color:"#64748b",margin:"0 0 9px" }}>Tambahkan +1 untuk setiap kondisi yang berlaku:</p>
        <div style={{ display:"flex",flexDirection:"column",gap:"7px" }}>
          {items.map(({k,title,desc})=>(
            <label key={k} style={{ display:"flex",alignItems:"flex-start",gap:"10px",padding:"9px 11px",borderRadius:"9px",cursor:"pointer",background:act[k]?"#fff7ed":"#f8fafc",border:`1px solid ${act[k]?"#fed7aa":"#e2e8f0"}`,transition:"all 0.15s" }}>
              <input type="checkbox" checked={act[k]} onChange={()=>handle(k)} style={{ marginTop:"2px",accentColor:"#ea580c",width:"14px",height:"14px" }} />
              <div>
                <p style={{ fontSize:"13px",fontWeight:"600",color:"#1e293b",margin:"0 0 1px" }}>{title}</p>
                <p style={{ fontSize:"11px",color:"#64748b",margin:0 }}>{desc}</p>
              </div>
            </label>
          ))}
        </div>
        <div style={{ borderTop:"1px solid #f1f5f9",paddingTop:"9px",marginTop:"9px",display:"flex",alignItems:"center",justifyContent:"space-between" }}>
          <span style={{ fontSize:"11px",fontWeight:"700",color:"#475569" }}>Activity Score:</span>
          <div style={{ background:activityScore>0?"#fff7ed":"#f8fafc",border:`1px solid ${activityScore>0?"#fed7aa":"#e2e8f0"}`,borderRadius:"8px",padding:"4px 16px",minWidth:"50px",textAlign:"center" }}>
            <span style={{ fontSize:"16px",fontWeight:"800",color:activityScore>0?"#ea580c":"#d1d5db" }}>{activityScore>0?`+${activityScore}`:"—"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// FINAL RESULTS
// ============================================================
interface FinalResultsProps { finalScore:number; riskLevel:string; scoreC:number; activityScore:number; }

export function FinalResults({ finalScore, riskLevel, scoreC, activityScore }: FinalResultsProps) {
  const P = finalScore===0
    ?{bg:"#f8fafc",bd:"#e2e8f0",tx:"#94a3b8",bbg:"#f1f5f9",bbd:"#e2e8f0",btx:"#94a3b8"}
    :finalScore<=1
    ?{bg:"#f0fdf4",bd:"#bbf7d0",tx:"#16a34a",bbg:"#dcfce7",bbd:"#86efac",btx:"#15803d"}
    :finalScore<=3
    ?{bg:"#fefce8",bd:"#fde68a",tx:"#ca8a04",bbg:"#fef9c3",bbd:"#fde047",btx:"#a16207"}
    :finalScore<=7
    ?{bg:"#fff7ed",bd:"#fed7aa",tx:"#ea580c",bbg:"#ffedd5",bbd:"#fdba74",btx:"#c2410c"}
    :finalScore<=10
    ?{bg:"#fef2f2",bd:"#fecaca",tx:"#dc2626",bbg:"#fee2e2",bbd:"#fca5a5",btx:"#b91c1c"}
    :{bg:"#fef2f2",bd:"#f87171",tx:"#991b1b",bbg:"#fee2e2",bbd:"#f87171",btx:"#7f1d1d"};
  const action =
    finalScore===0?"Lengkapi asesmen untuk melihat rekomendasi.":
    finalScore<=1?"Tidak diperlukan tindakan. Postur dapat diterima.":
    finalScore<=3?"Mungkin diperlukan perubahan. Investigasi lebih lanjut.":
    finalScore<=7?"Perubahan diperlukan segera. Terapkan perbaikan.":
    finalScore<=10?"Perubahan diperlukan SEKARANG. Risiko cedera tinggi.":
    "Perubahan DARURAT diperlukan. Risiko cedera sangat tinggi!";
  return (
    <div style={{ background:"#fff",border:"1px solid #fecaca",borderRadius:"12px",overflow:"hidden",boxShadow:"0 1px 3px rgba(0,0,0,0.06)" }}>
      <div style={{ background:"#dc2626",padding:"10px 14px" }}>
        <h3 style={{ color:"#fff",fontSize:"13px",fontWeight:"700",textTransform:"uppercase",letterSpacing:"0.06em",margin:0 }}>Final REBA Assessment</h3>
      </div>
      <div style={{ padding:"18px" }}>
        <div style={{ background:"#f8fafc",border:"1px solid #e2e8f0",borderRadius:"10px",padding:"11px 13px",marginBottom:"14px" }}>
          <p style={{ fontSize:"10px",fontWeight:"700",color:"#64748b",margin:"0 0 5px",textTransform:"uppercase" }}>Formula</p>
          <div style={{ fontFamily:"monospace",fontSize:"13px",color:"#1e293b" }}>
            REBA Score = Score C + Activity Score
            {scoreC>0&&<div style={{ marginTop:"4px",color:"#475569" }}>= {scoreC} + {activityScore} = <strong style={{ color:"#dc2626" }}>{finalScore}</strong></div>}
          </div>
        </div>
        <div style={{ background:P.bg,border:`2px solid ${P.bd}`,borderRadius:"12px",padding:"18px",display:"flex",alignItems:"center",justifyContent:"space-between",gap:"16px",flexWrap:"wrap",marginBottom:"12px" }}>
          <div style={{ textAlign:"center" }}>
            <p style={{ fontSize:"10px",fontWeight:"700",color:"#64748b",margin:"0 0 3px",textTransform:"uppercase" }}>Final Score</p>
            <span style={{ fontSize:"50px",fontWeight:"900",color:P.tx,lineHeight:1 }}>{finalScore>0?finalScore:"—"}</span>
          </div>
          <div style={{ flex:1,minWidth:"150px" }}>
            <div style={{ background:P.bbg,border:`1px solid ${P.bbd}`,borderRadius:"9px",padding:"9px 12px",marginBottom:"7px" }}>
              <p style={{ fontSize:"10px",fontWeight:"700",color:"#64748b",margin:"0 0 2px",textTransform:"uppercase" }}>Risk Level</p>
              <p style={{ fontSize:"16px",fontWeight:"800",color:P.btx,margin:0 }}>{finalScore>0?riskLevel:"Belum lengkap"}</p>
            </div>
            <p style={{ fontSize:"12px",color:"#475569",margin:0,lineHeight:"1.5" }}>{action}</p>
          </div>
        </div>
        <div>
          <p style={{ fontSize:"10px",fontWeight:"700",color:"#64748b",margin:"0 0 7px",textTransform:"uppercase" }}>REBA Scale</p>
          <div style={{ display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:"5px" }}>
            {[{r:"1",l:"Negligible",bg:"#dcfce7",bd:"#86efac",tx:"#15803d"},{r:"2–3",l:"Low",bg:"#fef9c3",bd:"#fde047",tx:"#a16207"},{r:"4–7",l:"Medium",bg:"#ffedd5",bd:"#fdba74",tx:"#c2410c"},{r:"8–10",l:"High",bg:"#fee2e2",bd:"#fca5a5",tx:"#b91c1c"},{r:"11+",l:"Very High",bg:"#fecaca",bd:"#f87171",tx:"#7f1d1d"}].map(({r,l,bg,bd,tx})=>(
              <div key={r} style={{ background:bg,border:`1px solid ${bd}`,borderRadius:"7px",padding:"5px 3px",textAlign:"center" }}>
                <p style={{ fontSize:"12px",fontWeight:"800",color:tx,margin:"0 0 1px" }}>{r}</p>
                <p style={{ fontSize:"9px",color:tx,margin:0,opacity:0.8 }}>{l}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// IMAGE UPLOADER — MediaPipe dioptimalkan (model kompleks-2, canvas preprocessing)
// ============================================================
interface ImageUploaderProps {
  onAutoScores: (s: ReturnType<typeof computeREBAFromLandmarks>) => void;
  onManualScores: (s: ReturnType<typeof computeREBAFromManualPoints>) => void;
  onImageUpload: (url: string) => void;
}

function ImageUploader({ onAutoScores, onManualScores, onImageUpload }: ImageUploaderProps) {
  const [imageUrl, setImageUrl]   = useState<string | null>(null);
  const imgRef                    = useRef<HTMLImageElement>(null);
  const [landmarks, setLandmarks] = useState<any[] | null>(null);
  const [loading, setLoading]     = useState(false);
  const [autoStatus, setAutoStatus] = useState<"idle"|"ok"|"fail">("idle");
  const poseRef                   = useRef<any>(null);
  const [showManual, setShowManual] = useState(false);
  const [manualDone, setManualDone] = useState(false);
  const fileInputRef              = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!window.Pose) { console.error("MediaPipe Pose not loaded"); return; }
    const pose = new window.Pose({
      locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`,
    });
    /**
     * Peningkatan akurasi MediaPipe:
     * - modelComplexity: 2  → model paling berat & akurat (Heavy Pose)
     * - minDetectionConfidence: 0.6 → threshold deteksi cukup ketat
     * - minTrackingConfidence: 0.6  → threshold tracking cukup ketat
     * - smoothLandmarks: true       → hasil lebih stabil
     */
    pose.setOptions({
      modelComplexity: 2,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
    pose.onResults((res: any) => {
      if (res.poseLandmarks && res.poseLandmarks.length > 0) {
        setLandmarks(res.poseLandmarks);
        setAutoStatus("ok");
      } else {
        setLandmarks(null);
        setAutoStatus("fail");
      }
      setLoading(false);
    });
    poseRef.current = pose;
    return () => pose.close();
  }, []);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url); setLandmarks(null);
    setShowManual(false); setManualDone(false); setAutoStatus("idle");
    onImageUpload(url);
  };

  /**
   * Preprocessing sebelum kirim ke MediaPipe:
   * Gambar di-render ke offscreen canvas dengan resolusi minimal 640 px lebar
   * agar model pose mendapat input yang cukup resolusinya.
   * Ini adalah cara yang paling efektif meningkatkan akurasi tanpa mengubah logika REBA.
   */
  const handleAnalyze = useCallback(async () => {
    if (!imgRef.current || !poseRef.current) return;
    setLoading(true); setAutoStatus("idle");
    const img = imgRef.current;

    const doSend = async () => {
      const MIN_W   = 640;
      const targetW = Math.max(img.naturalWidth, MIN_W);
      const scale   = targetW / img.naturalWidth;
      const targetH = Math.round(img.naturalHeight * scale);
      const off     = document.createElement("canvas");
      off.width     = targetW;
      off.height    = targetH;
      const ctx     = off.getContext("2d")!;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, 0, 0, targetW, targetH);
      await poseRef.current.send({ image: off });
    };

    if (img.complete && img.naturalWidth > 0) await doSend();
    else img.onload = async () => { await doSend(); };
  }, []);

  useEffect(() => {
    if (landmarks) onAutoScores(computeREBAFromLandmarks(landmarks));
  }, [landmarks, onAutoScores]);

  const handleManualWrap = useCallback((s: ReturnType<typeof computeREBAFromManualPoints>) => {
    onManualScores(s); setManualDone(true); setShowManual(false);
  }, [onManualScores]);

  return (
    <>
      {showManual && imageUrl && (
        <ManualLandmarker imageUrl={imageUrl} onScores={handleManualWrap} onClose={() => setShowManual(false)} />
      )}
      <div style={{ background:"#fff",border:"1px solid #e2e8f0",borderRadius:"14px",overflow:"hidden",boxShadow:"0 2px 8px rgba(0,0,0,0.07)" }}>
        {/* Header */}
        <div style={{ background:"linear-gradient(135deg,#1d4ed8 0%,#2563eb 100%)",padding:"13px 20px",display:"flex",alignItems:"center",gap:"12px" }}>
          <div style={{ width:"32px",height:"32px",borderRadius:"9px",background:"rgba(255,255,255,0.2)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:"15px" }}>📸</div>
          <div>
            <h3 style={{ color:"#fff",fontSize:"14px",fontWeight:"700",margin:0 }}>Analisis Foto Postur</h3>
            <p style={{ color:"rgba(255,255,255,0.7)",fontSize:"11px",margin:0 }}>Auto-detect (MediaPipe) atau tandai manual</p>
          </div>
        </div>

        <div style={{ padding:"18px" }}>
          {!imageUrl ? (
            <div onClick={()=>fileInputRef.current?.click()}
              style={{ border:"2px dashed #bfdbfe",borderRadius:"12px",padding:"40px 20px",textAlign:"center",cursor:"pointer",background:"#f8fafc",transition:"all 0.2s" }}
              onMouseEnter={e=>{e.currentTarget.style.borderColor="#2563eb";e.currentTarget.style.background="#eff6ff"}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor="#bfdbfe";e.currentTarget.style.background="#f8fafc"}}
            >
              <div style={{ fontSize:"38px",marginBottom:"10px" }}>🖼️</div>
              <p style={{ color:"#374151",fontSize:"14px",fontWeight:"600",margin:"0 0 4px" }}>Klik untuk upload foto</p>
              <p style={{ color:"#94a3b8",fontSize:"12px",margin:0 }}>JPG, PNG, WEBP — Tampak samping direkomendasikan</p>
            </div>
          ) : (
            <div style={{ display:"flex",gap:"16px",flexWrap:"wrap" }}>
              {/* Preview */}
              <div style={{ flex:"0 0 auto",position:"relative",borderRadius:"10px",overflow:"hidden",border:"1px solid #e2e8f0",boxShadow:"0 2px 8px rgba(0,0,0,0.08)" }}>
                <img ref={imgRef} src={imageUrl} alt="upload" crossOrigin="anonymous"
                  style={{ display:"block",maxHeight:"210px",maxWidth:"300px",height:"auto" }}
                  onError={()=>alert("Gagal memuat gambar.")} />
                {manualDone&&<div style={{ position:"absolute",bottom:"7px",left:"7px",background:"rgba(22,163,74,0.9)",borderRadius:"6px",padding:"2px 8px",color:"#fff",fontSize:"11px",fontWeight:"600" }}>✅ Manual selesai</div>}
                {autoStatus==="ok"&&<div style={{ position:"absolute",bottom:"7px",right:"7px",background:"rgba(37,99,235,0.9)",borderRadius:"6px",padding:"2px 8px",color:"#fff",fontSize:"11px",fontWeight:"600" }}>🤖 Auto OK</div>}
              </div>

              {/* Controls */}
              <div style={{ flex:"1 1 180px",display:"flex",flexDirection:"column",gap:"9px",justifyContent:"center" }}>
                <button onClick={handleAnalyze} disabled={loading} style={{
                  padding:"10px 14px",
                  background:loading?"#e2e8f0":"linear-gradient(135deg,#2563eb,#1d4ed8)",
                  border:"none",borderRadius:"9px",color:loading?"#94a3b8":"#fff",
                  cursor:loading?"not-allowed":"pointer",fontSize:"13px",fontWeight:"700",
                  display:"flex",alignItems:"center",gap:"8px",
                  boxShadow:loading?"none":"0 2px 8px rgba(37,99,235,0.25)",
                }}>
                  <span>{loading?"⏳":"🤖"}</span>
                  {loading?"Mendeteksi...":"Auto Analyze (MediaPipe)"}
                </button>

                {autoStatus==="fail"&&(
                  <div style={{ background:"#fef2f2",border:"1px solid #fecaca",borderRadius:"8px",padding:"7px 11px",fontSize:"12px",color:"#b91c1c" }}>
                    ⚠️ Pose tidak terdeteksi. Gunakan <strong>Manual Landmarking</strong>.
                  </div>
                )}

                <button onClick={()=>setShowManual(true)} style={{
                  padding:"10px 14px",
                  background:"linear-gradient(135deg,#059669,#047857)",
                  border:"none",borderRadius:"9px",color:"#fff",cursor:"pointer",
                  fontSize:"13px",fontWeight:"700",display:"flex",alignItems:"center",gap:"8px",
                  boxShadow:"0 2px 8px rgba(5,150,105,0.25)",
                }}>
                  <span>🎯</span>Manual Landmarking
                </button>

                <button onClick={()=>fileInputRef.current?.click()} style={{ padding:"8px 14px",background:"#fff",border:"1px solid #e2e8f0",borderRadius:"9px",color:"#64748b",cursor:"pointer",fontSize:"12px",fontWeight:"600" }}>
                  🔄 Ganti Foto
                </button>

                <p style={{ color:"#94a3b8",fontSize:"11px",margin:0,lineHeight:"1.5" }}>
                  💡 <strong style={{ color:"#059669" }}>Manual Landmarking</strong> memberi akurasi lebih tinggi bila auto-detect kurang tepat.
                </p>
              </div>
            </div>
          )}

          {/* Badges */}
          <div style={{ display:"flex",gap:"8px",marginTop:"13px",flexWrap:"wrap" }}>
            <span style={{ display:"flex",alignItems:"center",gap:"5px",padding:"3px 11px",background:"#eff6ff",border:"1px solid #bfdbfe",borderRadius:"99px",color:"#1e40af",fontSize:"11px",fontWeight:"500" }}>
              <span style={{ width:"6px",height:"6px",borderRadius:"50%",background:"#2563eb",display:"inline-block" }}/>
              Auto: MediaPipe (model kompleks-2, resolusi 640px+)
            </span>
            <span style={{ display:"flex",alignItems:"center",gap:"5px",padding:"3px 11px",background:"#ecfdf5",border:"1px solid #a7f3d0",borderRadius:"99px",color:"#065f46",fontSize:"11px",fontWeight:"500" }}>
              <span style={{ width:"6px",height:"6px",borderRadius:"50%",background:"#059669",display:"inline-block" }}/>
              Manual: Klik titik anatomi
            </span>
          </div>
        </div>
      </div>
      <input ref={fileInputRef} type="file" accept="image/*" onChange={handleUpload} style={{ display:"none" }} />
    </>
  );
}

// ============================================================
// REBA TABLES & OPTIONS
// ============================================================
const tableA: number[][][] = [
  [[1,2,3,4],[1,2,3,4],[3,3,5,6]],[[2,3,4,5],[3,4,5,6],[4,5,6,7]],
  [[2,4,5,6],[4,5,6,7],[5,6,7,8]],[[3,5,6,7],[5,6,7,8],[6,7,8,9]],
  [[4,6,7,8],[6,7,8,9],[7,8,9,9]],
];
const tableB: number[][][] = [
  [[1,2,2],[1,2,3]],[[1,2,3],[2,3,4]],[[3,4,5],[4,5,5]],[[4,5,5],[5,6,7]],[[7,8,8],[8,9,9]],
];
const tableC: number[][] = [
  [1,1,1,2,3,3,4,5,6,7,7,7],[1,2,2,3,4,4,5,6,6,7,7,8],
  [2,3,3,3,4,5,6,7,7,8,8,8],[3,4,4,4,5,6,7,8,8,9,9,9],
  [4,4,4,5,6,7,8,8,9,9,9,9],[6,6,6,7,8,8,9,9,10,10,10,10],
  [7,7,7,8,9,9,9,10,10,11,11,11],[8,8,8,9,10,10,10,10,10,11,11,11],
  [9,9,9,10,10,10,11,11,11,12,12,12],[10,10,10,11,11,11,11,12,12,12,12,12],
  [11,11,11,11,12,12,12,12,12,12,12,12],[12,12,12,12,12,12,12,12,12,12,12,12],
];

const neckPostures     = [{value:1,label:"0–20° flexion"},{value:2,label:">20° flex/ext"}];
const trunkPostures    = [{value:1,label:"Tegak"},{value:2,label:"0–20° flex/ext"},{value:3,label:"20–60° flexion"},{value:4,label:">60° flexion"}];
const legPostures      = [{value:1,label:"Kedua kaki ditopang"},{value:2,label:"Satu kaki ditopang"},{value:3,label:"Lutut 30–60°"},{value:4,label:"Lutut >60°"}];
const upperArmPostures = [{value:1,label:"−20°–20° flex/ext"},{value:2,label:">20° ext / 20–45° flex"},{value:3,label:"45–90° flexion"},{value:4,label:">90° flexion"}];
const lowerArmPostures = [{value:1,label:"60°–100°"},{value:2,label:"<60° atau >100°"}];
const wristPostures    = [{value:1,label:"0°–15° flex/ext"},{value:2,label:">15° flex/ext"}];
const loadOptions      = [{value:0,label:"Tanpa beban (0)"},{value:1,label:"≤5 kg (+1)"},{value:2,label:">5 kg (+2)"}];
const couplingOptions  = [{value:0,label:"Grip bagus (0)"},{value:1,label:"Grip sedang (+1)"},{value:2,label:"Grip buruk (+2)"}];

// Section header helper
function SectionHeader({ color, title, badge }: { color:string; title:string; badge?:string }) {
  return (
    <div style={{ display:"flex",alignItems:"center",gap:"10px",marginBottom:"14px",padding:"9px 14px",background:"#f8fafc",border:"1px solid #e2e8f0",borderLeft:`4px solid ${color}`,borderRadius:"8px" }}>
      <h2 style={{ fontSize:"14px",fontWeight:"700",color:"#1e293b",margin:0,flex:1 }}>{title}</h2>
      {badge&&<span style={{ padding:"2px 10px",borderRadius:"99px",background:`${color}18`,border:`1px solid ${color}33`,color,fontSize:"11px",fontWeight:"700" }}>{badge}</span>}
    </div>
  );
}

// ============================================================
// MAIN APP — LOGIKA REBA TIDAK BERUBAH
// ============================================================
export default function App() {
  const [groupA, setGroupA] = useState({
    neck:  { posture:0, twisted:false, sideBending:false },
    trunk: { posture:0, twisted:false, sideBending:false },
    legs:0, load:0,
  });
  const [groupB, setGroupB] = useState({
    upperArm: { posture:0, shoulderRaised:false, armAbducted:false, armSupported:false },
    lowerArm: 0,
    wrist:    { posture:0, wristTwisted:false, wristDeviated:false },
    coupling: 0,
  });
  const [activity, setActivity] = useState({ static:false, repetitive:false, rapid:false });
  const [, setUploadedImageUrl] = useState<string|null>(null);

  const handleNeckAdjust     = (t:string,v:boolean) => setGroupA(p=>({...p,neck:{...p.neck,[t]:v}}));
  const handleTrunkAdjust    = (t:string,v:boolean) => setGroupA(p=>({...p,trunk:{...p.trunk,[t]:v}}));
  const handleUpperArmAdjust = (t:string,v:boolean) => setGroupB(p=>({...p,upperArm:{...p.upperArm,[t]:v}}));
  const handleWristAdjust    = (t:string,v:boolean) => setGroupB(p=>({...p,wrist:{...p.wrist,[t]:v}}));

  const applyScores = useCallback((s: ReturnType<typeof computeREBAFromLandmarks>) => {
    setGroupA({ neck:s.neck, trunk:s.trunk, legs:s.legs, load:s.load });
    setGroupB({ upperArm:s.upperArm, lowerArm:s.lowerArm, wrist:s.wrist, coupling:s.coupling });
  }, []);

  const handleImageUpload = useCallback((url:string)=>{ setUploadedImageUrl(url); },[]);

  // REBA scoring logic — TIDAK BERUBAH
  const scoreA = useMemo(() => {
    if (!groupA.neck.posture||!groupA.trunk.posture||!groupA.legs) return 0;
    let n=groupA.neck.posture; if(groupA.neck.twisted)n++; if(groupA.neck.sideBending)n++;
    let t=groupA.trunk.posture; if(groupA.trunk.twisted)t++; if(groupA.trunk.sideBending)t++;
    n=Math.min(Math.max(n,1),3); t=Math.min(Math.max(t,1),5);
    return (tableA[t-1]?.[n-1]?.[groupA.legs-1]??0)+groupA.load;
  },[groupA]);

  const scoreB = useMemo(() => {
    if (!groupB.upperArm.posture||!groupB.lowerArm||!groupB.wrist.posture) return 0;
    let u=groupB.upperArm.posture;
    if(groupB.upperArm.shoulderRaised)u++; if(groupB.upperArm.armAbducted)u++; if(groupB.upperArm.armSupported)u--;
    let w=groupB.wrist.posture; if(groupB.wrist.wristTwisted)w++; if(groupB.wrist.wristDeviated)w++;
    u=Math.min(Math.max(u,1),5); w=Math.min(Math.max(w,1),3);
    return (tableB[u-1]?.[groupB.lowerArm-1]?.[w-1]??0)+groupB.coupling;
  },[groupB]);

  const scoreC = useMemo(()=>{
    if (!scoreA||!scoreB) return 0;
    const a=Math.min(Math.max(scoreA,1),12)-1, b=Math.min(Math.max(scoreB,1),12)-1;
    return tableC[a]?.[b]??0;
  },[scoreA,scoreB]);

  const activityScore = (activity.static?1:0)+(activity.repetitive?1:0)+(activity.rapid?1:0);
  const finalScore    = scoreC+activityScore;
  const riskLevel     = finalScore===0?"Incomplete":finalScore<=1?"Negligible":finalScore<=3?"Low":finalScore<=7?"Medium":finalScore<=10?"High":"Very High";
  const riskColor     = finalScore===0?"#94a3b8":finalScore<=1?"#16a34a":finalScore<=3?"#ca8a04":finalScore<=7?"#ea580c":"#dc2626";

  return (
    <div style={{ minHeight:"100vh",background:"#f1f5f9" }}>
      {/* Sticky Navbar */}
      <div style={{ background:"#fff",borderBottom:"1px solid #e2e8f0",padding:"0 24px",boxShadow:"0 1px 4px rgba(0,0,0,0.07)",position:"sticky",top:0,zIndex:40 }}>
        <div style={{ maxWidth:"1280px",margin:"0 auto",height:"58px",display:"flex",alignItems:"center",justifyContent:"space-between" }}>
          <div style={{ display:"flex",alignItems:"center",gap:"12px" }}>
            <div style={{ width:"36px",height:"36px",borderRadius:"9px",background:"linear-gradient(135deg,#2563eb,#1d4ed8)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:"18px",boxShadow:"0 2px 6px rgba(37,99,235,0.3)" }}>🦴</div>
            <div>
              <h1 style={{ color:"#1e293b",fontSize:"17px",fontWeight:"800",margin:0,letterSpacing:"-0.02em" }}>REBA Assessment Tool</h1>
              <p style={{ color:"#94a3b8",fontSize:"11px",margin:0 }}>Rapid Entire Body Assessment</p>
            </div>
          </div>
          {finalScore>0&&(
            <div style={{ display:"flex",alignItems:"center",gap:"8px",padding:"5px 13px",borderRadius:"99px",background:"#fff",border:`2px solid ${riskColor}33`,boxShadow:"0 1px 6px rgba(0,0,0,0.07)" }}>
              <span style={{ fontSize:"11px",color:"#64748b",fontWeight:"500" }}>REBA:</span>
              <span style={{ fontSize:"20px",fontWeight:"900",color:riskColor }}>{finalScore}</span>
              <span style={{ padding:"2px 8px",borderRadius:"99px",background:`${riskColor}18`,color:riskColor,fontSize:"11px",fontWeight:"700" }}>{riskLevel}</span>
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth:"1280px",margin:"0 auto",padding:"22px 16px",display:"flex",flexDirection:"column",gap:"18px" }}>

        <ImageUploader onAutoScores={applyScores} onManualScores={applyScores} onImageUpload={handleImageUpload} />

        {/* Group A card */}
        <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"18px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
          <SectionHeader color="#2563eb" title="Group A — Body Postures" badge="Neck · Trunk · Legs · Load" />
          <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",gap:"13px" }}>
            <AssessmentSection title="Neck" postureOptions={neckPostures} groupColor="blue"
              onChange={v=>setGroupA(s=>({...s,neck:{...s.neck,posture:v}}))} selectedValue={groupA.neck.posture}
              adjustments={{ twisted:groupA.neck.twisted, sideBending:groupA.neck.sideBending, onAdjustmentChange:handleNeckAdjust }} />
            <AssessmentSection title="Trunk" postureOptions={trunkPostures} groupColor="blue"
              onChange={v=>setGroupA(s=>({...s,trunk:{...s.trunk,posture:v}}))} selectedValue={groupA.trunk.posture}
              adjustments={{ twisted:groupA.trunk.twisted, sideBending:groupA.trunk.sideBending, onAdjustmentChange:handleTrunkAdjust }} />
            <AssessmentSection title="Legs" postureOptions={legPostures} groupColor="blue"
              onChange={v=>setGroupA(s=>({...s,legs:v}))} selectedValue={groupA.legs} />
            <AssessmentSection title="Load" postureOptions={loadOptions} groupColor="blue"
              onChange={v=>setGroupA(s=>({...s,load:v}))} selectedValue={groupA.load} />
          </div>
        </div>

        {/* Group B card */}
        <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"18px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
          <SectionHeader color="#059669" title="Group B — Arm & Wrist Postures" badge="Upper Arm · Lower Arm · Wrist · Coupling" />
          <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",gap:"13px" }}>
            <AssessmentSection title="Upper Arm" postureOptions={upperArmPostures} groupColor="emerald"
              onChange={v=>setGroupB(s=>({...s,upperArm:{...s.upperArm,posture:v}}))} selectedValue={groupB.upperArm.posture}
              adjustments={{ shoulderRaised:groupB.upperArm.shoulderRaised, armAbducted:groupB.upperArm.armAbducted, armSupported:groupB.upperArm.armSupported, onAdjustmentChange:handleUpperArmAdjust }} />
            <AssessmentSection title="Lower Arm" postureOptions={lowerArmPostures} groupColor="emerald"
              onChange={v=>setGroupB(s=>({...s,lowerArm:v}))} selectedValue={groupB.lowerArm} />
            <AssessmentSection title="Wrist" postureOptions={wristPostures} groupColor="emerald"
              onChange={v=>setGroupB(s=>({...s,wrist:{...s.wrist,posture:v}}))} selectedValue={groupB.wrist.posture}
              adjustments={{ wristTwisted:groupB.wrist.wristTwisted, wristDeviated:groupB.wrist.wristDeviated, onAdjustmentChange:handleWristAdjust }} />
            <AssessmentSection title="Coupling" postureOptions={couplingOptions} groupColor="emerald"
              onChange={v=>setGroupB(s=>({...s,coupling:v}))} selectedValue={groupB.coupling} />
          </div>
        </div>

        {/* Score Calculation card */}
        <div style={{ background:"#fff",borderRadius:"14px",border:"1px solid #e2e8f0",padding:"18px",boxShadow:"0 1px 4px rgba(0,0,0,0.06)" }}>
          <SectionHeader color="#7c3aed" title="Score Calculation" />
          <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(240px,1fr))",gap:"13px" }}>
            <ScoringTable title="Table A — Body Score" tableLabel="Score A" formula="Trunk + Neck + Legs + Load"
              components={["Neck","Trunk","Legs","Load"]} groupColor="blue" score={scoreA} />
            <ScoringTable title="Table B — Arm Score" tableLabel="Score B" formula="Upper Arm + Lower Arm + Wrist + Coupling"
              components={["Upper Arm","Lower Arm","Wrist","Coupling"]} groupColor="emerald" score={scoreB} />
            <CombinationTable scoreA={scoreA} scoreB={scoreB} scoreC={scoreC} />
          </div>
        </div>

        <ActivityScore activityScore={activityScore} onActivityChange={setActivity} />
        <FinalResults finalScore={finalScore} riskLevel={riskLevel} scoreC={scoreC} activityScore={activityScore} />

      </div>
    </div>
  );
}