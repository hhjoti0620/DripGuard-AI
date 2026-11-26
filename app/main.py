import asyncio
import base64
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.auth import (
    create_access_token,
    get_current_user,
    get_ws_user,
    hash_password,
    require_role,
    verify_password,
)
from app.clinical_engine import DripFlowEngine, EngineConfig, OnnxDropDetector
from app.clinical_rules import (
    evaluate_infusion_order,
    explain_telemetry_decision,
    validate_lifecycle_transition,
)
from app.database import (
    Alert,
    AuditLog,
    InfusionBed,
    InfusionSession,
    Patient,
    SessionLocal,
    TelemetryLog,
    User,
    atomic_bed_update,
    get_db,
    init_db,
    record_audit,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.onnx"
DEFAULTS_PATH = BASE_DIR / "engine_defaults.json"
VIDEOS_DIR = (BASE_DIR / "sample_videos").resolve()
STATIC_DIR = (BASE_DIR / "static").resolve()

VIDEOS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

DEFAULTS = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8")) if DEFAULTS_PATH.exists() else {}
DEFAULT_CONF = float(DEFAULTS.get("conf", 0.25))
DEFAULT_IOU = float(DEFAULTS.get("iou", 0.60))
DEFAULT_SIZE = int(DEFAULTS.get("input_size", 640))

VIDEO_PRESETS = {
    "15011120": {"roi": [0.36, 0.36, 0.65, 0.72], "corridor": [0.45, 0.36, 0.85, 0.72]},
    "6248839":  {"roi": [0.34, 0.25, 0.66, 0.72], "corridor": [0.25, 0.08, 0.75, 0.85]},
    "12076506": {"roi": [0.30, 0.25, 0.70, 0.75], "corridor": [0.25, 0.15, 0.75, 0.80]},
}


def seed_initial_data() -> None:
    init_db()
    db = SessionLocal()
    try:
        seed_pw = os.getenv("DRIPGUARD_DEMO_PASSWORD", "ChangeMe@2026")
        if db.query(User).count() == 0:
            db.add(User(username="nurse1", full_name="Senior ICU Nurse", hashed_password=hash_password(seed_pw), role="nurse"))
            db.add(User(username="doctor1", full_name="Attending Physician", hashed_password=hash_password(seed_pw), role="doctor"))
            db.add(User(username="admin", full_name="Clinical Engineer", hashed_password=hash_password(seed_pw), role="admin"))
            db.commit()

        if db.query(Patient).count() == 0:
            p1 = Patient(mrn="MRN-1001", full_name="Rahim Uddin", age_years=54, weight_kg=68.0, is_pediatric=False, ward_notes="Post-op hydration")
            p2 = Patient(mrn="MRN-1002", full_name="Baby Ayaan", age_years=3, weight_kg=14.0, is_pediatric=True, ward_notes="Pediatric micro-drip protocol")
            db.add_all([p1, p2])
            db.commit()
            db.refresh(p1)
            db.refresh(p2)

            vids = sorted([p.name for p in VIDEOS_DIR.glob("*.mp4")])
            vid1 = next((v for v in vids if "15011120" in v), vids[0] if vids else "webcam:0")
            vid2 = next((v for v in vids if "6248839" in v), vid1)
            pr1 = next((v for k, v in VIDEO_PRESETS.items() if k in vid1), {"roi": [0.36, 0.36, 0.65, 0.72], "corridor": [0.45, 0.36, 0.85, 0.72]})
            pr2 = next((v for k, v in VIDEO_PRESETS.items() if k in vid2), {"roi": [0.34, 0.25, 0.66, 0.72], "corridor": [0.25, 0.08, 0.75, 0.85]})

            b1 = InfusionBed(
                bed_code="ICU-BED-01",
                ward_name="ICU-A",
                patient_id=p1.id,
                fluid_type="Normal Saline 0.9% (500 mL)",
                target_volume_ml=500.0,
                target_duration_min=165.0,
                drop_factor=20.0,
                prescribed_dpm=60.0,
                video_source=vid1,
                roi_json=json.dumps(pr1["roi"]),
                corridor_json=json.dumps(pr1["corridor"]),
                lifecycle_state="RUNNING",
            )
            b2 = InfusionBed(
                bed_code="PED-BED-02",
                ward_name="PICU-B",
                patient_id=p2.id,
                fluid_type="Dextrose 5% in 0.45% NaCl",
                target_volume_ml=250.0,
                target_duration_min=300.0,
                drop_factor=60.0,
                prescribed_dpm=50.0,
                video_source=vid2,
                roi_json=json.dumps(pr2["roi"]),
                corridor_json=json.dumps(pr2["corridor"]),
                lifecycle_state="RUNNING",
            )
            db.add_all([b1, b2])
            db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_initial_data()
    yield
    await sync_hub.shutdown_all()


app = FastAPI(title="DripGuard AI Clinical Mission Control", version="2.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

detector: Optional[OnnxDropDetector] = None


def get_detector() -> OnnxDropDetector:
    global detector
    if detector is None:
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            raise HTTPException(
                status_code=503,
                detail=f"ONNX model file is missing or empty at {MODEL_PATH}. Place valid best.onnx in models/.",
            )
        detector = OnnxDropDetector(
            str(MODEL_PATH),
            conf=max(DEFAULT_CONF, 0.22),
            iou=DEFAULT_IOU,
            size=DEFAULT_SIZE,
            providers=("CPUExecutionProvider",),
        )
    return detector


def validate_safe_video_source(src: str) -> None:
    if src.startswith("webcam:"):
        idx_str = src.split(":", 1)[1]
        if not idx_str.isdigit() or int(idx_str) > 8:
            raise HTTPException(status_code=422, detail="Invalid webcam index (must be webcam:0 .. webcam:8)")
        return
    if "/" in src or "\\" in src or ".." in src:
        raise HTTPException(status_code=422, detail="Invalid video_source path (path traversal blocked)")
    candidate = (VIDEOS_DIR / src).resolve()
    if candidate.parent != VIDEOS_DIR or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Sample video not found: {src}")


class BedSyncHub:
    """
    A01 Shared State Manager: Runs ONE authoritative telemetry/inference worker per bed
    so multiple connected clients see identical real-time drop counts without duplicate DB writes.
    """

    def __init__(self) -> None:
        self.subscribers: Dict[int, Set[WebSocket]] = {}
        self.live_configs: Dict[int, dict] = {}
        self.reset_flags: Set[int] = set()
        self.source_reload_flags: Set[int] = set()
        self.workers: Dict[int, asyncio.Task] = {}

    async def broadcast_bed_event(self, bed_id: int, event: dict) -> None:
        for ws in list(self.subscribers.get(bed_id, set())):
            try:
                await ws.send_json(event)
            except Exception:
                self.subscribers.get(bed_id, set()).discard(ws)

    def ensure_worker(self, bed_id: int, started_by: str) -> None:
        task = self.workers.get(bed_id)
        if task is None or task.done():
            self.workers[bed_id] = asyncio.create_task(self._run_bed_worker(bed_id, started_by))

    async def shutdown_all(self) -> None:
        for task in list(self.workers.values()):
            task.cancel()
        self.workers.clear()

    async def _run_bed_worker(self, bed_id: int, started_by: str) -> None:
        try:
            det = get_detector()
        except HTTPException as exc:
            await self.broadcast_bed_event(bed_id, {"error": exc.detail})
            return

        db = SessionLocal()
        bed = db.query(InfusionBed).filter(InfusionBed.id == bed_id).first()
        if not bed:
            db.close()
            await self.broadcast_bed_event(bed_id, {"error": "Bed not found"})
            return

        src = bed.video_source
        roi = json.loads(bed.roi_json)
        corridor = tuple(json.loads(bed.corridor_json))
        cfg = EngineConfig(
            drop_factor=bed.drop_factor,
            prescribed_dpm=bed.prescribed_dpm,
            tolerance=bed.tolerance,
            corridor=corridor,
            min_conf=0.25,
        )
        session_rec = InfusionSession(bed_id=bed_id, started_by=started_by, state="RUNNING")
        db.add(session_rec)
        db.commit()
        db.refresh(session_rec)
        session_id = session_rec.id
        db.close()

        engine = DripFlowEngine(cfg)

        def open_capture(source_str: str):
            if source_str.startswith("webcam:"):
                return cv2.VideoCapture(int(source_str.split(":")[1])), False
            v_path = (VIDEOS_DIR / source_str).resolve()
            if v_path.parent != VIDEOS_DIR or not v_path.exists():
                return None, True
            return cv2.VideoCapture(str(v_path)), True

        cap, is_file = open_capture(src)
        if cap is None:
            await self.broadcast_bed_event(bed_id, {"error": f"Video not found: {src}"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1 or fps > 120:
            fps = 30.0

        # Real-time visual playback paced by wall-clock time so 0.1 vCPU never lags behind real time
        visual_stride = 4 if fps >= 45 else 2
        target_step_sec = visual_stride / fps

        frame_idx = 0
        last_db_log_idx = -999
        last_st = None
        last_xai = None
        last_infer_done_at = 0.0
        infer_task: Optional[asyncio.Task] = None

        try:
            while self.subscribers.get(bed_id):
                loop_start = time.monotonic()
                pushed = self.live_configs.pop(bed_id, None)
                if pushed:
                    engine.cfg.prescribed_dpm = pushed["prescribed_dpm"]
                    engine.cfg.drop_factor = float(pushed["drop_factor"])
                    engine.cfg.tolerance = float(pushed.get("tolerance", engine.cfg.tolerance))
                    roi = pushed["roi"]
                    corridor = tuple(pushed["corridor"])
                    engine.cfg.corridor = corridor
                    if bed_id in self.source_reload_flags:
                        self.source_reload_flags.discard(bed_id)
                        cap.release()
                        cap, is_file = open_capture(pushed["video_source"])
                        if cap is None:
                            await self.broadcast_bed_event(bed_id, {"error": f"Video not found: {pushed['video_source']}"})
                            break
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        if fps <= 1 or fps > 120:
                            fps = 30.0
                        visual_stride = 4 if fps >= 45 else 2
                        target_step_sec = visual_stride / fps
                        engine.reset()
                        frame_idx = 0
                        last_db_log_idx = -999
                        last_st = None
                        last_xai = None
                        last_infer_done_at = 0.0

                if bed_id in self.reset_flags:
                    self.reset_flags.discard(bed_id)
                    engine.reset()
                    frame_idx = 0
                    last_db_log_idx = -999
                    last_st = None
                    last_xai = None
                    last_infer_done_at = 0.0
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                ok, fr = cap.read()
                if not ok:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        engine.reset()
                        frame_idx = 0
                        last_db_log_idx = -999
                        continue
                    break

                current_idx = frame_idx
                frame_idx += 1

                if current_idx % visual_stride != 0:
                    continue

                t = current_idx / fps
                orig_H, orig_W = fr.shape[:2]
                out_w = 270
                out_h = max(1, int(orig_H * (out_w / orig_W)))
                vis = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                rx0 = max(0, min(out_w - 10, int(roi[0] * out_w)))
                ry0 = max(0, min(out_h - 10, int(roi[1] * out_h)))
                rx1 = max(rx0 + 10, min(out_w, int(roi[2] * out_w)))
                ry1 = max(ry0 + 10, min(out_h, int(roi[3] * out_h)))
                rw, rh = rx1 - rx0, ry1 - ry0

                # Collect completed inference and leave a 1.5s CPU breather so video stays 100% real-time
                if infer_task is not None and infer_task.done():
                    try:
                        inf_t, boxes_roi = infer_task.result()
                        last_st = engine.update(inf_t, boxes_roi)
                        last_xai = explain_telemetry_decision(
                            status=last_st.status,
                            dpm=last_st.dpm,
                            prescribed_dpm=engine.cfg.prescribed_dpm,
                            tolerance=engine.cfg.tolerance,
                            ml_hr=last_st.ml_hr,
                            drop_factor=engine.cfg.drop_factor,
                            raw_reason=last_st.reason,
                        )
                    except Exception:
                        pass
                    infer_task = None
                    last_infer_done_at = time.monotonic()

                now_mono = time.monotonic()
                if infer_task is None and (last_st is None or (now_mono - last_infer_done_at) >= 1.5):
                    roi_crop = vis[ry0:ry1, rx0:rx1].copy()
                    async def _bg_infer(crop, ts):
                        boxes = await asyncio.to_thread(det.detect, crop)
                        return ts, boxes
                    infer_task = asyncio.create_task(_bg_infer(roi_crop, t))

                if last_st is None:
                    last_st = engine.update(t, [])
                    last_xai = explain_telemetry_decision(
                        status=last_st.status,
                        dpm=last_st.dpm,
                        prescribed_dpm=engine.cfg.prescribed_dpm,
                        tolerance=engine.cfg.tolerance,
                        ml_hr=last_st.ml_hr,
                        drop_factor=engine.cfg.drop_factor,
                        raw_reason=last_st.reason,
                    )

                st = last_st
                xai = last_xai

                cv2.rectangle(vis, (rx0, ry0), (rx1, ry1), (220, 190, 80), 1)
                cx1 = int(rx0 + corridor[0] * rw)
                cy1 = int(ry0 + corridor[1] * rh)
                cx2 = int(rx0 + corridor[2] * rw)
                cy2 = int(ry0 + corridor[3] * rh)
                cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (56, 189, 248), 2)

                for x1, y1, x2, y2, cf in st.boxes:
                    bx1 = int(rx0 + x1 * rw)
                    by1 = int(ry0 + y1 * rh)
                    bx2 = int(rx0 + x2 * rw)
                    by2 = int(ry0 + y2 * rh)
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (102, 255, 0), 2)
                    cv2.putText(vis, f"{cf:.2f}", (bx2 + 4, by1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (102, 255, 0), 1, cv2.LINE_AA)

                _, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 48])
                b64_img = base64.b64encode(jpg.tobytes()).decode("ascii")

                if current_idx - last_db_log_idx >= int(fps * 2):
                    last_db_log_idx = current_idx
                    log_db = SessionLocal()
                    try:
                        log_db.add(TelemetryLog(
                            bed_id=bed_id,
                            total_drops=st.total_drops,
                            dpm=st.dpm,
                            ml_hr=st.ml_hr,
                            ml_infused=st.ml_infused,
                            status=st.status,
                            confidence=st.confidence,
                            reason=st.reason,
                        ))
                        if st.status in ("FAST", "SLOW", "STOPPED_OR_OCCLUDED"):
                            existing_open = (
                                log_db.query(Alert)
                                .filter(
                                    Alert.bed_id == bed_id,
                                    Alert.alert_type == st.status,
                                    Alert.acknowledged == False,
                                )
                                .first()
                            )
                            if not existing_open:
                                sev = "CRITICAL" if st.status in ("FAST", "STOPPED_OR_OCCLUDED") else "WARNING"
                                log_db.add(Alert(
                                    bed_id=bed_id,
                                    severity=sev,
                                    alert_type=st.status,
                                    message=xai["action_required"],
                                ))
                        log_db.commit()
                    finally:
                        log_db.close()

                await self.broadcast_bed_event(bed_id, {
                    "type": "telemetry",
                    "frame": b64_img,
                    "telemetry": {
                        "t": round(t, 2),
                        "total_drops": st.total_drops,
                        "dpm": st.dpm,
                        "ml_hr": st.ml_hr,
                        "ml_infused": st.ml_infused,
                        "status": st.status,
                        "reason": st.reason,
                        "confidence": st.confidence,
                        "new_event": st.new_event,
                        "xai": xai,
                    },
                })

                elapsed = time.monotonic() - loop_start
                sleep_time = max(0.02, target_step_sec - elapsed)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass
        finally:
            if infer_task is not None and not infer_task.done():
                infer_task.cancel()
            if cap is not None:
                cap.release()
            self.workers.pop(bed_id, None)
            end_db = SessionLocal()
            try:
                s = end_db.query(InfusionSession).filter(InfusionSession.id == session_id).first()
                if s:
                    s.ended_at = datetime.now(timezone.utc)
                    s.state = "COMPLETED"
                    end_db.commit()
            finally:
                end_db.close()


sync_hub = BedSyncHub()


def _validate_box(vals: List[float], name: str) -> List[float]:
    if len(vals) != 4 or any(not (0.0 <= v <= 1.0) for v in vals):
        raise ValueError(f"{name} must contain 4 normalized coordinates between 0.0 and 1.0")
    if vals[2] <= vals[0] + 0.02 or vals[3] <= vals[1] + 0.02:
        raise ValueError(f"{name} must satisfy x1 > x0 + 0.02 and y1 > y0 + 0.02")
    return vals


class PatientCreate(BaseModel):
    mrn: str = Field(..., min_length=2, max_length=40)
    full_name: str = Field(..., min_length=2, max_length=100)
    age_years: int = Field(..., ge=0, le=120)
    weight_kg: float = Field(..., gt=0.5, le=350.0)
    is_pediatric: bool = False
    ward_notes: str = ""


class OrderCheckRequest(BaseModel):
    prescribed_dpm: Optional[float] = Field(default=None, gt=0, le=300)
    drop_factor: float = Field(default=20.0, gt=0)
    target_volume_ml: float = Field(default=500.0, gt=0, le=5000)
    target_duration_min: float = Field(default=240.0, gt=0, le=10080)
    patient_id: Optional[int] = None


class BedCreate(BaseModel):
    bed_code: str = Field(..., min_length=2, max_length=32)
    ward_name: str = "ICU-A"
    patient_id: Optional[int] = None
    fluid_type: str = "Normal Saline 0.9%"
    target_volume_ml: float = Field(default=500.0, gt=0, le=5000)
    target_duration_min: float = Field(default=240.0, gt=0, le=10080)
    drop_factor: float = Field(default=20.0, gt=0)
    prescribed_dpm: Optional[float] = Field(default=60.0, gt=0, le=300)
    tolerance: float = Field(default=0.25, gt=0.01, le=0.9)
    video_source: str = "15011120_1080_1920_30fps.mp4"
    roi: List[float] = [0.36, 0.36, 0.65, 0.72]
    corridor: List[float] = [0.45, 0.36, 0.85, 0.72]

    @field_validator("roi")
    @classmethod
    def check_roi(cls, v):
        return _validate_box(v, "roi")

    @field_validator("corridor")
    @classmethod
    def check_corridor(cls, v):
        return _validate_box(v, "corridor")


class BedUpdate(BaseModel):
    ward_name: Optional[str] = None
    patient_id: Optional[int] = None
    fluid_type: Optional[str] = None
    target_volume_ml: Optional[float] = Field(default=None, gt=0, le=5000)
    target_duration_min: Optional[float] = Field(default=None, gt=0, le=10080)
    drop_factor: Optional[float] = Field(default=None, gt=0)
    prescribed_dpm: Optional[float] = Field(default=None, gt=0, le=300)
    clear_prescribed_dpm: bool = False
    tolerance: Optional[float] = Field(default=None, gt=0.01, le=0.9)
    video_source: Optional[str] = None
    roi: Optional[List[float]] = None
    corridor: Optional[List[float]] = None
    lifecycle_state: Optional[str] = None
    version_id: int

    @field_validator("roi")
    @classmethod
    def check_roi(cls, v):
        return _validate_box(v, "roi") if v is not None else v

    @field_validator("corridor")
    @classmethod
    def check_corridor(cls, v):
        return _validate_box(v, "corridor") if v is not None else v


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_ready": MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0,
        "defaults": DEFAULTS,
    }


@app.post("/api/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "role": user.role,
        "full_name": user.full_name,
    }


@app.get("/api/auth/me")
def get_me(user: User = Depends(get_current_user)):
    return {"id": user.id, "username": user.username, "full_name": user.full_name, "role": user.role}


@app.get("/api/videos")
def list_videos(user: User = Depends(get_current_user)):
    vids = sorted([p.name for p in VIDEOS_DIR.glob("*.mp4")])
    return {"videos": vids, "presets": VIDEO_PRESETS}


@app.post("/api/clinical/validate-order")
def validate_order(payload: OrderCheckRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first() if payload.patient_id else None
    ev = evaluate_infusion_order(
        prescribed_dpm=payload.prescribed_dpm,
        drop_factor=payload.drop_factor,
        target_volume_ml=payload.target_volume_ml,
        target_duration_min=payload.target_duration_min,
        patient_weight_kg=patient.weight_kg if patient else None,
        is_pediatric=patient.is_pediatric if patient else False,
    )
    return {
        "feasible": ev.feasible,
        "risk_level": ev.risk_level,
        "computed_dpm": ev.computed_dpm,
        "computed_ml_hr": ev.computed_ml_hr,
        "explanation": ev.explanation,
        "violated_constraints": ev.violated_constraints,
        "recommended_adjustment": ev.recommended_adjustment,
    }


@app.get("/api/patients")
def list_patients(q: Optional[str] = None, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    query = db.query(Patient)
    if q:
        like = f"%{q.strip()}%"
        query = query.filter(or_(Patient.full_name.ilike(like), Patient.mrn.ilike(like)))
    return [
        {
            "id": p.id,
            "mrn": p.mrn,
            "full_name": p.full_name,
            "age_years": p.age_years,
            "weight_kg": p.weight_kg,
            "is_pediatric": p.is_pediatric,
            "ward_notes": p.ward_notes,
        }
        for p in query.order_by(Patient.id.asc()).all()
    ]


@app.post("/api/patients", status_code=201)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db
