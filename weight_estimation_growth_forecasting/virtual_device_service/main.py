"""
Virtual Device Service - Pull-style Simulator API
- Simulates sensor readings with stateful drift
- Serves paired RGB+Depth images via round-robin selection over EXISTING pairs
- Logs all operations to SQLite
- Runs on port 8010
"""

import os
import re
import sqlite3
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ==================== CONSTANTS ====================

DEVICE_ID = "sim-01"
DB_PATH = "virtual_device_service/device_runtime.db"

RGB_DIR = "best_matches/RGBImages"
DEPTH_DIR = "best_matches/DepthImages"

# Sensor baselines
BASELINES = {
    "NORMAL": {"temperature_c": 23.5, "humidity_pct": 70.0, "ec_ms_cm": 1.6, "ph": 6.0},
    "STRESS": {"temperature_c": 28.0, "humidity_pct": 60.0, "ec_ms_cm": 2.2, "ph": 5.3},
}

# Sensor ranges (clamping)
RANGES = {
    "temperature_c": (15.0, 35.0),
    "humidity_pct": (30.0, 95.0),
    "ec_ms_cm": (0.8, 3.0),
    "ph": (4.5, 7.5),
}

# Drift parameters (max change per call)
DRIFT_AMOUNT = {"temperature_c": 0.3, "humidity_pct": 1.5, "ec_ms_cm": 0.05, "ph": 0.08}

# Filename patterns
RGB_PATTERN = re.compile(r"^RGB_(\d+)\.png$")
DEPTH_PATTERN = re.compile(r"^Depth_(\d+)\.png$")


# ==================== SCHEMAS ====================

class SensorReading(BaseModel):
    device_id: str
    zone_id: str
    timestamp: str
    temperature_c: float
    humidity_pct: float
    ec_ms_cm: float
    ph: float
    mode: str


class CaptureRequest(BaseModel):
    plant_id: str
    zone_id: str
    mode: str = "NORMAL"


class CaptureResponse(BaseModel):
    plant_id: str
    zone_id: str
    timestamp: str
    pair_index: int
    rgb_filename: str
    depth_filename: str
    rgb_url: str
    depth_url: str
    sensors: SensorReading


# ==================== STATE ====================

# Stateful sensor values per zone
sensor_state: Dict[str, Dict[str, float]] = {}

# Round-robin POSITION per zone (index into list of available pair indices)
zone_pair_pos: Dict[str, int] = {}


# ==================== DATABASE ====================

def init_db():
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            temperature_c REAL NOT NULL,
            humidity_pct REAL NOT NULL,
            ec_ms_cm REAL NOT NULL,
            ph REAL NOT NULL,
            mode TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS capture_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            plant_id TEXT NOT NULL,
            pair_index INTEGER NOT NULL,
            rgb_filename TEXT NOT NULL,
            depth_filename TEXT NOT NULL,
            mode TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def log_sensor_reading(reading: SensorReading):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sensor_logs (ts, zone_id, temperature_c, humidity_pct, ec_ms_cm, ph, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        reading.timestamp,
        reading.zone_id,
        reading.temperature_c,
        reading.humidity_pct,
        reading.ec_ms_cm,
        reading.ph,
        reading.mode,
    ))
    conn.commit()
    conn.close()


def log_capture(ts: str, zone_id: str, plant_id: str, pair_index: int, rgb_filename: str, depth_filename: str, mode: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO capture_logs (ts, zone_id, plant_id, pair_index, rgb_filename, depth_filename, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ts, zone_id, plant_id, pair_index, rgb_filename, depth_filename, mode))
    conn.commit()
    conn.close()


# ==================== HELPERS ====================

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def list_available_pair_indices() -> list[int]:
    """Return sorted indices k where BOTH RGB_k.png and Depth_k.png exist."""
    if not os.path.exists(RGB_DIR) or not os.path.exists(DEPTH_DIR):
        return []

    rgb_indices = set()
    for fn in os.listdir(RGB_DIR):
        m = RGB_PATTERN.match(fn)
        if m:
            rgb_indices.add(int(m.group(1)))

    depth_indices = set()
    for fn in os.listdir(DEPTH_DIR):
        m = DEPTH_PATTERN.match(fn)
        if m:
            depth_indices.add(int(m.group(1)))

    return sorted(rgb_indices.intersection(depth_indices))


def next_round_robin_pair(zone_id: str, pairs: list[int]) -> int:
    """Round-robin over the AVAILABLE PAIRS list."""
    if not pairs:
        raise ValueError("No paired RGB/Depth images available")

    pos = zone_pair_pos.get(zone_id, -1)
    pos = (pos + 1) % len(pairs)
    zone_pair_pos[zone_id] = pos
    return pairs[pos]


def generate_sensors(zone_id: str, mode: str) -> SensorReading:
    import random

    mode = (mode or "NORMAL").upper()
    if mode not in BASELINES:
        mode = "NORMAL"

    baseline = BASELINES[mode]

    if zone_id not in sensor_state:
        sensor_state[zone_id] = baseline.copy()

    current = sensor_state[zone_id]

    for key in ["temperature_c", "humidity_pct", "ec_ms_cm", "ph"]:
        target = baseline[key]
        current_val = current[key]
        drift_max = DRIFT_AMOUNT[key]

        drift = random.uniform(-drift_max, drift_max)
        pull = (target - current_val) * 0.1  # pull 10% back to baseline

        new_val = current_val + drift + pull
        lo, hi = RANGES[key]
        current[key] = clamp(new_val, lo, hi)

    ts = datetime.utcnow().isoformat() + "Z"
    return SensorReading(
        device_id=DEVICE_ID,
        zone_id=zone_id,
        timestamp=ts,
        temperature_c=round(current["temperature_c"], 2),
        humidity_pct=round(current["humidity_pct"], 2),
        ec_ms_cm=round(current["ec_ms_cm"], 2),
        ph=round(current["ph"], 2),
        mode=mode,
    )


def validate_filename(filename: str) -> bool:
    return re.match(r"^(RGB|Depth)_\d+\.png$", filename) is not None


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Virtual Device Service",
    description="Pull-style simulator API for hydroponic system testing",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    init_db()
    pairs = list_available_pair_indices()
    print(f"✓ Database initialized at {DB_PATH}")
    print(f"✓ RGB images directory: {RGB_DIR}")
    print(f"✓ Depth images directory: {DEPTH_DIR}")
    print(f"✓ Paired indices found: {len(pairs)} (example: {pairs[:10]})")


# ==================== ENDPOINTS ====================

@app.get("/device/health")
def health_check():
    return {"status": "ok"}


@app.get("/device/sensors", response_model=SensorReading)
def get_sensors(
    zone_id: str = Query(..., description="Zone identifier"),
    mode: str = Query("NORMAL", description="Sensor mode: NORMAL or STRESS"),
):
    reading = generate_sensors(zone_id, mode)
    log_sensor_reading(reading)
    return reading


@app.post("/device/capture", response_model=CaptureResponse)
@app.post("/device/capture/", response_model=CaptureResponse)
def capture_images(request: CaptureRequest, http_request: Request):
    pairs = list_available_pair_indices()
    if not pairs:
        raise HTTPException(
            status_code=503,
            detail="No valid RGB/Depth pairs found. Ensure RGB_k.png and Depth_k.png exist for the same k.",
        )

    pair_idx = next_round_robin_pair(request.zone_id, pairs)

    rgb_filename = f"RGB_{pair_idx}.png"
    depth_filename = f"Depth_{pair_idx}.png"

    rgb_path = os.path.join(RGB_DIR, rgb_filename)
    depth_path = os.path.join(DEPTH_DIR, depth_filename)

    if not os.path.exists(rgb_path):
        raise HTTPException(status_code=404, detail=f"RGB image not found: {rgb_filename}")
    if not os.path.exists(depth_path):
        raise HTTPException(status_code=404, detail=f"Depth image not found: {depth_filename}")

    sensors = generate_sensors(request.zone_id, request.mode)
    log_sensor_reading(sensors)

    base_url = str(http_request.base_url).rstrip("/")
    rgb_url = f"{base_url}/device/files/{rgb_filename}"
    depth_url = f"{base_url}/device/files/{depth_filename}"

    log_capture(
        ts=sensors.timestamp,
        zone_id=request.zone_id,
        plant_id=request.plant_id,
        pair_index=pair_idx,
        rgb_filename=rgb_filename,
        depth_filename=depth_filename,
        mode=request.mode,
    )

    return CaptureResponse(
        plant_id=request.plant_id,
        zone_id=request.zone_id,
        timestamp=sensors.timestamp,
        pair_index=pair_idx,
        rgb_filename=rgb_filename,
        depth_filename=depth_filename,
        rgb_url=rgb_url,
        depth_url=depth_url,
        sensors=sensors,
    )


@app.get("/device/files/{filename}")
def serve_image(filename: str):
    if not validate_filename(filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Must match RGB_<num>.png or Depth_<num>.png pattern.",
        )

    if filename.startswith("RGB_"):
        file_path = os.path.join(RGB_DIR, filename)
    elif filename.startswith("Depth_"):
        file_path = os.path.join(DEPTH_DIR, filename)
    else:
        raise HTTPException(status_code=400, detail="Invalid filename prefix")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="image/png", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)