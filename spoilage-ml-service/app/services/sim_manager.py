# app/services/sim_manager.py
from __future__ import annotations

import csv
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlmodel import Session

from app.db import engine
from app.models import SpoilagePrediction
from app.services.postprocess import make_status

STAGES = ["fresh", "slightly_aged", "near_spoilage", "spoiled"]


def plant_str_to_csv_int(plant_id: str) -> int:
    """
    CSV plant_id is numeric (0,1,2...) in your probs dataset.
    Allow UI to send "P-001" or "0".

    P-001 -> 0
    P-002 -> 1
    "0"   -> 0
    """
    s = str(plant_id).strip()
    if s.upper().startswith("P-"):
        n = int(s.split("-")[1])
        return max(0, n - 1)
    return int(float(s))


def csv_int_to_plant_str(n: int) -> str:
    # 0 -> P-001
    return f"P-{n+1:03d}"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def stage_from_probs(pf: float, ps: float, pn: float, pp: float) -> str:
    probs = {"fresh": pf, "slightly_aged": ps, "near_spoilage": pn, "spoiled": pp}
    return max(probs, key=probs.get)


@dataclass
class SimState:
    running: bool = False
    plant_id: str | None = None
    interval_sec: int = 15
    loop: bool = False
    last_row: dict[str, Any] | None = None
    started_at: str | None = None


class ProbReplaySimulator:
    """
    Replays rows from SIM_PROBS_CSV for a single plant.
    Also provides sample_row() for UI simulation.

    ✅ Important fix:
    - Do NOT use request-scoped Session from Depends(get_session).
    - Always create a NEW Session(engine) inside the background thread.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.state = SimState()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rows: list[dict[str, Any]] | None = None
        self._sim_images: list[str] | None = None

    # -------------------------
    # CSV + Image loading
    # -------------------------
    def _load_csv_once(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows

        path = Path(self.csv_path)

        # optional auto-fallback if you have sim-data vs sim_data mismatch
        if not path.exists():
            alt1 = Path(str(path).replace("sim_data", "sim-data"))
            alt2 = path.with_name(path.stem + " (2)" + path.suffix)

            if alt1.exists():
                path = alt1
            elif alt2.exists():
                path = alt2
            else:
                raise FileNotFoundError(
                    f"SIM_PROBS_CSV not found: {path.resolve()} "
                    f"(also tried {alt1.resolve()} and {alt2.resolve()})"
                )

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]

        if not rows:
            raise RuntimeError("SIM_PROBS_CSV loaded but has 0 rows")

        self._rows = rows
        return rows

    def _load_sim_images_once(self) -> list[str]:
        if self._sim_images is not None:
            return self._sim_images

        img_dir = Path("sim_images")
        if not img_dir.exists():
            self._sim_images = []
            return self._sim_images

        imgs = []
        for p in img_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.append(p.name)

        self._sim_images = imgs
        return imgs

    def _pick_existing_image(self, preferred_name: str | None) -> str | None:
        imgs = self._load_sim_images_once()
        if not imgs:
            return None

        if preferred_name:
            # exact match
            if preferred_name in imgs:
                return preferred_name
            # case-insensitive match
            low = preferred_name.lower()
            for x in imgs:
                if x.lower() == low:
                    return x

        # fallback random
        return random.choice(imgs)

    # -------------------------
    # ✅ UI sampling (NEW)
    # -------------------------
    def sample_row(self, plant_id: str | None = None, label: str | None = None) -> dict[str, Any]:
        rows = self._load_csv_once()

        filtered = rows

        if plant_id and str(plant_id).strip():
            try:
                pid_int = plant_str_to_csv_int(plant_id)
                filtered = [r for r in filtered if int(float(r.get("plant_id", -999))) == pid_int]
            except Exception:
                pass

        if label and str(label).strip():
            lab = str(label).strip()
            filtered = [r for r in filtered if (r.get("label") or "").strip() == lab]

        if not filtered:
            filtered = rows

        r = random.choice(filtered)

        # parse values
        pid_csv = int(float(r.get("plant_id", 0)))
        plant_str = csv_int_to_plant_str(pid_csv)

        temperature = safe_float(r.get("temperature"), 6.5)
        humidity = safe_float(r.get("humidity"), 91.0)

        img_name_csv = (r.get("image_name") or "").strip() or None
        chosen_img = self._pick_existing_image(img_name_csv)

        # return clean dict for API
        return {
            "plant_id": plant_str,
            "plant_id_csv": pid_csv,
            "temperature": temperature,
            "humidity": humidity,
            "label": (r.get("label") or "").strip(),
            "image_name": chosen_img,  # ✅ ensure exists in sim_images (or None)
            "remaining_days": safe_float(r.get("remaining_days"), 0.0),
        }

    # -------------------------
    # Replay logic (existing)
    # -------------------------
    def start(self, *, plant_id: str, interval_sec: int, loop: bool, reg):
        if self.state.running:
            return

        self.state.running = True
        self.state.plant_id = plant_id
        self.state.interval_sec = interval_sec
        self.state.loop = loop
        self.state.started_at = datetime.now(timezone.utc).isoformat()
        self._stop_event.clear()

        def runner():
            try:
                pid_csv = plant_str_to_csv_int(plant_id)
                rows = self._load_csv_once()

                plant_rows = [r for r in rows if int(float(r.get("plant_id", -999))) == pid_csv]
                if not plant_rows:
                    plant_rows = rows  # fallback

                i = 0
                while not self._stop_event.is_set():
                    if i >= len(plant_rows):
                        if loop:
                            i = 0
                        else:
                            break

                    r = plant_rows[i]
                    i += 1

                    temp = safe_float(r.get("temperature"), 6.5)
                    hum = safe_float(r.get("humidity"), 91.0)

                    pf = safe_float(r.get("p_fresh"), 0.0)
                    ps = safe_float(r.get("p_slightly_aged"), 0.0)
                    pn = safe_float(r.get("p_near_spoilage"), 0.0)
                    pp = safe_float(r.get("p_spoiled"), 0.0)

                    probs = {
                        "fresh": pf,
                        "slightly_aged": ps,
                        "near_spoilage": pn,
                        "spoiled": pp,
                    }

                    stage = stage_from_probs(pf, ps, pn, pp)
                    status = make_status(stage, probs)

                    # If you want regressor computed instead of CSV remaining_days:
                    # remaining = reg.predict(probs, temp, hum)
                    remaining = safe_float(r.get("remaining_days"), 0.0)

                    self.state.last_row = {
                        "plant_id": plant_id,
                        "temperature": temp,
                        "humidity": hum,
                        "stage": stage,
                        "remaining_days": remaining,
                        "probs": probs,
                    }

                    # insert into DB
                    try:
                        with Session(engine) as session:
                            session.add(
                                SpoilagePrediction(
                                    plant_id=plant_id,
                                    captured_at=datetime.now(timezone.utc),
                                    temperature=float(temp),
                                    humidity=float(hum),
                                    stage=stage,
                                    status=status,
                                    remaining_days=float(remaining),
                                    p_fresh=float(pf),
                                    p_slightly_aged=float(ps),
                                    p_near_spoilage=float(pn),
                                    p_spoiled=float(pp),
                                    image_url=None,
                                )
                            )
                            session.commit()
                    except Exception as e:
                        print("SIM insert failed:", e)

                    # sleep
                    for _ in range(max(1, interval_sec)):
                        if self._stop_event.is_set():
                            break
                        time.sleep(1)

            finally:
                self.state.running = False

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.state.running = False

    def status(self):
        return {
            "running": self.state.running,
            "plant_id": self.state.plant_id,
            "interval_sec": self.state.interval_sec,
            "loop": self.state.loop,
            "started_at": self.state.started_at,
            "last_row": self.state.last_row,
        }