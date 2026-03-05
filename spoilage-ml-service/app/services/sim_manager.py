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
    s = str(plant_id).strip()
    if s.upper().startswith("P-"):
        n = int(s.split("-")[1])
        return max(0, n - 1)
    return int(float(s))


def csv_int_to_plant_str(n: int) -> str:
    return f"P-{n+1:03d}"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
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
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.state = SimState()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rows: list[dict[str, Any]] | None = None

        self._sim_images: list[str] | None = None
        self._sim_images_lower: dict[str, str] | None = None  # lower->real

    # -------------------------
    # CSV + Image loading
    # -------------------------
    def _load_csv_once(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows

        path = Path(self.csv_path)

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
            self._sim_images_lower = {}
            return self._sim_images

        imgs = []
        lower_map = {}
        for p in img_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.append(p.name)
                lower_map[p.name.lower()] = p.name

        self._sim_images = imgs
        self._sim_images_lower = lower_map
        return imgs

    def _resolve_existing_image(self, name: str | None) -> str | None:
        self._load_sim_images_once()
        if not self._sim_images_lower:
            return None
        if not name:
            return None
        return self._sim_images_lower.get(name.lower())

    # -------------------------
    # ✅ UI sampling (day-aware, plant-sticky, deterministic)
    # -------------------------
    def sample_row(
        self,
        plant_id: str | None = None,
        label: str | None = None,
        day_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Plant-sticky + day-clamped + deterministic.

        Goals:
        - If plant_id is provided: NEVER switch to another plant (no cross-plant fallbacks).
        - If day_id is provided but not available for that plant: clamp to nearest previous day,
          else min day available.
        - For the same (plant_id, effective_day_id, label) return a stable image (no random flipping).
        - Only return images that exist in sim_images (case-insensitive). If none exist -> image_name=None.
        """
        rows = self._load_csv_once()
        self._load_sim_images_once()

        want_pid = None
        if plant_id and str(plant_id).strip():
            try:
                want_pid = plant_str_to_csv_int(plant_id)
            except Exception:
                want_pid = None

        want_label = (str(label).strip() if label and str(label).strip() else None)
        want_day = (int(day_id) if day_id is not None else None)

        def pid_of(r: dict[str, Any]) -> int:
            return safe_int(r.get("plant_id", -999), -999)

        def day_of(r: dict[str, Any]) -> int:
            return safe_int(r.get("day_id", -999), -999)

        def label_of(r: dict[str, Any]) -> str:
            return (r.get("label") or "").strip()

        def resolved_img_of(r: dict[str, Any]) -> str | None:
            img = (r.get("image_name") or "").strip() or None
            return self._resolve_existing_image(img)

        # 1) Restrict to plant if given
        if want_pid is not None:
            plant_rows = [r for r in rows if pid_of(r) == want_pid]
            if not plant_rows:
                # Plant not present in CSV -> fall back to any row (but still validate image)
                r = random.choice(rows)
                chosen_img = resolved_img_of(r)
                pid_csv = pid_of(r) if pid_of(r) >= 0 else 0
                return {
                    "plant_id": csv_int_to_plant_str(pid_csv),
                    "plant_id_csv": pid_csv,
                    "temperature": safe_float(r.get("temperature"), 6.5),
                    "humidity": safe_float(r.get("humidity"), 91.0),
                    "label": label_of(r),
                    "day_id": day_of(r) if day_of(r) >= 0 else 0,
                    "capture_date": (r.get("capture_date") or "").strip() or None,
                    "image_name": chosen_img,
                    "remaining_days": safe_float(r.get("remaining_days"), 0.0),
                }
        else:
            plant_rows = rows

        # 2) Clamp day within this plant
        effective_day = want_day
        if effective_day is not None:
            available_days = sorted({day_of(r) for r in plant_rows if day_of(r) >= 0})
            if available_days:
                if effective_day not in available_days:
                    prev_days = [d for d in available_days if d <= effective_day]
                    effective_day = max(prev_days) if prev_days else min(available_days)
            else:
                effective_day = None

        # 3) Filter candidates
        candidates = plant_rows
        if effective_day is not None:
            candidates = [r for r in candidates if day_of(r) == effective_day]

        if want_label:
            exact = [r for r in candidates if label_of(r) == want_label]
            if exact:
                candidates = exact

        if not candidates:
            # If plant_id given -> fallback inside SAME PLANT only
            candidates = plant_rows if want_pid is not None else rows

        # 4) Prefer rows that have images
        with_img: list[tuple[dict[str, Any], str]] = []
        for r in candidates:
            resolved = resolved_img_of(r)
            if resolved:
                with_img.append((r, resolved))

        # 5) Deterministic pick (stable)
        def stable_index(key: str, n: int) -> int:
            acc = 0
            for ch in key:
                acc = (acc * 131 + ord(ch)) % 2_147_483_647
            return acc % n if n > 0 else 0

        key = f"{want_pid}-{effective_day}-{want_label or ''}"

        if with_img:
            with_img.sort(key=lambda t: (day_of(t[0]), label_of(t[0]), t[1]))
            idx = stable_index(key, len(with_img))
            r, chosen_img = with_img[idx]
        else:
            candidates.sort(
                key=lambda r: (pid_of(r), day_of(r), label_of(r), (r.get("image_name") or ""))
            )
            idx = stable_index(key, len(candidates))
            r = candidates[idx]
            chosen_img = None

        pid_csv = pid_of(r)
        if pid_csv < 0:
            pid_csv = 0
        plant_str = csv_int_to_plant_str(pid_csv)

        return {
            "plant_id": plant_str,
            "plant_id_csv": pid_csv,
            "temperature": safe_float(r.get("temperature"), 6.5),
            "humidity": safe_float(r.get("humidity"), 91.0),
            "label": label_of(r),
            "day_id": day_of(r) if day_of(r) >= 0 else (effective_day or 0),
            "capture_date": (r.get("capture_date") or "").strip() or None,
            "image_name": chosen_img,
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

                plant_rows = [r for r in rows if safe_int(r.get("plant_id", -999), -999) == pid_csv]
                if not plant_rows:
                    plant_rows = rows

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

                    probs = {"fresh": pf, "slightly_aged": ps, "near_spoilage": pn, "spoiled": pp}
                    stage = stage_from_probs(pf, ps, pn, pp)
                    status = make_status(stage, probs)

                    remaining = safe_float(r.get("remaining_days"), 0.0)

                    self.state.last_row = {
                        "plant_id": plant_id,
                        "temperature": temp,
                        "humidity": hum,
                        "stage": stage,
                        "remaining_days": remaining,
                        "probs": probs,
                    }

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