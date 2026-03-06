from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from app.db import engine
from app.models import SpoilagePrediction
from app.services.postprocess import make_status

STAGES = ["fresh", "slightly_aged", "near_spoilage", "spoiled"]
STAGE_RANK = {s: i for i, s in enumerate(STAGES)}


def plant_str_to_csv_int(plant_id: str) -> int:
    s = str(plant_id).strip().upper()
    if s.startswith("SIM-"):
        s = s[4:]
    if s.startswith("P-"):
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


def preferred_labels_for_day(day_id: int | None) -> list[str]:
    if day_id is None or day_id <= 0:
        return ["fresh", "slightly_aged"]
    if day_id == 1:
        return ["slightly_aged", "near_spoilage"]
    if day_id == 2:
        return ["near_spoilage", "spoiled"]
    return ["spoiled"]


def stable_index(key: str, n: int) -> int:
    acc = 0
    for ch in key:
        acc = (acc * 131 + ord(ch)) % 2_147_483_647
    return acc % n if n > 0 else 0


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

    Behavior:
    - accepts P-001 and SIM-P-001
    - progression is day-aware
    - future sim rescans prefer more advanced images/stages
    - avoids reusing the last sim image if another valid candidate exists
    - still deterministic for the same plant/day stream
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.state = SimState()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rows: list[dict[str, Any]] | None = None

        self._sim_images: list[str] | None = None
        self._sim_images_lower: dict[str, str] | None = None

        self._project_root = Path(__file__).resolve().parents[2]

    # -------------------------
    # CSV + Image loading
    # -------------------------
    def _load_csv_once(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows

        path = Path(self.csv_path)
        if not path.is_absolute():
            path = (self._project_root / path).resolve()

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

        img_dir = (self._project_root / "sim_images").resolve()

        if not img_dir.exists() or not img_dir.is_dir():
            self._sim_images = []
            self._sim_images_lower = {}
            return self._sim_images

        imgs: list[str] = []
        lower_map: dict[str, str] = {}
        for p in img_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.append(p.name)
                lower_map[p.name.lower()] = p.name

        self._sim_images = imgs
        self._sim_images_lower = lower_map
        return imgs

    def _resolve_existing_image(self, name: str | None) -> str | None:
        self._load_sim_images_once()
        if not self._sim_images_lower or not name:
            return None

        cleaned = str(name).strip().replace("\\", "/").split("/")[-1]
        return self._sim_images_lower.get(cleaned.lower())

    def _get_last_sim_source_image(self, plant_id: str | None) -> str | None:
        if not plant_id:
            return None

        try:
            with Session(engine) as session:
                stmt = (
                    select(SpoilagePrediction)
                    .where(SpoilagePrediction.plant_id == plant_id)
                    .order_by(SpoilagePrediction.captured_at.desc())
                    .limit(1)
                )
                latest = session.exec(stmt).first()
                if not latest:
                    return None

                last_name = getattr(latest, "sim_source_image", None)
                if not last_name:
                    return None

                return self._resolve_existing_image(last_name) or last_name
        except Exception as e:
            print("Could not read last sim_source_image:", e)
            return None

    # -------------------------
    # UI sampling
    # -------------------------
    def sample_row(
        self,
        plant_id: str | None = None,
        label: str | None = None,
        day_id: int | None = None,
    ) -> dict[str, Any]:
        rows = self._load_csv_once()
        self._load_sim_images_once()

        want_pid = None
        if plant_id and str(plant_id).strip():
            try:
                want_pid = plant_str_to_csv_int(plant_id)
            except Exception:
                want_pid = None

        want_label = (str(label).strip().lower() if label and str(label).strip() else None)
        want_day = int(day_id) if day_id is not None else None

        def pid_of(r: dict[str, Any]) -> int:
            return safe_int(r.get("plant_id", -999), -999)

        def day_of(r: dict[str, Any]) -> int:
            return safe_int(r.get("day_id", -999), -999)

        def label_of(r: dict[str, Any]) -> str:
            return (r.get("label") or "").strip().lower()

        def resolved_img_of(r: dict[str, Any]) -> str | None:
            img = (r.get("image_name") or "").strip() or None
            return self._resolve_existing_image(img)

        if want_pid is not None:
            plant_rows = [r for r in rows if pid_of(r) == want_pid]
            if not plant_rows:
                plant_rows = rows
        else:
            plant_rows = rows

        available_days = sorted({day_of(r) for r in plant_rows if day_of(r) >= 0})
        effective_day = want_day

        if effective_day is None:
            effective_day = available_days[0] if available_days else 0

        preferred = preferred_labels_for_day(effective_day)
        last_used_image = self._get_last_sim_source_image(plant_id)

        def score_row(r: dict[str, Any]) -> tuple:
            d = day_of(r)
            lbl = label_of(r)
            img = resolved_img_of(r)

            has_image_penalty = 0 if img else 1
            label_penalty = 0 if lbl in preferred else 1

            if effective_day is None:
                day_distance = 0
                future_bias = 0
            else:
                day_distance = abs(d - effective_day)
                future_bias = 0 if d >= effective_day else 1

            if want_label:
                explicit_label_penalty = 0 if lbl == want_label else 1
            else:
                explicit_label_penalty = 0

            stage_rank = STAGE_RANK.get(lbl, 999)

            return (
                has_image_penalty,
                explicit_label_penalty,
                label_penalty,
                day_distance,
                future_bias,
                stage_rank,
            )

        scored = [(score_row(r), r, resolved_img_of(r)) for r in plant_rows]
        scored.sort(key=lambda x: (x[0], x[2] or ""))

        if not scored:
            raise RuntimeError("No simulation rows available")

        best_score = scored[0][0]
        best_group = [(r, img) for s, r, img in scored if s == best_score]

        # avoid reusing last image if possible
        if last_used_image:
            filtered_group = [
                (r, img)
                for r, img in best_group
                if img and img.lower() != last_used_image.lower()
            ]
            if filtered_group:
                best_group = filtered_group

        # deterministic pick among equally-good remaining candidates
        best_group.sort(key=lambda t: t[1] or "")
        key = f"{plant_id or ''}-{effective_day}-{want_label or ''}"
        idx = stable_index(key, len(best_group))
        r, chosen_img = best_group[idx]

        pid_csv = pid_of(r)
        if pid_csv < 0:
            pid_csv = 0

        return {
            "plant_id": csv_int_to_plant_str(pid_csv),
            "plant_id_csv": pid_csv,
            "temperature": safe_float(r.get("temperature"), 6.5),
            "humidity": safe_float(r.get("humidity"), 91.0),
            "label": label_of(r),
            "day_id": day_of(r) if day_of(r) >= 0 else effective_day,
            "capture_date": (r.get("capture_date") or "").strip() or None,
            "image_name": chosen_img,
            "remaining_days": safe_float(r.get("remaining_days"), 0.0),
        }

    # -------------------------
    # Replay logic
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
                                    sim_source_image=None,
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