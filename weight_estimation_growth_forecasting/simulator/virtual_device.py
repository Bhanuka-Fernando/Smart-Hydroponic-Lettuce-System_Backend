import os
import time
import random
import argparse
from datetime import datetime, timezone, timedelta

import requests


def lk_now_iso():
    lk = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(lk).isoformat()


def gen_sensors(mode: str):
    # Normal range (close to ideal)
    airT = 23.5 + random.uniform(-0.8, 0.8)
    RH = 70.0 + random.uniform(-3.0, 3.0)
    EC = 1.60 + random.uniform(-0.05, 0.05)
    pH = 6.00 + random.uniform(-0.07, 0.07)

    # Stress mode (intentionally off-optimal)
    if mode.upper() == "STRESS":
        airT = 28.0 + random.uniform(-1.0, 1.0)
        RH = 55.0 + random.uniform(-5.0, 5.0)
        EC = 2.20 + random.uniform(-0.10, 0.10)
        pH = 5.30 + random.uniform(-0.10, 0.10)

    return {"airT": airT, "RH": RH, "EC": EC, "pH": pH}


def list_rgb_files(rgb_dir: str):
    files = [f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    return files


def match_depth(rgb_name: str):
    # RGB_81.png -> Depth_81.png
    base = os.path.splitext(rgb_name)[0]
    if base.startswith("RGB_"):
        num = base.replace("RGB_", "")
        return f"Depth_{num}.png"
    # fallback: same base name
    return base + ".png"


def post_sensor(base_url: str, device_id: str, zone_id: str, mode: str):
    payload = {
        "device_id": device_id,
        "zone_id": zone_id,
        "ts": lk_now_iso(),
        **gen_sensors(mode),
    }
    r = requests.post(f"{base_url}/infer/iot/ingest", json=payload, timeout=15)
    print(f"[SENSOR] {r.status_code} {payload}")
    return r


def post_scan(base_url: str, device_id: str, plant_id: str, zone_id: str, rgb_path: str, depth_path: str):
    ts = lk_now_iso()

    data = {
        "device_id": device_id,
        "plant_id": plant_id,
        "zone_id": zone_id,
        "ts": ts,
    }

    files = {
        "rgb_image": (os.path.basename(rgb_path), open(rgb_path, "rb"), "image/png"),
        "depth_image": (os.path.basename(depth_path), open(depth_path, "rb"), "image/png"),
    }

    try:
        r = requests.post(f"{base_url}/infer/scans/ingest", data=data, files=files, timeout=120)
        print(f"[SCAN] {r.status_code} rgb={os.path.basename(rgb_path)} depth={os.path.basename(depth_path)}")
        if r.status_code != 200:
            print(r.text[:400])
        else:
            # print key outputs only (keep console clean)
            j = r.json()
            print(
                f"      A={j.get('A_proj_cm2'):.2f} cm2 | D={j.get('D_proj_cm'):.2f} cm | "
                f"W={j.get('weight_est_g'):.2f} g | A_next={j.get('A_next_cm2')} | D_next={j.get('D_next_cm')}"
            )
        return r
    finally:
        files["rgb_image"][1].close()
        files["depth_image"][1].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:8000")
    ap.add_argument("--device_id", default="DT-001")
    ap.add_argument("--zone_id", default="ZONE-A")
    ap.add_argument("--plant_id", default="PLANT-01")
    ap.add_argument("--rgb_dir", default="simulator/data/rgb")
    ap.add_argument("--depth_dir", default="simulator/data/depth")
    ap.add_argument("--mode", default="NORMAL", choices=["NORMAL", "STRESS"])
    ap.add_argument("--sensor_interval", type=int, default=30)   # seconds
    ap.add_argument("--scan_every", type=int, default=5)         # every N sensor loops
    args = ap.parse_args()

    rgb_files = list_rgb_files(args.rgb_dir)
    if not rgb_files:
        raise SystemExit(f"No RGB images found in {args.rgb_dir}")

    print("=== Virtual IoT Device Started ===")
    print("Base URL:", args.base_url)
    print("Mode:", args.mode)
    print("RGB count:", len(rgb_files))

    i = 0
    idx = 0

    while True:
        # 1) Send sensor reading
        post_sensor(args.base_url, args.device_id, args.zone_id, args.mode)

        # 2) Every N loops send scan
        if i % args.scan_every == 0:
            rgb_name = rgb_files[idx % len(rgb_files)]
            depth_name = match_depth(rgb_name)

            rgb_path = os.path.join(args.rgb_dir, rgb_name)
            depth_path = os.path.join(args.depth_dir, depth_name)

            if not os.path.exists(depth_path):
                print(f"[SCAN] Skipped (missing depth): {depth_name}")
            else:
                post_scan(args.base_url, args.device_id, args.plant_id, args.zone_id, rgb_path, depth_path)

            idx += 1

        i += 1
        time.sleep(args.sensor_interval)


if __name__ == "__main__":
    main()
