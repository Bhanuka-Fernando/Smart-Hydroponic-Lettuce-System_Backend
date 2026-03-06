import csv
from pathlib import Path
from collections import defaultdict

CSV_PATH = Path("sim_data/remaining_days_inputs_with_probs.csv")
SIM_IMG_DIR = Path("sim_images")

def safe_int(x, default=-1):
    try:
        return int(float(x))
    except Exception:
        return default

def resolve_image(name: str | None):
    if not name:
        return None
    cleaned = str(name).strip().replace("\\", "/").split("/")[-1]
    p = SIM_IMG_DIR / cleaned
    return cleaned if p.exists() else None

def main():
    rows_by_plant = defaultdict(list)

    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = safe_int(r.get("plant_id"))
            day_id = safe_int(r.get("day_id"))
            label = (r.get("label") or "").strip().lower()
            image_name = resolve_image(r.get("image_name"))
            rows_by_plant[pid].append({
                "day_id": day_id,
                "label": label,
                "image_name": image_name,
            })

    for pid, rows in sorted(rows_by_plant.items()):
        unique_imgs = sorted({r["image_name"] for r in rows if r["image_name"]})
        if unique_imgs:
            print(f"\nPlant CSV ID {pid}  ->  P-{pid+1:03d}")
            print(f"Unique images: {len(unique_imgs)}")
            for img in unique_imgs:
                print(f"  - {img}")

            print("Rows:")
            for r in sorted(rows, key=lambda x: (x["day_id"], x["label"], x["image_name"] or "")):
                print(
                    f"  day={r['day_id']}, label={r['label']}, image={r['image_name']}"
                )

if __name__ == "__main__":
    main()