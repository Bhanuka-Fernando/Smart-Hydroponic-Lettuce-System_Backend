import re

# ✅ Accept:
#   P-001
#   P-1
#   SIM-P-001
#   SIM-P-1
PLANT_ID_RE = re.compile(r"^(SIM-)?P-(\d{1,4})$", re.IGNORECASE)

def normalize_plant_id(plant_id: str) -> str:
    if plant_id is None:
        raise ValueError("plant_id is required")

    s = str(plant_id).strip().upper()
    m = PLANT_ID_RE.match(s)
    if not m:
        raise ValueError("plant_id must be like P-001 (or SIM-P-001 for simulation)")

    is_sim = m.group(1) is not None
    n = int(m.group(2))

    prefix = "SIM-" if is_sim else ""
    return f"{prefix}P-{n:03d}"

def plant_id_to_int(plant_id: str) -> int:
    if plant_id is None:
        raise ValueError("plant_id is required")

    s = str(plant_id).strip().upper()
    m = PLANT_ID_RE.match(s)
    if not m:
        raise ValueError("plant_id must be like P-001 (or SIM-P-001 for simulation)")

    return int(m.group(2))