import re

PLANT_ID_RE = re.compile(r"^P-(\d{3})$")

def normalize_plant_id(plant_id: str) -> str:
    plant_id = plant_id.strip().upper()
    m = PLANT_ID_RE.match(plant_id)
    if not m:
        raise ValueError("plant_id must be like P-001")
    return plant_id

def plant_id_to_int(plant_id: str) -> int:
    m = PLANT_ID_RE.match(plant_id.strip().upper())
    if not m:
        raise ValueError("plant_id must be like P-001")
    return int(m.group(1))
