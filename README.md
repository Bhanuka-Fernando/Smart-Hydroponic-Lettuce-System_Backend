````md
# Smart Hydroponic Lettuce System — Backend (Microservices)

Backend repository for the SLIIT group research project implemented for **Asia Plantations & Export (Pvt) Ltd**.  
The backend is built using a **microservices architecture** where each major component runs as an independent FastAPI service, plus a shared **Auth service**.

---

## What This Backend Solves

This system supports end-to-end decision-making for hydroponic lettuce production and post-harvest handling:

**Water Monitoring → Plant Health → Growth/Yield → Post-harvest Shelf-life**

Outputs include real-time alerts, scores, forecasts, and recommendations to reduce waste and improve harvest/storage planning.

---

## Services Included (All Components)

### 1) Leaf Health Detection Service
- Disease + nutrient deficiency classification
- Tipburn detection + severity estimation
- Health Score (0–100)
- Status output: **OK / WATCH / ACT NOW**

### 2) Spoilage Prediction Service
- Spoilage stage classification:
  - `fresh`, `slightly_aged`, `near_spoilage`, `spoiled`
- Remaining days / shelf-life estimation
- Status output + near-spoilage alerts

### 3) Water Quality Detection Service
- Sensor-based analysis (pH, EC/TDS, temperature, etc.)
- Water status scoring + alerts
- Action recommendations (e.g., adjust nutrients / check aeration)

### 4) Growth Monitoring & Weight Estimation Service
- Growth monitoring (growth indicators / trends)
- Weight estimation (vision + sensor support depending on model)
- Harvest readiness indicators / forecasts

### 5) Auth Service (Shared)
- JWT-based login / token issuing
- Used to protect endpoints of all ML services (optional for demos)

---

## Architecture Overview

- **Microservices:** each component is a separate FastAPI app on its own port
- **Independent deployment:** each service can be updated without affecting others
- **Shared Auth:** one authentication service issues JWT tokens
- **Clear REST APIs:** services are consumed by the mobile app / demo UIs via HTTP

---

## Tech Stack

* **FastAPI** (REST APIs + Swagger docs)
* **Uvicorn** (ASGI server)
* **TensorFlow / Keras** (CNN models)
* **scikit-learn / joblib** (regression models and utilities)
* **PyJWT** (JWT authentication)
* **pydantic / pydantic-settings** (schema + environment configs)
* **python-dotenv** (load `.env`)

---

## Ports (Example)

Update based on your actual setup:

| Service                 | Default Port |
| ----------------------- | ------------ |
| Auth Service            | 8000         |
| Leaf Health ML Service  | 8001         |
| Spoilage ML Service     | 8002         |
| Water Quality Service   | 8005         |
| Growth & Weight Service | 8006         |

---

## Setup Requirements

### Prerequisites

* Python **3.11+** (3.12 is OK if dependencies are compatible)
* Git
* (Optional) Docker if containerizing

---

## Running a Service (Standard Steps)

> Do these steps separately inside each service folder.

### 1) Create virtual environment

**Windows (PowerShell):**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Set environment variables

Create a `.env` inside the service directory.

Example `.env` for Spoilage service:

```env
# Auth
AUTH_ENABLED=false
JWT_SECRET=your_secret_key
JWT_ALGORITHM=HS256

# Model artifacts
STAGE_MODEL_PATH=artifacts/spoilage_stage_classifier.keras
STAGE_META_PATH=artifacts/spoilage_stage_classifier_meta.json
REG_MODEL_PATH=artifacts/remaining_days_linear.joblib
REG_META_PATH=artifacts/remaining_days_linear_meta.json
```

> Keep the model files inside each service’s `artifacts/` folder.

### 4) Start server

```bash
python -m uvicorn app.main:app --reload --port 8002
```

### 5) Open Swagger (Docs)

```txt
http://127.0.0.1:8002/docs
```

---

## Running All Services (Local Development)

Open separate terminals and run each service:

### Auth Service

```bash
cd auth-service
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

### Leaf Health Service

```bash
cd leaf-health-ml-service
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8001
```

### Spoilage Service

```bash
cd spoilage-ml-service
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8002
```

### Water Quality Service

```bash
cd water-ml-service
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8005
```

### Growth & Weight Service

```bash
cd growth-weight-ml-service
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8006
```

---

## Authentication (JWT)

If `AUTH_ENABLED=true`, protected endpoints require this header:

```
Authorization: Bearer <access_token>
```

### Get token using Swagger (recommended)

1. Open Auth Swagger:

   * `http://127.0.0.1:8000/docs`
2. Find `POST /auth/login` (or your login route)
3. Click **Try it out**
4. Provide login JSON
5. Execute → copy `access_token`

### Get token using cURL (example)

Adjust path/body based on your auth API:

```bash
curl -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"test@gmail.com\",\"password\":\"12345678\"}"
```

Expected response includes:

```json
{ "access_token": "..." }
```

---

## API Examples (Service-Level)

### A) Spoilage Prediction Service

#### 1) Health check

`GET /health`

#### 2) Full prediction (Stage + Remaining Days + Status)

`POST /spoilage/predict` (multipart/form-data)

**Fields:**

* `image` (file) — required
* `temperature` (float) — required
* `humidity` (float) — required
* `plant_id` (string) — optional or required depending on your schema
* `captured_at` (ISO datetime string) — optional (auto-generated if empty)

#### 3) Stage-only prediction

`POST /spoilage/stage-only` (multipart/form-data)

#### 4) Remaining-days-only

`POST /spoilage/remaining-days-only` (application/json)

Example request:

```json
{
  "stage_probs": {
    "fresh": 0.01,
    "slightly_aged": 0.10,
    "near_spoilage": 0.85,
    "spoiled": 0.04
  },
  "temperature": 7,
  "humidity": 93
}
```

---

### B) Leaf Health Detection Service (Typical)

Endpoints vary by implementation, but usually include:

* `POST /leaf-health/predict`
  Input: image (+ optional sensor context)
  Output: disease/deficiency class, tipburn severity, health score, status.

---

### C) Water Quality Detection Service (Typical)

* `POST /water/analyze`
  Input: sensor readings (pH, EC/TDS, temp, etc.)
  Output: water status score + alerts + recommendations.

---

### D) Growth & Weight Service (Typical)

* `POST /growth/predict`
* `POST /weight/estimate`
  Input: vision measurements and/or sensors
  Output: growth indicators, estimated weight, readiness.

---

## Demo UI (Optional)

Some services include an HTML demo UI for quick panel demos (example: Spoilage).

### Start the Demo UI

```bash
cd spoilage-ml-service/demo-ui
python -m http.server 5173
```

Open:

```txt
http://127.0.0.1:5173/index.html
```

> If the demo UI is calling the backend from the browser, you may need CORS enabled.

---

## CORS Setup (If Browser UI Calls Backend)

Add this in your service `app/main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Common Errors & Fixes

### 1) 422 Unprocessable Entity

Cause: request body does not match endpoint schema.

Fix:

* If endpoint expects `multipart/form-data`, don’t send JSON.
* Ensure required fields are included:

  * `image`, `temperature`, `humidity`, etc.

### 2) 401 Unauthorized

Cause: endpoint is protected and token missing/invalid.

Fix:

* Login to Auth service → copy `access_token`
* Add header:

  * `Authorization: Bearer <token>`

### 3) CORS errors in browser

Cause: browser blocks cross-origin requests.

Fix:

* enable CORS middleware (see section above)

### 4) Model loading warnings

* TensorFlow CPU optimization logs are normal.
* If scikit-learn warns about version mismatch, install the same sklearn version used during training.

---

## Contribution Notes

* Each member maintains their service code and model artifacts.
* Shared conventions:

  * consistent request/response schemas (Pydantic)
  * `/health` endpoints
  * `/docs` for Swagger per service

---

## License

Academic / research use. Update as needed for your repo policy.

```
::contentReference[oaicite:0]{index=0}
```
