# Docker Guide

This project is now organized as five Dockerized FastAPI services plus one shared PostgreSQL container.

## Services

| Service | Host URL | Container port |
| --- | --- | --- |
| Authentication | `http://localhost:8000` | `8000` |
| Disease service | `http://localhost:8001` | `8000` |
| Spoilage ML service | `http://localhost:8002` | `8000` |
| Water quality ML service | `http://localhost:8003` | `8000` |
| Weight estimation and growth forecasting | `http://localhost:8004` | `8000` |
| Virtual device simulator | `http://localhost:8010` | `8010` |
| PostgreSQL | `localhost:5432` | `5432` |

PostgreSQL creates these databases on first startup:

- `hydroponic_auth`
- `hydroponic_iot`

## Run Everything

From the repository root:

```bash
docker compose up --build
```

The ML services download large packages such as TensorFlow and PyTorch. If your connection is slow or unstable, build one service at a time:

```bash
COMPOSE_PARALLEL_LIMIT=1 docker compose build
docker compose up
```

Run in the background:

```bash
docker compose up --build -d
```

View logs:

```bash
docker compose logs -f
```

Stop containers:

```bash
docker compose down
```

Stop containers and remove database/upload volumes:

```bash
docker compose down -v
```

## Health Checks

After startup, test:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8010/device/health
```

## Important Configuration

The default local Docker database credentials are:

```text
user: hydroponic_user
password: hydroponic2026
host from containers: postgres
host from your laptop: localhost
```

The services receive environment variables from `docker-compose.yml`. For production, replace these placeholder values:

- `SECRET_KEY`
- `JWT_SECRET`
- `GOOGLE_CLIENT_ID`
- database password

## Runtime Data

The Compose file uses Docker volumes for runtime files:

- auth avatars/static files
- disease uploads and plant counter
- spoilage uploads
- growth uploads
- Postgres data

This keeps generated files out of the image and preserves them when containers restart.

## Model Artifacts

Some model files are already present:

- `spoilage-ml-service/artifacts/spoilage_stage_classifier.keras`
- `spoilage-ml-service/artifacts/remaining_days_linear.joblib`
- `water-quality-ml-service/artifacts/water_status_model.joblib`
- `water-quality-ml-service/artifacts/algae_warning_model.joblib`
- `weight_estimation_growth_forecasting/artifacts/weight_mlp_bundle.pt`

The disease service currently expects these files, but they are not present in this repo:

- `disease-service/artifacts/tipburn_best.pt`
- `disease-service/artifacts/cls_effnetv2_b1_best.pt`

Add those files before expecting `disease-service` to start successfully.

## Rebuild One Service

```bash
docker compose build water-quality-ml-service
docker compose up water-quality-ml-service
```

Replace `water-quality-ml-service` with any service name from `docker-compose.yml`.

If a build fails with `Temporary failure in name resolution` or `ReadTimeoutError`, rerun the same build command. Docker will reuse completed layers, so it usually resumes from the failed service instead of starting from zero.

The PyTorch services pin `torch==2.2.2` and `torchvision==0.17.2` intentionally. Leaving these unpinned can make Linux ARM builds pull large CUDA/NVIDIA packages that are not useful for local CPU-only Docker runs on a Mac. The disease service uses a newer Ultralytics release because its YOLO model references newer modules such as `A2C2f`.
