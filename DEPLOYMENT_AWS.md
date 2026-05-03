# AWS Backend Deployment Notes

This document records the working AWS deployment flow for the backend in `us-east-1`.

## Current deployed services

- Authentication
  - Health: `http://hydroponic-backend-alb-1973402541.us-east-1.elb.amazonaws.com/health`
- Disease
  - Health: `http://hydroponic-backend-alb-1973402541.us-east-1.elb.amazonaws.com:8001/health`
- Spoilage
  - Health: `http://hydroponic-backend-alb-1973402541.us-east-1.elb.amazonaws.com:8002/health`
- Water quality
  - Health: `http://hydroponic-backend-alb-1973402541.us-east-1.elb.amazonaws.com:8003/health`
- Weight growth
  - Health: `http://hydroponic-backend-alb-1973402541.us-east-1.elb.amazonaws.com:8004/health`

## AWS resources

- Region: `us-east-1`
- ECS cluster: `hydroponic-backend-cluster`
- RDS PostgreSQL instance: `hydroponic-postgres`
- RDS endpoint: `hydroponic-postgres.c6rgogaisvih.us-east-1.rds.amazonaws.com`
- VPC: `vpc-0cd44110d525ed9ee`
- ECS tasks SG: `sg-0b4acdd50c8de4858`
- ALB SG: `sg-00bb591511fe0d7fd`
- ALB: `hydroponic-backend-alb`
- ALB DNS: `hydroponic-backend-alb-1973402541.us-east-1.elb.amazonaws.com`

## Databases created in RDS

- `hydroponic_auth`
- `hydroponic_iot`

## Important build rule

Because deployment targets ECS/Fargate on `x86_64`, images must be built for `linux/amd64`.

Use:

```bash
docker buildx build --platform linux/amd64 -t <ecr-image-uri>:latest <service-dir> --push
```

If you build from Apple Silicon without `--platform linux/amd64`, ECS may fail with:

```text
exec /usr/local/bin/uvicorn: exec format error
```

## Deployment pattern used

For each service:

1. Build and push the image to ECR with `linux/amd64`.
2. Create a CloudWatch log group.
3. Register an ECS task definition.
4. Create an ALB target group.
5. Create an ALB listener on a public port.
6. Create an ECS Fargate service attached to the target group.
7. Open the corresponding inbound ALB security group port.
8. Verify ECS service state and health endpoint.

## ECR repositories

- `hydroponic-authentication`
- `hydroponic-disease-service`
- `hydroponic-spoilage-ml-service`
- `hydroponic-water-quality-ml-service`
- `hydroponic-weight-growth-service`

## ALB listener mapping

- Port `80` -> `hydroponic-authentication-tg`
- Port `8001` -> `hydroponic-disease-tg`
- Port `8002` -> `hydroponic-spoilage-tg`
- Port `8003` -> `hydroponic-water-tg`
- Port `8004` -> `hydroponic-weight-tg`

## ECS service names

- `hydroponic-authentication-service`
- `hydroponic-disease-service`
- `hydroponic-spoilage-ml-service`
- `hydroponic-water-quality-ml-service`
- `hydroponic-weight-growth-service`

## Task definition families

- `hydroponic-authentication`
- `hydroponic-disease-service`
- `hydroponic-spoilage-ml-service`
- `hydroponic-water-quality-ml-service`
- `hydroponic-weight-growth-service`

## Working task definition files in repo root

- [auth-taskdef.json](/Users/dulanimalka/Desktop/Smart-Hydroponic-Lettuce-System_Backend/auth-taskdef.json:1)
- [disease-taskdef.json](/Users/dulanimalka/Desktop/Smart-Hydroponic-Lettuce-System_Backend/disease-taskdef.json:1)
- [spoilage-taskdef.json](/Users/dulanimalka/Desktop/Smart-Hydroponic-Lettuce-System_Backend/spoilage-taskdef.json:1)
- [water-taskdef.json](/Users/dulanimalka/Desktop/Smart-Hydroponic-Lettuce-System_Backend/water-taskdef.json:1)
- [weight-taskdef.json](/Users/dulanimalka/Desktop/Smart-Hydroponic-Lettuce-System_Backend/weight-taskdef.json:1)

## Subnets used for ECS services

- `subnet-025723a39b639941c` (`us-east-1a`)
- `subnet-0503fb64c1b31abed` (`us-east-1b`)

## Security group rules that mattered

### RDS SG

Allowed PostgreSQL `5432` from:

- ECS tasks SG `sg-0b4acdd50c8de4858`

Temporary direct testing from CloudShell/local also used a public `5432` rule during setup. Tighten this later if not needed.

### ECS tasks SG

Allowed inbound app traffic from the ALB SG to container port `8000`.

### ALB SG

Allowed inbound public traffic on:

- `80`
- `8001`
- `8002`
- `8003`
- `8004`

## Useful verification commands

Check ECS service state:

```bash
aws ecs describe-services --cluster hydroponic-backend-cluster --services <service-name>
```

List tasks:

```bash
aws ecs list-tasks --cluster hydroponic-backend-cluster --service-name <service-name>
```

Check CloudWatch log streams:

```bash
aws logs describe-log-streams \
  --log-group-name <log-group> \
  --order-by LastEventTime \
  --descending \
  --max-items 5
```

Read logs:

```bash
aws logs get-log-events \
  --log-group-name <log-group> \
  --log-stream-name "<stream-name>"
```

## Health check responses seen

- Auth: `{"status":"ok"}`
- Disease: `{"ok":true}`
- Spoilage: `{"status":"ok"}`
- Water: `{"status":"ok","service":"water-quality-ml-service"}`
- Weight growth: `{"ok":true}`

## Next recommended steps

1. Keep this deployment state as the baseline.
2. Pull the latest branch only after noting which parts of this deployment worked.
3. Re-test locally after syncing the branch.
4. Rebuild changed services with `linux/amd64`.
5. Move large or missing model artifacts to S3 instead of relying on laptop-only files.
6. Add CI/CD after the manual deployment flow is fully understood.
