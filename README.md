<div align="center">

# 🖥️ Universal Model Server

**Production-grade · Framework-agnostic · ML Inference Platform**
Open-source alternative to NVIDIA Triton Inference Server

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-32%20passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](docker/)
[![gRPC](https://img.shields.io/badge/gRPC-supported-blueviolet?style=for-the-badge)](grpc/)

<br/>

> *"Serve any ML model, from any framework, with SLA guarantees — without vendor lock-in."*

</div>

---

## 📋 Table of Contents

- [What is UMS?](#-what-is-universal-model-server)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Register Your Own Model](#-registering-your-own-model)
- [Build Custom Pipelines](#-building-custom-pipelines)
- [Multi-Tenant Usage](#-multi-tenant-usage)
- [Shadow Deployment](#-shadow-deployment)
- [Model Optimization](#-model-optimization)
- [Drift Detection](#-drift-detection)
- [Benchmarking](#-benchmarking)
- [Security](#-security)
- [Docker Deployment](#-docker-deployment)
- [Running Tests](#-running-tests)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)

---

## 🌍 What is Universal Model Server?

Universal Model Server (UMS) is a **production-ready ML inference platform** built in Python with FastAPI. It solves the core problems that existing tools like NVIDIA Triton and TorchServe leave unsolved:

| Tool | Problem |
|------|---------|
| **NVIDIA Triton** | NVIDIA-only, vendor-locked |
| **TorchServe** | Only handles PyTorch models |
| **Both** | No SLA guarantees, no multi-tenant isolation, no drift detection |

UMS handles all of this — and adds INT8 quantization, statistical benchmarking, shadow deployment, and a KS-test drift detector on top.

---

## ✨ Features

| Feature | What it does |
|---------|-------------|
| **Framework-agnostic** | Serve PyTorch, ONNX, JAX models through one unified API |
| **SLA Scheduler** | Priority queue guarantees urgent=100ms, normal=500ms, batch=5s |
| **Multi-tenant** | Team A and Team B get isolated queues and rate limits |
| **Model Optimizer** | Auto INT8 quantization — 2-4x speedup, zero user code change |
| **Pipeline Engine** | Chain models: Preprocess → Model A → Model B → Postprocess |
| **Shadow Deployment** | Silently test new models against production traffic |
| **Drift Detector** | KS statistical test — alerts when model outputs shift |
| **Benchmark Suite** | P50/P95/P99 latency, throughput RPS, SLA breach rate |
| **Full Observability** | OpenTelemetry traces + Prometheus metrics + Grafana |
| **REST + gRPC** | Industry-standard protocols, both supported simultaneously |
| **API Key Auth** | Secure endpoints with API key verification |
| **CORS Protection** | Configurable origin whitelist |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Client Layer                        │
│              REST  ·  gRPC  ·  GraphQL                  │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│               API Key Authentication                    │
│          X-API-Key header verification                  │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   SLA Classifier                        │
│         urgent (100ms) · normal (500ms) · batch (5s)   │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              SLA-Aware Priority Queue                   │
│      deadline-sorted · tenant-isolated · rate-limited   │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   Router Engine                         │
│       GPU affinity · load balance · shadow split        │
└────────┬───────────────┬───────────────┬────────────────┘
         ↓               ↓               ↓
  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
  │   PyTorch   │ │    ONNX     │ │     JAX     │
  │   Runtime   │ │   Runtime   │ │   Runtime   │
  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
         └───────────────┼───────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  Pipeline DAG Engine                    │
│    Preprocess → Model A → Model B → Postprocess         │
│         parallel stages via asyncio.gather              │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│         Optimizer · Drift Monitor · Benchmark           │
│    INT8 quant  ·  KS test  ·  P50/P95/P99 tracking     │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Observability Stack                    │
│       OpenTelemetry · Prometheus · Grafana · Jaeger     │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- Git

### Step 1 — Clone & Install

```bash
git clone https://github.com/your-username/universal-model-server.git
cd universal-model-server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Setup Environment

```bash
# Copy example env file
cp .env.example .env
```

Open `.env` and fill in your values (defaults work for local development).

### Step 3 — Create Test Models

```bash
pip install scikit-learn skl2onnx
python scripts/create_dummy_models.py
```

Output:
```
Creating dummy ONNX model...
✅ models/echo.onnx created!
Creating dummy PyTorch model...
✅ models/classifier.pt created!
```

### Step 4 — Start the Server

```bash
python main.py
```

Output:
```
🚀 Universal Model Server v0.1.0 starting...
🔭 Tracing initialized
📦 Registered model: 'echo_model' [onnx on cpu]
📦 Registered model: 'classifier' [pytorch on cpu]
🔄 Loading registered models...
⚡ fresh | standard | 0.001MB → 0.001MB | 0.45s
✅ Loaded: echo_model
👁️  Drift monitoring enabled: 'echo_model'
✅ 2 model(s) ready.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 5 — Open Swagger UI

```
http://localhost:8000/docs
```

All endpoints are interactive here — no curl needed!

### Step 6 — Send Your First Request

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "echo_model",
    "inputs": {"float_input": [[1.0, 2.0]]},
    "priority": "urgent",
    "tenant_id": "team_a"
  }'
```

Response:
```json
{
  "request_id": "a3f9c1d2-...",
  "model_name": "echo_model",
  "outputs": {
    "output_label": [0],
    "output_probability": [{"0": 0.97, "1": 0.02}]
  },
  "status": "done",
  "latency_ms": 1.37,
  "tenant_id": "team_a",
  "served_by": "echo_model"
}
```

---

## 🔧 Configuration

All configuration is done via `.env` file. Copy `.env.example` and edit:

```env
# ── Server ────────────────────────────────────────────
APP_NAME=Universal Model Server
APP_VERSION=0.1.0
HOST=0.0.0.0
PORT=8000
DEBUG=false
WORKERS=1

# ── SLA Deadlines (milliseconds) ──────────────────────
SLA_URGENT_MS=100
SLA_NORMAL_MS=500
SLA_BATCH_MS=5000

# ── Queue ─────────────────────────────────────────────
MAX_QUEUE_SIZE=1000

# ── Device ────────────────────────────────────────────
# Options: cpu | cuda:0 | cuda:1
DEFAULT_DEVICE=cpu

# ── Observability ─────────────────────────────────────
ENABLE_TRACING=true
PROMETHEUS_PORT=9090

# ── CORS ──────────────────────────────────────────────
# Add your frontend domain here
ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:8080"]

# ── Authentication ────────────────────────────────────
# Set to true in production
API_KEY_ENABLED=false
# Comma-separated list of valid API keys
API_KEYS=your-secret-key-here

# ── Grafana ───────────────────────────────────────────
GRAFANA_ADMIN_PASSWORD=change-me-in-production
```

> ⚠️ **Never commit `.env` to Git.** It is already in `.gitignore`.

---

## 📡 API Reference

### Inference

#### `POST /infer` — Run inference on a model

**Request:**
```json
{
  "model_name": "echo_model",
  "inputs": {"float_input": [[1.0, 2.0]]},
  "priority": "urgent",
  "tenant_id": "team_a",
  "deadline_ms": 100
}
```

**Priority levels:**

| Value | SLA Deadline | Use Case |
|-------|-------------|----------|
| `urgent` | 100ms | Real-time user-facing requests |
| `normal` | 500ms | Standard API calls |
| `batch` | 5000ms | Background processing |

You can also override priority via HTTP header:
```bash
curl -H "X-Priority: urgent" -X POST http://localhost:8000/infer ...
```

**Response:**
```json
{
  "request_id": "a3f9c1d2-...",
  "model_name": "echo_model",
  "outputs": {"output_label": [0]},
  "status": "done",
  "latency_ms": 1.37,
  "tenant_id": "team_a",
  "served_by": "echo_model"
}
```

---

#### `POST /pipeline/{name}` — Run a multi-stage pipeline

```bash
curl -X POST http://localhost:8000/pipeline/preprocess_and_infer \
  -H "Content-Type: application/json" \
  -d '{"float_input": [[5.0, 10.0]]}'
```

**Response:**
```json
{
  "pipeline": "preprocess_and_infer",
  "results": {
    "normalize": {"float_input": [[0.33, 0.66]]},
    "format": {"formatted": true, "predictions": {}},
    "__pipeline_ms__": 3.21
  }
}
```

---

#### `POST /optimize/{model_name}` — Optimize a model

```bash
curl -X POST "http://localhost:8000/optimize/echo_model?level=aggressive"
```

**Optimization levels:**

| Level | Technique | Expected Speedup |
|-------|-----------|-----------------|
| `basic` | Graph cleanup only | 10–20% |
| `standard` | Full graph optimization | 20–40% |
| `aggressive` | INT8 quantization | 2–4x |

**Response:**
```json
{
  "model_name": "echo_model",
  "optimization_level": "aggressive",
  "original_size_mb": 0.45,
  "optimized_size_mb": 0.12,
  "size_reduction_pct": 73.3,
  "optimization_time_s": 1.24,
  "from_cache": false
}
```

> ✅ Optimized models are cached by file hash — running again returns instantly from cache.

---

### Monitoring & Observability

#### `GET /health` — Liveness probe
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2024-01-15T10:00:00Z"
}
```

#### `GET /ready` — Readiness probe (Kubernetes-compatible)
```bash
curl http://localhost:8000/ready
```
```json
{
  "ready": true,
  "checks": {"server": true, "model_registry": true}
}
```

#### `GET /metrics` — Prometheus metrics scrape endpoint
```bash
curl http://localhost:8000/metrics
```

#### `GET /benchmark/stats` — Live P50/P95/P99 latency
```bash
curl http://localhost:8000/benchmark/stats
```
```json
{
  "total_requests": 1500,
  "success_rate_pct": 99.8,
  "p50_ms": 1.2,
  "p95_ms": 4.8,
  "p99_ms": 12.3,
  "throughput_rps": 342.5,
  "sla_breach_100ms": 0.2,
  "sla_breach_500ms": 0.0
}
```

#### `GET /queue/stats` — Current queue depth
```bash
curl http://localhost:8000/queue/stats
```
```json
{
  "queue_size": 3,
  "tenant_counts": {"team_a": 150, "team_b": 89}
}
```

#### `GET /drift/summary` — Drift status for all models
```bash
curl http://localhost:8000/drift/summary
```
```json
{
  "echo_model": {
    "is_ready": true,
    "reference_samples": 200,
    "current_samples": 87,
    "total_alerts": 0,
    "latest_alert": null
  }
}
```

#### `GET /drift/alerts` — All drift alerts, newest first
```bash
curl http://localhost:8000/drift/alerts
```
```json
{
  "alerts": [{
    "model_name": "echo_model",
    "severity": "critical",
    "p_value": 0.0001,
    "ks_statistic": 0.82,
    "recommendation": "🚨 Immediate action required! Consider rolling back.",
    "timestamp": 1705312800.0
  }]
}
```

#### `GET /shadow/summary` — Shadow deployment comparison
```bash
curl http://localhost:8000/shadow/summary
```
```json
{
  "total_shadow_requests": 45,
  "avg_latency_diff_ms": -2.3,
  "shadow_ratio": 0.1,
  "verdict": "✅ Shadow faster"
}
```

#### Other endpoints
```bash
GET /models           # List all registered models
GET /pipelines        # List all registered pipelines
```

---

## 📦 Registering Your Own Model

### Step 1 — Export your model

**PyTorch → TorchScript:**
```python
import torch

class MyModel(torch.nn.Module):
    def forward(self, x):
        return x * 2.0

model = MyModel()
scripted = torch.jit.script(model)
scripted.save("models/my_model.pt")
```

**Scikit-learn → ONNX:**
```python
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

clf = RandomForestClassifier().fit(X_train, y_train)
initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

with open("models/my_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### Step 2 — Register in `configs/models.yaml`

```yaml
models:
  - name: "my_model"
    runtime: "onnx"           # pytorch | onnx | jax
    model_path: "models/my_model.onnx"
    device: "cpu"             # cpu | cuda:0 | cuda:1
    max_batch_size: 32
    shadow_model: null        # set to another model name for shadow deployment
```

### Step 3 — Restart & call

```bash
python main.py

curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my_model",
    "inputs": {"float_input": [[1.0, 2.0, 3.0]]}
  }'
```

---

## 🔗 Building Custom Pipelines

Add your pipeline inside `_setup_pipelines()` in `core/server.py`:

```python
from pipeline.dag import PipelineDAG

async def my_preprocess(inputs: dict) -> dict:
    # normalize, resize, tokenize — whatever you need
    return {"processed": inputs}

async def my_postprocess(inputs: dict) -> dict:
    return {"final": inputs}

my_pipeline = (
    PipelineDAG("my_pipeline")
    .add_stage("preprocess", my_preprocess)
    .add_stage("postprocess", my_postprocess, depends_on=["preprocess"])
)
register_pipeline(my_pipeline)
```

**Parallel stages** — stages with no shared dependency run simultaneously:

```python
parallel_pipeline = (
    PipelineDAG("ensemble")
    .add_stage("model_a", run_model_a)          # ─┐ run together
    .add_stage("model_b", run_model_b)          # ─┘ (asyncio.gather)
    .add_stage("merge", merge_results,
               depends_on=["model_a", "model_b"])  # runs after both
)
```

Call it:
```bash
curl -X POST http://localhost:8000/pipeline/my_pipeline \
  -H "Content-Type: application/json" \
  -d '{"float_input": [[1.0, 2.0]]}'
```

---

## 👥 Multi-Tenant Usage

Each team gets an **isolated queue and rate limit** — one team's traffic never slows down another.

```bash
# Team A — urgent, 200 RPS limit
curl -X POST http://localhost:8000/infer \
  -d '{"model_name": "echo_model", "inputs": {...},
       "priority": "urgent", "tenant_id": "team_a"}'

# Team B — normal, 50 RPS limit, completely independent
curl -X POST http://localhost:8000/infer \
  -d '{"model_name": "echo_model", "inputs": {...},
       "priority": "normal", "tenant_id": "team_b"}'
```

Configure per-tenant limits in `configs/tenants.yaml`:

```yaml
tenants:
  team_a:
    max_requests_per_second: 200
  team_b:
    max_requests_per_second: 50
  team_c:
    max_requests_per_second: 500
```

Check live usage:
```bash
curl http://localhost:8000/queue/stats
```

---

## 🔮 Shadow Deployment

Test a new model version **silently against real traffic** — users always get the champion's response, the challenger runs in the background.

### Step 1 — Register both models in `configs/models.yaml`

```yaml
models:
  - name: "my_model_v1"          # champion
    runtime: "onnx"
    model_path: "models/my_model_v1.onnx"
    device: "cpu"
    shadow_model: "my_model_v2"  # ← challenger

  - name: "my_model_v2"          # challenger (shadow)
    runtime: "onnx"
    model_path: "models/my_model_v2.onnx"
    device: "cpu"
```

### Step 2 — Send traffic normally

```bash
curl -X POST http://localhost:8000/infer \
  -d '{"model_name": "my_model_v1", "inputs": {...}}'
```

10% of requests automatically hit the shadow model. Users get `v1` responses.

### Step 3 — Check comparison

```bash
curl http://localhost:8000/shadow/summary
# {
#   "total_shadow_requests": 120,
#   "avg_latency_diff_ms": -5.2,
#   "verdict": "✅ Shadow faster"
# }
```

### Step 4 — Promote challenger

When confident, update `configs/models.yaml` — change `v1` path to `v2`. Restart server.

---

## ⚡ Model Optimization

The optimizer automatically runs on ONNX models at startup (standard level). You can manually trigger any level:

```bash
# Basic — safe, always recommended
curl -X POST "http://localhost:8000/optimize/echo_model?level=basic"

# Standard — recommended for production
curl -X POST "http://localhost:8000/optimize/echo_model?level=standard"

# Aggressive — INT8 quantization, maximum speedup
curl -X POST "http://localhost:8000/optimize/echo_model?level=aggressive"
```

Optimized models are **cached by file hash** — running optimization twice returns instantly from cache the second time.

---

## 📊 Drift Detection

UMS uses the **Kolmogorov-Smirnov (KS) statistical test** to monitor model output distributions in real time.

**How it works:**
1. First 200 requests → builds reference distribution
2. Every 50 subsequent requests → KS test against reference
3. p-value < 0.05 → `warning` alert
4. p-value < 0.01 → `critical` alert + rollback recommendation

```bash
# Check drift status
curl http://localhost:8000/drift/summary

# Check all alerts
curl http://localhost:8000/drift/alerts

# Check specific model
curl http://localhost:8000/drift/summary/echo_model
```

When a critical alert fires, you'll see in terminal:
```
🚨 DRIFT DETECTED [echo_model] severity=critical p=0.0001 ks=0.82
   → 🚨 Immediate action required! Consider rolling back to previous model version.
```

---

## 🔒 Security

### API Key Authentication

Enable in `.env`:
```env
API_KEY_ENABLED=true
API_KEYS=key-team-a-abc123,key-team-b-xyz789
```

Send with every request:
```bash
curl -X POST http://localhost:8000/infer \
  -H "X-API-Key: key-team-a-abc123" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "echo_model", "inputs": {...}}'
```

Without a valid key, the server returns:
```json
{"detail": "Invalid API key"}
```

### CORS

Only allow specific origins in `.env`:
```env
# Development
ALLOWED_ORIGINS=["http://localhost:3000"]

# Production
ALLOWED_ORIGINS=["https://your-app.com"]
```

### Docker Secrets

Grafana password and API keys are read from `.env` — never hardcoded. Make sure `.env` is in `.gitignore` (it already is).

---

## 📈 Benchmarking

### Live Stats (built-in)

After sending some requests:
```bash
curl http://localhost:8000/benchmark/stats
```

### Locust Load Test

```bash
# Terminal 1 — start server
python main.py

# Terminal 2 — run load test
locust -f benchmark/locustfile.py --host=http://localhost:8000

# Open Locust UI
# http://localhost:8089
# Set: Users=100, Spawn rate=10, Duration=60s
```

### CLI Benchmark Runner

```bash
python scripts/benchmark.py \
  --model echo_model \
  --concurrency 20 \
  --duration 30 \
  --priority urgent
```

Output:
```
╔══════════════════════════════════════════════╗
║         BENCHMARK REPORT                     ║
╠══════════════════════════════════════════════╣
║  Model       : echo_model                   ║
║  Concurrency : 20                            ║
║  Duration    : 30.0                          ║
╠══════════════════════════════════════════════╣
║  Total Req   : 4821                          ║
║  Success     : 99.8 %                        ║
║  Throughput  : 160.70 rps                    ║
╠══════════════════════════════════════════════╣
║  P50 Latency : 1.20 ms                       ║
║  P95 Latency : 4.80 ms                       ║
║  P99 Latency : 12.30 ms                      ║
╚══════════════════════════════════════════════╝
```

---

## 🐳 Docker Deployment

### Server Only

```bash
# Build image
docker build -t universal-model-server -f docker/Dockerfile .

# Run
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/configs:/app/configs \
  universal-model-server
```

### Full Stack (Server + Prometheus + Grafana)

```bash
docker-compose -f docker/docker-compose.yml up
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Model Server | http://localhost:8000 | — |
| Swagger UI | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / from `.env` |

> ⚠️ Make sure your `.env` has `GRAFANA_ADMIN_PASSWORD` set before running compose.

---

## 🧪 Running Tests

```bash
# All 32 tests
pytest tests/ -v

# By module
pytest tests/test_runtime.py -v        # 3 tests  — schemas + runtime factory
pytest tests/test_scheduler.py -v      # 5 tests  — queue + rate limiter
pytest tests/test_pipeline.py -v       # 3 tests  — DAG + shadow
pytest tests/test_optimizer.py -v      # 4 tests  — quantization + cache
pytest tests/test_benchmark.py -v      # 5 tests  — P50/P95/P99
pytest tests/test_drift.py -v          # 7 tests  — KS test + monitor
pytest tests/test_integration.py -v    # 5 tests  — end-to-end API
```

Expected:
```
32 passed, 0 errors ✅
```

---

## 📁 Project Structure

```
universal-model-server/
│
├── core/
│   ├── config.py              # Pydantic settings, .env support
│   ├── schemas.py             # InferenceRequest/Response models
│   ├── registry.py            # Model registry (YAML-based)
│   ├── auth.py                # API key authentication
│   └── server.py              # FastAPI app + all endpoints
│
├── scheduler/
│   ├── priority_queue.py      # Deadline-sorted asyncio.PriorityQueue
│   ├── tenant_limiter.py      # Sliding window rate limiter per team
│   └── sla_classifier.py     # Priority tagging from headers/body
│
├── router/
│   └── engine.py              # Main orchestrator — routing + metrics + drift
│
├── runtimes/
│   ├── base.py                # Abstract BaseRuntime interface
│   ├── pytorch_runtime.py     # TorchScript inference
│   ├── onnx_runtime.py        # ONNX Runtime inference
│   └── jax_runtime.py         # JAX/XLA inference
│
├── pipeline/
│   ├── dag.py                 # DAG executor with parallel stage support
│   ├── stages.py              # Built-in stages (normalize, format)
│   ├── batcher.py             # Dynamic batching engine
│   └── registry.py            # Pipeline registry
│
├── shadow/
│   └── manager.py             # Champion vs challenger traffic split
│
├── optimizer/
│   ├── engine.py              # INT8 quantization + graph optimization
│   └── cache.py               # File-hash based optimization cache
│
├── benchmark/
│   ├── stats.py               # LatencyTracker — P50/P95/P99, SLA breach
│   ├── runner.py              # Async concurrent benchmark runner
│   └── locustfile.py          # Locust load test definition
│
├── drift/
│   ├── detector.py            # KS test drift detector per model
│   └── monitor.py             # Multi-model drift monitor + alerts
│
├── observability/
│   ├── health.py              # /health + /ready endpoints
│   ├── metrics.py             # Prometheus counters/histograms/gauges
│   └── tracing.py             # OpenTelemetry tracer setup
│
├── grpc/
│   ├── server.py              # gRPC async server
│   └── serving.proto          # Protobuf definitions
│
├── tests/                     # 32 tests, all passing
├── docker/                    # Dockerfile + docker-compose + prometheus.yml
├── configs/
│   ├── models.yaml            # Model registry config
│   └── tenants.yaml           # Per-tenant SLA + quota config
├── scripts/
│   ├── create_dummy_models.py # Generate test models
│   └── benchmark.py           # CLI benchmark runner
├── models/                    # Model files (gitignored)
│
├── main.py                    # Server entrypoint
├── requirements.txt           # All dependencies
├── .env.example               # Environment template (copy to .env)
├── conftest.py                # pytest path configuration
├── pytest.ini                 # pytest settings
└── .gitignore                 # Ignores models/, venv/, .env, cache/
```

---

## 🗺️ Roadmap

- [ ] JAX runtime — full implementation
- [ ] gRPC — full implementation
- [ ] Kubernetes Helm chart
- [ ] React monitoring dashboard
- [ ] Auto-retraining trigger on critical drift
- [ ] WebSocket streaming for long inference
- [ ] Plugin system for custom runtimes
- [ ] Model versioning + rollback API
- [ ] Per-tenant Grafana dashboards

---

## 🤝 Built With

| Technology | Role |
|-----------|------|
| [FastAPI](https://fastapi.tiangolo.com) | Async HTTP server |
| [Pydantic v2](https://docs.pydantic.dev) | Schema validation |
| [ONNX Runtime](https://onnxruntime.ai) | Cross-framework inference |
| [PyTorch](https://pytorch.org) | TorchScript inference |
| [SciPy](https://scipy.org) | KS test for drift detection |
| [OpenTelemetry](https://opentelemetry.io) | Distributed tracing |
| [Prometheus](https://prometheus.io) | Metrics collection |
| [Grafana](https://grafana.com) | Metrics visualization |
| [Locust](https://locust.io) | Load testing |
| [Docker](https://docker.com) | Containerization |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built from scratch — a production-grade ML systems portfolio project.**

*If this helped you, please give it a ⭐*

</div>
