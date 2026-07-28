# LLMBench

Stress tester for LLM APIs — measures token throughput under concurrent load.

## Quick Start

```bash
./install.sh                # one-time setup (installs uv + deps)
source run.sh               # activates venv + loads .env
python app.py               # run benchmark
```

## Prerequisites

- Python ≥ 3.10
- UV (auto-installed by `install.sh`)

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Required: `API_URL`, `API_KEY`, `MODEL`.

## Usage

```bash
# Interactive
python app.py

# With args
python app.py --duration 120 --max-concurrency 32 --max-tokens 512
```

## GPU (H100 / DGX Spark)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=0
./run.sh
```

## Output

- `stress_result.json` — full benchmark report
- `stress_result.csv`  — per-request details
