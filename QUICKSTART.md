# Quickstart — reproducing CS269 Flow Planner experiments

This document lists every dependency, environment requirement, and known limitation needed to reproduce any experiment in this repository.

---

## 1. Runtime requirements

| Resource | Required | Notes |
|---|---|---|
| GPU | NVIDIA T4 or better | Free Colab T4 is sufficient for inference + small-scale training |
| Python | 3.10–3.12 | Colab's default 3.12 works |
| Disk | ~25 GB on Colab `/content/` | npz cache + checkpoints |
| Google Drive | Required for persistent storage | Mount at `/content/drive` |
| DagsHub account | Required for the preprocessed nuPlan cache | Used to mount `jialic/dagshub-drive` |

---

## 2. Python dependencies

All dependencies live in a single file, `flow_planner/requirements.txt`. The fastest setup path is:

```bash
pip install -r flow_planner/requirements.txt
```

The file pins the core training stack (`torch==2.3.0`, `tensorboard==2.11.2`, `hydra-core==1.3.2`, `omegaconf==2.3.0`, `einops==0.8.0`, `scipy==1.13.1`, `timm==1.0.10`, `flow-matching`) and adds the notebook extras (`dagshub`, `matplotlib`, `numpy`, `pytorch-lightning`).

The `flow-matching` package is critical — without it, Hydra cannot resolve `flow_planner.model.flow_planner_model.flow_utils.flow_ode.FlowODE` and every model-instantiation step will fail.

---

## 3. Data access

The preprocessed nuPlan-mini cache (80,000 `.npz` files) lives on DagsHub. Mount it once per session:

```python
import dagshub.colab, dagshub.storage

dagshub.colab.login()                                       # opens browser OAuth link
mount_path = dagshub.storage.mount('jialic/dagshub-drive')  # ~1 min, no further input

import pathlib
data_dir = pathlib.Path(mount_path) / 'data'                # 80k .npz files live here
```

The DagsHub mount is read-only and FUSE-backed, so file reads are slow. Notebooks copy a small deterministic subset (50–5,000 scenarios) to local Colab storage before training or evaluation.

---

## 4. Repository setup

Every notebook clones this repository fresh into the Colab runtime:

```python
import subprocess, pathlib, sys

REPO_DIR = '/content/CS269FlowPlannerProject'
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run([
        'git', 'clone', '--depth', '1',
        'https://github.com/wimaan3/CS269FlowPlannerProject.git',
        REPO_DIR,
    ], check=True)

FP_DIR = f'{REPO_DIR}/flow_planner'   # the vendored package directory
sys.path.insert(0, FP_DIR)
sys.path.insert(0, REPO_DIR)
```

---

## 5. Notebook catalog

| Notebook | Purpose | Approximate runtime |
|---|---|---|
| `paper_baseline.ipynb` | Pristine upstream Waypoints baseline + minimal Frenet | 2 h on T4 |
| `motion_representations.ipynb` | Single-seed paired comparison across all four representations | 4 h on T4 |
| `v8_team.ipynb` | Team Colab runner — setup, preprocess, train, evaluate | 4 h on T4 |
| `v8_frenet_fixes.ipynb` | Frenet V0–V11 inference-variant ablation | 30 min on T4 |
| `cs269_dagshub_best_ever_frenet.ipynb` | Best-ever Frenet at 5,000 training scenarios | 2 h on L4 / 4 h on T4 |
| `cs269_frenet_every_advantage.ipynb` | Adds Fix B+C and Fix A1+B+C variants | 4 h on L4 / 8 h on T4 |
| `scale_ablation_full.ipynb` | Full 5k Frenet scale ablation | 2 h on L4 / 4 h on T4 |
| `recover_all.ipynb` | Multi-seed paired comparison (seeds 269/1337/2026) | 8+ h on T4 |

To render BEV figures from trained checkpoints, use the environment-agnostic script `scripts/generate_paper_bevs.py` (runs locally, on Colab, or on a cluster):

```bash
python scripts/generate_paper_bevs.py \
    --checkpoints-dir /path/to/checkpoints \
    --cache-dir       /path/to/preprocessed_cache \
    --output-dir      /path/to/paper_bevs \
    --eval-dirs       /path/to/eval_jsons
```

See the script's docstring for setup instructions in both Colab and local environments. Typical runtime is 30 seconds per checkpoint on a T4 GPU.

---

## 6. Regenerating normalization stats from data

Three Hydra normalization configs ship with the repository:

| Config | Used by |
|---|---|
| `flow_planner/script/normalization_stats/waypoints_norm_stats.yaml` | Waypoints representation |
| `flow_planner/script/normalization_stats/frenet_norm_stats_v1.yaml` | Frenet representation (paper headline) |
| `flow_planner/script/normalization_stats/va_norm_stats.yaml` | Velocity and Acceleration representations |

The `va_norm_stats.yaml` shipped with the repository uses conservative defaults
based on typical nuPlan urban-driving statistics. To replace it with values
measured from your actual preprocessed cache:

```bash
python scripts/measure_va_norm_stats.py \
    --cache-dir   /path/to/preprocessed_cache \
    --output-yaml flow_planner/flow_planner/script/normalization_stats/va_norm_stats.yaml
```

For Frenet, the equivalent script is `scripts/measure_frenet_stats.py` followed
by `scripts/update_frenet_norm_stats.py`.

## 7. Other reproducibility notes

| Item | Affects | Notes |
|---|---|---|
| nuPlan raw `.db` files | Training from scratch only | Use the preprocessed DagsHub cache instead. |
| Vertex AI Colab Enterprise billing | Original 5k Frenet training runs | Hosted T4 Colab works for all experiments; use a longer wall-clock budget. |

---

## 8. Storage layout expected by notebooks

Notebooks expect this Drive layout. Adjust `SEARCH_DIRS` constants in the notebook if your paths differ.

```
MyDrive/cs269/
├── checkpoints/                          # Training output checkpoints
├── motion_representations/
│   ├── checkpoints/                      # 4 motion-rep checkpoints at seed 269
│   └── results/                          # Per-checkpoint eval_*.json files
├── v8_frenet_fixes/
│   ├── checkpoints/                      # V0–V11 inference-variant checkpoints
│   └── results/                          # V0–V11 eval JSONs + matrix CSV
└── paper_bevs/                           # Output of cs269_generate_paper_bevs.ipynb
```

---

## 9. Verification cell (paste before running any training/inference notebook)

Run this before any heavy work to confirm your environment is correct:

```python
import sys, pathlib, importlib

# 1. Check critical packages
for pkg in ['flow_matching', 'hydra', 'omegaconf', 'torch', 'dagshub']:
    try:
        importlib.import_module(pkg)
        print(f'  installed   {pkg}')
    except ImportError:
        print(f'  MISSING     {pkg} (re-run pip install cell)')

# 2. Check the vendored flow_planner package is importable
sys.path.insert(0, '/content/CS269FlowPlannerProject/flow_planner')
try:
    importlib.import_module('flow_planner.model.flow_planner_model.flow_utils.flow_ode')
    print('  resolvable  flow_planner.model.flow_planner_model.flow_utils.flow_ode.FlowODE')
except Exception as e:
    print(f'  FAIL        {e}')

# 3. Check Drive is mounted
if pathlib.Path('/content/drive/MyDrive').exists():
    print('  mounted     /content/drive/MyDrive')
else:
    print('  NOT MOUNTED Drive (run drive.mount cell)')

# 4. Check GPU
import torch
print(f'  GPU         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only — change runtime to T4"}')
```

---

## 10. Repository layout

```
CS269FlowPlannerProject/
├── README.md                 high-level project description
├── QUICKSTART.md             this file
├── flow_planner/             vendored fork of Flow Planner with audit patches
├── notebooks/                experiment notebooks (Colab-runnable)
├── scripts/                  helper scripts (log-split, viz, normalization)
├── tests/                    unit tests for the audit patches
└── bevs/                     pre-rendered BEV figures + catalog
```
