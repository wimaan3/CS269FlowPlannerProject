# cs269-flow-planner

Trajectory representation ablation in flow-matching driving planners.

We vendor the official **Flow Planner** code ([Tan et al., NeurIPS 2025](https://arxiv.org/abs/2510.11083), MIT licensed) and ablate the trajectory representation it predicts while holding architecture, training schedule, and hyperparameters fixed.

**Representations tested**: Waypoints `(x, y, cos θ, sin θ)` · Frenet `(s, d, cos θ, sin θ)` · Velocity `(ẋ, ẏ, cos θ, sin θ)` · Acceleration `(ẍ, ÿ, cos θ, sin θ)`.

---

## Getting started

For every dependency, environment requirement, and a verification cell, see [`QUICKSTART.md`](./QUICKSTART.md) before running any notebook.

---

## Repository layout

```
cs269-flow-planner/
├── README.md                 this file
├── QUICKSTART.md             environment setup and verification
├── flow_planner/             vendored fork of Flow Planner with audit patches
├── notebooks/                experiment notebooks (Colab-runnable)
├── scripts/                  helper scripts (log-split, viz, normalization)
├── tests/                    unit tests for the audit patches
└── bevs/                     pre-rendered BEV figures
    ├── static/               4-scene and 16-scene PNG snapshots
    ├── rollouts/             animated GIF rollouts
    └── matrices/             summary bar/grid figures
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| [`paper_baseline.ipynb`](notebooks/paper_baseline.ipynb) | Pristine upstream Waypoints baseline + minimal Frenet |
| [`motion_representations.ipynb`](notebooks/motion_representations.ipynb) | Single-seed paired comparison across all four representations |
| [`recover_all.ipynb`](notebooks/recover_all.ipynb) | Multi-seed paired comparison (seeds 269, 1337, 2026) |
| [`v8_team.ipynb`](notebooks/v8_team.ipynb) | Team Colab runner — end-to-end training pipeline |
| [`v8_frenet_fixes.ipynb`](notebooks/v8_frenet_fixes.ipynb) | Frenet V0–V11 inference-variant ablation |
| [`scale_ablation_full.ipynb`](notebooks/scale_ablation_full.ipynb) | Full 5,000-scenario Frenet scale ablation |
| [`cs269_dagshub_best_ever_frenet.ipynb`](notebooks/cs269_dagshub_best_ever_frenet.ipynb) | Best-ever Frenet 5k training (paper headline) |
| [`cs269_frenet_every_advantage.ipynb`](notebooks/cs269_frenet_every_advantage.ipynb) | Adds Fix B+C and Fix A1+B+C variants |

To render the paper's BEV figures from trained checkpoints, run [`scripts/generate_paper_bevs.py`](scripts/generate_paper_bevs.py) — environment-agnostic, takes checkpoint directory and cache directory as arguments. See the script's docstring for setup instructions in both Colab and local environments.

---

## Headline results

| Representation | ADE (m) | FDE (m) | Source |
|---|---|---|---|
| Waypoints (1.5k, seed=269) | 4.19 | 8.63 | `motion_waypoints_seed269.ckpt` |
| Acceleration (1.5k, seed=269) | 19.98 | 35.60 | `motion_acceleration_seed269.ckpt` |
| Velocity (1.5k, seed=269) | 22.30 | 39.38 | `motion_velocity_seed269.ckpt` |
| Frenet — Fix B+C (5k, seed=42) | **21.00** | 26.42 | `frenet_seed42_fixedCB.ckpt` |
| Frenet — Original (5k, seed=42) | 22.66 | 27.50 | `frenet_seed42.ckpt` |

---

## Audit patches applied to upstream Flow Planner

Applied uniformly to all four representations to fix upstream bugs without favoring any representation:

1. Convert heading `θ` → `(cos θ, sin θ)` before finite-difference (fixes 2π wrap)
2. Re-fit Frenet normalization stats on the preprocessed cache (`frenet_norm_stats_v1`)
3. `tanh(d/3)` lateral clipping for stability when `|d|` spikes
4. Route-aware multi-segment "smart centerline" picker (`FRENET_SMART_CENTERLINE=1`)
5. `CenterlineEncoder` module for centerline conditioning at training and inference
6. Single-GPU DDP cleanup guard (`if torch.distributed.is_initialized()`)
7. Hydra override prefix fix (`++` for keys already in the YAML)
8. Lazy `wandb` imports
9. `protobuf < 4` pinned (wandb compatibility)
10. GitHub token and DagsHub auth fallback for Colab

See [`flow_planner/CS269_NOTES.md`](flow_planner/CS269_NOTES.md) for file-by-file modification notes.

---

## License

Our additions are MIT-licensed. The vendored `flow_planner/` directory retains its original MIT license from the paper authors (`flow_planner/LICENSE`).

---

## References

- T. Tan et al. *Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling*. NeurIPS 2025. arXiv:2510.11083.
- Y. Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023. arXiv:2210.02747.
- H. Caesar et al. *nuPlan: A Closed-loop ML-based Planning Benchmark for Autonomous Vehicles*. CVPR Workshop 2021. arXiv:2106.11810.
- M. Werling et al. *Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenét Frame*. ICRA 2010.
