# Security policy

This is a research repository for a UCLA CS 269 final project. It is not a
production-grade application. Nevertheless we take security seriously and
treat known dependency vulnerabilities as quality bugs.

## Threat model

The training pipeline runs offline on single-host machines (Colab, a local
GPU box, or a DagsHub workspace). It consumes:

- Trusted training data (nuPlan-mini, downloaded once via DagsHub)
- Trusted model checkpoints (produced by our own training runs)
- Hydra YAML configs in the repo

It **does not** accept network input at training or inference time, run
inference servers exposed to the public, parse untrusted user-supplied
files, or deserialize attacker-controlled pickles. Checkpoint loads use
`torch.load(weights_only=True)` with a controlled fallback for legacy
files (see [`security` commit `7ec9033`](https://github.com/wimaan3/CS269FlowPlannerProject/commit/7ec9033)).

## Reported vulnerabilities and our response

We track upstream CVEs via `pip-audit` and GitHub Dependabot. Current state
(after [`security` commit](https://github.com/wimaan3/CS269FlowPlannerProject/commits/main/flow_planner/requirements.txt)
bumping `tensorboard`, `torch`, `pytorch-lightning`):

| Package | Pinned | Open CVEs | Why we did not bump further |
|---|---|---|---|
| `torch` | 2.9.1 | 3 (PYSEC-2026-139, CVE-2025-3000, CVE-2025-3001) | First two have no upstream fix; CVE-2025-3001 fix is in torch 2.10.0, which would force a numerical-reproducibility break against the eval JSONs already shipped under [`results/`](results/). All three require parsing a malicious tensor / model file, which our threat model rules out. |
| `pytorch-lightning` | 2.5.5 | 1 (CVE-2026-31221) | LightningCLI vulnerability requiring a hostile config file. We don't invoke `LightningCLI` anywhere; checked via `git grep LightningCLI`. |

We have closed:

- `tensorboard==2.11.2` → `==2.17.1` (clears CVE-2023-25658, CVE-2023-25662)
- `torch==2.3.0` → `==2.9.1` (clears 18 prior CVEs)
- All 5 previously-unpinned deps (`flow-matching`, `dagshub`, `matplotlib`,
  `numpy`, `pytorch-lightning`) — Dependabot can now alert on them.

## Reporting

If you find a vulnerability that affects the offline training threat model
above, open an issue on GitHub. Use the `security` label.
