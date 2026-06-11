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
after the dependency bumps in
[`flow_planner/requirements.txt`](flow_planner/requirements.txt):

### Closed by bumping

- `tensorboard==2.11.2` → `==2.17.1` — clears CVE-2023-25658 (path traversal),
  CVE-2023-25662 (open redirect)
- `torch==2.3.0` → `==2.12.0` — clears 19 prior CVEs (full list in commit
  `b73ca15` and follow-up)
- `pytorch-lightning==2.4.0` → `==2.5.5` — tracks the torch line
- All 5 previously-unpinned deps (`flow-matching`, `dagshub`, `matplotlib`,
  `numpy`, `pytorch-lightning`) are now pinned so Dependabot can alert on them.

### Residual CVEs (do not apply to this codebase)

| Advisory | Package | Vulnerable function | Why it does not apply here |
|---|---|---|---|
| CVE-2025-3000 (GHSA-rrmf-rvhw-rf47) | `torch` | `torch.jit.script` memory corruption | We do not call `torch.jit.script` anywhere. Verified via `git grep "torch\.jit\.script"` → 0 hits. The Flow Planner code never JIT-compiles models; flow-matching uses eager mode throughout. |
| CVE-2026-31221 (GHSA-75m9-98v2-hjpm) | `pytorch-lightning` | `LightningModule.load_from_checkpoint` insecure deserialization | We do not call `load_from_checkpoint` anywhere. Verified via `git grep "load_from_checkpoint"` → 0 hits. We load checkpoints directly with `torch.load(weights_only=True)` (see commit `7ec9033`), which is unaffected. |

Both of these have no upstream fix released — the only way to "fix" them is
to avoid the vulnerable function, which we already do.

### Mitigations in code

- `torch.load(weights_only=True)` everywhere checkpoints are loaded
  (`inference_eval.py`, `visualize_bev.py`, `visualize_bev_gif.py`) — see
  commit `7ec9033`.
- No `eval()` / `pickle.loads()` of user input anywhere in the repo.
- No `subprocess.shell=True` calls touching user input.

## Reporting

If you find a vulnerability that affects the offline training threat model
above, open an issue on GitHub. Use the `security` label.
