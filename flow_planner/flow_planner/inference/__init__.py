"""Inference-time hooks for Flow Planner.

Each submodule under hooks/ exposes a single `apply(model)` function that
monkey-patches the model's forward_inference to implement a specific
inference-time technique (multi-frame ensembling, best-of-N verifier rerank,
etc.). Hooks are opt-in: inference_eval.py imports a hook only when the
FRENET_INFERENCE_HOOK environment variable names its module path.

Adding a new hook:
    1. Create flow_planner/inference/hooks/my_hook.py with an `apply(model)`
       function that wraps model.forward_inference.
    2. Set FRENET_INFERENCE_HOOK=flow_planner.inference.hooks.my_hook before
       running inference_eval.py.
    3. Optional hook-specific knobs go in env vars (e.g.
       FRENET_INFERENCE_K=4, FRENET_INFERENCE_N=16).
"""
