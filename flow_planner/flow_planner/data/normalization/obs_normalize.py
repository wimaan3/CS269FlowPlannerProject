import torch
from flow_planner.core.common.json_utils import openjson
from copy import deepcopy

class ObservationNormalizer:
    
    def __init__(self, config):
        self._ndt = {
                k: {
                    "mean": torch.tensor(config[k]['mean'], dtype=torch.float32),
                    "std": torch.tensor(config[k]['std'], dtype=torch.float32),
                }
                for k in config.keys()
                if k not in ("ego", "neighbor")
            }

        # Optional: store ego/neighbor if needed
        self.ego = config['ego'] if "ego" in config else None
        self.neighbor = config['neighbor'] if "neighbor" in config else None
        
    def __call__(self, data):
        data = deepcopy(data)
        for k, v in self._ndt.items():
            mask = torch.sum(torch.ne(data.__dict__[k], 0), dim=-1) == 0
            data.__dict__[k] = (data.__dict__[k] - v["mean"].to(data.__dict__[k].device)) / v["std"].to(data.__dict__[k].device)
            data.__dict__[k][mask] = 0
        return data

    def inverse(self, data):
        # Symmetry fix: ``__call__`` indexes ``data.__dict__[k]`` (works for the
        # NuPlanDataSample dataclass), but the original ``inverse`` indexed
        # ``data[k]`` — raising TypeError because NuPlanDataSample is not
        # subscriptable. Currently no caller exercises this path, but keeping
        # the asymmetry guarantees a future viz/debug script crashes.
        data = deepcopy(data)
        for k, v in self._ndt.items():
            mask = torch.sum(torch.ne(data.__dict__[k], 0), dim=-1) == 0
            data.__dict__[k] = (
                data.__dict__[k] * v["std"].to(data.__dict__[k].device)
                + v["mean"].to(data.__dict__[k].device)
            )
            data.__dict__[k][mask] = 0
        return data