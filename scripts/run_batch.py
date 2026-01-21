"""
Launch multiple `neural_spline.main` runs sequentially on a single GPU.

You can keep all run settings inside this script (see DEFAULT_CONFIG) or pass a
JSON config via `--config path/to/config.json`.

Config schema:
{
  "base_args": { "model": "...", "hidden-dim": 128, ... },
  "grid": { "num-layers": [4,6], "max-knots": [32,64], ... }
}

Each run uses `base_args` plus one value from every list in `grid`
(cartesian product).
"""

from __future__ import annotations

import argparse
import json
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_CONFIG = {
    "base_args": {
        "model": "Stanford_armadillo",
        "hidden-dim": 128,
        "max-seg-insertions": 32,
    },
    "grid": {
        "quantile-length": [0.1, 0.3, 0.5],
        "lr": [0.01, 0.001, 0.0005],
        "hidden-dim": [128, 64],
    },
}


def load_config(config_path: Optional[str]) -> Dict:
    """Load JSON config from a file, or fall back to DEFAULT_CONFIG."""
    if config_path is None:
        return DEFAULT_CONFIG
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    return json.loads(path.read_text())


def iter_grid(grid: Dict[str, List]) -> Iterable[Dict[str, object]]:
    """Yield dicts representing each point in the cartesian product of the grid."""
    items = list(grid.items())
    keys = [k for k, _ in items]
    values = [v for _, v in items]
    for combo in itertools.product(*values):
        yield {k: v for k, v in zip(keys, combo)}


def build_save_dir(run_args: Dict[str, object], idx: int, total: int, grid: Dict[str, List]) -> str:
    """
    Construct a per-run save directory if not provided.
    Prefers: outputs/<model>/d{num_layers}_{hidden_dim}/k{max_knots}
    Falls back to: outputs/run_{idx:03d}_of_{total}
    """

    grid_str = "_".join(f"{k}={run_args.get(k)}" for k in grid.keys() if k in run_args)
    base = Path("outputs")
    model = run_args.get("model")
    
    return str(base / str(model) /grid_str)


def to_cli_args(args_dict: Dict[str, object]) -> List[str]:
    """Convert a dict of flag -> value to a flat CLI arg list."""
    flat: List[str] = []
    for key, val in args_dict.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                flat.append(flag)
            continue
        flat.extend([flag, str(val)])
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch launcher for neural_spline.main")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--config", help="Path to JSON config. If omitted, DEFAULT_CONFIG is used.")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional args forwarded to neural_spline.main")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    base_args = dict(config.get("base_args", {}))
    grid = config.get("grid", {})

    if not grid:
        raise SystemExit("Config grid is empty.")

    configs = list(iter_grid(grid))
    total = len(configs)
    if total == 0:
        print("No configurations to run.")
        return

    for idx, overrides in enumerate(configs, 1):
        run_args = {**base_args, **overrides}

        # Ensure save-dir exists in args; auto-generate if missing.
        if "save-dir" not in run_args:
            run_args["save-dir"] = build_save_dir(run_args, idx, total, grid)

        cmd = [args.python, "-m", "neural_spline.main"]
        cmd += to_cli_args(run_args)

        if args.extra_args:
            cmd.extend(args.extra_args)

        if Path(run_args["save-dir"]).exists():
            print(f"Save directory {run_args['save-dir']} already exists. Skipping.")
            continue

        prefix = f"[{idx}/{total}] "
        print(prefix + " ".join(cmd))

        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"{prefix}Run failed with exit code {exc.returncode}. Stopping.")
            sys.exit(exc.returncode)
        except KeyboardInterrupt:
            print(f"{prefix}Interrupted. Stopping.")
            sys.exit(1)


if __name__ == "__main__":
    main()
