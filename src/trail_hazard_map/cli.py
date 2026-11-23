import argparse
import sys
from pathlib import Path

import yaml


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trail-hazard-map")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--config", default="config.yaml")
    d.add_argument("--out", default=None)
    r = sub.add_parser("run")
    r.add_argument("--trail")
    r.add_argument("--area")
    r.add_argument("--images")
    r.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        from trail_hazard_map.demo.pipeline import run_demo

        cfg = load_config(args.config)
        run_demo(cfg, args.out or Path(cfg["out_dir"]) / "demo")
        return 0
    print("run needs the GPU stack (Mapillary, VGGT, gsplat, Grounded-SAM-2); not part of the demo build. Use: trail-hazard-map demo", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
