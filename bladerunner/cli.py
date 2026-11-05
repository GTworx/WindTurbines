"""Command-line interface for BladeRunner.

Commands:
  train       Build/refresh the defect taxonomy from the Defects/ directory.
  inspect     Inspect a turbine directory (Inspector Agent only).
  run         Full pipeline: train -> inspect -> report (Orchestrator).
  reexamine   Re-run inspection for new/delta defect classes on cached images.
  report      (Re)generate wiki + HTML dashboard from a stored results.json.
  status      Show checkpoint state.
  init-config Write an example config file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config, DEFAULTS
from .state import CheckpointState


def _load_config(args) -> Config:
    project_root = args.project_root or "."
    data = dict(DEFAULTS)
    data["project_root"] = project_root
    cfg = Config(data=data)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        from .config import _deep_merge

        cfg = Config(data=_deep_merge(cfg.data, user_cfg))
    return cfg


def cmd_init_config(args) -> int:
    out = Path(args.output or "bladerunner.example.json")
    out.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
    print(f"Wrote example config to {out}")
    return 0


def cmd_train(args) -> int:
    cfg = _load_config(args)
    from .agents import DefectTrainingAgent
    from .logging_setup import BladeLogger

    BladeLogger.setup(cfg.log_file)
    agent = DefectTrainingAgent(cfg)
    result = agent.run()
    print(json.dumps(result, indent=2))
    return 0


def cmd_inspect(args) -> int:
    cfg = _load_config(args)
    cfg.data["backend"] = args.backend or cfg.backend
    cfg = Config(data=cfg.data)
    from .agents import BladeInspectorAgent
    from .logging_setup import BladeLogger
    from .vision import get_backend
    from .state import CheckpointState

    BladeLogger.setup(cfg.log_file)
    vision = get_backend(cfg.backend, config=cfg)
    state = CheckpointState(cfg.checkpoint_file)
    agent = BladeInspectorAgent(cfg, vision=vision, run_id=args.run_id, state=state)
    report = agent.inspect_turbine(Path(args.turbine_dir), confidence_threshold=args.confidence_threshold)
    print(json.dumps({"turbine": report.to_dict()}, indent=2, default=str))
    print(f"Results: {agent.run_dir / 'results.json'}")
    return 0


def cmd_run(args) -> int:
    cfg = _load_config(args)
    cfg.data["backend"] = args.backend or cfg.backend
    cfg = Config(data=cfg.data)
    from .orchestrator import OrchestratorAgent

    orch = OrchestratorAgent(cfg, run_id=args.run_id, backend=cfg.backend)
    summary = orch.run(
        Path(args.turbine_dir),
        confidence_threshold=args.confidence_threshold,
        train=not args.no_train,
        report=not args.no_report,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, default=str))
    artifacts = summary.get("artifacts", {})
    if artifacts.get("dashboard"):
        print(f"\nDashboard: {artifacts['dashboard']}")
        print(f"Wiki index: {artifacts.get('wiki_index')}")
    return 0


def cmd_reexamine(args) -> int:
    cfg = _load_config(args)
    cfg.data["backend"] = args.backend or cfg.backend
    cfg = Config(data=cfg.data)
    from .agents import ReexamineAgent
    from .logging_setup import BladeLogger
    from .vision import get_backend
    from .state import CheckpointState

    BladeLogger.setup(cfg.log_file)
    vision = get_backend(cfg.backend, config=cfg)
    agent = ReexamineAgent(cfg, vision=vision, run_id=args.run_id, state=CheckpointState(cfg.checkpoint_file))
    delta = [c.strip() for c in args.delta_classes.split(",") if c.strip()]
    report = agent.run(
        Path(args.turbine_dir),
        Path(args.prior_results) if args.prior_results else None,
        delta,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps({"turbine": report.to_dict(), "delta_classes": delta}, indent=2, default=str))
    return 0


def cmd_report(args) -> int:
    cfg = _load_config(args)
    from .agents import ReportingAgent
    from .logging_setup import BladeLogger
    from .models import Detection, BladeReport, ImageResult, TurbineReport

    BladeLogger.setup(cfg.log_file)
    with open(args.results, "r", encoding="utf-8") as f:
        raw = json.load(f).get("turbine", {})
    turbine = TurbineReport(turbine_id=raw.get("turbine_id", ""))
    for br_raw in raw.get("blade_reports", []):
        br = BladeReport(blade_id=br_raw["blade_id"], turbine_id=raw.get("turbine_id", ""))
        for ir_raw in br_raw.get("image_results", []):
            ir = ImageResult(image_path=ir_raw["image_path"], image_name=ir_raw["image_name"])
            for d in ir_raw.get("detections", []):
                ir.detections.append(Detection(**d))
            br.image_results.append(ir)
        turbine.blade_reports.append(br)
    agent = ReportingAgent(cfg, run_id=args.run_id)
    artifacts = agent.generate(turbine)
    print(json.dumps(artifacts, indent=2))
    return 0


def cmd_status(args) -> int:
    cfg = _load_config(args)
    state = CheckpointState(cfg.checkpoint_file).load()
    print(json.dumps(state.data, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bladerunner", description="BladeRunner wind turbine blade inspection framework.")
    p.add_argument("--version", action="version", version=f"bladerunner {__version__}")
    p.add_argument("--config", help="Path to a BladeRunner JSON config file.")
    p.add_argument("--project-root", help="Project root directory (default: current dir).")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("train", help="Build/refresh the defect taxonomy.")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("inspect", help="Inspect a turbine directory.")
    sp.add_argument("--turbine-dir", required=True)
    sp.add_argument("--confidence-threshold", type=float, default=None)
    sp.add_argument("--backend", choices=["mock", "local"], default=None)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("run", help="Run the full pipeline (train -> inspect -> report).")
    sp.add_argument("--turbine-dir", required=True)
    sp.add_argument("--confidence-threshold", type=float, default=None)
    sp.add_argument("--backend", choices=["mock", "local"], default=None)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--no-train", action="store_true", help="Skip taxonomy build.")
    sp.add_argument("--no-report", action="store_true", help="Skip report generation.")
    sp.add_argument("--resume", action="store_true", help="Reuse cached results if present.")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("reexamine", help="Re-examine cached images for new defect classes.")
    sp.add_argument("--turbine-dir", required=True)
    sp.add_argument("--delta-classes", required=True, help="Comma-separated new defect classes.")
    sp.add_argument("--prior-results", default=None, help="Path to a prior results.json.")
    sp.add_argument("--confidence-threshold", type=float, default=None)
    sp.add_argument("--backend", choices=["mock", "local"], default=None)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_reexamine)

    sp = sub.add_parser("report", help="Generate wiki + dashboard from a results.json.")
    sp.add_argument("--results", required=True, help="Path to a results.json.")
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("status", help="Show checkpoint state.")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("init-config", help="Write an example config file.")
    sp.add_argument("--output", default=None)
    sp.set_defaults(func=cmd_init_config)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the CLI
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
