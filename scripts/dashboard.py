#!/usr/bin/env python3
"""
Launch the AutoConjecture Phase 4 interactive dashboard.

Usage:
    python3 scripts/dashboard.py [--port PORT] [--checkpoint-dir DIR] [--log-dir DIR]

The dashboard opens in your browser at http://localhost:8501 (default port).
"""
import argparse
import os
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="AutoConjecture Phase 4: launch the monitoring dashboard"
    )
    p.add_argument("--port", type=int, default=8501, help="Port to run on (default: 8501)")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override the default checkpoint directory shown in the sidebar",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override the default log directory shown in the sidebar",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open a browser window",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dashboard_module = os.path.join(project_root, "src", "monitoring", "dashboard.py")

    if not os.path.isfile(dashboard_module):
        print(f"Error: dashboard module not found at {dashboard_module}", file=sys.stderr)
        sys.exit(1)

    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        dashboard_module,
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
        "--theme.base", "light",
        "--theme.primaryColor", "#4C78A8",
    ]

    # Pass overrides via environment variables that dashboard.py reads
    env = os.environ.copy()
    if args.checkpoint_dir:
        env["AUTOCONJECTURE_CHECKPOINT_DIR"] = os.path.abspath(args.checkpoint_dir)
    if args.log_dir:
        env["AUTOCONJECTURE_LOG_DIR"] = os.path.abspath(args.log_dir)

    print("=" * 60)
    print("AutoConjecture Phase 4 — Monitoring Dashboard")
    print("=" * 60)
    print(f"  URL: http://localhost:{args.port}")
    print(f"  Checkpoint dir: {args.checkpoint_dir or os.path.join(project_root, 'data', 'checkpoints')}")
    print(f"  Log dir:        {args.log_dir or os.path.join(project_root, 'data', 'logs')}")
    print("  Press Ctrl-C to stop.")
    print("=" * 60)

    try:
        subprocess.run(cmd, env=env, check=True, cwd=project_root)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except FileNotFoundError:
        print(
            "\nError: 'streamlit' not found.\n"
            "Install it with:  pip install streamlit>=1.28.0",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
