#!/usr/bin/env python3
"""
Setup script for Lean 4 + Mathlib integration (Track B).

Checks whether Lean 4 / elan / lake / Mathlib are installed and provides
actionable instructions for anything missing.

Usage:
    python scripts/setup_lean4.py
    python scripts/setup_lean4.py --create-project /path/to/new_project
"""
import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def run(cmd, cwd=None) -> tuple:
    """Run a command, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=cwd
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, "", "Timeout"


def check_elan():
    code, out, err = run(["elan", "--version"])
    if code == 0:
        print(f"  ✓ elan: {out}")
        return True
    print("  ✗ elan not found.")
    print("    Install: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh")
    return False


def check_lean():
    code, out, err = run(["lean", "--version"])
    if code == 0:
        print(f"  ✓ lean: {out}")
        return True
    print("  ✗ lean not found.")
    print("    Run: elan toolchain install leanprover/lean4:stable && elan default leanprover/lean4:stable")
    return False


def check_lake():
    code, out, err = run(["lake", "--version"])
    if code == 0:
        print(f"  ✓ lake: {out}")
        return True
    print("  ✗ lake not found (usually bundled with lean via elan).")
    return False


def check_mathlib(project_dir=None):
    """Check whether Mathlib is available in the given project."""
    if project_dir is None:
        print("  - Mathlib: no project directory specified (use --project-dir).")
        return False

    lockfile = os.path.join(project_dir, "lake-manifest.json")
    if not os.path.isfile(lockfile):
        print(f"  ✗ Mathlib: lake-manifest.json not found in {project_dir}.")
        print(f"    Run: cd {project_dir} && lake update")
        return False

    import json
    try:
        with open(lockfile) as f:
            manifest = json.load(f)
        packages = [p.get("name", "") for p in manifest.get("packages", [])]
        if "mathlib" in packages or "Mathlib" in packages:
            print(f"  ✓ Mathlib found in {project_dir}")
            return True
        else:
            print(f"  ✗ Mathlib not in lake-manifest. Add it to lakefile.lean and run `lake update`.")
            return False
    except Exception as e:
        print(f"  ✗ Could not read lake-manifest.json: {e}")
        return False


def check_repl():
    code, out, err = run(["repl", "--help"])
    if code == 0:
        print("  ✓ lean4-repl found (persistent mode available).")
        return True
    print("  - lean4-repl not found (oneshot mode only).")
    print("    For faster batch evaluation, install:")
    print("    https://github.com/leanprover-community/repl")
    return False


def create_project(path: str):
    """Create a new Lake project with Mathlib at the given path."""
    print(f"\nCreating Lake project at {path}...")

    # Create directory
    os.makedirs(path, exist_ok=True)

    # Write lakefile.lean
    lakefile = os.path.join(path, "lakefile.lean")
    with open(lakefile, "w") as f:
        f.write("""import Lake
open Lake DSL

package autoconj_lean4 where
  name := "autoconj_lean4"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

lean_lib AutoConjLean4 where
  globs := #[.path `AutoConjLean4]
""")

    # Write lean-toolchain
    toolchain = os.path.join(path, "lean-toolchain")
    # Get current lean toolchain
    code, out, _ = run(["lean", "--version"])
    version_str = "leanprover/lean4:stable"
    if code == 0 and "version" in out.lower():
        # Try to extract version number
        import re
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        if m:
            version_str = f"leanprover/lean4:v{m.group(1)}"
    with open(toolchain, "w") as f:
        f.write(version_str + "\n")

    # Create a minimal .lean file
    src_dir = os.path.join(path, "AutoConjLean4")
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, "Basic.lean"), "w") as f:
        f.write("import Mathlib\nopen Nat\n\n-- AutoConjecture Lean 4 integration\n")

    print(f"  ✓ Created {lakefile}")
    print(f"  ✓ Created {toolchain} ({version_str})")
    print(f"\nNext steps:")
    print(f"  cd {path}")
    print(f"  lake update          # Downloads Mathlib (may take 10-30 min first time)")
    print(f"  lake build           # Compiles Mathlib cache")
    print(f"\nThen run:")
    print(f"  python scripts/run_lean4_eval.py --project-dir {path} --check-only")


def main():
    parser = argparse.ArgumentParser(description="Set up Lean 4 + Mathlib for Track B")
    parser.add_argument(
        "--project-dir", type=str, default=None,
        help="Lake project directory to check for Mathlib",
    )
    parser.add_argument(
        "--create-project", type=str, default=None,
        help="Create a new Lake project with Mathlib at this path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Lean 4 Setup Check")
    print("=" * 60)

    ok_elan = check_elan()
    ok_lean = check_lean()
    ok_lake = check_lake()
    ok_mathlib = check_mathlib(args.project_dir)
    ok_repl = check_repl()

    print("\nSummary:")
    all_ok = ok_lean and ok_lake
    if all_ok:
        print("  ✓ Core Lean 4 tools are installed.")
        print("  → Run Track B evaluation: python scripts/run_lean4_eval.py --check-only")
        if not ok_mathlib:
            print(
                "\n  NOTE: Mathlib not found. Without Mathlib, only basic tactics "
                "(decide, norm_num) are available.\n"
                "  Create a project: python scripts/setup_lean4.py --create-project /path/to/proj"
            )
    else:
        print("  ✗ Some Lean 4 components are missing. See instructions above.")

    if args.create_project:
        create_project(args.create_project)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
