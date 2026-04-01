"""
Lean 4 REPL / subprocess interface.

Supports two modes:
  - "oneshot"    : Write a .lean file to a temp dir and invoke `lean` on it.
                   Simple but incurs Mathlib compilation overhead every call.
  - "persistent" : Maintain a long-running Lean 4 REPL process that accepts
                   JSON-line commands (compatible with the lean4-repl package).
                   Reuses the Mathlib environment across calls; much faster
                   for batch evaluation.

If Lean 4 is not installed, all methods raise Lean4NotAvailable.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional


class Lean4NotAvailable(RuntimeError):
    """Raised when Lean 4 is not installed or not on PATH."""
    pass


@dataclass
class Lean4Result:
    """Result of one Lean 4 check or tactic application."""
    success: bool
    error_message: Optional[str]
    proof_term: Optional[str]   # The completed proof string if success
    time_taken_seconds: float
    env_id: int = 0             # Environment ID (persistent mode only)
    raw_output: str = ""


class Lean4REPLInterface:
    """
    Manages communication with Lean 4 for theorem checking.

    Usage (one-shot mode, no Lean 4 REPL package required):
        with Lean4REPLInterface(mode="oneshot") as lean:
            result = lean.check_theorem("theorem t : 1 + 1 = 2 := by norm_num")

    Usage (persistent mode, requires lean4-repl):
        with Lean4REPLInterface(mode="persistent") as lean:
            result = lean.check_theorem("theorem t : 1 + 1 = 2 := by norm_num")
    """

    # Mathlib import header inserted at the top of every generated file
    MATHLIB_HEADER = "import Mathlib\nopen Nat\n\n"
    # Minimal header without Mathlib (faster, but fewer tactics available)
    MINIMAL_HEADER = "-- Lean 4 (no Mathlib)\nopen Nat\n\n"

    def __init__(
        self,
        lean_executable: str = "lean",
        lake_executable: str = "lake",
        mathlib_path: Optional[str] = None,
        timeout_seconds: float = 60.0,
        mode: str = "oneshot",        # "oneshot" | "persistent"
        use_mathlib: bool = True,
        project_dir: Optional[str] = None,
    ):
        """
        Args:
            lean_executable: Path to `lean` binary (default: searches PATH).
            lake_executable: Path to `lake` binary (default: searches PATH).
            mathlib_path: Path to a Mathlib4 checkout (used by oneshot mode).
            timeout_seconds: Per-call timeout.
            mode: "oneshot" or "persistent".
            use_mathlib: If False, uses a minimal header (no Mathlib import).
            project_dir: Lake project directory with Mathlib as a dependency.
                         If provided, oneshot mode uses `lake env lean`.
        """
        self.lean_executable = lean_executable
        self.lake_executable = lake_executable
        self.mathlib_path = mathlib_path
        self.timeout = timeout_seconds
        self.mode = mode
        self.use_mathlib = use_mathlib
        self.project_dir = project_dir

        self._available: Optional[bool] = None   # Cached availability check
        self._repl_proc: Optional[subprocess.Popen] = None
        self._repl_lock = threading.Lock()
        self._env_id: int = 0

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "Lean4REPLInterface":
        self._check_available()
        if self.mode == "persistent":
            self._start_repl()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── Public API ─────────────────────────────────────────────────────────────

    def check_theorem(
        self,
        lean_code: str,
        extra_tactics: Optional[list] = None,
    ) -> Lean4Result:
        """
        Attempt to verify a Lean 4 theorem declaration.

        Args:
            lean_code: Complete Lean 4 theorem declaration, e.g.
                       "theorem t (x : Nat) : x + 0 = x := by simp"
            extra_tactics: Additional tactics to try if the given one fails
                           (oneshot mode only, appended as fallback `by <tac>`).

        Returns:
            Lean4Result with success=True if Lean accepts the declaration.
        """
        self._check_available()
        if self.mode == "persistent" and self._repl_proc is not None:
            return self._check_persistent(lean_code)
        return self._check_oneshot(lean_code)

    def check_expression(self, lean_type: str, tactic: str = "decide") -> Lean4Result:
        """
        Check a Lean 4 proposition given as a type string.
        Wraps it in a fresh theorem declaration automatically.
        """
        code = f"theorem _check : {lean_type} := by {tactic}"
        return self.check_theorem(code)

    def is_available(self) -> bool:
        """Return True if Lean 4 is installed and accessible."""
        try:
            self._check_available()
            return True
        except Lean4NotAvailable:
            return False

    def stop(self):
        """Terminate the persistent REPL subprocess if running."""
        if self._repl_proc is not None:
            try:
                self._repl_proc.terminate()
                self._repl_proc.wait(timeout=5)
            except Exception:
                pass
            self._repl_proc = None

    # ── One-shot mode ──────────────────────────────────────────────────────────

    def _check_oneshot(self, lean_code: str) -> Lean4Result:
        """Write lean_code to a temp file and invoke `lean` on it."""
        header = self.MATHLIB_HEADER if self.use_mathlib else self.MINIMAL_HEADER
        full_code = header + lean_code

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False
        ) as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            t0 = time.time()
            if self.project_dir and os.path.isdir(self.project_dir):
                cmd = [self.lake_executable, "env", self.lean_executable, tmp_path]
                cwd = self.project_dir
            else:
                cmd = [self.lean_executable, tmp_path]
                cwd = None

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
            )
            elapsed = time.time() - t0
            stdout = result.stdout
            stderr = result.stderr
            combined = (stdout + "\n" + stderr).strip()

            success = result.returncode == 0 and "error" not in combined.lower()
            error_msg = None if success else combined[:500]

            return Lean4Result(
                success=success,
                error_message=error_msg,
                proof_term=lean_code if success else None,
                time_taken_seconds=elapsed,
                raw_output=combined,
            )

        except subprocess.TimeoutExpired:
            return Lean4Result(
                success=False,
                error_message="timeout",
                proof_term=None,
                time_taken_seconds=self.timeout,
                raw_output="",
            )
        except Exception as e:
            return Lean4Result(
                success=False,
                error_message=str(e),
                proof_term=None,
                time_taken_seconds=0.0,
                raw_output="",
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Persistent REPL mode ───────────────────────────────────────────────────

    def _start_repl(self):
        """
        Launch a lean4-repl subprocess that accepts JSON-line commands.

        Requires the `repl` executable from https://github.com/leanprover-community/repl
        to be on PATH (or bundled in the project_dir).
        """
        repl_cmd = self._find_repl_executable()
        if repl_cmd is None:
            print(
                "[Lean4] lean4-repl not found; falling back to oneshot mode.",
                flush=True,
            )
            self.mode = "oneshot"
            return

        try:
            self._repl_proc = subprocess.Popen(
                [repl_cmd],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_dir,
            )
            # Send a no-op to verify the REPL is alive
            test = self._send_repl_cmd("#check Nat", env=0)
            if not test.get("env"):
                raise RuntimeError("REPL did not respond to test command")
            self._env_id = test.get("env", 0)
            print("[Lean4] Persistent REPL started.", flush=True)
        except Exception as e:
            print(f"[Lean4] Failed to start persistent REPL: {e}. Using oneshot.", flush=True)
            self.stop()
            self.mode = "oneshot"

    def _find_repl_executable(self) -> Optional[str]:
        """Locate the lean4-repl executable."""
        candidates = ["repl", "lean4-repl"]
        if self.project_dir:
            candidates.insert(0, os.path.join(self.project_dir, ".lake", "build", "bin", "repl"))
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
            try:
                subprocess.run([c, "--help"], capture_output=True, timeout=5)
                return c
            except Exception:
                continue
        return None

    def _check_persistent(self, lean_code: str) -> Lean4Result:
        """Send a declaration to the persistent REPL and parse the response."""
        with self._repl_lock:
            t0 = time.time()
            try:
                resp = self._send_repl_cmd(lean_code, env=self._env_id)
                elapsed = time.time() - t0

                messages = resp.get("messages", [])
                errors = [m for m in messages if m.get("severity") == "error"]

                if errors:
                    return Lean4Result(
                        success=False,
                        error_message="; ".join(m.get("data", "") for m in errors)[:500],
                        proof_term=None,
                        time_taken_seconds=elapsed,
                        env_id=self._env_id,
                        raw_output=json.dumps(resp),
                    )

                new_env = resp.get("env", self._env_id)
                self._env_id = new_env
                return Lean4Result(
                    success=True,
                    error_message=None,
                    proof_term=lean_code,
                    time_taken_seconds=elapsed,
                    env_id=new_env,
                    raw_output=json.dumps(resp),
                )
            except Exception as e:
                # REPL crashed; restart and fall back to oneshot for this call
                self.stop()
                self.mode = "oneshot"
                return self._check_oneshot(lean_code)

    def _send_repl_cmd(self, cmd: str, env: int = 0) -> dict:
        """Send one JSON-line command to the REPL and read the response."""
        if self._repl_proc is None or self._repl_proc.poll() is not None:
            raise RuntimeError("REPL process is not running")

        payload = json.dumps({"cmd": cmd, "env": env}) + "\n"
        self._repl_proc.stdin.write(payload)
        self._repl_proc.stdin.flush()

        # Read until we get a complete JSON object (ends with newline)
        line = self._repl_proc.stdout.readline()
        if not line:
            raise RuntimeError("REPL returned empty response")
        return json.loads(line)

    # ── Availability check ─────────────────────────────────────────────────────

    def _check_available(self):
        if self._available is True:
            return
        if self._available is False:
            raise Lean4NotAvailable(
                "Lean 4 is not available. Install it from https://leanprover.github.io/lean4/doc/setup.html"
            )
        try:
            result = subprocess.run(
                [self.lean_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and "Lean" in result.stdout:
                self._available = True
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        self._available = False
        raise Lean4NotAvailable(
            f"Lean 4 executable '{self.lean_executable}' not found. "
            "Install via elan: https://leanprover.github.io/lean4/doc/setup.html"
        )
