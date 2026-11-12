"""Build commands inside a read-only, network-isolated bubblewrap namespace."""
from __future__ import annotations

import os
import shutil


_SYSTEM_BINDS = ("/usr", "/bin", "/lib", "/lib64", "/sbin")
MAX_SANDBOX_TMP_BYTES = 64 * 1024 * 1024
MAX_SANDBOX_FILE_BYTES = 16 * 1024 * 1024
MAX_SANDBOX_PROCESSES = 64
MAX_SANDBOX_OPEN_FILES = 256
MAX_SANDBOX_CPU_SECONDS = 60


def build_readonly_sandbox_command(workspace_root: str, command: str) -> list[str]:
    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    if bwrap is None or prlimit is None:
        raise RuntimeError(
            "Read-only workflow commands require bubblewrap and prlimit."
        )
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Sandbox command must not be empty")

    workspace = os.path.realpath(os.path.abspath(workspace_root))
    if not os.path.isdir(workspace):
        raise ValueError("Sandbox workspace must be an existing directory")

    args = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    for path in _SYSTEM_BINDS:
        if os.path.exists(path):
            args.extend(("--ro-bind", path, path))
    args.extend((
        "--proc", "/proc",
        "--dev", "/dev",
        "--size", str(MAX_SANDBOX_TMP_BYTES),
        "--tmpfs", "/tmp",
        "--dir", "/workspace",
        "--ro-bind", workspace, "/workspace",
        "--chdir", "/workspace",
        "--clearenv",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "HOME", "/tmp",
        "--setenv", "WORKFLOW_ROOT", "/workspace",
        "--", prlimit,
        f"--nproc={MAX_SANDBOX_PROCESSES}:{MAX_SANDBOX_PROCESSES}",
        f"--nofile={MAX_SANDBOX_OPEN_FILES}:{MAX_SANDBOX_OPEN_FILES}",
        f"--fsize={MAX_SANDBOX_FILE_BYTES}:{MAX_SANDBOX_FILE_BYTES}",
        f"--cpu={MAX_SANDBOX_CPU_SECONDS}:{MAX_SANDBOX_CPU_SECONDS}",
        "--", "/bin/sh", "-c", command,
    ))
    return args


__all__ = ["build_readonly_sandbox_command"]