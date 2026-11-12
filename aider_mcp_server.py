#!/usr/bin/env python3
"""Aider wrapped as a FastMCP tool for surgical code edits.

The main LLM never reads the full file: Aider builds a tree-sitter repo map,
applies search/replace patches, and commits to git. Token savings on large
repos are typically 4-10x vs. full-file rewrite.

Install:
    pip install aider-chat fastmcp

Run:
    python aider_mcp_server.py    # stdio transport

Then add to .chainlit/config.toml:
    [[features.mcp.servers]]
        name = "aider"
        type = "stdio"
        command = "python /workspaces/idk/aider_mcp_server.py"
        env = { GEMINI_API_KEY = "..." }
"""
from __future__ import annotations

import os
from pathlib import Path

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from fastmcp import FastMCP

mcp = FastMCP("aider-editor")

# Default model: read from env, fall back to Gemini
DEFAULT_MODEL = os.environ.get("AIDER_MODEL", "gemini/gemini-2.5-flash")


@mcp.tool
def surgical_edit(repo_path: str, files: list[str], instruction: str) -> str:
    """Apply a surgical edit using Aider's tree-sitter repo map + search/replace diffs.

    Args:
        repo_path: Absolute path to a local git repository.
        files: List of files to include in Aider's active workspace.
        instruction: Natural-language instruction (e.g. "change the timeout to 60s").

    Returns:
        The Aider diff/response as text, or an ERROR string.

    Notes:
        - Requires a git repository; Aider uses git to compute diffs and commit.
        - Only the repo map is read into context, not full files. This is the
          token-saving property.
        - The change is auto-committed to git, so you can review with `git log`.
    """
    repo = Path(repo_path).resolve()
    if not (repo / ".git").is_dir():
        return f"ERROR: {repo} is not a git repository (no .git directory)"

    file_paths = []
    for f in files:
        p = repo / f if not Path(f).is_absolute() else Path(f)
        if not p.is_file():
            return f"ERROR: file not found: {p}"
        file_paths.append(str(p))

    # Silent IO — no terminal output, but keep the diff text for the caller
    io = InputOutput(yes=True, pretty=False, fancy_input=False)

    try:
        model = Model(DEFAULT_MODEL)
        coder = Coder.create(
            main_model=model,
            fnames=file_paths,
            io=io,
            auto_commits=True,
            show_diffs=True,
            stream=False,
        )
        result = coder.run(with_message=instruction)
        return f"Aider result:\n{result}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    mcp.run()