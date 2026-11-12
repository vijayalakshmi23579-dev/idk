"""Focused, offline tests for memory path and freshness behavior."""
import os
import tempfile
from unittest.mock import patch

from memdir import (
    MemoryFile,
    get_memory_dir,
    memory_freshness_note,
    validate_memory_path,
    write_memory,
)


def main():
    for path in ("relative/path", "/", "/a", "//server/share", "bad\0path"):
        try:
            validate_memory_path(path)
            raise AssertionError(f"accepted unsafe memory path: {path!r}")
        except ValueError:
            pass
    print("Memory path validation OK")

    with tempfile.TemporaryDirectory() as temporary:
        project = os.path.join(temporary, "project")
        os.makedirs(project)
        with patch.dict(os.environ, {}, clear=True):
            resolved = get_memory_dir(project)
            assert resolved == os.path.join(project, ".claude", "memory")
            try:
                get_memory_dir(project, project_setting="/tmp/redirect")
                raise AssertionError("project setting redirected memory path")
            except ValueError:
                pass

        remote_dir = os.path.join(temporary, "remote-memory")
        with patch.dict(os.environ, {
            "CLAUDE_CODE_REMOTE": "1", "REMOTE_MEMORY_DIR": remote_dir,
        }, clear=True):
            assert get_memory_dir(project) == remote_dir

        memory = MemoryFile(
            path="unused.md", name="Old note", description="test",
            type="project", mtime=1_000.0,
        )
        assert memory_freshness_note(memory, now=1_000.0 + 86_401) is not None
        assert memory_freshness_note(memory, now=1_000.0 + 86_400) is None
        print("Memory freshness threshold OK")

        target = write_memory(resolved, "Stable Note", "stable", "project", "body")
        with patch("memdir.os.replace", side_effect=AssertionError("rewrote unchanged memory")):
            assert write_memory(resolved, "Stable Note", "stable", "project", "body") == target
        print("Unchanged memory write skipped OK")

        root = os.path.join(temporary, "symlink-project")
        outside = os.path.join(temporary, "outside")
        os.makedirs(os.path.join(root, ".claude"))
        os.makedirs(outside)
        os.rmdir(os.path.join(root, ".claude"))
        try:
            os.symlink(outside, os.path.join(root, ".claude"))
        except OSError:
            print("Symlink escape check skipped (symlinks unavailable)")
        else:
            try:
                get_memory_dir(root)
                raise AssertionError("project memory symlink escaped project")
            except ValueError:
                pass
            print("Project memory symlink escape blocked OK")


if __name__ == "__main__":
    main()