"""File and provenance primitives for the external benchmark recipe (stdlib only)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class BusyError(ValueError):
    """A second writer must not append diagnostics to another process's active run."""


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def decode_json(text: str, origin="JSON"):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {origin}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"invalid JSON numeric constant {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=constant)


def read_json(path: Path):
    return decode_json(path.read_text(), str(path))


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def atomic_file(path: Path, write) -> None:
    """Publish a complete file without replacing an existing result, even after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            write(output)
            output.flush()
            os.fsync(output.fileno())
        # link is create-new and atomic; replace/rename could overwrite a competing writer.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path: Path):
    """POSIX process lock; released by the OS on crash, with no stale-lock deletion needed."""
    import fcntl
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BusyError(f"another process is using {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def dataset_hashes(root: Path, split="test") -> dict[str, str]:
    if not split or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in split):
        raise ValueError("split must be one ASCII filename component")
    return {name: sha256(root / name) for name in ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv")}


def source_hashes() -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))}


def versions(names) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def provenance(binary: Path | None = None) -> dict:
    def git(*args, cwd=REPO):
        try:
            return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    try:
        physical_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        physical_memory = None
    cpu_brand = platform.processor() or None
    if platform.system() == "Darwin":
        try:
            cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                                               stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "source_commit": git("rev-parse", "HEAD"),
        "cpu_brand": cpu_brand, "physical_memory_bytes": physical_memory,
        "python_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"], cwd=REPO
        )).hexdigest() if git("rev-parse", "HEAD") else None,
        "recipe_sha256": source_hashes(),
        "revision_is_build_attestation": False,
        "cubecl_commit": git("rev-parse", "HEAD", cwd=REPO.parent / "cubecl"),
        "cubecl_tracked_status": git("status", "--porcelain", "--untracked-files=no", cwd=REPO.parent / "cubecl"),
    }
    if binary:
        result.update(binary_sha256=sha256(binary), binary_name=binary.name)
    return result


def capture(output: Path, binary: Path | None = None) -> None:
    """Snapshot the small owned recipe and tracked source delta, including untracked helpers."""
    write_json(output / "provenance.json", provenance(binary))
    paths = [*HERE.glob("*.py"), *HERE.glob("specs/*.json"),
             REPO / "Cargo.lock", REPO / "Cargo.toml", REPO / "crates/cubeql-code/Cargo.toml",
             REPO / "crates/cubeql-code/examples/benchmark_replay.rs"]
    for path in paths:
        if not path.is_file():
            continue
        target = output / "recipe-source" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as out:
            out.write(path.read_bytes())
    for label, root in [("cubeql", REPO), ("cubecl", REPO.parent / "cubecl")]:
        try:
            patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        with (output / f"{label}-tracked.patch").open("xb") as out:
            out.write(patch)


def start_directory(output: Path, request: dict, resume=False) -> None:
    if output.exists():
        if not resume or not output.is_dir():
            raise ValueError(f"output already exists: {output}; use a new directory")
        if (output / "manifest.json").exists():
            raise ValueError("run already completed; verify it instead of running it again")
        if read_json(output / "request.json") != request:
            raise ValueError("resume configuration, inputs, model or recipe changed")
        return
    if resume:
        raise ValueError("--resume requires an existing incomplete run")
    output.mkdir(parents=True)
    write_json(output / "request.json", request)


def failure(output: Path, error: Exception) -> None:
    """Retain every failed attempt instead of replacing the preceding diagnostic."""
    for attempt in range(1, 10000):
        path = output / f"failure-{attempt:04}.json"
        try:
            write_json(path, {"status": "failed", "error": str(error), "error_type": type(error).__name__,
                              "recorded_utc": datetime.now(timezone.utc).isoformat()})
            return
        except FileExistsError:
            continue
    raise ValueError("too many failed attempts")


def seal(directory: Path, kind: str, excludes=()) -> None:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"archive contains a symlink: {path}")
        relative = str(path.relative_to(directory))
        if path.is_file() and relative not in {"manifest.json", *excludes}:
            files[relative] = sha256(path)
    manifest = {"format_version": 1, "kind": kind, "status": "complete",
                "files_sha256": files, "mutable_exclusions": list(excludes)}
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    atomic_file(directory / "manifest.json", lambda out: out.write(payload))


def verify_seal(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("unsupported or incomplete archive")
    expected = manifest["files_sha256"]
    if not expected:
        raise ValueError("empty archive manifest")
    for name, digest in expected.items():
        path = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError(f"invalid archive member: {name}")
        if sha256(path) != digest:
            raise ValueError(f"checksum mismatch: {name}")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"archive contains a symlink: {path}")
        relative = str(path.relative_to(directory))
        if path.is_file() and relative not in {"manifest.json", *manifest.get("mutable_exclusions", [])}:
            actual.add(relative)
    if actual != set(expected):
        raise ValueError("archive members differ from manifest")
    return {"status": "verified", "kind": manifest["kind"], "files": len(expected)}


def run_process(command: list[str], output: Path) -> None:
    """No short inference timeout; retain stdout, stderr and exit status before raising."""
    with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    write_json(output / "execution.json", {"command": command, "started_utc": started,
               "finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": result.returncode})
    if result.returncode:
        raise ValueError(f"command failed ({result.returncode}); see {output / 'stderr.log'}")
