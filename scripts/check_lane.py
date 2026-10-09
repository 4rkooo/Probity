"""Person 2 lane checker: run before every lane merge.

    uv run python scripts/check_lane.py --lane b --base person2/trueframe

Fails when any change since the merge base with ``--base`` (committed, staged, unstaged, or
untracked) is outside the lane allowlist in ``scripts/lanes.json``, when a frozen file changed,
when golden or frozen hashes/signatures mismatch, or when the full ``pytest -q`` is not green.

``--write-hashes`` regenerates ``golden_hashes.json`` and ``frozen_hashes.json``; integrator only
(those files are frozen for every lane, so a lane that runs it still fails the scope check).

Pure Python, Windows-safe: pathlib only, subprocess argument lists, no shell. Text files are
hashed with CRLF normalized to LF so ``core.autocrlf`` checkouts hash the same as the repo blobs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LANES_FILE = REPO / "scripts" / "lanes.json"
GOLDEN_FILE = REPO / "scripts" / "golden_hashes.json"
FROZEN_FILE = REPO / "scripts" / "frozen_hashes.json"

GOLDEN_ROOTS = ("fixtures/demo", "fixtures/synthetic")
FROZEN_FILES = (
    "src/probity/domain/models.py",
    "src/probity/domain/enums.py",
    "src/probity/ports.py",
    "config/policy.demo.yaml",
    "contracts/openapi.json",
    "contracts/schemas/*.schema.json",
    "fixtures/schema/*.schema.json",
    "pyproject.toml",
    "uv.lock",
    "src/probity/reconstruction/types.py",
    "src/probity/eval/synth.py",
    "tests/unit/reconstruction/conftest.py",
    "tests/unit/reconstruction/test_guards.py",
    "scripts/check_lane.py",
    "scripts/lanes.json",
)
SIGNATURE_MODULES = (
    "src/probity/reconstruction/align.py",
    "src/probity/reconstruction/color.py",
    "src/probity/reconstruction/fuse.py",
    "src/probity/reconstruction/provenance.py",
    "src/probity/reconstruction/integrity.py",
    "src/probity/reconstruction/baseline.py",
    "src/probity/eval/metrics.py",
)
TEXT_SUFFIXES = frozenset({
    ".json", ".yaml", ".yml", ".md", ".py", ".txt", ".toml", ".lock", ".csv", ".ps1", ".cfg",
    ".ini", ".html",
})


# ------------------------------------------------------------------------------------------------
# Hashing and signatures (also imported by tests/unit/reconstruction/test_guards.py)
# ------------------------------------------------------------------------------------------------


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitattributes", ".gitignore"}


def normalized_sha256(path: Path) -> str:
    data = path.read_bytes()
    if is_text(path):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def rel(path: Path, root: Path = REPO) -> str:
    return path.relative_to(root).as_posix()


def golden_files(root: Path = REPO) -> list[Path]:
    out: list[Path] = []
    for top in GOLDEN_ROOTS:
        base = root / top
        if base.is_dir():
            out.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(out, key=lambda p: rel(p, root))


def frozen_files(root: Path = REPO) -> list[Path]:
    out: list[Path] = []
    for pattern in FROZEN_FILES:
        out.extend(p for p in root.glob(pattern) if p.is_file())
    return sorted(set(out), key=lambda p: rel(p, root))


def hash_map(paths: Iterable[Path], root: Path = REPO) -> dict[str, str]:
    return {rel(p, root): normalized_sha256(p) for p in paths}


def _args_sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}({ast.unparse(node.args)}){ret}"


def public_signatures(source: str) -> dict[str, str]:
    """Top-level public functions and classes: argument lists, annotations, return types, and
    class attribute annotations / method signatures. Docstrings and bodies are ignored."""
    sigs: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                sigs[node.name] = _args_sig(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            parts = [f"bases=({', '.join(ast.unparse(b) for b in node.bases)})"]
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    parts.append(f"{item.target.id}: {ast.unparse(item.annotation)}")
                elif (isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                      and not item.name.startswith("_")):
                    parts.append(f"def {item.name}{_args_sig(item)}")
            sigs[node.name] = "; ".join(parts)
    return sigs


def signature_map(root: Path = REPO) -> dict[str, dict[str, str]]:
    return {m: public_signatures((root / m).read_text(encoding="utf-8"))
            for m in SIGNATURE_MODULES if (root / m).is_file()}


def compare_hashes(recorded: dict[str, str], actual: dict[str, str]) -> list[str]:
    problems = [f"missing: {p}" for p in sorted(set(recorded) - set(actual))]
    problems += [f"unexpected: {p}" for p in sorted(set(actual) - set(recorded))]
    problems += [f"changed: {p}" for p in sorted(set(recorded) & set(actual))
                 if recorded[p] != actual[p]]
    return problems


def compare_signatures(recorded: dict[str, dict[str, str]],
                       actual: dict[str, dict[str, str]]) -> list[str]:
    """Every recorded public name must still exist with an identical signature (additions ok)."""
    problems: list[str] = []
    for module, names in sorted(recorded.items()):
        have = actual.get(module)
        if have is None:
            problems.append(f"missing module: {module}")
            continue
        for name, sig in sorted(names.items()):
            if name not in have:
                problems.append(f"{module}: {name} removed")
            elif have[name] != sig:
                problems.append(f"{module}: {name} changed\n      frozen: {sig}\n      now:    "
                                f"{have[name]}")
    return problems


def golden_problems(root: Path = REPO) -> list[str]:
    recorded = json.loads((root / "scripts" / "golden_hashes.json").read_text(encoding="utf-8"))
    return compare_hashes(recorded["files"], hash_map(golden_files(root), root))


def frozen_problems(root: Path = REPO) -> list[str]:
    recorded = json.loads((root / "scripts" / "frozen_hashes.json").read_text(encoding="utf-8"))
    problems = compare_hashes(recorded["files"], hash_map(frozen_files(root), root))
    return problems + compare_signatures(recorded["signatures"], signature_map(root))


def write_json(path: Path, data: object) -> None:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_hashes(root: Path = REPO) -> None:
    note = "SHA-256; text files hashed with CRLF normalized to LF (see scripts/check_lane.py)"
    write_json(root / "scripts" / "golden_hashes.json",
               {"version": 1, "normalization": note, "roots": list(GOLDEN_ROOTS),
                "files": hash_map(golden_files(root), root)})
    write_json(root / "scripts" / "frozen_hashes.json",
               {"version": 1, "normalization": note, "files": hash_map(frozen_files(root), root),
                "signatures": signature_map(root)})


# ------------------------------------------------------------------------------------------------
# Lane scope
# ------------------------------------------------------------------------------------------------


def glob_regex(pattern: str) -> re.Pattern[str]:
    """``**`` spans directories, ``*`` and ``?`` stay within one path segment."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(f"^{out}$")


def matches(path: str, patterns: Sequence[str]) -> bool:
    return any(glob_regex(p).match(path) for p in patterns)


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"git {' '.join(args)} failed: {msg}")
    return proc.stdout.decode("utf-8", errors="replace")


def _names(output: str) -> set[str]:
    return {n for n in output.split("\0") if n}


def changed_files(base: str) -> tuple[str, set[str]]:
    git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
    merge_base = git("merge-base", base, "HEAD").strip()
    changed = _names(git("diff", "--name-only", "--no-renames", "-z", merge_base, "HEAD"))
    changed |= _names(git("diff", "--name-only", "--no-renames", "-z", "HEAD"))
    changed |= _names(git("ls-files", "--others", "--exclude-standard", "-z"))
    return merge_base, changed


@dataclass
class Check:
    name: str
    problems: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


def scope_check(lane: dict[str, list[str]], frozen: list[str], changed: set[str]) -> Check:
    check = Check("scope: changes inside lane allowlist, no frozen file touched")
    exempt = lane.get("frozen_exempt", [])
    for path in sorted(changed):
        if matches(path, frozen) and not matches(path, exempt):
            check.problems.append(f"FROZEN file changed: {path}")
        elif not matches(path, lane["allow"]) and not matches(path, exempt):
            check.problems.append(f"outside lane allowlist: {path}")
    return check


def run_tests() -> Check:
    check = Check("tests: full `pytest -q` green")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=REPO, capture_output=True, check=False)
    if proc.returncode != 0:
        tail = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()[-25:]
        check.problems = [f"pytest exit code {proc.returncode}", *tail]
    return check


def main(argv: Sequence[str] | None = None) -> int:
    config = json.loads(LANES_FILE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=sorted(config["lanes"]))
    parser.add_argument("--base", default=config["base"])
    parser.add_argument("--skip-tests", action="store_true", help="debug only; never for merges")
    parser.add_argument("--write-hashes", action="store_true", help="integrator only")
    args = parser.parse_args(argv)

    if args.write_hashes:
        write_hashes()
        print("wrote scripts/golden_hashes.json and scripts/frozen_hashes.json")
        return 0
    if not args.lane:
        parser.error("--lane is required")

    lane = config["lanes"][args.lane]
    merge_base, changed = changed_files(args.base)
    checks = [scope_check(lane, config["frozen"], changed)]
    checks.append(Check("golden fixture hashes (scripts/golden_hashes.json)", golden_problems()))
    checks.append(Check("frozen contract hashes + signatures (scripts/frozen_hashes.json)",
                        frozen_problems()))
    checks.append(Check("tests: full `pytest -q` green", skipped=True) if args.skip_tests
                  else run_tests())

    print(f"Lane {args.lane}: {lane['title']}")
    print(f"Base {args.base} (merge base {merge_base[:12]}); {len(changed)} changed path(s)")
    for path in sorted(changed):
        print(f"  {path}")
    print()
    for check in checks:
        status = "SKIP" if check.skipped else ("PASS" if check.ok else "FAIL")
        print(f"[{status}] {check.name}")
        for problem in check.problems:
            print(f"    {problem}")
    failed = [c for c in checks if not c.ok or c.skipped]
    print()
    if failed:
        print(f"LANE CHECK FAILED ({len(failed)} of {len(checks)} checks not passing)")
        return 1
    print("LANE CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
