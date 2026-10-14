"""Verified Continuous Learning: paired, independently checked replay.

The replay executable is an OPERATOR-PINNED TRUST BOUNDARY: this module does not
pretend that subprocess/chdir is a security sandbox. In production the pinned
runner MUST isolate agent execution (e.g. through Medusa containment), report
tool-observed results, and never expose eval assertions to the candidate.
Without a configured, SHA256-pinned runner the gate stays closed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MAX_SUITE_BYTES = 1_000_000
MAX_FILES = 20_000
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RESULT_BYTES = 128 * 1024
ALLOWED_OPS = frozenset({"eq", "contains", "not_contains", "exists"})
FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*$")
CASE_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,100}$")

class EvaluationError(Exception):
    """A failure to produce trustworthy evaluation evidence; always hold."""

@dataclass(frozen=True)
class Check:
    field: str
    op: str
    expected: Any = None
    critical: bool = False

@dataclass(frozen=True)
class Case:
    case_id: str
    prompt: str
    checks: tuple[Check, ...]

@dataclass(frozen=True)
class Suite:
    version: str
    cases: tuple[Case, ...]
    digest: str

@dataclass(frozen=True)
class Policy:
    min_cases: int = 10
    min_gain: float = 0.05
    min_candidate_score: float = 0.85
    min_improved_cases: int = 2
    max_regressions: int = 0

    def digest(self) -> str:
        return _hash(json.dumps(self.__dict__, sort_keys=True).encode())

def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _json(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))

def load_suite(path: Path) -> Suite:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_SUITE_BYTES:
        raise EvaluationError("suite missing, symlinked, or oversized")
    data = path.read_bytes()
    try:
        raw = _json(data)
        if set(raw) != {"version", "cases"} or not isinstance(raw["cases"], list):
            raise ValueError("invalid suite shape")
        if not isinstance(raw["version"], str) or not CASE_RE.fullmatch(raw["version"]):
            raise ValueError("invalid suite version")
        cases = []
        seen = set()
        for item in raw["cases"]:
            if set(item) != {"id", "prompt", "checks"}:
                raise ValueError("invalid case fields")
            cid, prompt = item["id"], item["prompt"]
            if not isinstance(cid, str) or not CASE_RE.fullmatch(cid) or cid in seen:
                raise ValueError("duplicate or invalid case id")
            if not isinstance(prompt, str) or not (1 <= len(prompt) <= 12000):
                raise ValueError("invalid case prompt")
            if not isinstance(item["checks"], list) or not item["checks"]:
                raise ValueError("case must contain checks")
            seen.add(cid)
            checks = []
            for c in item["checks"]:
                if not isinstance(c, dict) or not {"field", "op"} <= set(c) or set(c) - {"field", "op", "expected", "critical"}:
                    raise ValueError("invalid check shape")
                if not isinstance(c["field"], str) or not FIELD_RE.fullmatch(c["field"]):
                    raise ValueError("unsafe field name")
                if c["op"] not in ALLOWED_OPS or (c["op"] != "exists" and "expected" not in c):
                    raise ValueError("invalid check operation")
                if not isinstance(c.get("critical", False), bool):
                    raise ValueError("critical must be boolean")
                checks.append(Check(c["field"], c["op"], c.get("expected"), c.get("critical", False)))
            cases.append(Case(cid, prompt, tuple(checks)))
        if len(cases) > 500:
            raise ValueError("suite too large")
        return Suite(raw["version"], tuple(cases), _hash(data))
    except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid evaluation suite: {exc}") from exc

def _lookup(output: Mapping[str, Any], field: str) -> tuple[bool, Any]:
    value: Any = output
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value

def score_case(case: Case, output: Mapping[str, Any]) -> dict:
    if not isinstance(output, dict):
        raise EvaluationError("runner output is not an object")
    passed = 0
    critical_failures = []
    failed_checks = []
    for check in case.checks:
        exists, value = _lookup(output, check.field)
        if check.op == "exists":
            ok = exists and value is not None
        elif not exists:
            ok = False
        elif check.op == "eq":
            ok = type(value) is type(check.expected) and value == check.expected
        elif check.op in ("contains", "not_contains"):
            if not isinstance(value, str) or not isinstance(check.expected, str):
                ok = False
            else:
                found = check.expected in value
                ok = found if check.op == "contains" else not found
        else:
            raise EvaluationError("unknown check operation")
        passed += int(ok)
        if not ok:
            failed_checks.append(check.field)
            if check.critical:
                critical_failures.append(check.field)
    return {"case_id": case.case_id, "score": passed / len(case.checks),
            "failed_checks": failed_checks, "critical_failures": critical_failures}

def decide(baseline: list[dict], candidate: list[dict], policy: Policy = Policy()) -> dict:
    if len(baseline) != len(candidate) or len(baseline) < policy.min_cases:
        return {"decision": "hold", "reason": "insufficient or mismatched paired cases"}
    if [r["case_id"] for r in baseline] != [r["case_id"] for r in candidate]:
        return {"decision": "hold", "reason": "case order or identity mismatch"}
    if any(r["critical_failures"] for r in candidate):
        return {"decision": "reject", "reason": "candidate violated a critical invariant"}
    deltas = [c["score"] - b["score"] for b, c in zip(baseline, candidate)]
    improved = sum(d > 1e-9 for d in deltas)
    regressed = sum(d < -1e-9 for d in deltas)
    gain = sum(deltas) / len(deltas)
    candidate_score = sum(r["score"] for r in candidate) / len(candidate)
    summary = {"gain": round(gain, 6), "candidate_score": round(candidate_score, 6),
               "improved": improved, "regressed": regressed, "cases": len(deltas)}
    if regressed > policy.max_regressions:
        return {**summary, "decision": "reject", "reason": "paired-case regression"}
    if (gain < policy.min_gain or improved < policy.min_improved_cases
            or candidate_score < policy.min_candidate_score):
        return {**summary, "decision": "hold", "reason": "positive improvement not established"}
    return {**summary, "decision": "promote", "reason": "paired improvement and safety gates passed"}

def file_hash(path: Path) -> str:
    return _hash(path.read_bytes()) if path.is_file() and not path.is_symlink() else ""

def target_relative(target_path: str, skills_root: Path) -> Path:
    """Only existing, regular, non-symlinked SKILL.md files can be auto-patched."""
    root = skills_root.resolve(strict=True)
    raw = Path(target_path)
    if not raw.is_absolute() or raw.name != "SKILL.md" or raw.is_symlink():
        raise EvaluationError("automatic learning requires an existing SKILL.md")
    try:
        rel = raw.resolve(strict=True).relative_to(root)
    except (ValueError, OSError) as exc:
        raise EvaluationError("target escapes approved skills directory") from exc
    if len(rel.parts) < 2 or any(p in (".", "..") for p in rel.parts):
        raise EvaluationError("invalid skills target")
    return rel

def apply_candidate_copy(path: Path, diff: str) -> None:
    if not isinstance(diff, str) or not (1 <= len(diff) <= 50_000):
        raise EvaluationError("invalid patch size")
    old = path.read_text(encoding="utf-8")
    if diff.startswith("APPEND:"):
        new = old + "\n" + diff[len("APPEND:"):].lstrip("\n")
    elif diff.startswith("<<<\n") and diff.endswith("\n>>>") and "\n===\n" in diff:
        before, after = diff[4:-4].split("\n===\n", 1)
        if not before or old.count(before) != 1:
            raise EvaluationError("candidate patch context ambiguous or stale")
        new = old.replace(before, after, 1)
    else:
        raise EvaluationError("only explicit APPEND or exact old/new patches are auto-evaluable")
    if new == old:
        raise EvaluationError("candidate changes nothing")
    path.write_text(new, encoding="utf-8")

def _copy_limited(src: Path, dst: Path) -> None:
    """Create real copies, reject symlinks/special files/oversized snapshots."""
    if not src.is_dir() or src.is_symlink():
        raise EvaluationError("input root missing or symlinked")
    count, size = 0, 0
    for directory, dirs, files in os.walk(src, followlinks=False):
        parent = Path(directory)
        dirs.sort()
        files.sort()
        for name in dirs + files:
            if (parent / name).is_symlink():
                raise EvaluationError("symlink in agent configuration")
        dest = dst / parent.relative_to(src)
        dest.mkdir(parents=True, exist_ok=True)
        for name in files:
            path = parent / name
            if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
                raise EvaluationError("special file in agent configuration")
            count += 1
            size += path.stat().st_size
            if count > MAX_FILES or size > MAX_INPUT_BYTES:
                raise EvaluationError("agent configuration exceeds snapshot limits")
            shutil.copyfile(path, dest / name)

def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in sorted(dirs + files):
            p = parent / name
            if p.is_symlink():
                raise EvaluationError("runner altered snapshot with symlink")
        for name in sorted(files):
            p = parent / name
            if not p.is_file():
                raise EvaluationError("runner inserted non-file")
            h.update(str(p.relative_to(root)).encode())
            h.update(b"\0")
            h.update(bytes.fromhex(file_hash(p)))
    return h.hexdigest()

def _runner(binary: Path, expected_hash: str, case: Case,
            skills: Path, memories: Path, seed: int, workdir: Path) -> dict:
    if not binary.is_file() or binary.is_symlink() or file_hash(binary) != expected_hash:
        raise EvaluationError("trusted runner missing or SHA256 does not match")
    workdir.mkdir(parents=True, exist_ok=True)
    input_file, result_file = workdir / "input.json", workdir / "result.json"
    input_file.write_text(json.dumps({"case_id": case.case_id, "prompt": case.prompt}))
    # Runner never sees the checks, evaluation policy, or expected answer.
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir),
           "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [str(binary), "--input", str(input_file), "--skills-dir", str(skills),
               "--memories-dir", str(memories), "--output", str(result_file),
               "--seed", str(seed)]
    try:
        result = subprocess.run(command, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvaluationError(f"trusted runner failed: {type(exc).__name__}") from exc
    if result.returncode != 0 or not result_file.is_file() or result_file.is_symlink():
        raise EvaluationError(f"trusted runner returned failure (code {result.returncode})")
    if result_file.stat().st_size > MAX_RESULT_BYTES:
        raise EvaluationError("runner output too large")
    try:
        output = _json(result_file.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise EvaluationError("runner returned invalid JSON") from exc
    if not isinstance(output, dict) or not isinstance(output.get("observations"), dict):
        raise EvaluationError("runner must return tool-observed observations")
    return output

def evaluate(proposal: dict, *, suite_path: Path, runner_path: Path,
             runner_sha256: str, skills_root: Path, memories_root: Path,
             policy: Policy = Policy()) -> dict:
    if proposal.get("target_kind") != "skill_patch":
        raise EvaluationError("automatic promotion supports skill_patch only")
    if not re.fullmatch(r"[a-f0-9]{64}", runner_sha256):
        raise EvaluationError("operator-pinned runner hash required")
    suite = load_suite(suite_path)
    if len(suite.cases) < policy.min_cases:
        raise EvaluationError("suite too small for automatic promotion")
    rel = target_relative(proposal["target_path"], skills_root)
    target = skills_root / rel
    baseline_file_sha = file_hash(target)
    if not baseline_file_sha:
        raise EvaluationError("baseline file missing")
    with tempfile.TemporaryDirectory(prefix="sri-vcl-") as temp:
        root = Path(temp)
        baseline, candidate = root / "baseline", root / "candidate"
        for name, dest in [("skills", baseline / "skills"), ("memories", baseline / "memories")]:
            _copy_limited(skills_root if name == "skills" else memories_root, dest)
        for name in ("skills", "memories"):
            _copy_limited(baseline / name, candidate / name)
        apply_candidate_copy(candidate / "skills" / rel, proposal["diff"])
        candidate_sha = file_hash(candidate / "skills" / rel)
        baseline_scores, candidate_scores = [], []
        for index, case in enumerate(suite.cases):
            for tag, source, scores in (("baseline", baseline, baseline_scores),
                                        ("candidate", candidate, candidate_scores)):
                work = root / "runs" / str(index) / tag
                for name in ("skills", "memories"):
                    _copy_limited(source / name, work / name)
                before = (tree_hash(work / "skills"), tree_hash(work / "memories"))
                output = _runner(runner_path, runner_sha256, case, work / "skills",
                                 work / "memories", index, work / "run")
                after = (tree_hash(work / "skills"), tree_hash(work / "memories"))
                if before != after:
                    raise EvaluationError("runner mutated evaluation input snapshot")
                scores.append(score_case(case, output))
        verdict = decide(baseline_scores, candidate_scores, policy)
        return {**verdict, "suite_version": suite.version, "suite_sha256": suite.digest,
                "runner_sha256": runner_sha256, "policy_sha256": policy.digest(),
                "baseline_file_sha256": baseline_file_sha, "candidate_file_sha256": candidate_sha,
                "diff_sha256": _hash(proposal["diff"].encode()),
                "baseline": baseline_scores, "candidate": candidate_scores}
