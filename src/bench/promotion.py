"""Automatic, fail-closed promotion using a durable evaluation authorization.

Opt in with SRI_VCL_ENABLED=1. Without a trusted replay runner and curated suite,
proposals are HELD, not applied. Policies, runner pinning and benchmark assertions
are operator-owned, never candidate-generated or candidate-editable.
"""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import db
from bench.vcl import EvaluationError, Policy, evaluate, file_hash, tree_hash, target_relative

SCHEMA = """
CREATE TABLE IF NOT EXISTS vcl_receipts (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 proposal_id INTEGER NOT NULL,
 recorded_at REAL NOT NULL,
 decision TEXT NOT NULL,
 reason TEXT NOT NULL,
 proof_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vcl_claims (
 proposal_id INTEGER PRIMARY KEY,
 claimed_at REAL NOT NULL,
 reason TEXT NOT NULL DEFAULT 'evaluating'
);
"""

def enabled() -> bool:
    return os.environ.get("SRI_VCL_ENABLED") == "1"

def initialize() -> None:
    with db.conn() as conn:
        conn.executescript(SCHEMA)

def _claim(proposal_id: int) -> bool:
    """Atomically reserve one proposal; a crash leaves it held for inspection.

    This deliberately favors fail-closed recovery over automatic retry when
    a worker crashes halfway through a potentially side-effecting replay.
    """
    with db.conn() as conn:
        result = conn.execute(
            "UPDATE proposals SET status='evaluating' WHERE id=? AND status='pending'",
            (proposal_id,),
        )
        if result.rowcount == 1:
            conn.execute("INSERT INTO vcl_claims (proposal_id,claimed_at) VALUES (?,?)",
                         (proposal_id, time.time()))
            return True
        return False

def recover_interrupted(max_age_seconds: int = 3600) -> int:
    """Hold abandoned evaluations, never auto-retry potentially side-effecting runs."""
    initialize()
    with db.conn() as conn:
        abandoned = conn.execute(
            """SELECT proposal_id FROM vcl_claims WHERE claimed_at < ? AND
               proposal_id IN (SELECT id FROM proposals WHERE status='evaluating')""",
            (time.time() - max_age_seconds,)).fetchall()
        for row in abandoned:
            pid = row["proposal_id"]
            conn.execute("UPDATE proposals SET status='held' WHERE id=? AND status='evaluating'", (pid,))
            conn.execute("UPDATE vcl_claims SET reason='abandoned' WHERE proposal_id=?", (pid,))
            conn.execute("INSERT INTO vcl_receipts (proposal_id,recorded_at,decision,reason,proof_json) "
                         "VALUES (?,?,?,?,?)", (pid,time.time(),'hold','interrupted evaluation', '{}'))
        return len(abandoned)

def _receipt(proposal_id: int, decision: str, reason: str, proof: dict) -> None:
    with db.conn() as conn:
        conn.execute(
            "INSERT INTO vcl_receipts (proposal_id, recorded_at, decision, reason, proof_json) VALUES (?, ?, ?, ?, ?)",
            (proposal_id, time.time(), decision, reason, json.dumps(proof, sort_keys=True)),
        )
        conn.execute("UPDATE vcl_claims SET reason='completed' WHERE proposal_id=?", (proposal_id,))
        if decision == "promote":
            conn.execute("UPDATE proposals SET status='merged', merged_at=? WHERE id=? AND status='evaluating'",
                         (time.time(), proposal_id))
        else:
            conn.execute("UPDATE proposals SET status='held' WHERE id=? AND status='evaluating'",
                         (proposal_id,))
    # Feedback from genuine comparative evaluations reaches the existing
    # proposer through lessons_learned. Keep holdout case details private:
    # the proposer sees only aggregate outcomes and broad failure classes.
    if proof.get("suite_sha256"):
        category = "gap" if decision in ("reject", "hold") else "lesson"
        gain = proof.get("gain", "not_computable")
        message = (f"Verified evaluation proposal #{proposal_id}: {decision}; "
                   f"reason={reason}; aggregate_gain={gain}; "
                   f"candidate_score={proof.get('candidate_score', 'not_computable')}. "
                   "Use fresh session evidence before proposing a correction.")
        db.add_lesson(category=category, content=message, source=f"vcl:{proposal_id}")

def run() -> dict:
    if not enabled():
        return {"enabled": False}
    initialize()
    recovered = recover_interrupted()
    with db.conn() as conn:
        rows = conn.execute(
            """SELECT p.*, j.verdict AS judge_verdict, j.judge_error AS judge_error
               FROM proposals p JOIN judge_verdicts j ON j.id=(
                 SELECT id FROM judge_verdicts WHERE proposal_id=p.id ORDER BY id DESC LIMIT 1)
               WHERE p.status='pending' ORDER BY p.id"""
        ).fetchall()
    summary = {"promote": 0, "reject": 0, "hold": 0, "recovered": recovered}
    for row in rows:
        p = dict(row)
        if not _claim(int(p["id"])):
            continue  # another evaluator claimed it
        try:
            if p["judge_verdict"] != "approve" or p["judge_error"]:
                decision = "hold"
                report = {"reason": "independent judge did not approve cleanly"}
            elif p["target_kind"] != "skill_patch":
                decision = "hold"
                report = {"reason": "target kind requires a separate operator policy"}
            else:
                required = ["SRI_VCL_SUITE", "SRI_VCL_RUNNER", "SRI_VCL_RUNNER_SHA256"]
                missing = [name for name in required if not os.environ.get(name)]
                if missing:
                    raise EvaluationError("missing trusted evaluation configuration: " + ", ".join(missing))
                home = Path.home() / ".hermes"
                report = evaluate(
                    p, suite_path=Path(os.environ["SRI_VCL_SUITE"]),
                    runner_path=Path(os.environ["SRI_VCL_RUNNER"]),
                    runner_sha256=os.environ["SRI_VCL_RUNNER_SHA256"],
                    skills_root=home / "skills", memories_root=home / "memories")
                decision = report["decision"]
        except (EvaluationError, OSError, ValueError, KeyError) as exc:
            decision = "hold"
            report = {"reason": f"unverifiable: {exc}"}
        _receipt(int(p["id"]), decision, str(report.get("reason", "")), report)
        summary[decision] += 1
    return summary

def authorization(proposal: dict) -> bool:
    """Check proof binds the exact approved patch to the live baseline.

    This is checked directly before mutation, including for manually merged
    proposals: Telegram approvals cannot bypass evaluation when VCL is enabled.
    """
    if not enabled():
        return True
    initialize()
    with db.conn() as conn:
        row = conn.execute(
            "SELECT decision, proof_json FROM vcl_receipts WHERE proposal_id=? ORDER BY id DESC LIMIT 1",
            (proposal["id"],),
        ).fetchone()
    if row is None or row["decision"] != "promote":
        return False
    try:
        proof = json.loads(row["proof_json"])
        return (
            proof.get("decision") == "promote"
            and proposal.get("target_kind") == "skill_patch"
            and proof.get("target_relative_path") == target_relative(
                proposal["target_path"], Path.home() / ".hermes" / "skills"
            ).as_posix()
            and proof.get("diff_sha256") == hashlib.sha256(proposal["diff"].encode()).hexdigest()
            and proof.get("baseline_file_sha256") == file_hash(Path(proposal["target_path"]))
            and proof.get("baseline_skills_tree_sha256") == tree_hash(Path.home() / ".hermes" / "skills")
            and proof.get("baseline_memories_tree_sha256") == tree_hash(Path.home() / ".hermes" / "memories")
            and proof.get("suite_sha256") == file_hash(Path(os.environ["SRI_VCL_SUITE"]))
            and proof.get("runner_sha256") == file_hash(Path(os.environ["SRI_VCL_RUNNER"]))
            and proof.get("policy_sha256") == Policy().digest()
        )
    except (ValueError, KeyError, OSError, TypeError):
        return False

def post_apply_matches(proposal: dict) -> bool:
    if not enabled():
        return True
    with db.conn() as conn:
        row = conn.execute("SELECT proof_json FROM vcl_receipts WHERE proposal_id=? ORDER BY id DESC LIMIT 1",
                           (proposal["id"],)).fetchone()
    if row is None:
        return False
    try:
        return json.loads(row["proof_json"]).get("candidate_file_sha256") == file_hash(Path(proposal["target_path"]))
    except (ValueError, TypeError):
        return False
