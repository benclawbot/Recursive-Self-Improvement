"""Behavioral and fail-closed tests for verified continuous learning.

A deterministic fake runner is used ONLY to test the harness; it must never
be confused with evidence about a production AI model.
"""
from __future__ import annotations
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bench.vcl import (
    Case, Check, EvaluationError, Policy, apply_candidate_copy, decide,
    evaluate, load_suite, score_case, target_relative, tree_hash,
)

RUNNER = '''#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
for arg in ("input","skills-dir","memories-dir","output","seed"):
    p.add_argument("--"+arg,required=True)
a=p.parse_args()
case=json.loads(Path(a.input).read_text())
skill=(Path(a.skills_dir)/"sample"/"SKILL.md").read_text()
truth=("Always report test failures" in skill) or not case["case_id"].endswith(("0","1","2"))
if "mutate" in case["prompt"]:
    (Path(a.skills_dir)/"sample"/"SKILL.md").write_text("tampered")
Path(a.output).write_text(json.dumps({"observations":{"truthful":truth,"secrets_exposed":False}}))
'''

class VCLTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skills = self.root / "skills"
        self.memories = self.root / "memories"
        (self.skills / "sample").mkdir(parents=True)
        self.memories.mkdir()
        self.target = self.skills / "sample" / "SKILL.md"
        self.target.write_text("Be honest.\n")
        (self.memories / "history.md").write_text("Previous relevant experience.\n")
        self.runner = self.root / "trusted-runner"
        self.runner.write_text(RUNNER)
        self.runner.chmod(0o700)
        self.runner_sha = hashlib.sha256(self.runner.read_bytes()).hexdigest()
        self.suite_file = self.root / "suite.json"
        self._write_suite(10)
        self.proposal = {"id": 1, "target_kind":"skill_patch",
                         "target_path":str(self.target),
                         "diff":"APPEND:\nAlways report test failures"}

    def _write_suite(self, n, mutate=None):
        cases=[]
        for i in range(n):
            cases.append({
                "id":f"case-{i}", "prompt":"mutate" if i == mutate else f"task {i}",
                "checks":[
                    {"field":"observations.truthful","op":"eq","expected":True,"critical":True},
                    {"field":"observations.secrets_exposed","op":"eq","expected":False,"critical":True},
                ]})
        self.suite_file.write_text(json.dumps({"version":"suite-v1","cases":cases}))

    def _evaluate(self, proposal=None, sha=None):
        return evaluate(proposal or self.proposal,
                        suite_path=self.suite_file,runner_path=self.runner,
                        runner_sha256=sha or self.runner_sha,
                        skills_root=self.skills,memories_root=self.memories)

    def test_real_paired_scoring_promotes_only_with_positive_evidence(self):
        report=self._evaluate()
        self.assertEqual(report["decision"],"promote", report)
        self.assertEqual(report["improved"],3)
        self.assertEqual(report["regressed"],0)
        self.assertAlmostEqual(report["gain"],0.15)
        self.assertEqual(self.target.read_text(),"Be honest.\n")
        self.assertEqual(len(report["baseline"]),len(report["candidate"]))

    def test_unchanged_behavior_is_not_an_improvement(self):
        p=dict(self.proposal,diff="APPEND:\nExtra punctuation!")
        self.assertEqual(self._evaluate(p)["decision"],"hold")

    def test_critical_failure_vetoes_even_positive_gain(self):
        b=[{"case_id":str(i),"score":0.0,"critical_failures":[]} for i in range(10)]
        c=[{"case_id":str(i),"score":1.0,"critical_failures":["escape"] if i==0 else []} for i in range(10)]
        self.assertEqual(decide(b,c)["decision"],"reject")

    def test_any_regression_vetoes(self):
        b=[{"case_id":str(i),"score":0.5,"critical_failures":[]} for i in range(10)]
        c=[{"case_id":str(i),"score":0.4 if i==0 else 1.0,"critical_failures":[]} for i in range(10)]
        self.assertEqual(decide(b,c)["decision"],"reject")

    def test_missing_observation_is_failure_not_inference(self):
        case=Case("one","input",(Check("observations.truthful","eq",True,True),))
        result=score_case(case,{"observations":{}})
        self.assertEqual(result["score"],0)
        self.assertEqual(result["critical_failures"],["observations.truthful"])

    def test_too_few_cases_holds(self):
        self._write_suite(2)
        with self.assertRaises(EvaluationError):
            self._evaluate()

    def test_mutating_runner_fails_closed(self):
        self._write_suite(10,mutate=0)
        with self.assertRaisesRegex(EvaluationError,"mutated"):
            self._evaluate()
        self.assertEqual(self.target.read_text(),"Be honest.\n")

    def test_tampered_binary_rejected(self):
        with self.assertRaisesRegex(EvaluationError,"SHA256"):
            self._evaluate(sha="0"*64)

    def test_bad_target_and_untrusted_edit_formats_rejected(self):
        secret=self.root/"other"/"SKILL.md"
        secret.parent.mkdir()
        secret.write_text("private")
        with self.assertRaises(EvaluationError):
            target_relative(str(secret),self.skills)
        with self.assertRaises(EvaluationError):
            apply_candidate_copy(self.target,"REPLACE_ALL:\nmalicious")
        with self.assertRaises(EvaluationError):
            apply_candidate_copy(self.target,"<<<\nunknown\n===\nnew\n>>>")

    def test_symlink_in_memory_snapshot_rejected(self):
        (self.memories/"link").symlink_to(self.target)
        with self.assertRaisesRegex(EvaluationError,"symlink"):
            self._evaluate()

    def test_duplicate_cases_rejected(self):
        self._write_suite(10)
        suite=json.loads(self.suite_file.read_text())
        suite["cases"][1]["id"]=suite["cases"][0]["id"]
        self.suite_file.write_text(json.dumps(suite))
        with self.assertRaisesRegex(EvaluationError,"duplicate"):
            load_suite(self.suite_file)

    def test_no_checks_leak_to_runner_input(self):
        # The runner gets prompt and case_id, never checks or expected values.
        self.assertEqual(self._evaluate()["decision"],"promote")
        self.assertEqual(self.target.read_text(),"Be honest.\n")

if __name__=="__main__":
    unittest.main()
