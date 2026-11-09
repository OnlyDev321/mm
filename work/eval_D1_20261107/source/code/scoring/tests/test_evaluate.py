"""Offline integration tests: no credentials or external API calls are used."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portable_evaluate", ROOT / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
CONFIG = ROOT.parent / "config.yaml"


def fixture_rows():
    """Construct schema fixtures; these are never benchmark measurements."""
    rows = []
    for subject in ev.SUBJECTS:
        for number in range(1, 31):
            is_open = len(rows) >= 847
            rows.append(dict(
                schema_version="0", synthetic=False,
                id=f"validation_{subject}_{number}", subject=subject,
                subfield=None, topic_difficulty=None, img_type=None,
                question_type="open" if is_open else "multiple-choice",
                gold="reference" if is_open else "A",
                options=[] if is_open else ["first option", "second option"],
                options_raw="[]" if is_open else "['first option', 'second option']",
                image_indices=[1], prompt="Fixture question <image 1>",
                status="ok", reason=None, error=None,
                raw_text="reference" if is_open else "A",
                output_token_ids=[10], output_tokens=1,
                finish_reason="stop", stop_reason=None, max_new_tokens=2,
                num_prompt_tokens=10, precomputed_prompt_tokens=10,
            ))
    rows[0].update(raw_text="unfinished reasoning", finish_reason="length",
                   output_token_ids=[10, 11], output_tokens=2)
    rows[1]["raw_text"] = "Cannot determine."
    return rows


class FakeHTTP:
    status = 200
    headers = {"x-request-id": "offline-fixture"}

    def __init__(self, text="Z", finish="stop"):
        self.payload = dict(model="gpt-4.1-mini-2025-04-14",
            choices=[dict(message=dict(content=text), finish_reason=finish)],
            usage=dict(prompt_tokens=10, completion_tokens=1, total_tokens=11,
                       prompt_tokens_details=dict(cached_tokens=2)))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.rows = fixture_rows()
        self.raw = self.base / "source.jsonl"
        ev.write_rows(self.raw, self.rows)
        self.out = self.base / "run"
        self.network = patch.object(ev.urllib.request, "urlopen", side_effect=AssertionError("Unexpected network"))
        self.blocked = self.network.start()
        self.addCleanup(self.network.stop)

    def prepare(self, out=None, raw=None, model=None):
        return ev.prepare(raw or self.raw, out or self.out, CONFIG, model)

    def test_prepare_offline_dynamic_routing_pending_and_move(self):
        before = self.raw.read_bytes()
        summary = self.prepare()
        self.assertEqual(summary["judge_count"], 2)
        self.assertEqual(summary["automatic_count"], 898)
        self.assertEqual(summary["overall"]["pending"], 2)
        self.assertIsNone(summary["overall"]["accuracy"])
        self.assertEqual(before, self.raw.read_bytes())
        moved = self.base / "moved"
        shutil.move(str(self.out), str(moved))
        self.assertEqual(ev.summarize(moved)["overall"], summary["overall"])
        self.blocked.assert_not_called()

    def test_length_uses_the_same_routing_as_stop(self):
        parsers = ev.load_parsers()
        cases = [
            (self.rows[2], "A", "auto", "A"),
            (self.rows[-1], "reference", "auto", "A"),
            (self.rows[2], "A and B considered; first option. Final Answer: B", "judge_conflict", None),
            (self.rows[2], "unfinished reasoning", "judge_rule_failure", None),
            (self.rows[-1], "unfinished reasoning", "judge_rule_failure", None),
        ]
        for template, text, route, answer in cases:
            for finish in ("stop", "length"):
                with self.subTest(text=text, finish=finish):
                    row = dict(template, raw_text=text, finish_reason=finish)
                    item = ev.derive_item(row, parsers)
                    self.assertEqual((item["route"], item["auto_answer"]), (route, answer))

    def test_z_is_completed_and_resume_sends_only_remaining(self):
        self.prepare()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-secret"}), \
             patch.object(ev.urllib.request, "urlopen", return_value=FakeHTTP("Z")) as api:
            first = ev.run(self.out, limit=1)
            second = ev.run(self.out)
            third = ev.run(self.out)
        self.assertEqual(api.call_count, 2)
        self.assertEqual(first["overall"]["pending"], 1)
        self.assertEqual(second["overall"]["pending"], 0)
        self.assertEqual(third["response_count"], 2)
        self.assertEqual(third["overall"]["extraction_failures"], 2)
        self.assertAlmostEqual(third["estimated_recorded_response_cost_usd"], 0.0000100)
        for path in self.out.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"offline-test-secret", path.read_bytes())

    def test_uncertain_failure_blocks_retry(self):
        self.prepare()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-secret"}), \
             patch.object(ev.urllib.request, "urlopen", side_effect=TimeoutError("offline-test-secret")) as api:
            with self.assertRaises(RuntimeError):
                ev.run(self.out, limit=1)
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                ev.run(self.out)
        self.assertEqual(api.call_count, 1)
        self.assertEqual(ev.summarize(self.out)["overall"]["pending"], 2)
        self.assertNotIn("offline-test-secret", (self.out / "errors.jsonl").read_text())

    def test_nonstop_valid_label_is_incorrect(self):
        self.prepare()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-secret"}), \
             patch.object(ev.urllib.request, "urlopen", return_value=FakeHTTP("A", "length")):
            summary = ev.run(self.out)
        self.assertEqual(summary["overall"]["extraction_failures"], 2)
        self.assertEqual(summary["overall"]["correct"], 898)

    def test_mutated_snapshot_rejected_before_network(self):
        self.prepare()
        with (self.out / "input" / "raw.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            ev.run(self.out)
        self.blocked.assert_not_called()

    def test_lock_prevents_second_executor(self):
        self.prepare()
        with ev.execution_lock(self.out):
            with self.assertRaisesRegex(ValueError, "locked"):
                ev.run(self.out)
        self.blocked.assert_not_called()

    def test_compare_pending_then_reject_question_and_model_change(self):
        self.prepare()
        right = self.base / "right"
        self.prepare(out=right)
        result = ev.compare(self.out, right, self.base / "comparison")
        self.assertEqual(result["state"], "partial")
        self.assertIsNone(result["right_minus_left_accuracy_pp"])
        self.assertFalse(result["cap_only_causal_effect_verified"])
        different_model = self.base / "other-model"
        self.prepare(out=different_model, model="gpt-4o-mini-2024-07-18")
        with self.assertRaisesRegex(ValueError, "settings/policy differ"):
            ev.compare(self.out, different_model, self.base / "comparison2")
        self.rows[2]["prompt"] = "Different question"
        ev.write_rows(self.raw, self.rows)
        different_question = self.base / "other-question"
        self.prepare(out=different_question)
        with self.assertRaisesRegex(ValueError, "input differs: prompt"):
            ev.compare(self.out, different_question, self.base / "comparison3")

    def test_duplicate_or_inconsistent_raw_rejected(self):
        bad = copy.deepcopy(self.rows)
        bad[1]["id"] = bad[0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ev.validate_raw(bad)
        bad = copy.deepcopy(self.rows)
        bad[0]["output_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "token|length"):
            ev.validate_raw(bad)
        bad = copy.deepcopy(self.rows)
        bad[0]["max_new_tokens"] = 3
        with self.assertRaisesRegex(ValueError, "length"):
            ev.validate_raw(bad)
        bad = copy.deepcopy(self.rows)
        bad[0]["num_prompt_tokens"] = 11
        with self.assertRaisesRegex(ValueError, "prompt token"):
            ev.validate_raw(bad)
        bad = copy.deepcopy(self.rows)
        bad[-1]["options"] = ["unexpected"]
        with self.assertRaisesRegex(ValueError, "open options"):
            ev.validate_raw(bad)


if __name__ == "__main__":
    unittest.main()
