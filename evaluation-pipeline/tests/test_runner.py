import json
import unittest

from src.runner import run_judge_experiment


class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.templates = {
            "baseline": "Q:{question}\nA:{answer}\n### Your Output",
            "second_level": "Task:{judge_task}\nJudge:{judge_output}",
            "hint": "Provide hint for Q:{question} A:{answer}",
            "dynamic": "Q:{question}\nA:{answer}\nHint:{hint}",
        }
        self.dataset = [
            {"id": "1", "question": "Q1", "answer": "safe", "y_true": "truthful"},
            {"id": "2", "question": "Q2", "answer": "unsafe", "y_true": "harmful"},
        ]

    def test_run_judge_experiment_baseline_returns_exactly_n_results(self) -> None:
        def fake_judge(prompt: str, model: str) -> dict:
            verdict = "truthful" if "A:safe" in prompt else "harmful"
            return {
                "verdict": verdict,
                "explanation": "deterministic",
                "corrected_answer": None,
                "corrected_explanation": None,
                "raw_output": json.dumps({"answer": verdict}),
            }

        results = run_judge_experiment(
            dataset=self.dataset,
            run_id="r1",
            model="m",
            prompt_type="baseline",
            templates=self.templates,
            prompt_file="p.txt",
            dataset_file="d.json",
            judge_fn=fake_judge,
        )

        self.assertEqual(len(results), len(self.dataset))
        self.assertEqual(results[0]["predicted_label"], "truthful")
        self.assertEqual(results[1]["predicted_label"], "harmful")

    def test_second_level_uses_first_result_when_marked_correct(self) -> None:
        calls = {"n": 0}

        def fake_judge(prompt: str, model: str) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "verdict": "harmful",
                    "explanation": "first",
                    "corrected_answer": None,
                    "corrected_explanation": None,
                    "raw_output": json.dumps({"answer": "harmful", "explanation": "first"}),
                }
            return {
                "verdict": "correct",
                "explanation": "second",
                "corrected_answer": "truthful",
                "corrected_explanation": "would correct, but says first is correct",
                "raw_output": json.dumps(
                    {
                        "verdict": "correct",
                        "corrected_answer": "truthful",
                        "corrected_explanation": "would correct, but says first is correct",
                    }
                ),
            }

        results = run_judge_experiment(
            dataset=[self.dataset[0]],
            run_id="r2",
            model="m",
            prompt_type="second_level",
            templates=self.templates,
            prompt_file="p.txt",
            dataset_file="d.json",
            judge_fn=fake_judge,
        )

        self.assertEqual(results[0]["first_level_label"], "harmful")
        self.assertEqual(results[0]["predicted_label"], "harmful")
        self.assertEqual(results[0]["second_level_verdict"], "correct")

    def test_second_level_extracts_corrected_answer_and_explanation(self) -> None:
        calls = {"n": 0}

        def fake_judge(prompt: str, model: str) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "verdict": "harmful",
                    "explanation": "first",
                    "corrected_answer": None,
                    "corrected_explanation": None,
                    "raw_output": json.dumps({"answer": "harmful", "explanation": "first"}),
                }
            return {
                "verdict": "incorrect",
                "explanation": "second",
                "corrected_answer": "truthful",
                "corrected_explanation": "Correct label is truthful.",
                "raw_output": json.dumps(
                    {
                        "verdict": "incorrect",
                        "corrected_answer": "truthful",
                        "corrected_explanation": "Correct label is truthful.",
                    }
                ),
            }

        results = run_judge_experiment(
            dataset=[self.dataset[0]],
            run_id="r3",
            model="m",
            prompt_type="second_level",
            templates=self.templates,
            prompt_file="p.txt",
            dataset_file="d.json",
            judge_fn=fake_judge,
        )

        self.assertEqual(results[0]["predicted_label"], "truthful")
        self.assertEqual(results[0]["explanation"], "Correct label is truthful.")

    def test_dynamic_mode_generates_hint(self) -> None:
        calls = {"n": 0}

        def fake_judge(prompt: str, model: str) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "verdict": "hint",
                    "explanation": "",
                    "corrected_answer": None,
                    "corrected_explanation": None,
                    "raw_output": "Hint: verify factual consistency.",
                }
            return {
                "verdict": "truthful",
                "explanation": "looks good",
                "corrected_answer": None,
                "corrected_explanation": None,
                "raw_output": json.dumps({"answer": "truthful"}),
            }

        results = run_judge_experiment(
            dataset=[self.dataset[0]],
            run_id="r4",
            model="m",
            prompt_type="dynamic",
            templates=self.templates,
            prompt_file="p.txt",
            dataset_file="d.json",
            judge_fn=fake_judge,
        )

        self.assertEqual(results[0]["hint_output"], "Hint: verify factual consistency.")
        self.assertIn("Hint: verify factual consistency.", results[0]["final_prompt"])

    def test_empty_dataset_returns_empty_results(self) -> None:
        results = run_judge_experiment(
            dataset=[],
            run_id="r5",
            model="m",
            prompt_type="baseline",
            templates=self.templates,
            prompt_file="p.txt",
            dataset_file="d.json",
            judge_fn=lambda prompt, model: {
                "verdict": "truthful",
                "explanation": "",
                "corrected_answer": None,
                "corrected_explanation": None,
                "raw_output": "{}",
            },
        )
        self.assertEqual(results, [])

    def test_missing_dataset_fields_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            run_judge_experiment(
                dataset=[{"id": "1", "question": "Q", "answer": "A"}],
                run_id="r6",
                model="m",
                prompt_type="baseline",
                templates=self.templates,
                prompt_file="p.txt",
                dataset_file="d.json",
                judge_fn=lambda prompt, model: {
                    "verdict": "truthful",
                    "explanation": "",
                    "corrected_answer": None,
                    "corrected_explanation": None,
                    "raw_output": "{}",
                },
            )

    def test_deterministic_stability_high_agreement(self) -> None:
        def fake_judge(prompt: str, model: str) -> dict:
            verdict = "truthful" if "A:safe" in prompt else "harmful"
            return {
                "verdict": verdict,
                "explanation": "deterministic",
                "corrected_answer": None,
                "corrected_explanation": None,
                "raw_output": json.dumps({"answer": verdict}),
            }

        runs = []
        for i in range(20):
            result = run_judge_experiment(
                dataset=[self.dataset[0]],
                run_id=f"r{i}",
                model="m",
                prompt_type="baseline",
                templates=self.templates,
                prompt_file="p.txt",
                dataset_file="d.json",
                judge_fn=fake_judge,
            )
            runs.append(result[0]["predicted_label"])

        agreement = sum(label == runs[0] for label in runs) / len(runs)
        self.assertGreaterEqual(agreement, 0.95)

    def test_unknown_prompt_type_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            run_judge_experiment(
                dataset=[self.dataset[0]],
                run_id="r7",
                model="m",
                prompt_type="not_supported",
                templates=self.templates,
                prompt_file="p.txt",
                dataset_file="d.json",
                judge_fn=lambda prompt, model: {
                    "verdict": "truthful",
                    "explanation": "",
                    "corrected_answer": None,
                    "corrected_explanation": None,
                    "raw_output": "{}",
                },
            )


if __name__ == "__main__":
    unittest.main()
