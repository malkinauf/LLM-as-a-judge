import unittest

from src.prompts import build_experiment_prompt


class TestPrompts(unittest.TestCase):
    def setUp(self) -> None:
        self.templates = {
            "baseline": "Q: {question} A: {answer}",
            "second_level": "Task: {judge_task} Output: {judge_output}",
            "hint": "Hint for {question} / {answer}",
            "dynamic": "Q: {question} A: {answer} H: {hint}",
        }

    def test_build_experiment_prompt_baseline_second_level_dynamic(self) -> None:
        baseline = build_experiment_prompt(
            "baseline",
            self.templates,
            {"question": "What?", "answer": "Because."},
        )
        second_level = build_experiment_prompt(
            "second_level",
            self.templates,
            {"judge_task": "task", "judge_output": "output"},
        )
        dynamic = build_experiment_prompt(
            "dynamic",
            self.templates,
            {"question": "What?", "answer": "Because.", "hint": "Use safety."},
        )

        self.assertEqual(baseline, "Q: What? A: Because.")
        self.assertEqual(second_level, "Task: task Output: output")
        self.assertEqual(dynamic, "Q: What? A: Because. H: Use safety.")

    def test_unknown_prompt_type_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_experiment_prompt("unknown", self.templates, {})


if __name__ == "__main__":
    unittest.main()
