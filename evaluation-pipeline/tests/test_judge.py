import json
import unittest

from src.judge import judge_response


class TestJudge(unittest.TestCase):
    def test_judge_response_with_valid_input_returns_structured_dict(self) -> None:
        def fake_chat(model: str, prompt: str) -> dict:
            return {
                "message": {
                    "content": json.dumps(
                        {"answer": "[truthful]", "explanation": "Matches known facts."}
                    )
                }
            }

        result = judge_response("prompt", "model", chat_fn=fake_chat)

        self.assertEqual(result["verdict"], "truthful")
        self.assertEqual(result["explanation"], "Matches known facts.")
        self.assertIsNotNone(result["raw_output"])

    def test_judge_response_extracts_second_level_corrections(self) -> None:
        def fake_chat(model: str, prompt: str) -> dict:
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "verdict": "incorrect",
                            "corrected_answer": "harmful",
                            "corrected_explanation": "The answer is unsafe.",
                        }
                    )
                }
            }

        result = judge_response("prompt", "model", chat_fn=fake_chat)

        self.assertEqual(result["verdict"], "incorrect")
        self.assertEqual(result["corrected_answer"], "harmful")
        self.assertEqual(result["corrected_explanation"], "The answer is unsafe.")

    def test_judge_response_with_missing_fields_is_robust(self) -> None:
        def fake_chat(model: str, prompt: str) -> dict:
            return {"message": {"content": "{}"}}

        result = judge_response("prompt", "model", chat_fn=fake_chat)

        self.assertEqual(result["verdict"], "invalid_label")
        self.assertEqual(result["explanation"], "")
        self.assertIsNotNone(result["raw_output"])


if __name__ == "__main__":
    unittest.main()
