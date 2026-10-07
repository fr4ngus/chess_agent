import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import explain_moment


MOMENT = {
    "move_number": 7,
    "side": "White",
    "fen": "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "played_move": "Ba4",
    "best_move": "Bxc6",
    "classification": "mistake",
    "loss": 102,
    "best_line": ["Bxc6", "dxc6", "Nxe5"],
}
EXPLANATION = {
    "summary": "Ba4 missed the stronger exchange on c6.",
    "why_it_matters": "White gave up over a pawn of evaluation.",
    "better_plan": "Play Bxc6, then consider Nxe5 as in the engine line.",
    "lesson": "Look for forcing exchanges before retreating a threatened piece.",
}


class ExplainMomentTests(unittest.TestCase):
    def test_load_and_select_moment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            moments_path = Path(directory) / "moments.json"
            moments_path.write_text(json.dumps([MOMENT]), encoding="utf-8")

            moments = explain_moment.load_moments(moments_path)

        self.assertEqual(explain_moment.select_moment(moments, 7), MOMENT)

    def test_load_moments_rejects_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            moments_path = Path(directory) / "moments.json"
            moments_path.write_text('[{"move_number": 7}]', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                explain_moment.load_moments(moments_path)

    def test_select_moment_reports_missing_move(self) -> None:
        with self.assertRaisesRegex(ValueError, "No coaching moment"):
            explain_moment.select_moment([MOMENT], 8)

    def test_validate_explanation_rejects_incomplete_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            explain_moment.validate_explanation('{"summary": "Only one field"}')

    def test_request_explanation_parses_valid_ollama_response(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps({"response": json.dumps(EXPLANATION)}).encode(
            "utf-8"
        )
        response.__enter__.return_value = response

        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            explanation = explain_moment.request_explanation(
                "http://ollama.example:11434", "model", "prompt", 10
            )

        self.assertEqual(explanation, EXPLANATION)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://ollama.example:11434/api/generate")
        self.assertEqual(json.loads(request.data)["model"], "model")


if __name__ == "__main__":
    unittest.main()
