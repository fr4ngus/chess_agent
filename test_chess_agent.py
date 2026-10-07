import unittest
from pathlib import Path
from unittest.mock import patch

import chess_agent


MOMENT = {
    "move_number": 7,
    "side": "White",
    "fen": "test-fen",
    "played_move": "Ba4",
    "best_move": "Bxc6",
    "classification": "mistake",
    "loss": 102,
    "best_line": ["Bxc6", "dxc6", "Nxe5"],
}


class ChessAgentTests(unittest.TestCase):
    def test_validate_action_accepts_supported_actions(self) -> None:
        self.assertEqual(
            chess_agent.validate_action({"name": "list_moments", "arguments": {}}),
            {"name": "list_moments", "arguments": {}},
        )
        self.assertEqual(
            chess_agent.validate_action(
                {"name": "explain_moment", "arguments": {"move_number": 7}}
            ),
            {"name": "explain_moment", "arguments": {"move_number": 7}},
        )
        self.assertEqual(
            chess_agent.validate_action(
                {"name": "answer", "arguments": {"answer": "Hello"}}
            ),
            {"name": "answer", "arguments": {"answer": "Hello"}},
        )
        self.assertEqual(
            chess_agent.validate_action(
                {"name": "analyze_pgn", "arguments": {"pgn_path": "game.pgn"}}
            ),
            {"name": "analyze_pgn", "arguments": {"pgn_path": "game.pgn"}},
        )

    def test_validate_action_rejects_extra_or_unsafe_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not accept arguments"):
            chess_agent.validate_action(
                {"name": "list_moments", "arguments": {"path": "/etc/passwd"}}
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            chess_agent.validate_action(
                {"name": "explain_moment", "arguments": {"move_number": 0}}
            )
        with self.assertRaisesRegex(ValueError, "unsupported action"):
            chess_agent.validate_action(
                {"name": "run_shell", "arguments": {"command": "rm -rf /"}}
            )
        with self.assertRaisesRegex(ValueError, "project directory"):
            chess_agent.validate_action(
                {"name": "analyze_pgn", "arguments": {"pgn_path": "../outside.pgn"}}
            )
        with self.assertRaisesRegex(ValueError, "end with .pgn"):
            chess_agent.validate_action(
                {"name": "analyze_pgn", "arguments": {"pgn_path": "game.txt"}}
            )

    def test_list_moments_omits_fen(self) -> None:
        result = chess_agent.list_moments([MOMENT])

        self.assertEqual(result["moments"][0]["move_number"], 7)
        self.assertNotIn("fen", result["moments"][0])

    def test_answer_question_executes_one_tool_then_answers(self) -> None:
        actions = [
            {"name": "list_moments", "arguments": {}},
            {
                "name": "answer",
                "arguments": {"answer": "Move 7 was the largest issue."},
            },
        ]
        with patch("chess_agent.request_action", side_effect=actions) as request_action:
            with patch("chess_agent.execute_action", return_value={"moments": []}) as execute:
                answer = chess_agent.answer_question(
                    "What went wrong?", [], Path("moments.json"), "host", "model", 10
                )

        self.assertEqual(answer, "Move 7 was the largest issue.")
        execute.assert_called_once_with(
            {
                "name": "list_moments",
                "arguments": {},
            },
            Path("moments.json"),
            "host",
            "model",
            10,
            chess_agent.coach.DEFAULT_ENGINE,
            12,
        )
        self.assertEqual(request_action.call_count, 2)

    def test_execute_analyze_pgn_calls_coach_with_project_file(self) -> None:
        with patch("chess_agent.coach.analyze_game", return_value=[MOMENT]) as analyze_game:
            result = chess_agent.execute_action(
                {
                    "name": "analyze_pgn",
                    "arguments": {"pgn_path": "sample_game.pgn"},
                },
                Path("moments.json"),
                "host",
                "model",
                10,
                "/stockfish",
                1,
            )

        self.assertEqual(result["analyzed_pgn"], "sample_game.pgn")
        self.assertEqual(result["coaching_moment_count"], 1)
        analyze_game.assert_called_once_with(
            chess_agent.PROJECT_PATH / "sample_game.pgn",
            "/stockfish",
            1,
            Path("moments.json"),
        )


if __name__ == "__main__":
    unittest.main()
