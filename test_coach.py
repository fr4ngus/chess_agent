import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.engine

import coach


class CoachTests(unittest.TestCase):
    def test_classify_loss_thresholds(self) -> None:
        self.assertIsNone(coach.classify_loss(49))
        self.assertEqual(coach.classify_loss(50), "inaccuracy")
        self.assertEqual(coach.classify_loss(100), "mistake")
        self.assertEqual(coach.classify_loss(300), "blunder")

    def test_write_moments_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "moments.json"
            moments = [{"played_move": "e4", "loss": 50}]

            coach.write_moments(output_path, moments)

            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), moments)
            self.assertFalse(output_path.with_suffix(".json.tmp").exists())

    def test_empty_game_writes_empty_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            pgn_path = directory_path / "empty.pgn"
            output_path = directory_path / "moments.json"
            pgn_path.write_text('[Result "*"]\n\n*\n', encoding="utf-8")

            with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen_uci:
                moments = coach.analyze_game(pgn_path, "stockfish", 1, output_path)

            self.assertEqual(moments, [])
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), [])
            popen_uci.assert_called_once_with("stockfish")

    def test_missing_pgn_is_reported_before_starting_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "moments.json"

            with self.assertRaisesRegex(ValueError, "PGN file not found"):
                coach.analyze_game(Path(directory) / "missing.pgn", "stockfish", 1, output_path)


if __name__ == "__main__":
    unittest.main()
