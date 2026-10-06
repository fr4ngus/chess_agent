#!/usr/bin/env python3
"""Analyze a PGN with Stockfish and identify the largest evaluation drops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

DEFAULT_ENGINE = "/opt/homebrew/bin/stockfish"
OUTPUT_PATH = Path(__file__).with_name("coaching_moments.json")


def score_for_player(info: dict, player: chess.Color) -> int:
    """Return an engine score in centipawns, positive when player is better."""
    return info["score"].pov(player).score(mate_score=10_000)


def classify_loss(loss: int) -> str | None:
    """Classify a move by the centipawn value it loses."""
    if loss >= 300:
        return "blunder"
    if loss >= 100:
        return "mistake"
    if loss >= 50:
        return "inaccuracy"
    return None


def write_moments(output_path: Path, moments: list[dict]) -> None:
    """Write output atomically so interrupted runs do not leave invalid JSON."""
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(moments, output, indent=2)
        output.write("\n")
    temporary_path.replace(output_path)


def analyze_game(
    pgn_path: Path, engine_path: str, depth: int, output_path: Path
) -> list[dict]:
    if depth < 1:
        raise ValueError("Engine depth must be at least 1.")
    if not pgn_path.is_file():
        raise ValueError(f"PGN file not found: {pgn_path}")

    with pgn_path.open(encoding="utf-8") as pgn_file:
        game = chess.pgn.read_game(pgn_file)

    if game is None:
        raise ValueError(f"No game found in {pgn_path}")

    board = game.board()
    mistakes: list[tuple[int, str, str, int, str]] = []
    moments: list[dict] = []

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for move_number, move in enumerate(game.mainline_moves(), start=1):
            player = board.turn
            before = score_for_player(
                engine.analyse(board, chess.engine.Limit(depth=depth)), player
            )
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            if not info.get("pv"):
                raise RuntimeError("Stockfish returned no principal variation.")
            fen = board.fen()
            san = board.san(move)
            best_move = board.san(info["pv"][0])

            pv_board = board.copy()
            pv_san = []

            for pv_move in info["pv"][:5]:
                pv_san.append(pv_board.san(pv_move))
                pv_board.push(pv_move)
            print(" ".join(pv_san))
            board.push(move)
            after = score_for_player(
                engine.analyse(board, chess.engine.Limit(depth=depth)), player
            )
            loss = before - after
            classification = classify_loss(loss)
            if classification is not None:
                moment = {
                    "move_number": move_number,
                    "side": "White" if player == chess.WHITE else "Black",
                    "fen": fen,
                    "played_move": san,
                    "best_move": best_move,
                    "classification": classification,
                    "loss": loss,
                    "best_line": pv_san,
                }
                moments.append(moment)
                print(moment)
            side = "White" if player == chess.WHITE else "Black"
            label = f" [{classification}]" if classification else ""
            print(
                f"{move_number:>2}. {side:<5} {san:<8} {before / 100:+.2f} -> {after / 100:+.2f} ({loss / 100:+.2f}){label}"
            )
            if classification:
                mistakes.append((move_number, side, san, loss, classification))

    write_moments(output_path, moments)

    print("\nCoaching moments")
    if not mistakes:
        print("No move lost at least 0.50 pawns at this engine depth.")
        return moments

    for move_number, side, san, loss, classification in sorted(
        mistakes, key=lambda item: item[3], reverse=True
    )[:3]:
        print(
            f"- Move {move_number}: {side} played {san}, a {classification}, losing about {loss / 100:.2f} pawns."
        )

    return moments


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a chess PGN with Stockfish.")
    parser.add_argument(
        "pgn", nargs="?", default="sample_game.pgn", help="path to a PGN file"
    )
    parser.add_argument(
        "--engine", default=DEFAULT_ENGINE, help="path to the Stockfish executable"
    )
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"path for coaching moments JSON (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    try:
        analyze_game(Path(args.pgn), args.engine, args.depth, args.output)
    except (OSError, ValueError, chess.engine.EngineError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
