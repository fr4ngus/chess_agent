#!/usr/bin/env python3
"""Explain a Stockfish coaching moment with a local or remote Ollama model."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_HOST = "http://192.168.1.54:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_MOMENTS_PATH = Path(__file__).with_name("coaching_moments.json")
REQUIRED_MOMENT_FIELDS = {
    "move_number",
    "side",
    "fen",
    "played_move",
    "best_move",
    "classification",
    "loss",
    "best_line",
}
REQUIRED_EXPLANATION_FIELDS = {
    "summary",
    "why_it_matters",
    "better_plan",
    "lesson",
}


def load_moments(moments_path: Path) -> list[dict[str, Any]]:
    """Load and validate the coaching moments written by coach.py."""
    if not moments_path.is_file():
        raise ValueError(f"Coaching moments file not found: {moments_path}")

    try:
        with moments_path.open(encoding="utf-8") as moments_file:
            moments = json.load(moments_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {moments_path}: {error.msg}") from error

    if not isinstance(moments, list):
        raise ValueError(f"Coaching moments must be a JSON list: {moments_path}")

    for index, moment in enumerate(moments, start=1):
        if not isinstance(moment, dict):
            raise ValueError(f"Moment {index} must be a JSON object.")
        missing_fields = REQUIRED_MOMENT_FIELDS - moment.keys()
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(f"Moment {index} is missing required fields: {fields}")
        if not isinstance(moment["move_number"], int):
            raise ValueError(f"Moment {index} has an invalid move_number.")
        if not isinstance(moment["best_line"], list):
            raise ValueError(f"Moment {index} has an invalid best_line.")

    return moments


def select_moment(moments: list[dict[str, Any]], move_number: int) -> dict[str, Any]:
    """Return the coaching moment for one half-move number."""
    for moment in moments:
        if moment["move_number"] == move_number:
            return moment
    raise ValueError(f"No coaching moment found for move number {move_number}.")


def build_prompt(moment: dict[str, Any]) -> str:
    """Provide only Stockfish-derived facts and constrain the model's role."""
    facts = json.dumps(moment, indent=2)
    return f"""You are a concise chess coach. Explain the Stockfish analysis below to a club-level player.

Use only the supplied facts. Do not claim tactical details that are not supported by the best line. Do not give a different best move. The evaluation loss is in centipawns.

Return exactly one JSON object with these four non-empty string fields:
- summary
- why_it_matters
- better_plan
- lesson

Stockfish facts:
{facts}
"""


def request_explanation(host: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    """Request JSON-only coaching advice from Ollama's generate endpoint."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Ollama at {host}: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError(f"Ollama request timed out after {timeout:g} seconds.") from error

    try:
        response_data = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON.") from error

    if not isinstance(response_data, dict):
        raise RuntimeError("Ollama returned an invalid response object.")
    if response_data.get("error"):
        raise RuntimeError(f"Ollama error: {response_data['error']}")
    model_response = response_data.get("response")
    if not isinstance(model_response, str):
        raise RuntimeError("Ollama response did not contain text.")

    return validate_explanation(model_response)


def validate_explanation(model_response: str) -> dict[str, str]:
    """Validate the JSON response before presenting it as coaching advice."""
    try:
        explanation = json.loads(model_response)
    except json.JSONDecodeError as error:
        raise ValueError("Model did not return valid JSON.") from error

    if not isinstance(explanation, dict):
        raise ValueError("Model response must be a JSON object.")
    missing_fields = REQUIRED_EXPLANATION_FIELDS - explanation.keys()
    if missing_fields:
        fields = ", ".join(sorted(missing_fields))
        raise ValueError(f"Model response is missing required fields: {fields}")

    validated: dict[str, str] = {}
    for field in REQUIRED_EXPLANATION_FIELDS:
        value = explanation[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Model response field {field!r} must be a non-empty string.")
        validated[field] = value.strip()
    return validated


def print_explanation(moment: dict[str, Any], explanation: dict[str, str]) -> None:
    """Print a readable terminal view after the model output is validated."""
    loss_in_pawns = moment["loss"] / 100
    print(
        f"Move {moment['move_number']} ({moment['side']}): "
        f"{moment['played_move']} was a {moment['classification']} "
        f"({loss_in_pawns:.2f} pawns lost)."
    )
    print(f"\nSummary\n{explanation['summary']}")
    print(f"\nWhy It Matters\n{explanation['why_it_matters']}")
    print(f"\nBetter Plan\n{explanation['better_plan']}")
    print(f"\nLesson\n{explanation['lesson']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain a Stockfish coaching moment with Ollama."
    )
    parser.add_argument("move_number", type=int, help="half-move number from coaching_moments.json")
    parser.add_argument(
        "--moments",
        type=Path,
        default=DEFAULT_MOMENTS_PATH,
        help=f"path to coaching moments JSON (default: {DEFAULT_MOMENTS_PATH})",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Ollama server URL (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="request timeout in seconds (default: 120)"
    )
    args = parser.parse_args()

    try:
        if args.timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")
        moments = load_moments(args.moments)
        moment = select_moment(moments, args.move_number)
        explanation = request_explanation(
            args.host, args.model, build_prompt(moment), args.timeout
        )
        print_explanation(moment, explanation)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
