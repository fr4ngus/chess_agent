#!/usr/bin/env python3
"""A bounded terminal chess coach agent using a remote Ollama model."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import coach
import explain_moment

DEFAULT_HOST = explain_moment.DEFAULT_HOST
DEFAULT_MODEL = explain_moment.DEFAULT_MODEL
DEFAULT_MOMENTS_PATH = explain_moment.DEFAULT_MOMENTS_PATH
PROJECT_PATH = Path(__file__).resolve().parent
TOOL_SCHEMAS = [
    {
        "name": "list_moments",
        "description": "List the saved chess coaching moments.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "explain_moment",
        "description": "Explain one saved coaching moment by half-move number.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_number": {
                    "type": "integer",
                    "description": "A positive half-move number.",
                }
            },
            "required": ["move_number"],
            "additionalProperties": False,
        },
    },
    {
        "name": "analyze_pgn",
        "description": "Analyze a .pgn filename located directly in the project directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "pgn_path": {
                    "type": "string",
                    "description": "A relative .pgn filename, such as my_game.pgn.",
                }
            },
            "required": ["pgn_path"],
            "additionalProperties": False,
        },
    },
]


def build_agent_prompt(
    question: str,
    history: list[dict[str, str]],
    tool_result: dict[str, Any] | None = None,
) -> str:
    """Ask for one constrained action; Python, not the model, executes tools."""
    tool_section = json.dumps(TOOL_SCHEMAS, indent=2)
    result_section = ""
    if tool_result is not None:
        result_section = f"""\nA tool has already been executed. Use its result to answer the current
user question. Do not choose another tool. Return only this JSON object:
{{"name": "answer", "arguments": {{"answer": "..."}}}}

Tool result:
{json.dumps(tool_result, indent=2)}
"""

    return f"""You are a concise chess coaching assistant.

Available tools:
{tool_section}

Return exactly one JSON object. To call a tool, use:
{{"name": "tool_name", "arguments": {{...}}}}
For a direct response, use:
{{"name": "answer", "arguments": {{"answer": "..."}}}}

Never invent a move number. Do not request files, shell commands, network access,
or any action outside this list.

Conversation history:
{json.dumps(history, indent=2)}

User question: {question}
{result_section}"""


def request_action(
    host: str, model: str, prompt: str, timeout: float, debug: bool = False
) -> dict[str, Any]:
    """Request one JSON action from Ollama."""
    if debug:
        print(f"\n[debug] Sending prompt:\n{prompt}")
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
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
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach Ollama at {host}: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError(
            f"Ollama request timed out after {timeout:g} seconds."
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON.") from error

    if not isinstance(response_data, dict):
        raise RuntimeError("Ollama returned an invalid response object.")
    if response_data.get("error"):
        raise RuntimeError(f"Ollama error: {response_data['error']}")
    response_text = response_data.get("response")
    if not isinstance(response_text, str):
        raise RuntimeError("Ollama response did not contain text.")
    if debug:
        print(f"\n[debug] Raw model response:\n{response_text}")

    try:
        action = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("Model did not return valid action JSON.") from error
    action = validate_action(action)
    if debug:
        print(f"\n[debug] Validated action:\n{json.dumps(action, indent=2)}")
    return action


def validate_action(action: Any) -> dict[str, Any]:
    """Allow only the exact action shapes supported by this program."""
    if not isinstance(action, dict):
        raise ValueError("Model action must be a JSON object.")

    if set(action) != {"name", "arguments"}:
        raise ValueError("Model action requires only name and arguments.")
    action_name = action.get("name")
    arguments = action.get("arguments")
    if not isinstance(action_name, str):
        raise ValueError("Model action name must be a string.")
    if not isinstance(arguments, dict):
        raise ValueError("Model action arguments must be a JSON object.")

    if action_name == "list_moments":
        if arguments:
            raise ValueError("list_moments does not accept arguments.")
    elif action_name == "explain_moment":
        if set(arguments) != {"move_number"}:
            raise ValueError("explain_moment requires only move_number.")
        if (
            not isinstance(arguments["move_number"], int)
            or arguments["move_number"] < 1
        ):
            raise ValueError("explain_moment move_number must be a positive integer.")
    elif action_name == "analyze_pgn":
        if set(arguments) != {"pgn_path"}:
            raise ValueError("analyze_pgn requires only pgn_path.")
        if not isinstance(arguments["pgn_path"], str):
            raise ValueError("analyze_pgn pgn_path must be a string.")
        validate_pgn_path(arguments["pgn_path"])
    elif action_name == "answer":
        if set(arguments) != {"answer"}:
            raise ValueError("answer requires only answer text.")
        if not isinstance(arguments["answer"], str) or not arguments["answer"].strip():
            raise ValueError("answer must be a non-empty string.")
    else:
        raise ValueError("Model requested an unsupported action.")
    return action


def validate_pgn_path(pgn_path: str) -> Path:
    """Limit agent analysis to PGNs stored directly in the project directory."""
    path = Path(pgn_path)
    if path.suffix.lower() != ".pgn":
        raise ValueError("analyze_pgn pgn_path must end with .pgn.")
    if path.is_absolute() or len(path.parts) != 1 or path.name != pgn_path:
        raise ValueError(
            "analyze_pgn pgn_path must be a filename in the project directory."
        )
    resolved_path = (PROJECT_PATH / path).resolve()
    if resolved_path.parent != PROJECT_PATH:
        raise ValueError("analyze_pgn pgn_path must stay in the project directory.")
    return resolved_path


def list_moments(moments: list[dict[str, Any]]) -> dict[str, Any]:
    """Return small, factual records so the model need not receive full FENs."""
    return {
        "moments": [
            {
                "move_number": moment["move_number"],
                "side": moment["side"],
                "played_move": moment["played_move"],
                "best_move": moment["best_move"],
                "classification": moment["classification"],
                "loss_in_pawns": moment["loss"] / 100,
            }
            for moment in moments
        ]
    }


def execute_action(
    action: dict[str, Any],
    moments_path: Path,
    host: str,
    model: str,
    timeout: float,
    engine_path: str = coach.DEFAULT_ENGINE,
    depth: int = 12,
) -> dict[str, Any]:
    """Execute approved local tools using validated arguments only."""
    if action["name"] == "analyze_pgn":
        pgn_path = validate_pgn_path(action["arguments"]["pgn_path"])
        moments = coach.analyze_game(pgn_path, engine_path, depth, moments_path)
        return {
            "analyzed_pgn": pgn_path.name,
            "coaching_moment_count": len(moments),
            "moments": list_moments(moments)["moments"],
        }

    moments = explain_moment.load_moments(moments_path)
    if action["name"] == "list_moments":
        return list_moments(moments)
    if action["name"] == "explain_moment":
        moment = explain_moment.select_moment(
            moments, action["arguments"]["move_number"]
        )
        explanation = explain_moment.request_explanation(
            host, model, explain_moment.build_prompt(moment), timeout
        )
        return {"moment": moment, "explanation": explanation}
    raise ValueError("answer is not a tool action.")


def answer_question(
    question: str,
    history: list[dict[str, str]],
    moments_path: Path,
    host: str,
    model: str,
    timeout: float,
    engine_path: str = coach.DEFAULT_ENGINE,
    depth: int = 12,
    debug: bool = False,
) -> str:
    """Run at most one tool call, then require the model to give its final answer."""
    action = request_action(
        host, model, build_agent_prompt(question, history), timeout, debug
    )
    if action["name"] == "answer":
        answer = action["arguments"]["answer"].strip()
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer

    tool_result = execute_action(
        action, moments_path, host, model, timeout, engine_path, depth
    )
    if debug:
        print(f"\n[debug] Tool result:\n{json.dumps(tool_result, indent=2)}")
    final_action = request_action(
        host,
        model,
        build_agent_prompt(question, history, tool_result),
        timeout,
        debug,
    )
    if final_action["name"] != "answer":
        raise RuntimeError(
            "Model did not provide a final answer after the tool result."
        )
    answer = final_action["arguments"]["answer"].strip()
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with a bounded chess coaching agent."
    )
    parser.add_argument("--moments", type=Path, default=DEFAULT_MOMENTS_PATH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--engine", default=coach.DEFAULT_ENGINE)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print model prompts, responses, and tool results",
    )
    args = parser.parse_args()

    if args.timeout <= 0 or args.depth < 1:
        parser.error("Timeout and engine depth must be greater than zero.")

    print("Chess coach agent. Ask about your analyzed game. Type 'quit' to exit.")
    history = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            print()
            return
        if question.lower() in {"quit", "exit"}:
            return
        if not question:
            continue
        try:
            answer = answer_question(
                question,
                history,
                args.moments,
                args.host,
                args.model,
                args.timeout,
                args.engine,
                args.depth,
                args.debug,
            )
            print(f"\nCoach: {answer}")
        except (OSError, RuntimeError, ValueError) as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
