# Chess Coach

This first command-line tool reads a PGN, asks Stockfish to evaluate each move,
and reports the three biggest evaluation drops.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the sample game

```bash
python coach.py
```

Analyze your own PGN:

```bash
python coach.py path/to/your-game.pgn
```

Options:

```bash
python coach.py --depth 14 --engine /opt/homebrew/bin/stockfish
```

Engine evaluations are shown in pawns from the perspective of the player who
made the move. A positive loss value means that move worsened their position.

## Explain a coaching moment with Ollama

The explanation command sends Stockfish's saved facts to an Ollama model and
validates the model's JSON response before displaying it. It does not ask the
model to calculate chess moves.

The defaults use the remote Ollama server at `http://192.168.1.54:11434` and
the `qwen3:8b` model:

```bash
python explain_moment.py 7
```

Choose a different moments file, server, or model when needed:

```bash
python explain_moment.py 7 \
  --moments coaching_moments.json \
  --host http://192.168.1.54:11434 \
  --model qwen3:8b
```

The Ollama server must be reachable from this computer and have the requested
model installed. Check the available models with:

```bash
curl http://192.168.1.54:11434/api/tags
```

## Chat with the coaching agent

The agent can analyze a PGN, inspect saved coaching moments, and request an
explanation for one move. The model chooses from a small fixed action set;
Python validates the choice and permits at most one tool call per question.

```bash
python chess_agent.py
```

Example questions:

```text
What coaching moments are available?
Analyze sample_game.pgn.
Why was move 7 a mistake?
What is the most important lesson from this game?
```

Use `quit` or `exit` to end the chat. You can override the remote Ollama server
or model with `--host` and `--model`, just as with `explain_moment.py`. The
`analyze_pgn` tool accepts only `.pgn` filenames placed directly in this project
directory; it cannot read arbitrary paths.
