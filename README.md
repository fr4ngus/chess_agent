# Chess Coach: Step 1

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
