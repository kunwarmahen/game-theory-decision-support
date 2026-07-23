# Game Theory Decision Analyzer

A web application that analyzes strategic situations with game-theory principles and
local or cloud LLMs. It returns stakeholders, weighted outcomes, an interactive
decision tree whose **expected values are computed deterministically by backward
induction**, and — for simultaneous games — a **payoff matrix with Nash equilibria
solved in code**. The app calculates the optimal strategy rather than asking the model
to guess it.

Runs fully locally with **Gemma via Ollama** by default (private, free), with an
optional OpenAI provider.

## Highlights

- **One app, pluggable providers** — local models (Gemma/Ollama) and OpenAI behind a
  single provider layer (`providers.py`). No more duplicated app files.
- **Reliable JSON via structured outputs** — the LLM is *constrained* to a JSON schema
  (Ollama `format` / OpenAI `response_format`), then validated with Pydantic. No fragile
  string scraping.
- **Real backward induction** — chance nodes are expected-value-weighted, decision nodes
  maximize, and the EV-optimal path is highlighted (`game_theory.py`).
- **Payoff matrix + Nash equilibria** — when the situation is a simultaneous two-player
  game, the model produces a normal-form payoff matrix and the server computes the
  equilibria: **pure-strategy** NE for any nxm game plus the **mixed-strategy** NE for
  2x2 games. Equilibrium cells are highlighted in the matrix.
- **Interactive decision tree** — rendered in the browser with vis-network (color-coded
  node types, probability edge labels, per-node EV, optimal path). No server-side image
  generation, so `matplotlib`/`networkx`/`graphviz` are no longer needed.
- **Model dropdown** — auto-populated from your locally installed Ollama models.

## Architecture

```
app.py          FastAPI endpoints (/, /api/models, /api/analyze)
models.py       Pydantic schemas — single source of truth for the JSON contract
prompts.py      System + analysis prompt
providers.py    LLMProvider base + OllamaProvider + OpenAIProvider + model listing
game_theory.py  Probability normalization, backward-induction EV, Nash equilibria
config.py       Settings (provider/model defaults, from .env)
templates/
  index.html    Single-page UI (provider toggle, model dropdown, interactive tree)
```

## Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com) installed and running, with at least one chat model:
  ```bash
  ollama pull gemma4:12b   # or any Gemma / Qwen / Llama chat model you prefer
  ```

### Run
```bash
cd /home/mahen/Documents/ai/game_theory/decision_support
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # optional — edit defaults
python app.py             # serves on http://localhost:8000
```

Open `http://localhost:8000`, describe your situation, pick a model, and click **Analyze**.

> Local models can take ~30–90s per analysis depending on model size and hardware. The
> smaller Gemma variants (e.g. `e4b`) are noticeably faster than the larger ones.

## Configuration (`.env`)

```ini
APP_HOST=0.0.0.0
APP_PORT=8000

# Provider: "ollama" (local Gemma) or "openai"
DEFAULT_PROVIDER=ollama

# Ollama
DEFAULT_OLLAMA_URL=http://localhost:11434
DEFAULT_MODEL_NAME=gemma4:12b

# OpenAI (optional — leave blank to disable). Read from the server env; never sent
# from the browser.
OPENAI_API_KEY=
DEFAULT_OPENAI_MODEL=gpt-4o

TEMPERATURE=0.2
LOG_LEVEL=INFO
CORS_ORIGINS=*
```

## API

Interactive docs at `http://localhost:8000/docs`.

**GET** `/api/models` — installed (chat-capable) Ollama models + provider availability.

**POST** `/api/analyze`
```json
{
  "query": "Your strategic situation",
  "provider": "ollama",
  "model_name": "gemma4:12b",
  "ollama_url": "http://localhost:11434"
}
```
Response (abridged):
```json
{
  "stakeholders": ["..."],
  "summary": "...",
  "outcomes": [{"description": "...", "probability": 0.35, "key_factors": ["..."], "recommendation": "..."}],
  "recommended_outcome": "...",
  "decision_tree": [
    {"id": "root", "label": "...", "type": "decision", "children": ["a","b"], "expected_value": 90.0, "is_optimal": true}
  ],
  "optimal_decision": "...",
  "optimal_expected_value": 90.0,
  "payoff_matrix": {
    "applicable": true,
    "player_row": "Chain A", "player_col": "Chain B",
    "row_strategies": ["High Price", "Price War"],
    "col_strategies": ["High Price", "Price War"],
    "cells": [{"row_strategy": "High Price", "col_strategy": "High Price", "row_payoff": 80, "col_payoff": 80}]
  },
  "nash_equilibria": [
    {"kind": "pure", "profile": "Price War / High Price", "row_payoff": 100, "col_payoff": 40, "description": "..."},
    {"kind": "mixed", "profile": "Chain A: 33% High Price / 67% Price War | ...", "row_mix": [0.33, 0.67], "col_mix": [0.33, 0.67]}
  ]
}
```

> `payoff_matrix` is `null` (and `nash_equilibria` empty) for one-sided decisions under
> uncertainty — those are represented by the decision tree instead.

## How the expected value is computed

The LLM assigns **payoffs** to terminal outcome nodes and **probabilities** to chance
branches. The server then walks the tree bottom-up (`game_theory.compute_expected_values`):

- **outcome (leaf):** value = payoff
- **chance node:** value = Σ (child probability × child value)
- **decision node:** value = max(child values); the maximizing branch is marked optimal

This yields a defensible expected value for every node and identifies the optimal
strategy — independent of whatever the model "thinks" the best choice is.

### Node types in the tree
- **Decision** (blue box) — a choice the actor controls
- **Chance** (amber diamond) — an uncertain event carrying a probability
- **Outcome** (green ellipse) — a terminal result carrying a payoff

## Payoff matrix & Nash equilibria

For **simultaneous** strategic interactions (two players each choosing a discrete
strategy at the same time — price wars, standoffs, negotiation as concede/hold, etc.),
the model emits a normal-form payoff matrix and the server solves it in code
(`game_theory.compute_nash_equilibria`):

- **Pure-strategy NE** (any nxm game): a strategy profile where neither player can
  improve by unilaterally deviating (mutual best response). These cells are highlighted
  in the matrix table.
- **Mixed-strategy NE** (2x2 games): the interior equilibrium where each player
  randomizes to keep the other indifferent — essential for games with no pure NE
  (e.g. Matching Pennies).

The model only builds the game; it never identifies the equilibrium itself. Verified
against textbook games — Prisoner's Dilemma (pure), Matching Pennies (mixed 50/50), and
Battle of the Sexes (two pure + one mixed).

**Scope:** the matrix and mixed-NE solver target two-player games (mixed NE specifically
for 2x2). Pure-strategy NE works for any nxm matrix.

## Production deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for systemd, Nginx reverse proxy, and TLS.

## Troubleshooting

- **No models in the dropdown** — ensure Ollama is running (`ollama serve`) and the URL
  in Settings matches; only chat-capable models are listed (embedding models are hidden).
- **OpenAI option disabled** — set `OPENAI_API_KEY` in `.env` and restart.
- **Port in use** — change `APP_PORT` in `.env`.

## Acknowledgments

Built with [FastAPI](https://fastapi.tiangolo.com/), [Tailwind CSS](https://tailwindcss.com/),
[vis-network](https://visjs.org/), and [Ollama](https://ollama.com/).
