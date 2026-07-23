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
- **Interactive sensitivity analysis** — the probabilities and payoffs are the model's
  *estimates*, so you can edit them in the UI and the expected values, optimal path and
  Nash equilibria re-solve instantly (via `/api/recompute`, which runs the same engine
  with no LLM call). Shows the EV delta vs the original and resets in one click.
- **Model dropdown** — auto-populated from your locally installed Ollama models.

## Architecture

```
app.py          FastAPI endpoints (/, /api/models, /api/analyze, /api/recompute)
models.py       Pydantic schemas — single source of truth for the JSON contract
prompts.py      System + analysis prompt
providers.py    LLMProvider base + OllamaProvider + OpenAIProvider + model listing
game_theory.py  Probability normalization, backward-induction EV, Nash equilibria
config.py       Settings (provider/model defaults, from .env)
templates/
  index.html    Single-page UI (provider toggle, model dropdown, interactive tree,
                payoff matrix, sensitivity analysis)
tests/
  test_game_theory.py   Decision-tree engine: normalization, repair, backward induction
  test_nash.py          Nash equilibria against classic games
pytest.ini      Test configuration
```

The LLM is used **only** to model the situation. Every number that drives a conclusion —
expected values, the optimal path, Nash equilibria — is computed in `game_theory.py` and
covered by tests.

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

## Try it — example scenarios

Paste any of these into the query box. Each exercises a different part of the app.

**1. Sequential decision under uncertainty → decision tree + expected value**
> I'm negotiating salary for a new job. They offered $80k but market rate is ~$95k.
> I have a competing offer and strong performance reviews. Should I push hard, or take
> a soft collaborative approach?

You should get a decision tree with the EV-optimal path highlighted in green, and a
recommended strategy computed by backward induction.

**2. Simultaneous game → payoff matrix + Nash equilibria**
> Two rival coffee chains must simultaneously decide whether to launch an aggressive
> price war or keep prices high this quarter. Analyze the strategic interaction.

This triggers the normal-form representation: a payoff matrix with Nash equilibrium
cells highlighted, listed as PURE and/or MIXED below the table.

**3. A classic game — sanity-check the solver**
> Two suspects are interrogated separately and each can stay silent or betray the other.
> Betraying while the other stays silent goes free; both betraying means a medium
> sentence; both silent means a light sentence.

A Prisoner's-Dilemma shape should yield exactly one pure equilibrium: both betray.

**4. Sensitivity analysis — stress-test the answer**
After any analysis, open **"Adjust assumptions"** under the decision tree, or edit the
payoff numbers directly in the matrix cells:
- Lower the payoff of the recommended branch and watch the optimal path (green) flip to
  a different strategy, with the EV delta shown against the model's original estimate.
- In scenario 2, make a price war mutually devastating (e.g. set both payoffs to `-5`)
  and the equilibrium structure changes — cooperation becomes stable.
- Click **"reset to model's estimates"** to restore the original analysis.

**5. Non-applicable matrix (expected behaviour)**
> Should I repaint my house this year or wait until next spring?

A one-sided decision has no second player, so the payoff matrix section is correctly
hidden and only the decision tree is shown.

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

**POST** `/api/recompute` — re-solve edited assumptions with no LLM call. Accepts
`{"decision_tree": [...], "payoff_matrix": {...}}` and returns the same shape as
`/api/analyze` with expected values, optimal path and Nash equilibria recomputed. This
powers the sensitivity analysis.

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

## Tests

The deterministic engine (backward induction, Nash equilibria, tree repair) is covered
by unit tests. They are pure computation — no model or network needed, so they run in
well under a second.

```bash
pip install pytest    # already in requirements.txt
pytest                # run everything

pytest tests/test_nash.py -v          # just the Nash equilibrium tests
pytest tests/test_game_theory.py -v   # just the decision-tree engine
pytest -k optimal_path -v             # run tests matching a name
```

### What's covered

| Area | Tests |
|---|---|
| **Nash equilibria** | Prisoner's Dilemma (one pure NE), Matching Pennies (no pure, mixed 50/50), Battle of the Sexes (two pure + 2/3-1/3 mixed), dominant strategies, incomplete matrix, non-applicable matrix |
| **Backward induction** | Leaf = payoff, chance = probability-weighted, decision = max, equal-weight fallback for missing probabilities, cycle guard on malformed trees |
| **Optimal path** | Losing branches and the unchosen decision's subtree are *not* highlighted; stale flags cleared on recompute |
| **Tree repair** | Duplicate-id merging, dangling and self-reference removal, orphan subtree reattachment |
| **Normalization** | Outcome and sibling probabilities rescaled to sum to 1; all-zero falls back to equal weights |
| **Integration** | `process_analysis` over a deliberately malformed tree |

### Manual API checks

The engine endpoint needs no LLM, so you can exercise it directly:

```bash
# List installed models / provider availability
curl -s localhost:8000/api/models

# Solve a Prisoner's Dilemma matrix (expect one pure NE: Defect / Defect)
curl -s -X POST localhost:8000/api/recompute -H 'Content-Type: application/json' -d '{
 "payoff_matrix":{"applicable":true,"player_row":"A","player_col":"B",
  "row_strategies":["Coop","Defect"],"col_strategies":["Coop","Defect"],
  "cells":[{"row_strategy":"Coop","col_strategy":"Coop","row_payoff":3,"col_payoff":3},
           {"row_strategy":"Coop","col_strategy":"Defect","row_payoff":0,"col_payoff":5},
           {"row_strategy":"Defect","col_strategy":"Coop","row_payoff":5,"col_payoff":0},
           {"row_strategy":"Defect","col_strategy":"Defect","row_payoff":1,"col_payoff":1}]}}'

# Backward induction on a small tree (expect optimal_decision "Push hard", EV 100)
curl -s -X POST localhost:8000/api/recompute -H 'Content-Type: application/json' -d '{
 "decision_tree":[
  {"id":"root","label":"Choose","type":"decision","children":["hi","lo"]},
  {"id":"hi","label":"Push hard","type":"outcome","children":[],"payoff":100},
  {"id":"lo","label":"Play safe","type":"outcome","children":[],"payoff":10}]}'

# Full analysis through a local model (slow — 30-90s)
curl -s -X POST localhost:8000/api/analyze -H 'Content-Type: application/json' \
  -d '{"query":"Two rival chains decide simultaneously whether to start a price war.","provider":"ollama"}'
```

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
