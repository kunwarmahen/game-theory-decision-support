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
- **Dominance & Pareto efficiency** — an equilibrium tells you a cell is stable but not
  *why*. Dominated strategies are struck through in the matrix ("never play this, and
  here's what beats it"), Pareto-efficient cells are outlined, and when the equilibrium
  is *not* Pareto efficient the app says so in plain language — the Prisoner's Dilemma
  insight, stated rather than left for you to spot.
- **Risk preferences** — maximizing raw expected value assumes you are indifferent
  between a coin-flip and its average. A slider runs from risk-seeking through neutral
  to risk-averse, and the recommendation is re-derived on the **certainty equivalent**:
  the guaranteed payoff worth the same as the gamble to you.
- **Printable report** — one click produces a print/PDF version of the analysis, with
  the controls stripped and a masthead recording the question, model, risk setting and
  whether assumptions were adjusted.
- **Interactive decision tree** — rendered in the browser with vis-network (color-coded
  node types, probability edge labels, per-node EV, optimal path). No server-side image
  generation, so `matplotlib`/`networkx`/`graphviz` are no longer needed.
- **Structural sanity checks** — models often type a *gamble* as a "decision", which
  would make backward induction take `max()` and silently discard the downside. Branch
  probabilities mean chance, not choice, so such nodes are reclassified before anything
  is computed — and every repair is reported to you rather than applied silently.
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
game_theory.py  Probability normalization, backward-induction EV, risk preferences,
                Nash equilibria, dominance, Pareto efficiency
config.py       Settings (provider/model defaults, from .env)
templates/
  index.html    Single-page UI (provider toggle, model dropdown, interactive tree,
                payoff matrix, sensitivity analysis, risk slider, print report)
tests/
  test_game_theory.py   Decision-tree engine: normalization, repair, backward induction
  test_nash.py          Nash equilibria against classic games
  test_dominance.py     Dominated strategies, Pareto efficiency, computed game notes
  test_risk.py          Utility transform, certainty equivalents, risk-adjusted choice
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
`{"decision_tree": [...], "payoff_matrix": {...}, "risk_aversion": 0.0}` and returns the
same shape as `/api/analyze` with expected values, optimal path, certainty equivalents,
equilibria, dominance and Pareto results recomputed. This powers the sensitivity
analysis and the risk slider. `risk_aversion` runs from `-1` (risk-seeking) through `0`
(risk-neutral, the default) to `1` (strongly risk-averse).

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
  ],
  "dominated_strategies": [
    {"player": "row", "player_name": "Chain A", "strategy": "High Price", "dominated_by": "Price War", "kind": "strict", "description": "..."}
  ],
  "pareto_efficient": [
    {"row_strategy": "High Price", "col_strategy": "High Price", "row_payoff": 80, "col_payoff": 80}
  ],
  "game_notes": ["The equilibrium '...' is not Pareto efficient: both players do better at '...'"],
  "risk_aversion": 0.0,
  "optimal_certainty_equivalent": null
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

### Risk preferences (beyond expected value)

Plain expected value is **risk-neutral**: a 50/50 of `+100` / `−100` scores identically
to a certain `0`. Few people decide real stakes that way, so the sensitivity panel has a
risk-appetite slider. With it off centre the engine also computes a **certainty
equivalent** for every node — the guaranteed payoff you'd accept instead of that node's
gamble — and decisions are resolved on *that* rather than on raw EV.

The transform is exponential (constant absolute risk aversion) utility,
`u(x) = 1 − e^(−x/R)`. CARA is what makes it usable inside backward induction: folding
certainty equivalents up the tree gives the same answer as evaluating the whole compound
lottery at once, so every node's number stays in payoff units and stays comparable to
its EV. (`test_risk.py` pins that equivalence — the recursion is only valid for this
utility family.)

The slider is **dimensionless** on purpose. Payoffs are on an arbitrary per-analysis
scale, so the server converts the setting into a risk-tolerance constant `R` sized to
the payoff spread actually in play; the same slider position therefore means the same
thing whatever scale the model invented. At full aversion `R` is half the spread.

The headline EV is the expected value **of the path risk preference chose** — not an
independently maximized one, which would describe a strategy the tool isn't
recommending. The gap between the two is the *risk premium*: what you're paying for
certainty.

### Node types in the tree
- **Decision** (blue box) — a choice the actor controls
- **Chance** (amber diamond) — an uncertain event carrying a probability
- **Outcome** (green ellipse) — a terminal result carrying a payoff

### Why node types matter (and are corrected)

The type decides how a node is valued: a decision takes `max()` of its branches, a chance
node takes the probability-weighted average. So a gamble mislabeled as a decision keeps
only its best branch and **loses its downside entirely**.

Concretely — "Push hard" leading to a 50/50 of `+100` / `−100`:

| Typed as | Value | Recommendation |
|---|---|---|
| `decision` (wrong) | `max(100, −100)` = **100** | push hard — the risk has vanished |
| `chance` (correct) | `0.5x100 + 0.5x(−100)` = **0** | accept the safe branch instead |

`infer_node_types` applies these rules before any value is computed:
1. A non-root `decision` whose branches carry probabilities is reclassified to `chance`.
2. A root `decision` keeps its type (it *is* the choice being analysed); spurious branch
   probabilities are dropped.
3. A `decision`/`chance` node with no branches becomes a terminal `outcome`.
4. A `chance` node with no branch probabilities is flagged — equal likelihood is assumed.

Repairs appear in a **warnings panel** above the results and in the `warnings` field of
the API response, so the analysis is never quietly rewritten underneath you.

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

### Dominance and Pareto efficiency

An equilibrium says a cell is stable; it doesn't say *why*, or whether it's any good.
Two further computations do (`find_dominated_strategies`, `find_pareto_efficient`):

- **Dominated strategies** — a strategy no rational player should choose because another
  always beats it. This is the actual mechanism behind the Prisoner's Dilemma: each
  player's own arithmetic rules out cooperation, with no reasoning about the opponent
  needed. **Strict** dominance (better against *every* opponent strategy) and **weak**
  dominance (never worse, better somewhere) are reported separately, because "never worse
  and sometimes better" is a softer argument and shouldn't be presented with the same
  confidence. Dominated rows and columns are struck through in the matrix, labelled with
  what beats them.
- **Pareto-efficient cells** — outcomes where no player can be made better off without
  making the other worse off. Outlined in amber.

The contrast between the two is the most instructive output the tool produces. When the
equilibrium is *not* Pareto efficient, the app states it directly, names the outcome both
players would prefer, and notes that reaching it takes a binding agreement or repetition
and trust:

> The equilibrium 'Confess / Confess' (1, 1) is not Pareto efficient: both players do
> better at 'Stay silent / Stay silent' (3, 3), but neither can move there alone.

A player with one strategy that strictly beats all their others is also called out —
their choice is settled before any strategic reasoning begins.

## Tests

The deterministic engine (backward induction, Nash equilibria, tree repair) is covered
by unit tests. They are pure computation — no model or network needed, so they run in
well under a second.

```bash
pip install pytest    # already in requirements.txt
pytest                # run everything

pytest tests/test_nash.py -v          # just the Nash equilibrium tests
pytest tests/test_game_theory.py -v   # just the decision-tree engine
pytest tests/test_dominance.py -v     # dominance, Pareto efficiency, game notes
pytest tests/test_risk.py -v          # risk preferences and certainty equivalents
pytest -k optimal_path -v             # run tests matching a name
```

### What's covered

| Area | Tests |
|---|---|
| **Nash equilibria** | Prisoner's Dilemma (one pure NE), Matching Pennies (no pure, mixed 50/50), Battle of the Sexes (two pure + 2/3-1/3 mixed), dominant strategies, incomplete matrix, non-applicable matrix |
| **Backward induction** | Leaf = payoff, chance = probability-weighted, decision = max, equal-weight fallback for missing probabilities, cycle guard on malformed trees |
| **Optimal path** | Losing branches and the unchosen decision's subtree are *not* highlighted; stale flags cleared on recompute |
| **Tree repair** | Duplicate-id merging, dangling and self-reference removal, orphan subtree reattachment |
| **Node-type inference** | Gamble-typed-as-decision reclassified (and proven to stop discarding downside), root keeps its type, childless nodes become outcomes, unweighted chance nodes flagged, clean trees produce no warnings |
| **Normalization** | Outcome and sibling probabilities rescaled to sum to 1; all-zero falls back to equal weights |
| **Dominance** | Prisoner's Dilemma strict dominance for both players, weak dominance flagged separately, ties are not dominance, strict dominator preferred over weak, no false positives in Matching Pennies, non-square matrices |
| **Pareto efficiency** | PD equilibrium excluded, cells beaten on one side only excluded, all cells efficient in zero-sum, duplicate optima both kept |
| **Game notes** | Dominant strategy named per player, inefficiency note names the mutually preferred outcome, no false claims when dominance is partial |
| **Risk preferences** | CE below/above the mean when averse/seeking, certain outcomes worth face value, high tolerance converges on risk-neutral, no overflow at extreme payoff/tolerance ratios, tolerance scales with payoff spread, recommendation flips with appetite, nested lotteries fold consistently (the CARA property), stale CEs cleared on recompute |
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

# Node-type correction: "Push hard" is typed decision but has 50/50 branches.
# Expect it reclassified to chance, EV 40, and "Accept offer" recommended --
# NOT the 100 you would get if the downside were discarded.
curl -s -X POST localhost:8000/api/recompute -H 'Content-Type: application/json' -d '{
 "decision_tree":[
  {"id":"root","label":"Negotiate?","type":"decision","children":["push","safe"]},
  {"id":"push","label":"Push hard","type":"decision","children":["win","lose"]},
  {"id":"win","label":"They concede","type":"outcome","children":[],"probability":0.5,"payoff":100},
  {"id":"lose","label":"Offer rescinded","type":"outcome","children":[],"probability":0.5,"payoff":-100},
  {"id":"safe","label":"Accept offer","type":"outcome","children":[],"payoff":40}]}'

# Full analysis through a local model (slow — 30-90s)
curl -s -X POST localhost:8000/api/analyze -H 'Content-Type: application/json' \
  -d '{"query":"Two rival chains decide simultaneously whether to start a price war.","provider":"ollama"}'
```

## Roadmap

Planned work, known limitations, and implementation notes live in
[ROADMAP.md](ROADMAP.md) — next up is multi-sample confidence (every probability is a
single draw from a stochastic model) and staged progress feedback during long local runs.

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
