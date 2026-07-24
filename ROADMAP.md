# Roadmap / Backlog

Outstanding work, roughly in priority order. Each item notes *why* it matters, a sketch
of the implementation, and where it would live.

Guiding principle for this project: **the LLM models the situation; the code computes the
conclusions.** Prefer additions that are deterministic and testable over ones that ask the
model for an answer we could calculate ourselves.

---

## Tier 1 — done

1. **Dominant / dominated strategies + Pareto efficiency** — shipped. Strict and weak
   dominance reported separately, Pareto-efficient cells computed, and a plain-language
   note when the equilibrium is not Pareto efficient. See `find_dominated_strategies`,
   `find_pareto_efficient`, `describe_game` and `tests/test_dominance.py`.
2. **Risk preferences** — shipped. Exponential (CARA) utility with a dimensionless
   slider scaled to the payoff spread, certainty equivalents on every node, decisions
   resolved on the risk-adjusted value. See `compute_expected_values`,
   `risk_tolerance_for` and `tests/test_risk.py`.
3. **Export & shareable report** — shipped as a print/PDF view of the live DOM with a
   provenance masthead. **Markdown and raw-JSON export were deliberately not built**; if
   re-importing an analysis into `/api/recompute` turns out to matter, the JSON half is
   the piece to add.

---

## Tier 2 — meaningful, more involved

### 4. Multi-sample confidence
**Why:** every probability and payoff is a single draw from a stochastic model. One run
looks authoritative but isn't. Showing the spread would communicate real uncertainty.

**Sketch:** run the analysis N times (N=3-5) concurrently, cluster the outcomes, and
report mean +/- range for probabilities and EV. Note that local models make this N times
slower — worth gating behind a "thorough" toggle. Consider a fixed `seed` option for
reproducible single runs.

**Effort:** medium. Latency is the real constraint.

### 5. Streaming / better progress feedback
**Why:** local models take 30-90s and the UI shows only a spinner; long waits feel broken.

**Sketch:** Ollama supports `"stream": true`. Either stream tokens to a progress area, or
(simpler and more useful) report staged progress — contacting model / generating /
validating / computing — plus an elapsed timer. Streaming structured output is awkward to
render mid-flight, so staged status is probably the better cost/benefit.

**Effort:** medium.

### 6. Analysis history
**Why:** analyses vanish on reload, so you can't compare two framings of the same problem.

**Sketch:** persist to `localStorage` (zero infra) or SQLite for multi-user. Sidebar of
recent analyses; clicking one restores it, including adjusted assumptions. Naturally
enables a diff view between two analyses.

**Effort:** medium.

---

## Tier 3 — deeper game theory

### 7. Beyond two-player / 2x2 equilibria
**Current limits:** the payoff matrix is two-player only. Pure NE works for any nxm, but
mixed NE is implemented only for 2x2 (`_mixed_2x2`, via the indifference method).

**Sketch:** support mixed equilibria for larger games via support enumeration, and
n-player normal form (payoffs as an n-dimensional structure, which also needs a different
UI than a table). Consider whether `nashpy` is worth the dependency versus keeping the
solver self-contained and dependency-free.

**Effort:** large. Only worth it if real scenarios demand it.

### 8. Sequential games / subgame perfect equilibrium
**Why:** the decision tree currently models *one* actor's choices against chance. Real
strategic sequences have another player choosing at intermediate nodes.

**Sketch:** allow tree nodes to be owned by a player, then compute subgame perfect
equilibrium by backward induction over player-owned nodes. This is the natural bridge
between the tree view and the matrix view, and would let one situation be analysed both
ways consistently.

**Effort:** large, but it is the most intellectually valuable extension.

---

## Tier 4 — engineering hygiene

### 9. Hardening for non-local deployment
- `CORS_ORIGINS` defaults to `*` — fine locally, should be tightened before exposure.
- No rate limiting or request size caps on `/api/analyze`; a long query burns minutes of
  local GPU time. Add a max query length and a simple per-IP limit.
- No authentication — assumed to be a trusted network. Document or address.

### 10. Observability
Request IDs, per-stage timing (model latency vs compute), and a `/health` endpoint that
reports Ollama reachability and the configured model.

### 11. Model-quality eval harness
**Why:** development surfaced recurring malformed output — duplicate node ids, orphan
subtrees, dangling child refs, truncated JSON, and empty responses from thinking models.
These are handled defensively (`sanitize_tree`, `reconnect_orphan_roots`,
`infer_node_types`), but we have no measurement of *how often* each occurs per model.

**Sketch:** a script running a fixed set of scenarios against several models, recording
how many repairs each triggers. That turns "gemma4:12b seems okay" into evidence, and
guides which model to default to.

**Effort:** medium. Genuinely useful given how much repair logic exists.

### 12. Frontend polish
Dark mode, accessibility pass (the payoff matrix needs proper table semantics and labels
for the inline inputs), and a mobile layout check for the tree and matrix.

---

## Known limitations (accepted, documented for future readers)

- **Node-type inference is heuristic.** `infer_node_types` detects a mislabeled gamble by
  the presence of branch probabilities. A gamble typed as a `decision` whose branches have
  *no* probabilities is indistinguishable from a genuine choice without semantic
  understanding, so it is not corrected. This was observed in real output.
- **Payoffs are on an arbitrary scale.** The model invents a utility scale per analysis,
  so expected values are comparable *within* an analysis but never across analyses.
- **Probabilities are re-normalized.** Sibling probabilities are rescaled to sum to 1, so
  edited values may shift from exactly what was typed.
- **Mixed NE only for 2x2**, and only interior equilibria (`0 < p < 1`); degenerate cases
  return no mixed equilibrium.
- **Dominance is single-round.** Strategies dominated only *after* another dominated
  strategy is removed are not found; iterated elimination (IESDS) was considered and
  deferred, since it needs its own UI to show the rounds. Reported findings are capped at
  one per dominated strategy — the useful conclusion is "never play this", not every
  route to it.
- **Risk preference applies to the decision tree only.** Nash equilibria are computed on
  raw matrix payoffs, per the usual convention that those are already utilities. A
  monotonic risk transform would rarely move a pure NE anyway, but it *would* shift mixed
  equilibrium probabilities — which is exactly why applying it there would mislead.
- **The risk slider's curvature (`_RISK_CURVATURE`) is a judgement call**, not a measured
  preference. It sets how sharply aversion bites at full deflection; it is tuned so the
  whole slider is useful rather than to any particular decision-theoretic standard.
- **The printed report is the live DOM.** That guarantees it matches the screen, but it
  also means print fidelity depends on the browser: the decision tree prints as the
  canvas was drawn, so a large tree may print small.
- **Thinking mode is disabled** for Ollama (`think: false`) because thinking models spent
  their token budget reasoning and returned empty content. If a future model reasons
  better with it on, this is worth revisiting.
