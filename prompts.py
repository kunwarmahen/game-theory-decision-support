"""Prompt construction for the game-theory analyzer (single source of truth)."""

SYSTEM_PROMPT = (
    "You are an expert in game theory and decision analysis, grounded in Nash "
    "Equilibrium, Expected Value Theory, and strategic decision making. You reason "
    "carefully about incentives, information, and payoffs, and you always respond "
    "with a single valid JSON object that conforms to the provided schema."
)


def build_analysis_prompt(user_query: str) -> str:
    """Build the user prompt for a strategic-situation analysis.

    The JSON *shape* is enforced separately via structured outputs, so this prompt
    focuses on reasoning quality and the semantics of each field.
    """
    return f"""Analyze the following strategic situation and return your analysis as JSON.

SITUATION:
{user_query}

Think like a game theorist:
- Identify the players/stakeholders and their incentives.
- Consider the strategic interaction (who moves, what they know, how they respond).
- Estimate realistic probabilities for uncertain events.
- Assign numeric PAYOFFS to terminal outcomes so the best strategy can be computed
  by expected value.

Fill in these fields:

- "stakeholders": the key players/parties involved.
- "summary": a concise, insightful summary of the strategic situation and the core tension.
- "outcomes": 3-5 distinct possible outcomes. Each has:
    - "description", "probability" (0.0-1.0, across all outcomes summing to ~1.0),
      2-4 "key_factors", and an actionable "recommendation".
- "recommended_outcome": the most strategically favorable outcome and why.
- "decision_tree": 5-12 nodes modeling the decision as a tree. Node rules:
    - The root should be a "decision" node (a choice the actor controls; NO probability).
    - "chance" nodes represent uncertain events and MUST have a "probability".
      A parent's chance children should have probabilities summing to ~1.0.
    - "outcome" nodes are terminal leaves (empty "children") and MUST have a numeric
      "payoff" representing the utility of that end-state (higher is better; negatives allowed).
    - Every node needs a unique "id", a human-readable "label", a "type", and a
      "children" list of child ids.

- "payoff_matrix": OPTIONAL normal-form (simultaneous) game representation.
    - Set "applicable": true ONLY when the situation is a strategic interaction between
      TWO players who each choose from a small set of discrete strategies at the same time
      (e.g. two competitors setting prices, negotiation as concede/hold, an arms race,
      Prisoner's-Dilemma-style standoffs). For one-sided decisions under uncertainty, set
      "applicable": false and leave the other matrix fields empty.
    - When applicable: name the two players ("player_row", "player_col"), list each
      player's strategies ("row_strategies", "col_strategies", ideally 2 each), and provide
      one "cells" entry for EVERY (row_strategy, col_strategy) combination with the payoff
      to each player. Higher payoff = better. Do NOT try to identify the equilibrium
      yourself — it is computed automatically.

CONNECTIVITY (critical): the decision_tree MUST be a single, fully connected tree.
- There is EXACTLY ONE root node (the initial decision) that no other node lists as a child.
- Every other node must appear EXACTLY ONCE in some parent's "children" list — no orphan
  or disconnected nodes, and no node referenced by a "children" list that doesn't exist.
- Keep labels short (a few words); details belong in the outcomes section, not the node label.

Give realistic, well-reasoned estimates. Payoffs should reflect the relative
desirability of outcomes so that expected-value comparisons are meaningful."""
