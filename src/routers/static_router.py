import re
from src.agent import actions
from src.env.engine import GreenEngine

class StaticHeuristicRouter:
    """
    Route 1: Static Heuristic Router (Advisor Baseline).
    Inspects query syntax upfront to assign a static trajectory.
    """
    COMPLEX_PATTERNS = [
        r"compare", r"difference", r"percentage change", r"versus", r"vs\.?",
        r"cumulative", r"ratio", r"growth", r"after accounting for"
    ]

    def __init__(self, engine: GreenEngine):
        self.engine = engine

    def classify_query(self, query: str) -> str:
        """Determines if query is LIGHT or HEAVY based on static rules."""
        # Rule 1: Multi-year detection (e.g., "in 2015 compare to 2014")
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", query)
        if len(set(years)) > 1:
            return "HEAVY"

        # Rule 2: Keyword triggering
        q_lower = query.lower()
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, q_lower):
                return "HEAVY"

        # Rule 3: String length threshold (> 120 chars)
        if len(query) > 120:
            return "HEAVY"

        return "LIGHT"

    def solve(self, start_state: dict) -> dict:
        query = start_state.get("question", "")
        trajectory = self.classify_query(query)
        current_state = start_state

        if trajectory == "LIGHT":
            # Tau_light: Keyword Retrieval -> Fast SLM Generation
            current_state = self.engine.step(current_state, actions.ACTION_RET_KEY)
            current_state = self.engine.step(current_state, actions.ACTION_GEN_SLM)
        else:
            # Tau_heavy: Vector Retrieval -> Heavy LLM Generation
            current_state = self.engine.step(current_state, actions.ACTION_RET_VEC)
            current_state = self.engine.step(current_state, actions.ACTION_GEN_LLM)

        return current_state