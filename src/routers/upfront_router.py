import json
import re
from src.agent import actions
from src.env.engine import GreenEngine

CLASSIFICATION_PROMPT = """You are a financial query complexity auditor.
Analyze the provided QUERY and classify its execution complexity.

- SIMPLE: Single metric lookup, direct fact retrieval, or single-figure question.
- COMPLEX: Requires multi-year comparisons, percentage change calculations, cross-referencing tables with text, or multi-step arithmetic.

QUERY: {query}

Respond ONLY with a valid JSON object:
{{
  "complexity": "SIMPLE" or "COMPLEX",
  "reason": "Brief justification"
}}
"""

class UpfrontClassifierRouter:
    """
    Route 2: Upfront SLM Classifier Router (Adaptive-RAG Baseline).
    Uses an SLM to predict complexity from query semantics before retrieval.
    """
    def __init__(self, engine: GreenEngine, slm_client):
        self.engine = engine
        self.slm_client = slm_client

    def classify_query(self, query: str) -> str:
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        raw_resp = self.slm_client.generate(prompt)

        # Parse JSON output
        json_match = re.search(r'\{.*?\}', raw_resp, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return data.get("complexity", "COMPLEX").upper()
            except Exception:
                pass

        # String fallback if JSON parsing fails
        return "SIMPLE" if "SIMPLE" in raw_resp.upper() and "COMPLEX" not in raw_resp.upper() else "COMPLEX"

    def solve(self, start_state: dict) -> dict:
        query = start_state.get("question", "")
        classification = self.classify_query(query)
        current_state = start_state

        if classification == "SIMPLE":
            # Tau_light: Keyword Retrieval -> Fast SLM Generation
            current_state = self.engine.step(current_state, actions.ACTION_RET_KEY)
            current_state = self.engine.step(current_state, actions.ACTION_GEN_SLM)
        else:
            # Tau_heavy: Vector Retrieval -> Heavy LLM Generation
            current_state = self.engine.step(current_state, actions.ACTION_RET_VEC)
            current_state = self.engine.step(current_state, actions.ACTION_GEN_LLM)

        return current_state