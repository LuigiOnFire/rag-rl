import json
import re

from src.agent import actions
from src.env.engine import GreenEngine
# from src.routers.sufficiency import evaluate_sufficiency 

SUFFICIENCY_PROMPT = """You are a financial data auditor determining data completeness.
Analyze the QUERY and RETRIEVED CONTEXT.

Determine if the CONTEXT contains all raw facts, numbers, and figures required to compute or answer the QUERY. 

CRITICAL RULE: If the context contains the raw figures needed to perform a calculation (e.g., values for both 2014 and 2015), mark it as SUFFICIENT. Do NOT require the final calculated result to be explicitly written.

QUERY: {query}
RETRIEVED CONTEXT: {context}

Respond ONLY with a valid JSON object:
{{
  "status": "SUFFICIENT" or "INSUFFICIENT",
  "missing_info": "Brief explanation of missing raw data if INSUFFICIENT, else null"
}}
"""

class ThriftyEarlyExitRouter:
    def __init__(self, retriever):
        self.engine = GreenEngine(retriever=retriever)

    def solve(self, start_state: dict, slm_client) -> dict:
        """
        Executes Route 3 (Dynamic Early Exit) without ground truth access.
        """
        current_state = start_state

        # Step 1: Execute Light Retrieval (RET_KEY)
        current_state = self.engine.step(current_state, actions.ACTION_RET_KEY)

        # Step 2: Runtime Sufficiency Check
        obs = evaluate_sufficiency(
            slm_client, 
            query=current_state['question'], 
            context=current_state.get('context', '')
        )

        # Step 3: Policy Branching
        if obs['is_sufficient']:
            # Early Exit: Fast SLM Generation
            current_state = self.engine.step(current_state, actions.ACTION_GEN_SLM)
        else:
            # Escalation: Heavy Vector Retrieval + Heavy LLM Generation
            current_state = self.engine.step(current_state, actions.ACTION_RET_VEC)
            current_state = self.engine.step(current_state, actions.ACTION_GEN_LLM)

        return current_state

def parse_sufficiency_json(response_text: str) -> dict:
    """Extracts and parses JSON safely from SLM output."""
    # Handle markdown code blocks (```json ... ```)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        raw_json = json_match.group(1)
    else:
        # Fallback: look for first and last curly braces
        brace_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        raw_json = brace_match.group(0) if brace_match else response_text

    try:
        data = json.loads(raw_json)
        status = str(data.get("status", "")).upper()
        return {
            "is_sufficient": status == "SUFFICIENT",
            "missing_info": data.get("missing_info"),
            "raw": data
        }
    except Exception:
        # Robust fallback using string inspection if JSON decoding fails
        upper_resp = response_text.upper()
        is_sufficient = "SUFFICIENT" in upper_resp and "INSUFFICIENT" not in upper_resp
        return {
            "is_sufficient": is_sufficient,
            "missing_info": "JSON parsing error; string heuristic applied.",
            "raw": response_text
        }

def evaluate_sufficiency(slm_client, query: str, context: str) -> dict:
    """Executes sufficiency evaluation over query and context."""
    prompt = SUFFICIENCY_PROMPT.format(query=query, context=context)
    # Replace .generate() with your SLM wrapper invocation
    response = slm_client.generate(prompt)
    return parse_sufficiency_json(response)