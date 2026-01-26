from typing import Any, Dict

def format_state_for_prompt(state: Dict[str, Any]) -> str:
    """
    The Single Source of Truth for how the Agent sees the world.
    Used by:
      1. Training (dataset.py) - to format history
      2. Inference (test/live) - to format live state
    """
    # 1. Header
    out = f"Goal: {state.get('main_query', 'Unknown')}\n"
    out += f"Status: {state.get('status', 'SOLVING')}\n"
    out += f"Scratchpad: {state.get('scratchpad', '')}\n\n"
    
    # 2. History
    out += "History:\n"
    history = state.get('history', [])
    if not history:
        out += "(None)\n"
    else:
        for i, item in enumerate(history):
            act = item.get('action_name', 'UNKNOWN')
            obs = item.get('observation', '')
            # Truncate long observations to save context window
            if len(str(obs)) > 200:
                obs = str(obs)[:200] + "... (truncated)"
            out += f"{i+1}. {act} -> {obs}\n"
    
    # 3. Subqueries
    out += "\nSub-Tasks:\n"
    subs = state.get('subqueries', [])
    for sub in subs:
        status = sub.get('status', 'PENDING')
        q = sub.get('question', '')
        ans = sub.get('answer')
        docs = len(sub.get('documents', []))
        
        # STRICT FORMAT: "[STATUS] Question" (No bullets!)
        line = f"[{status}] {q}"
        if ans: line += f" (Ans: {ans})"
        if docs > 0: line += f" [Docs: {docs}]"
        out += line + "\n"

        # 0: "GEN_SLM",
        # 1: "GEN_LLM",
        # 2: "RET_KEY",
        # 3: "RET_VEC",
        # 4: "GRD_SLM",
        # 5: "GRD_LLM",
        # 6: "RWT_SLM",
        # 7: "DEC_SLM",
        # 8: "DEC_LLM",
        # 9: "FAIL"

        out += """
AVAILABLE ACTIONS:
Type only the corresponding action ID (0-9) and provide the required input as specified.
------------------
[0] GEN_SLM (Answer Question with Small LLM)
   - Usage: You have sufficient information to answer the Main Goal.
   - Input: LEAVE EMPTY (The system will handle the prompt).

[1] GEN_LLM (Answer Question with Large LLM)
   - Usage: You have sufficient information to answer the Main Goal.
   - Input: LEAVE EMPTY (The system will handle the prompt).

[2] RET_KEY (Keyword Search)
   - Usage: Find specific facts, names, or dates.
   - Input: 2-4 keywords ONLY.
   - BAD: "Who is the CEO of Apple?" (Too verbose)
   - GOOD: "Apple CEO"

[3] RET_VEC (Dense/Concept Search)
   - Usage: Find explanations or broader concepts.
   - Input: A short natural language phrase.
   - BAD: "Compare the economic policies of..." (Too complex)
   - GOOD: "economic policies comparison"

[4] GRD_SLM (Grade/Verify with Small LLM)
   - Usage: Use this to quick-check if retrieved documents are actually relevant to the query.
   - Input: LEAVE EMPTY (The system will verify the last retrieval).

[5] GRD_LLM (Grade/Verify with Large LLM)
   - Usage: Use this for a deep consistency check. Verifies if a generated answer is hallucinated or unsupported by context.
   - Input: LEAVE EMPTY (The system will verify the last generation).

[6] RWT_SLM (Rewrite Query)
   - Usage: The current query is ambiguous, too conversational, or failed to yield results.
   - Input: The improved, standalone version of the query.
   - BAD: "Search for that again"
   - GOOD: "2024 population statistics France"

[7] DEC_SLM (Decompose into Sub-Tasks)
   - Usage: Call upon an SLM to break down a complex goal into simpler sub-tasks.
   - Input: LEAVE EMPTY (The system will handle the prompt).
   - Input: A single, simple sub-question.
------------------
"""
        print("Added action instructions to prompt.")
        return out