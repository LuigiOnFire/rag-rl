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
        
    out += "\nTask: Select the next best Action and Argument.\nAnswer:"
    return out