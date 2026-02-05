from typing import Any, Dict

def format_state_for_prompt(state):
   """
   Reconstructs the exact input prompt from the saved state dictionary.
   Now includes DOCUMENTS/CONTEXT so the model can actually answer!
   """
   # 1. Header
   # Uses .get('question') which matches your JSON's "question" key
   prompt = f"Goal: {state.get('question', 'Unknown')}\n"
   prompt += f"Status: {state.get('status', 'SOLVING')}\n"
   prompt += f"Scratchpad: {state.get('scratchpad', '')}\n\n"
   
   # 2. History
   history = state.get('history', [])
   prompt += "History:\n"
   if not history:
      prompt += "(None)\n"
   else:
      for item in history:
         a_name = item.get('action_name', 'Unknown')
         obs = item.get('observation', '')
         arg = item.get('argument', '')
         arg_str = f"({arg})" if arg else ""
         prompt += f"- Action: {a_name}{arg_str} | Result: {obs}\n"
         
   prompt += "\n"

   # --- NEW: DOCUMENTS SECTION ---
   # This was missing! The model needs to see what it found.
   documents = state.get('documents', [])
   if documents:
      prompt += "CONTEXT (Retrieved Documents):\n"
      for i, doc in enumerate(documents):
         title = doc.get('title', 'Unknown')
         content = doc.get('content', '')
         # Limit content length to prevent context overflow if needed
         prompt += f"[{i+1}] {title}: {content}\n"
      prompt += "\n"
   # ------------------------------
   
   # 3. Sub-Tasks (Stack)
   subqueries = state.get('subqueries', [])
   prompt += "Sub-Tasks:\n"
   if not subqueries:
      prompt += "(None)\n"
   else:
      for i, sub in enumerate(reversed(subqueries)):
            status_tag = "[ACTIVE]" if i == 0 else "[PENDING]"
            prompt += f"{status_tag} {sub['question']}\n"

   prompt += "\n"
   
   prompt += """
AVAILABLE ACTIONS:
Type only the corresponding action ID (0-9) and provide the required input as specified.
------------------
[0] GEN_SLM (Answer Question with Small LLM)
   - Usage: You have sufficient information to answer the Main Goal.

[1] GEN_LLM (Answer Question with Large LLM)
   - Usage: You have sufficient information to answer the Main Goal.

[2] RET_KEY (Keyword Search)
   - Usage: Find specific facts, names, or dates.

[3] RET_VEC (Dense/Concept Search)
   - Usage: Find explanations or broader concepts.

[4] GRD_SLM (Grade/Verify with Small LLM)
   - Usage: Use this to quick-check if retrieved documents are actually relevant to the query.

[5] GRD_LLM (Grade/Verify with Large LLM)
   - Usage: Use this for a deep consistency check. Verifies if a generated answer is hallucinated or unsupported by context.

[6] RWT_SLM (Rewrite Query)
   - Usage: The current query is ambiguous, too conversational, or failed to yield results.

[7] DEC_SLM (Decompose into Sub-Tasks)
- Usage: Call upon an SLM to break down a complex goal into simpler sub-tasks.

[8] DEC_LLM (Decompose into Sub-Tasks with Large LLM)
- Usage: Call upon an LLM to break down a complex goal into simpler sub-tasks.

[9] FAIL (Abort)
- Usage: Use this to abort the current task if it is unsolvable.
------------------
"""
   
   return prompt
