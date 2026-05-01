from typing import Any, Dict

def format_state_for_prompt(state):
   """
   Reconstructs the exact input prompt from the saved state dictionary.
   Includes DOCUMENTS/CONTEXT so the model can see what information it has access to when making decisions.
   """
   # 1. Header
   # Uses .get('question') which matches your JSON's "question" key
   prompt = f"Goal: {state.get('question', 'Unknown')}\n"
   prompt += f"Status: {state.get('status', 'SOLVING')}\n"
   prompt += f"Strategy (Long-term path to answer the Goal): {state.get('strategy', 'None')}\n"
   prompt += f"Plan (Short-term immediate next steps): {state.get('plan', 'None')}\n\n"
   
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

   documents = state.get('documents', [])
   if documents:
      prompt += "CONTEXT (Retrieved Documents):\n"
      for i, doc in enumerate(documents):
         title = doc.get('title', 'Unknown')
         content = doc.get('content', '')
         # Limit content length to prevent context overflow if needed
         prompt += f"[{i+1}] {title}: {content}\n"
      prompt += "\n"
   
   # 3. Sub-Tasks (Plan Order)
   subqueries = state.get('subqueries', [])
   prompt += "Sub-Tasks (in order):\n"
   if not subqueries:
      prompt += "(None)\n"
   else:
      for sub in subqueries:
            status_tag = f"[{sub.get('status', 'PENDING')}]"
            prompt += f"{status_tag} {sub['question']}\n"

   prompt += "\n"
   
   prompt += """
AVAILABLE ACTIONS:
Type only the corresponding action ID (0-8).
------------------
[0] GEN_SLM (Answer Question with Small LLM)
   - Usage: Answer the main query or the current active subquery cost-efficiently with a smaller language model.

[1] GEN_LLM (Answer Question with Large LLM)
   - Usage: Answer the main query or the current active subquery with the large language model.

[2] RET_KEY (Keyword Search)
   - Usage: Prompt an SLM to generate a search term to find specific facts, names, or dates.

[3] RET_VEC (Dense/Concept Search)
   - Usage: Prompt an LLM to generate a search term to find explanations or broader concepts.

[4] RSN_SLM (Reasoning Pass with Small LLM)
   - Usage: Think briefly about what is missing and choose the next action.

[5] RSN_LLM (Reasoning Pass with Large LLM)
   - Usage: Think more deeply to determine the best next action or strategy.

[6] DECOMPOSE_LLM (Decompose into Sub-Tasks)
   - Usage: Break down a multi-hop query into a list of simpler questions.

[7] DECOMPOSE_W_REASONING (Decompose with Reasoning)
   - Usage: Perform a deep reasoning trace, then decompose into sub-questions.

[8] FAIL (Abort)
   - Usage: Use this to abort the current task if it is unsolvable.
------------------
"""
   
   return prompt
