import ollama
from typing import TypedDict, Any
from src.env.state import GreenState

# Models
# MODEL_SLM = "llama-3.2-1b-instruct" 
MODEL_SLM = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
MODEL_LLM = "llama3:8b"

class LLMWorker:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        
        response = ollama.chat(model=self.model_name, messages=messages)
        return response['message']['content']

slm_worker = LLMWorker(MODEL_SLM)
llm_worker = LLMWorker(MODEL_LLM)

def _get_active_subquery(state: GreenState):
    # Find last active or pending
    for sub in reversed(state['subqueries']):
        if sub['status'] in ["ACTIVE", "PENDING"]:
            return sub
    return state['subqueries'][0] # Fallback to main

def generate_search_query(state: GreenState) -> str:
    """
    Focus on state['subqueries'][-1] (The active problem) or the specific active one.
    """
    active_sub = _get_active_subquery(state)
    target_q = active_sub['question']
    
    prompt = f"""
Current Task: {target_q}
Context from previous steps: {state['scratchpad']}

Generate a specific search query to find information for this task.
Output ONLY the search query.
"""
    return slm_worker.generate(prompt).strip().strip('"')

def _format_history(history: list) -> str:
    out = []
    for h in history:
        # Assuming h is GreenHistoryItem or similar dict
        out.append(f"Action: {h.get('action_name', 'UNKNOWN')} -> Obs: {h.get('observation', '')}")
    return "\n".join(out)

def generate_plan(state: GreenState, use_llm: bool = False) -> str:
    """
    Generates a step-by-step plan.
    If use_llm=True, uses the 8B model for complex reasoning.
    """
    # 1. Select the Brain
    worker = llm_worker if use_llm else slm_worker
    
    # 2. Contextual Prompt
    # We need to see the main query and what we've already done.
    history_str = _format_history(state['history'])
    main_query = state['main_query']
    
    prompt = f"""
    Task: Break down the Main Question into 2-4 simple, independent sub-questions search queries.
    Constraint: Return ONLY the numbered list. No intro, no filler.
    
    Main Question: "{main_query}"
    
    Context (What we've done so far):
    {history_str}
    
    Plan:
    """
    
    # 3. Generate
    raw_plan = worker.generate(prompt)
    
    # 4. Cleaning (Crucial for weak models)
    # Remove "Here is the plan:" prefixes
    clean_plan = raw_plan.replace("Here is the plan:", "").strip()
    
    return clean_plan

def generate_rewrite(state: GreenState) -> str:
    # Not strictly used in new flow as described, but good to keep hook.
    active_sub = _get_active_subquery(state)
    prompt = f"""
Refine this query: {active_sub['question']}
Based on recent history: {state['history'][-3:]}
Output ONLY the rewritten query.
"""
    return slm_worker.generate(prompt).strip()

def generate_grade(state: GreenState, doc_text: str, use_llm: bool = False) -> str:
    active_sub = _get_active_subquery(state)
    worker = llm_worker if use_llm else slm_worker
    prompt = f"""
Question: {active_sub['question']}
Document: {doc_text}

Is this document useful for answering the question?
Reply with EXACTLY one word: "Relevant" or "Irrelevant".
"""
    result = worker.generate(prompt).strip().lower()
    return "Relevant" if "relevant" in result else "Irrelevant"

def generate_answer(state: GreenState, use_llm: bool = False) -> str:
    """
    Synthesize from state['subqueries'] (The gathered facts).
    """
    active_sub = _get_active_subquery(state)
    worker = llm_worker if use_llm else slm_worker
    
    # Gather context from all gathered docs (or just relevant ones)
    context_str = ""
    for sub in state['subqueries']:
        for doc in sub['documents']:
            if doc['relevance'] == "RELEVANT" or doc['relevance'] == "UNKNOWN":
                context_str += f"- {doc['content']}\n"
    
    prompt = f"""
Question: {active_sub['question']}
Gathered Facts:
{context_str}

Provide the final answer to the question using the gathered facts.
Output ONLY the answer.
"""
    return worker.generate(prompt).strip()
