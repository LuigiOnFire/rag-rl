import ollama
from typing import List, Any
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

def _format_history(state: GreenState) -> str:
    hist_str = ""
    for step in state.history:
        # TODO: Lookup mnemonic from ID
        hist_str += f"Action: {step['action_id']} | Arg: {step['argument']} | Obs: {step['observation']}\n"
    return hist_str

def generate_search_query(state: GreenState, history: str) -> str:
    """
    Method for RET (2,3).
    Input Context: History + Previous Plan
    Argument: Specific Search String
    """
    # Simple prompt
    prompt = f"""
Current Question: {state.question}
History:
{history}

Based on the history, generate a specific, focused search query to find the next necessary piece of information.
Output ONLY the search query.
"""
    return slm_worker.generate(prompt).strip().strip('"')

def generate_plan(state: GreenState, history: str, use_llm: bool = False) -> str:
    """
    Method for DEC (7,8).
    Input Context: History (Ambiguous Query)
    Argument: List of Sub-questions
    """
    worker = llm_worker if use_llm else slm_worker
    prompt = f"""
The question "{state.question}" is complex.
History:
{history}

Decompose the question into a step-by-step plan of simple sub-questions.
Format: 1. [Sub-question] 2. [Sub-question] ...
"""
    return worker.generate(prompt).strip()

def generate_rewrite(state: GreenState, history: str) -> str:
    """
    Method for RWT (6).
    Input Context: History + Previous Observation
    Logic: Integrate Obs into Query.
    """
    prompt = f"""
Original Question: {state.question}
History:
{history}

The previous search result needs to be integrated into the query to find the next step.
Rewrite the query to be more specific based on the LAST observation.
Output ONLY the rewritten query.
"""
    return slm_worker.generate(prompt).strip().strip('"')

def generate_grade(state: GreenState, doc_text: str, use_llm: bool = False) -> str:
    """
    Method for GRD (4,5).
    Input Context: History + Last Retrieved Docs
    Argument: "Relevant" / "Irrelevant"
    """
    worker = llm_worker if use_llm else slm_worker
    prompt = f"""
Question: {state.question}
Document: {doc_text}

Is this document useful for answering the question?
Reply with EXACTLY one word: "Relevant" or "Irrelevant".
"""
    result = worker.generate(prompt).strip().lower()
    return "Relevant" if "relevant" in result else "Irrelevant"

def generate_answer(state: GreenState, history: str, use_llm: bool = False) -> str:
    """
    Method for GEN (0,1).
    Input Context: Full History
    Argument: Final Answer Text
    """
    worker = llm_worker if use_llm else slm_worker
    prompt = f"""
Question: {state.question}
Search History:
{history}

Based on the gathered information, provide the final answer to the question.
Output ONLY the answer.
"""
    return worker.generate(prompt).strip()
