import ollama
from typing import TypedDict, Any
from src.env.state import GreenState

# Models
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
        
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"Worker Error: {str(e)}"

slm_worker = LLMWorker(MODEL_SLM)
llm_worker = LLMWorker(MODEL_LLM)

def _get_active_subquery(state: GreenState):
    # Find last active or pending
    for sub in reversed(state['subqueries']):
        if sub['status'] in ["ACTIVE", "PENDING"]:
            return sub
    return state['subqueries'][0] # Fallback

# --- CORE SKILLS (Director Delegates These) ---

def generate_grade(state: GreenState, doc_text: str, use_llm: bool = False) -> str:
    """
    Action 4/5: The Director says 'Check this doc', the Worker reads it.
    """
    active_sub = _get_active_subquery(state)
    worker = llm_worker if use_llm else slm_worker
    
    prompt = f"""
    Task: Check if the Document contains information relevant to the Question.
    Question: "{active_sub['question']}"
    
    Document:
    "{doc_text[:2000]}" ... (truncated)
    
    Instruction: Reply with EXACTLY one word: "Relevant" or "Irrelevant".
    """
    
    result = worker.generate(prompt).strip().lower()
    return "Relevant" if "relevant" in result else "Irrelevant"

def generate_answer(state: GreenState, use_llm: bool = False) -> str:
    """
    Action 0/1: Synthesize facts into an answer.
    """
    active_sub = _get_active_subquery(state)
    worker = llm_worker if use_llm else slm_worker
    
    # 1. Gather Context
    context_str = ""
    found_docs = False
    
    # We look at all subqueries to build a "Knowledge Base" for the answer
    for sub in state['subqueries']:
        for doc in sub['documents']:
            # Leniency: Include UNKNOWN docs if we haven't graded them yet
            if doc.get('relevance', 'UNKNOWN') in ["RELEVANT", "UNKNOWN"]:
                context_str += f"- {doc['content']}\n"
                found_docs = True
    
    if not found_docs:
        context_str = "No external documents found. Rely on internal knowledge."

    # Check the documents
    print(f" Worker context length: {len(context_str)} characters.")
    print(f" Worker context preview:\n{context_str[:500]}...\n")

    # 2. Prompt
    prompt = f"""
    Question: {active_sub['question']}
    
    Gathered Facts:
    {context_str}
    
    Instruction: Provide a direct, concise answer to the question based ONLY on the Gathered Facts. 
    If the facts are insufficient, say "I cannot answer based on available info."
    """
    
    return worker.generate(prompt).strip()