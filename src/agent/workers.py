import ollama
from typing import List, TypedDict, Any
from src.env.state import GreenState, get_active_subquery
from src.env.retriever import EphemeralRetriever
import logging
import os

# Intialize a "Null" logger at first
# The handler will be set later by a higher level module
trace_logger = logging.getLogger("LLM_TRACE")
trace_logger.addHandler(logging.NullHandler())

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
            response_text = response['message']['content']
            # Log the interaction
            # We capture exactly what went in and what came out
            log_entry = (                
                f"MODEL: {self.model_name}\n"
                f" === INPUT (What LLM saw) === \n"
                f" {system}\n{prompt}\n"
                f" === OUTPUT (What LLM replied) === \n"
                f"{response_text}" 
            )
            trace_logger.debug(log_entry)

            return response_text
        except Exception as e:
            error_msg = f"Worker Error: {str(e)}"
            trace_logger.error(f"MODEL: {self.model_name}\nERROR: {error_msg}")
            return error_msg

slm_worker = LLMWorker(MODEL_SLM)
llm_worker = LLMWorker(MODEL_LLM)

# --- LOGGING UTILITY ---
def configure_worker_logging(log_path: str):
    """
    This function is called from the top level script (e.g. 02_trajectory.py)
    to direct LLM input/output logs to a specific file.
    """
    trace_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicate logs
    if trace_logger.hasHandlers():
        trace_logger.handlers.clear()
    
    # Making the directory should be handled by the top level function that made the path

    # Create the specific file handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')    
    file_handler.setFormatter(formatter)


    trace_logger.addHandler(file_handler)
    print(f" LLM Trace Logging intialized at: {log_path}")


# --- CORE SKILLS (Director Delegates These) ---
def generate_answer(state: GreenState, use_llm: bool = False) -> str:
    """
    Action 0/1: Synthesize facts into an answer.
    """
    worker = llm_worker if use_llm else slm_worker

    # Determine the active query by using the ID
    active_sub_query = get_active_subquery(state)
    if active_sub_query is None:
        query = state['question']
    else:
        query = active_sub_query['question']
    
    # 1. Gather Context
    context_str = ""
    found_docs = False
    
    # We look at all subqueries to build a "Knowledge Base" for the answer

    # First get top level docs
    for doc in state['documents']:
        # Leniency: Include UNKNOWN docs if we haven't graded them yet
        if doc.get('relevance', 'UNKNOWN') in ["RELEVANT", "UNKNOWN"]:
            context_str += f"- {doc['content']}\n"
            found_docs = True

    # Then get subquery docs
    for sub in state['subqueries']:
        for doc in sub['documents']:
            # Leniency: Include UNKNOWN docs if we haven't graded them yet
            if doc.get('relevance', 'UNKNOWN') in ["RELEVANT", "UNKNOWN"]:
                context_str += f"- {doc['content']}\n"
                found_docs = True
    
    if not found_docs:
        context_str = "No external documents found. Rely on internal knowledge."

    # Check the documents
    # print(f" Worker context length: {len(context_str)} characters.")
    # print(f" Worker context preview:\n{context_str[:500]}...\n")

    # 2. Prompt
    prompt = f"""
    Question: {query}
    
    Gathered Facts:
    {context_str}
    
    Instruction: Provide a direct, concise answer to the question based ONLY on the Gathered Facts. 
    If the facts are insufficient, say "I cannot answer based on available info."
    """
    
    return worker.generate(prompt).strip()

def generate_query_for_keyword_search(state: GreenState, use_llm: bool = False) -> str:
    """
    Look at the state and output a search string.
    Does NOT connect to any database.
    """
    active_sub = get_active_subquery(state)

    if active_sub is not None:
        active_query = active_sub['question']
    else:
        active_query = state['question']

    known_info = None
    if "documents" in active_query and active_sub is not None:
        known_info = "\n".join([f"- {d['title']} : {d['content'][:100]}..." for d in active_sub['documents']])

    if use_llm:
        worker = llm_worker
    else:
        worker = slm_worker

    if known_info is None:
        prompt = f"""
        Task: Create a concise search query to find information relevant to the Question.
        Question: "{active_query}"
        
        Constraint: Keep it under 10 words. Focus on key terms.
        
        Search Query:
        """
        
    else:
        prompt = f"""
        Task: Create a concise search query to find information relevant to the Question.
        Question: "{active_query}"
        
        We have already found these pages:
        {known_info}

        Constraint: Keep it under 10 words. Focus on key terms not already covered by Known Information.
        
        Search Query:
        """
    
    return worker.generate(prompt).strip()

def generate_query_for_vector_search(state: GreenState, use_llm: bool = False) -> str:
    """
    Look at the state and output a search string.
    Does NOT connect to any database.
    """
    active_sub = get_active_subquery(state)

    if active_sub is not None:
        active_query = active_sub['question']
    else:
        active_query = state['question']

    known_info = None
    if "documents" in active_query and active_sub is not None:
        known_info = "\n".join([f"- {d['title']} : {d['content'][:100]}..." for d in active_sub['documents']])

    if use_llm:
        worker = llm_worker
    else:
        worker = slm_worker

    if known_info is None:
        prompt = f"""
    Task: Create a query for a vector search to find information relevant to the Question.
    Question: "{active_query}"
    
    Constraint: Keep it under 10 words. Focus on key terms.
    
    Search Query:
    """
        
    else:
        prompt = f"""
    Task: Create a query for a vector search to find information relevant to the Question.
    Question: "{active_query}"
    
    We have already found these pages:
    {known_info}

    Constraint: Keep it under 10 words. Focus on key terms not already covered by Known Information.
    
    Search Query:
    """
    
    return worker.generate(prompt).strip()
    
def generate_grade(state: GreenState, doc_text: str, use_llm: bool = False) -> str:
    """
    Action 4/5: The Director says 'Check this doc', the Worker reads it.
    """
    active_sub = get_active_subquery(state)

    if active_sub is not None:
        active_query = active_sub['question']
    else:
        active_query = state['question']

    worker = llm_worker if use_llm else slm_worker
    
    prompt = f"""
    Task: Check if the Document contains information relevant to the Question.
    Question: "{active_query}"
    
    Document:
    "{doc_text[:2000]}" ... (truncated)
    
    Instruction: Reply with EXACTLY one word: "Relevant" or "Irrelevant".
    """
    
    result = worker.generate(prompt).strip().lower()
    return "Relevant" if "relevant" in result else "Irrelevant"

def generate_rewrite(state: GreenState) -> str:
    # Not strictly used in new flow as described, but good to keep hook.
    active_sub = get_active_subquery(state)

    if active_sub is None:
        return ""

    prompt = f"""
    Refine this query: {active_sub['question']}
    Based on recent history: {state['history'][-3:]}
    Output ONLY the rewritten query.
    """
    return slm_worker.generate(prompt).strip()

def _format_history(history: List[Any]) -> str:
    """Formats the conversation history for the worker context."""
    out = []
    for h in history:
        # Assuming h is a dict with action_name and observation
        name = h.get('action_name', 'UNKNOWN')
        obs = h.get('observation', '')
        out.append(f"Action: {name} -> Obs: {obs}")
    return "\n".join(out)

def generate_plan(state: GreenState, use_llm: bool = False) -> str:
    """
    Action 7/8 (Optional Support): Generates a step-by-step plan if the Director
    delegates the planning process entirely.
    """
    worker = llm_worker if use_llm else slm_worker
    
    question = state['question']
    
    # 1. Gather Context
    context_str = ""
    found_docs = False

    # First get top level docs
    for doc in state['documents']:
        # Leniency: Include UNKNOWN docs if we haven't graded them yet
        if doc.get('relevance', 'UNKNOWN') in ["RELEVANT", "UNKNOWN"]:
            context_str += f"- {doc['content']}\n"
            found_docs = True

    # Then get subquery docs
    for sub in state['subqueries']:
        for doc in sub['documents']:
            # Leniency: Include UNKNOWN docs if we haven't graded them yet
            if doc.get('relevance', 'UNKNOWN') in ["RELEVANT", "UNKNOWN"]:
                context_str += f"- {doc['content']}\n"
                found_docs = True
    
    if not found_docs:
        context_str = "No external documents found. Rely on internal knowledge."

    
    prompt = f"""
    Task: Break down the Main Question into 2-4 simple, independent sub-questions or search queries.
    Constraint: Return ONLY the numbered list. No intro, no filler.
    
    Main Question: "{question}"
        
    Plan:
    """
    
    # 2. Generate
    raw_plan = worker.generate(prompt)
    
    # 3. Cleaning
    clean_plan = raw_plan.replace("Here is the plan:", "").strip()
    
    return clean_plan