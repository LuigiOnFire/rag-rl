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

# Grab the variable, or provide a safe default just in case
LLM_NAME = os.getenv("LLM_MODEL", "llama3:8b")
SLM_NAME = os.getenv("SLM_MODEL", "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

class LLMWorker:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            # Add options={"temperature": 0.0} to lock the model to greedy decoding
            response = ollama.chat(
                model=self.model_name, 
                messages=messages,
                options={
                    "temperature": 0.0, 
                    "num_predict": 500,
                    "num_ctx": 8192
                } # <-- Forces it to stop after 500 tokens
            )
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

slm_worker = LLMWorker(SLM_NAME)
llm_worker = LLMWorker(LLM_NAME)


def _brain_context(state: GreenState, task_focus: str) -> str:
    """Render the agent's current strategy and plan with task-specific guidance."""
    strategy = state.get('strategy', 'None')
    plan = state.get('plan', 'None')
    return (
        f"Task Focus: {task_focus}\n"
        "Strategy = long-term path to solve the Goal (what information to find, in what order).\n"
        "Plan = immediate next steps for this task (what to do right now).\n"
        f"Current Strategy: {strategy}\n"
        f"Current Plan: {plan}\n"
    )

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
    is_subquery = active_sub_query is not None

    if active_sub_query is None:
        question_text = state['question']
    else:
        question_text = active_sub_query['question']

    # 1. Gather Sub-Task History (Q&A)
    # --------------------------------
    # This creates a log of what the agent has already solved.
    history_parts = []
    for i, sq in enumerate(state.get('subqueries', [])):
        # Only include answered queries to avoid confusing the model
        if sq.get('status') == "ANSWERED" and sq.get('answer'):
            history_parts.append(f"Sub-question {i+1}: {sq['question']}")
            history_parts.append(f"Answer: {sq['answer']}\n")
    qa_section = "\n".join(history_parts)

    # 2. Gather Relevant Documents (Facts)
    # ------------------------------------
    # Collect docs from the main state
    all_docs = state.get('documents', [])
    
    doc_parts = []
    for doc in all_docs:
        # Leniency: Include UNKNOWN (ungraded) or RELEVANT docs
        # We explicitly exclude IRRELEVANT docs to save context window
        if doc.get('relevance', 'UNKNOWN') in ["RELEVANT", "UNKNOWN"]:
            title = doc.get('title', 'Unknown Source')
            content = doc.get('content', '').strip()
            doc_parts.append(f"- [{title}]: {content}")
    
    # 3. Handle Fallback & Assembly
    # -----------------------------
    if doc_parts:
        facts_section = "\n".join(doc_parts)
    else:
        # This specific string triggers the "NO_CONTEXT" logic in your prompt
        facts_section = "No relevant documents found."

    # Combine nicely with headers
    context_blocks = []
    if qa_section:
        context_blocks.append(f"### COMPLETED SUB-TASKS:\n{qa_section}")
    
    context_blocks.append(f"### GATHERED FACTS:\n{facts_section}")
    
    context_str = "\n\n".join(context_blocks)
    # Check the documents
    # print(f" Worker context length: {len(context_str)} characters.")
    # print(f" Worker context preview:\n{context_str[:500]}...\n")

    # 4. Prompt Generation
    # -------------------
    if is_subquery:
        # PURE EXTRACTION: For intermediate steps. No comparing facts or forcing "winning entities".
        prompt = f"""
### INSTRUCTION
Extract the exact answer to the Sub-Question from the Context below.
Output ONLY the specific entity, date, or fact requested. Be brief and concise. Do not write complete sentences.
If the answer is not in the context, output exactly: "No relevant information found."

### CURRENT TASK
{_brain_context(state, "Sub-query answer extraction")}
Context:
{context_str}

Sub-Question: "{question_text}"

### ANSWER
"""
    else:
        # SYNTHESIS & COMPARISON: For the final answer. Strictly forces a winning entity.
        prompt = f"""
### INSTRUCTION
Extract the exact answer to the Main User Question from the Context below.
Output ONLY the entity (name, date, number). Do not write complete sentences. Be robotic and concise.
Do not list facts. Compare the facts and output ONLY the final winning entity.

### EXAMPLES
Question: "What is the capital of France?"
Good Answer: "Paris"

Question: "Who won the 1996 World Series?"
Good Answer: "New York Yankees"

### CURRENT TASK
{_brain_context(state, "Final answer synthesis")}
Context:
{context_str}

Main Question: "{question_text}"

### ANSWER
"""
    
    return worker.generate(prompt).strip()

def generate_query_for_keyword_search(state: GreenState, use_llm: bool = False) -> str:
    """
    Look at the state and output a search string.
    Does NOT connect to any database.
    """
    worker = llm_worker if use_llm else slm_worker
    
    # 1. Determine the active question
    active_sub = get_active_subquery(state)
    active_query = active_sub['question'] if active_sub is not None else state['question']

    # 2. Format Known Information (Targeting the correct document list)
    target_docs = state.get('documents', [])
    known_info_str = ""
    
    if target_docs:
        doc_titles = "\n".join([f"- {d.get('title')}..." for d in target_docs])
        known_info_str = f"We have already found these pages:\n{doc_titles}\n"

    # 3. Format Previous Searches (The Taboo List)
    prev_searches_str = ""
    prev_searches = state.get('prev_searches', [])
    
    if prev_searches:
        search_bullets = "\n".join([f"- {s}" for s in prev_searches])
        prev_searches_str = (
            f"Previous searches we already tried (DO NOT REPEAT THESE):\n"
            f"{search_bullets}\n"
        )

    # 4. Build Dynamic Constraints
    constraints = ["Keep it under 10 words.", "Focus on key terms."]
    if known_info_str:
        constraints.append("Do not search for concepts already covered by the pages found.")
    if prev_searches_str:
        constraints.append("You MUST try a new semantic angle or completely different keywords.")
        
    constraint_text = " ".join(constraints)

    # 5. Final Master Prompt
    # Had some trouble with models doing SQL or code generation, so I modified the promt.
    prompt = f"""
Task: Create a query for a keyword search to find information relevant to the Question.
Do NOT write SQL. Do NOT write code.
Question: "{active_query}"

{_brain_context(state, "Keyword search query")}

{known_info_str}
{prev_searches_str}
Constraint: {constraint_text}

Search Query:
"""
    return worker.generate(prompt).strip()

def generate_query_for_vector_search(state: GreenState, use_llm: bool = False) -> str:
    """
    Look at the state and output a search string.
    Does NOT connect to any database.
    """
    worker = llm_worker if use_llm else slm_worker
    
    # 1. Determine the active question
    active_sub = get_active_subquery(state)
    active_query = active_sub['question'] if active_sub is not None else state['question']

    # 2. Format Known Information (Targeting the correct document list)
    target_docs = state.get('documents', [])
    known_info_str = ""
    
    if target_docs:
        doc_titles = "\n".join([f"- {d.get('title')}..." for d in target_docs])
        known_info_str = f"We have already found these pages:\n{doc_titles}\n"

    # 3. Format Previous Searches (The Taboo List)
    prev_searches_str = ""
    prev_searches = state.get('prev_searches', [])
    
    if prev_searches:
        search_bullets = "\n".join([f"- {s}" for s in prev_searches])
        prev_searches_str = (
            f"Previous searches we already tried (DO NOT REPEAT THESE):\n"
            f"{search_bullets}\n"
        )

    # 4. Build Dynamic Constraints
    constraints = ["Keep it under 10 words.", "Focus on key terms."]
    if known_info_str:
        constraints.append("Do not search for concepts already covered by the pages found.")
    if prev_searches_str:
        constraints.append("You MUST try a new semantic angle or completely different keywords.")
        
    constraint_text = " ".join(constraints)

    # 5. Final Master Prompt
    prompt = f"""
Task: Create a query for a vector search to find information relevant to the Question.
Do NOT write SQL. Do NOT write code. Do NOT use markdown formatting.
Question: "{active_query}"

{_brain_context(state, "Vector search query")}

{known_info_str}
{prev_searches_str}
Constraint: {constraint_text}

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

{_brain_context(state, "Document relevance check")}

Document:
"{doc_text[:2000]}" ... (truncated)

Instruction: Reply with EXACTLY one word: "Relevant" or "Irrelevant".
    """
    
    result = worker.generate(prompt).strip().lower()
    return "Relevant" if "relevant" in result else "Irrelevant"

def _format_history(history: List[Any]) -> str:
    """Formats the conversation history for the worker context."""
    out = []
    for h in history:
        # Assuming h is a dict with action_name and observation
        name = h.get('action_name', 'UNKNOWN')
        obs = h.get('observation', '')
        out.append(f"Action: {name} -> Obs: {obs}")
    return "\n".join(out)

def generate_rewrite(state: GreenState) -> str:
        active_sub = get_active_subquery(state)
        if active_sub is None:
            return ""

        # Gather the answers we already know
        resolved_context = []
        for i, sq in enumerate(state.get('subqueries', [])):
            if sq.get('status') == "ANSWERED" and sq.get('answer'):
                resolved_context.append(f"- {sq['question']} -> {sq['answer']}")
        
        context_str = "\n".join(resolved_context) if resolved_context else "None"

        # Had SQL problems here to so I modified the prompt.
        prompt = f"""
    Task: Update the Original Question to be more specific by injecting information from the Known Facts.
    Do NOT write SQL. Do NOT write code. Write a normal English sentence.    
    Resolved Queries:
    {context_str}

    {_brain_context(state, "Rewrite subquery")}

    Target Query to Rewrite: "{active_sub['question']}"
    
    Constraint: Output ONLY the rewritten query. Do not answer it. If no rewrite is needed, output the original Target Query.
    
    Rewritten Query:"""
        
        return slm_worker.generate(prompt).strip()

def generate_reasoning(state: GreenState, use_llm: bool = False) -> str:
    """
    Action 4/5: Generates a short-term plan (SLM) or long-term strategy (LLM).
    """
    worker = llm_worker if use_llm else slm_worker
    
    question = state['question']
    
    # 1. Gather Context (Documents)
    context_str = ""
    found_docs = False
    for doc in state.get('documents', []):
        if doc.get('relevance', 'UNKNOWN') in ["RELEVANT", "UNKNOWN"]:
            context_str += f"- {doc['content']}\n"
            found_docs = True

    if not found_docs:
        context_str = "No external documents found yet."

    # 2. Gather Recent History (Crucial for Reasoning)
    # Get the last 3 actions so the model knows what just succeeded/failed
    history_str = ""
    recent_history = state.get("history", [])[-3:] 
    for entry in recent_history:
        history_str += f"- Action: {entry.get('action_name')} | Result: {entry.get('observation')}\n"
        
    if not history_str:
        history_str = "No previous actions taken."

    # 3. Identify Active Sub-task (if any)
    active_sub = None
    for sub in state.get('subqueries', []):
        if sub.get('status') == 'ACTIVE':
            active_sub = sub['question']
            break
    
    task_focus = f"\nCurrently Active Sub-Task: {active_sub}" if active_sub else ""

    # 4. Build the Dynamic Prompt
    base_prompt = (
        f"Main Question: {question}{task_focus}\n\n"
        f"{_brain_context(state, 'Reasoning pass')}\n"
        f"Recent History (Last 3 actions):\n{history_str}\n"
        f"Current Knowledge Context:\n{context_str}\n"
    )

    if use_llm:
        # RSN_LLM: Long-term Strategy
        instruction = (
            "You are a strategic planning AI. Your job is to set the long-term strategy. "
            "Analyze the Main Question, the Current Knowledge, and the Recent History. "
            "Write a concise, high-level strategy (2-4 sentences) outlining the exact sequence of "
            "information we still need to find, and how we should approach it. "
            "Do not output a numbered list. Return ONLY the strategy text."
        )
    else:
        # RSN_SLM: Short-term Plan
        instruction = (
            "You are a tactical AI assistant. Your job is to determine the immediate next step. "
            "Look at the Recent History and Current Knowledge. Write exactly 1 or 2 sentences stating "
            "what our VERY NEXT action should be (e.g., 'We need to run a Keyword Search for X' or "
            "'We have the answer, generate the final response'). "
            "Return ONLY the plan text."
        )

    full_prompt = f"{base_prompt}\n\n{instruction}"
    
    # 5. Call the worker
    reasoning_text = worker.generate(full_prompt).strip()
    
    return reasoning_text
def generate_plan(state: GreenState, reasoning_mode = False) -> str:
    """
    Action 6/7 (Optional Support): Generates a step-by-step plan if the Director
    delegates the planning process entirely.
    """   
    question = state['question']
    worker = llm_worker

    
    # 1. Gather Context
    context_str = ""
    found_docs = False

    # First get top level docs
    for doc in state['documents']:
        # Leniency: Include UNKNOWN docs if we haven't graded them yet
        if doc.get('relevance', 'UNKNOWN') in ["RELEVANT", "UNKNOWN"]:
            context_str += f"- {doc['content']}\n"
            found_docs = True

    if not found_docs:
        context_str = "No external documents found. Rely on internal knowledge."
    
    # 2. Build the Dynamic Prompt
    base_prompt = (
        "You are an expert AI search planner.\n"
        f"Main Question: {question}\n"
        f"{_brain_context(state, 'Decomposition planning')}\n"
        f"Context:\n{context_str}\n\n"
        "Task: Break down the Main Question into 2-4 simple, independent search queries."
        "CRITICAL: Every sub-query must be fully self-contained. NEVER use pronouns (he, she, it, they). "
        "Always repeat the specific names or entities involved."
    )

    if reasoning_mode:
        # DEC_RSN: Force the reasoning trace and the explicit separator
        instruction = (
            "INSTRUCTIONS:\n"
            "1. First, analyze the dependencies of the query and context. Write a brief step-by-step reasoning trace of what information is missing.\n"
            "2. Second, you MUST write the exact word `---SUBQUERIES---` on a new line.\n"
            "3. Finally, provide the sub-queries as a numbered list. Do not add any text after the list."
        )
    else:
        # DEC_LLM: Zero-shot, strict list generation
        instruction = (
            "INSTRUCTIONS:\n"
            "Return ONLY a valid numbered list of the sub-queries. No intro, no filler, no reasoning."
        )

    full_prompt = f"{base_prompt}\n\n{instruction}"
    
    # 2. Generate
    plan = worker.generate(full_prompt)
    
    return plan