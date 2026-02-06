from typing import List, Literal, TypedDict, Optional, Any

class Document(TypedDict):
    title: str
    content: str
    relevance: Literal["UNKNOWN", "RELEVANT", "IRRELEVANT"]

class SubQuery(TypedDict):
    id: int
    question: str
    status: Literal["PENDING", "ACTIVE", "ANSWERED", "FAILED"]
    answer: Optional[str]        # The extracted fact (e.g., "Shirley Temple")
    documents: List[Document]    # The raw search hits

class GreenHistoryItem(TypedDict):
    """Lightweight history item (No pre_state recursion)."""
    action_id: int
    action_name: str    # "RET_KEY"
    argument: str       # "What is the capital?"
    observation: str    # "Found 3 docs..."
    cost: float

class GreenState(TypedDict):
    # 1. High Level
    question: str
    # ground_truth: str  # The reference answer, for checking success during search/training. This is no lonerg stored in the state, as the agent should not have direct access to it.
    status: Literal["SOLVING", "SOLVED", "FAILED"]
    total_joules: float
    documents: List[Document]    # The raw search hits
    answer: Optional[str]        # The final answer, once solved


    # 2. The Brain (Reasoning Traces)
    scratchpad: str  # e.g., "I need to check X because Y failed..."

    # 3. The Plan
    subqueries: List[SubQuery]

    # 4. History (Full History of Actions Taken for SFT)
    # Changed to Lightweight Items to prevent recursion bloat
    history: List[GreenHistoryItem]
    
    # 5. Metadata
    # Commented out for now
    # I don't forsee this being used
    # judge_log: Optional[str]

def create_initial_state(question: str, ground_truth: str = "") -> GreenState:
    return {
        "question": question,
        "status": "SOLVING",
        "total_joules": 0.0,
        "scratchpad": "Goal: Answer the main query.",
        "subqueries": [],
        "history": [],
        "documents": [],
        "answer": None,
        # "judge_log": None
    }

def get_active_subquery(state: GreenState):
    # Find last active or pending
    # NO LONGER sets state as active
    for sub in reversed(state['subqueries']):
        if sub['status'] in ["ACTIVE", "PENDING"]:
            return sub
    
    # If there are no subqueries, return None
    return None