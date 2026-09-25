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

class GreenHistoryItem(TypedDict):
    """Lightweight history item (No pre_state recursion)."""
    action_id: int
    action_name: str    # "RET_KEY"
    argument: str       # "What is the capital?"
    observation: str    # "Found 3 docs..."
    cost: float
    input_state_size: int
    output_state_size: int
    duration_seconds: float

class GreenState(TypedDict):
    # 1. High Level
    question: str
    # ground_truth: str  # The reference answer, for checking success during search/training. This is no lonerg stored in the state, as the agent should not have direct access to it.
    status: Literal["SOLVING", "SOLVED", "FAILED"]
    total_joules: float
    documents: List[Document]    # The raw search hits
    answer: Optional[str]        # The final answer, once solved
    prev_searches: List[str]  # The raw search queries made, for reference and potential reuse in subqueries

    # 2. The Brain (Reasoning Traces)
    strategy: str # a longer term strategy to persist between actions, but can be overwritten
    plan: str # a shorter term plan that is replaced after one action

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
        "strategy": "No strategy formed yet. Consider using a Reasoning Pass.",
        "plan": "No immediate plan formed yet.",
        "subqueries": [],
        "history": [],
        "documents": [],
        "answer": None,
        "prev_searches": [],
        # "judge_log": None
    }

def get_active_subquery(state: GreenState):
    # Find first active or pending in natural order
    # NO LONGER sets state as active
    for sub in state['subqueries']:
        if sub['status'] in ["ACTIVE", "PENDING"]:
            return sub
    
    # If there are no subqueries, return None
    return None