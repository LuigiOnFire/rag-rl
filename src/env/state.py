from typing import List, Literal, TypedDict, Optional, Any

class Document(TypedDict):
    title: str
    content: str
    relevance: Literal["UNKNOWN", "RELEVANT", "IRRELEVANT"]

class SubQuery(TypedDict):
    id: str
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
    main_query: str
    ground_truth: str  # The reference answer, for checking success during search/training.
    status: Literal["SOLVING", "SOLVED", "FAILED"]
    total_joules: float

    # 2. The Brain (Reasoning Traces)
    scratchpad: str  # e.g., "I need to check X because Y failed..."

    # 3. The Plan
    subqueries: List[SubQuery]

    # 4. History (Full History of Actions Taken for SFT)
    # Changed to Lightweight Items to prevent recursion bloat
    history: List[GreenHistoryItem]
    
    # 5. Metadata
    judge_log: Optional[str]

def create_initial_state(question: str, ground_truth: str = "") -> GreenState:
    return {
        "main_query": question,
        "ground_truth": ground_truth,
        "status": "SOLVING",
        "total_joules": 0.0,
        "scratchpad": "Goal: Answer the main query.",
        "subqueries": [{
            "id": "1",
            "question": question,
            "status": "ACTIVE",
            "answer": None,
            "documents": []
        }],
        "history": [],
        "judge_log": None
    }
