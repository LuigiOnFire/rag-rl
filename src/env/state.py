from typing import TypedDict, List, Optional, Any

class GreenStep(TypedDict):
    action_id: int        # The Strategy (e.g., 2 for RET_KEY)
    argument: str         # The Tactics (e.g., "Shirley Temple political office")
    observation: str      # The Result (e.g., "Found 3 docs...")
    cost: float           # Energy cost of this step (Joules)

class GreenState:
    def __init__(self, question: str, ground_truth: str = ""):
        self.question = question
        self.ground_truth = ground_truth
        self.history: List[GreenStep] = []
        self.context: str = "" # Accumulated context
        self.depth: int = 0
        self.dec_count: int = 0
    
    @property
    def last_step(self) -> Optional[GreenStep]:
        return self.history[-1] if self.history else None

    @property
    def last_action(self) -> Optional[int]:
        return self.history[-1]['action_id'] if self.history else None
    
    @property
    def last_observation(self) -> str:
        return self.history[-1]['observation'] if self.history else ""

    def add_step(self, step: GreenStep):
        self.history.append(step)
        self.depth += 1
