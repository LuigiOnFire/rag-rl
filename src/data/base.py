from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator

class BaseStreamer(ABC):
    def __init__(self, split: str = "train", limit: int = 0):
        self.split = split
        self.limit = limit

    @abstractmethod
    def stream(self) -> Iterator[Dict[str, Any]]:
        """
        MUST yield dictionaries with exactly:
        {
            "question": str,
            "answer": str,
            "corpus": List[Dict[str, str]] # [{'title': '...', 'content': '...'}]
        }
        """
        pass