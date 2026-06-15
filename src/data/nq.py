import logging
from typing import Dict, Generator, Any, Optional
from datasets import load_dataset, Dataset

from .base import BaseStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NQStreamer(BaseStreamer):
    """
    Streams the Natural Questions (Open-Domain) dataset for the GreenRAG Oracle.
    Uses 'nq_open' which aligns with standard DPR/Adaptive-RAG evaluation,
    providing organic web queries and short answers without massive HTML contexts.
    """
    def __init__(self, setting: str = "fullwiki", split: str = "train", limit: Optional[int] = None):        
        """
        Args:
            split: 'train' or 'validation' 
            limit: If set, only load the first N samples
        """
        self.split = split
        self.limit = limit
        self.dataset: Optional[Any] = None
            
        try:
            logger.info(f"Loading NQ Open split='{split}'...")
            self.dataset = load_dataset("nq_open", split=self.split)

            self.total_size: int = len(self.dataset)  # full dataset size before any limit
            
            if self.limit:
                logger.info(f"Limiting dataset to first {self.limit} samples.")
                self.dataset = self.dataset.select(range(min(self.limit, len(self.dataset))))
                
        except Exception as e:
            logger.error(f"Failed to load NQ Open: {e}")
            raise e

    def stream(self, shuffle: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Yields cleaned samples one by one.
        
        Args:
            shuffle: If True, iterate in a random order (new seed each call).
        """
        if self.dataset is None:
             raise ValueError("Dataset not initialized.")
        dataset = self.dataset.shuffle() if shuffle else self.dataset
        
        # Enumerate to inject an ID since nq_open lacks native IDs
        for i, row in enumerate(dataset):
            yield self._process_row(i, row)

    def _process_row(self, index: int, row: Any) -> Dict:
        """
        Converts HuggingFace nq_open format to GreenRAG format.
        """
        # nq_open provides: 'question' (str) and 'answer' (list of acceptable strings)
        answers = row.get("answer", [])
        
        processed = {
            # nq_open does not have explicit IDs like SQuAD, so we generate a deterministic fallback
            # This is crucial for your .jsonl checkpointing system!
            "id": f"nq_{self.split}_{index}",
            "question": row.get("question", ""),
            # NQ often provides multiple acceptable answers; we take the primary one
            "answer": answers[0] if answers else "", 
            "gold_titles": [],
            "corpus": []  # Like SQuAD, nq_open relies on external retrieval
        }
        
        return processed