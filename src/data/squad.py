import logging
from typing import Dict, Generator, Any, Optional
from datasets import load_dataset, Dataset

from .base import BaseStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQuADStreamer(BaseStreamer):
    """
    Streams the SQuAD v1.1 dataset for the GreenRAG Oracle.
    Note: SQuAD provides only questions and answers, no corpus or supporting facts.
    """
    def __init__(self, split: str = "train", limit: Optional[int] = None):        
        """
        Args:
            split: 'train' or 'validation' (test not available)
            limit: If set, only load the first N samples
        """
        self.split = split
        self.limit = limit
        self.dataset: Optional[Any] = None
            
        try:
            logger.info(f"Loading SQuAD split='{split}'...")
            self.dataset = load_dataset("squad", split=self.split)

            self.total_size: int = len(self.dataset)  # full dataset size before any limit
            
            if self.limit:
                logger.info(f"Limiting dataset to first {self.limit} samples.")
                self.dataset = self.dataset.select(range(min(self.limit, len(self.dataset))))
                
        except Exception as e:
            logger.error(f"Failed to load SQuAD: {e}")
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
        for row in dataset:
            yield self._process_row(row)

    def _process_row(self, row: Any) -> Dict:
        """
        Converts HuggingFace SQuAD format to GreenRAG format.
        """
        # SQuAD provides: question, id, answers (list of dicts with 'text' and 'answer_start')
        answers = row.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        
        processed = {
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "answer": answer_texts[0] if answer_texts else "",
            "gold_titles": [],
            "corpus": []  # SQuAD has no context corpus
        }
        
        return processed
