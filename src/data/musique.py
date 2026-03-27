import logging
from typing import List, Dict, Generator, Any, Optional
from datasets import load_dataset, Dataset

from .base import BaseStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusiqueStreamer(BaseStreamer):
    """
    Streams the MuSiQue dataset for the GreenRAG Oracle.
    """
    def __init__(self, setting: str = "fullwiki", split: str = "train", limit: Optional[int] = None, **kwargs):
        """
        Args:
            setting: Option for default or fullwiki modes.
            split: 'train', 'validation', or 'test'
            limit: If set, only load the first N samples (useful for debugging)
            kwargs: Extra args for dataset loading.
        """
        self.setting = setting
        self.split = split
        self.limit = limit
        self.dataset: Optional[Any] = None
            
        try:
            logger.info(f"Loading MuSiQue dataset split='{split}'...")
            # MuSiQue requires named config 'default' if we use dgslibisey/musique, or we default to the standard HF path.
            self.dataset = load_dataset("dgslibisey/musique", "default", split=self.split, trust_remote_code=True, **kwargs)
            self.total_size: int = len(self.dataset) 
            
            if self.limit:
                logger.info(f"Limiting dataset to first {self.limit} samples.")
                self.dataset = self.dataset.select(range(self.limit))
                
        except Exception as e:
            logger.error(f"Failed to load MuSiQue: {e}")
            raise e

    def stream(self, shuffle: bool = False) -> Generator[Dict[str, Any], None, None]:
        if self.dataset is None:
             raise ValueError("Dataset not initialized.")
        dataset = self.dataset.shuffle() if shuffle else self.dataset
        for row in dataset:
            yield self._process_row(row)

    def _process_row(self, row: Any) -> Dict:
        """
        Converts HuggingFace format to GreenRAG format.
        """
        processed = {
            "id": row.get("id", ""),
            "question": row["question"],
            "answer": row["answer"],
            "gold_titles": [], # Store titles of supporting facts
            "corpus": []       # Empty if fullwiki
        }

        # If we have paragraphs provided, populate them conditionally
        if self.setting != "fullwiki" and "paragraphs" in row:
             for para in row["paragraphs"]:
                 title = para.get("title", "")
                 text = para.get("paragraph_text", "")
                 processed["corpus"].append(f"{title}: {text}")

        # Try to pull ground truth from "question_decomposition" if relevant in the MuSiQue schema format
        if "question_decomposition" in row:
             for step in row["question_decomposition"]:
                  if "paragraph_support" in step:
                       for support in step["paragraph_support"]:
                            processed["gold_titles"].append(support.get("title", ""))
             processed["gold_titles"] = list(set(processed["gold_titles"]))

        return processed
