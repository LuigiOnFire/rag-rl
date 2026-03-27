import logging
from typing import Dict, Any, Generator, Optional
from datasets import load_dataset

from .base import BaseStreamer

logger = logging.getLogger(__name__)

class TwoWikiStreamer(BaseStreamer):
    """
    Streams the 2WikiMultihopQA dataset for the GreenRAG Oracle.
    """
    def __init__(self, setting: str = "fullwiki", split: str = "train", limit: Optional[int] = None, **kwargs):
        """
        Args:
            setting: Option for default or fullwiki modes.
            split: 'train', 'validation', or 'test'
            limit: If set, only load the first N samples (useful for debugging)
            kwargs: Extra args for dataset loading (e.g., path="framolfese/2WikiMultihopQAs")
        """
        self.setting = setting
        self.split = split
        self.limit = limit
        
        # You can override the HF path via kwargs, e.g. path="framolfese/2WikiMultihopQA" 
        dataset_path = kwargs.pop("path", "framolfese/2WikiMultihopQA")
        dataset_name = kwargs.pop("name", None)

        try:
            logger.info(f"Loading 2Wiki dataset from {dataset_path} split='{split}'...")
            if dataset_name:
                self.dataset = load_dataset(dataset_path, dataset_name, split=self.split, trust_remote_code=True, **kwargs)
            else:
                self.dataset = load_dataset(dataset_path, split=self.split, trust_remote_code=True, **kwargs)
                
            self.total_size: int = len(self.dataset) 
            
            if self.limit:
                logger.info(f"Limiting dataset to first {self.limit} samples.")
                self.dataset = self.dataset.select(range(self.limit))
                
        except Exception as e:
            logger.error(f"Failed to load 2Wiki: {e}")
            raise e

    def stream(self, shuffle: bool = False) -> Generator[Dict[str, Any], None, None]:
        if not hasattr(self, "dataset") or self.dataset is None:
             raise ValueError("Dataset not initialized.")
        dataset = self.dataset.shuffle() if shuffle else self.dataset
        for row in dataset:
            yield self._process_row(row)

    def _process_row(self, row: Any) -> Dict:
        """
        Converts HuggingFace format to GreenRAG format.
        2Wiki uses a very similar format to HotpotQA.
        """
        processed = {
            "id": row.get("_id", row.get("id", "")),
            "question": row["question"],
            "answer": row["answer"],
            "gold_titles": [], # Store titles of supporting facts
            "corpus": []       # Empty if fullwiki
        }

        # If we have context provided
        if self.setting != "fullwiki" and "context" in row:
             titles = row["context"]["title"]
             sentences_lists = row["context"]["sentences"]
             
             for title, sentences in zip(titles, sentences_lists):
                 # Join sentences into one block of text per document
                 full_text = f"{title}: {' '.join(sentences)}"
                 processed["corpus"].append(full_text)

        # Try to pull ground truth from "supporting_facts"
        if "supporting_facts" in row:
             processed["gold_titles"] = list(set(row["supporting_facts"]["title"]))

        return processed
