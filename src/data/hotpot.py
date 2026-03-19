import logging
from typing import List, Dict, Generator, Any, Optional
from datasets import load_dataset, Dataset

from .base import BaseStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotpotQAStreamer(BaseStreamer):
    """
    Streams the HotpotQA dataset for the GreenRAG Oracle.
    Supports both 'distractor' (sandbox) and 'fullwiki' (online/real-world) settings.
    """
    def __init__(self, setting: str = "distractor", split: str = "train", limit: Optional[int] = None):        
        """
        Args:
            setting: 'distractor' (provides 10 paragraphs) or 'fullwiki' (provides only Q&A)
            split: 'train', 'validation', or 'test'
            limit: If set, only load the first N samples (useful for debugging)
        """
        if setting not in ["distractor", "fullwiki"]:
            raise ValueError("Setting must be 'distractor' or 'fullwiki'")

        self.setting = setting
        self.split = split
        self.limit = limit
        self.dataset: Optional[Any] = None
            
        try:
            logger.info(f"Loading HotpotQA ({setting}) split='{split}'...")
            # We dynamically pass the setting here
            self.dataset = load_dataset("hotpot_qa", self.setting, split=self.split) 
            self.total_size: int = len(self.dataset)  # full dataset size before any limit
            
            if self.limit:
                logger.info(f"Limiting dataset to first {self.limit} samples.")
                self.dataset = self.dataset.select(range(self.limit))
                
        except Exception as e:
            logger.error(f"Failed to load HotpotQA: {e}")
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
        Converts HuggingFace format to GreenRAG format.
        """
        processed = {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "gold_titles": [], # Store titles of supporting facts for Hindsight eval
            "corpus": []       # The text for the Retriever (Empty if fullwiki!)
        }

        # 1. Parse Context (The Haystack)
        # Only inject the pre-packaged context if we are in the distractor sandbox.
        # If we are in fullwiki mode, we intentionally leave 'corpus' empty to force live retrieval.
        if self.setting == "distractor" and "context" in row:
            titles = row["context"]["title"]
            sentences_lists = row["context"]["sentences"]
            
            for title, sentences in zip(titles, sentences_lists):
                # Join sentences into one block of text per document
                full_text = f"{title}: {' '.join(sentences)}"
                processed["corpus"].append(full_text)

        # 2. Parse Supporting Facts (The Cheat Sheet for HER)
        # We just need the titles to know which docs were the "Gold" ones.
        if "supporting_facts" in row:
            processed["gold_titles"] = list(set(row["supporting_facts"]["title"]))

        return processed

# --- Verification Block ---
if __name__ == "__main__":
    print("Testing HotpotQA Streamer (Distractor Sandbox)...")
    streamer_sandbox = HotpotQAStreamer(setting="distractor", split="train", limit=1)
    for sample in streamer_sandbox.stream():
        print(f"Q: {sample['question']}")
        print(f"Corpus Size: {len(sample['corpus'])} docs (Agent reads this instantly)")
        print(f"Gold Docs: {sample['gold_titles']}\n")

    print("Testing HotpotQA Streamer (Fullwiki Online Mode)...")
    streamer_online = HotpotQAStreamer(setting="fullwiki", split="train", limit=1)
    for sample in streamer_online.stream():
        print(f"Q: {sample['question']}")
        print(f"Corpus Size: {len(sample['corpus'])} docs (Agent must use SEARCH tool!)")
        print(f"Gold Docs: {sample['gold_titles']}")