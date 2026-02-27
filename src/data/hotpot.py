import logging
from typing import List, Dict, Generator, Any, Optional
from datasets import load_dataset, Dataset

from .base import BaseStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotpotQAStreamer(BaseStreamer):
    """
    Streams the HotpotQA dataset (Distractor setting) for the GreenRAG Oracle.
    """
    def __init__(self, split: str = "train", limit: Optional[int] = None):
        """
        Args:
            split: 'train', 'validation', or 'test'
            limit: If set, only load the first N samples (useful for debugging)
        """
        self.split = split
        self.limit = limit
        self.dataset: Optional[Any] = None
            
        try:
            # We use 'distractor' because it provides the Gold Paragraphs + Hard Negatives
            # This is ideal for our "Ephemeral Index" strategy.
            logger.info(f"Loading HotpotQA (distractor) split='{split}'...")
            self.dataset = load_dataset("hotpot_qa", "distractor", split=split) # NOTE: I removed trust_remote_code=True due to a hugging face error.
            
            if self.limit:
                logger.info(f"Limiting dataset to first {self.limit} samples.")
                self.dataset = self.dataset.select(range(self.limit))
                
        except Exception as e:
            logger.error(f"Failed to load HotpotQA: {e}")
            raise e

    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields cleaned samples one by one.
        """
        if self.dataset is None:
             raise ValueError("Dataset not initialized.")
        for row in self.dataset:
            yield self._process_row(row)

    def _process_row(self, row: Any) -> Dict:
        """
        Converts HuggingFace format to GreenRAG format.
        
        HF 'context' is: {'title': ['T1', 'T2'], 'sentences': [['S1', 'S2'], ['S1']]}
        We flatten this into a format ready for our Ephemeral Retriever.
        """
        processed = {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "gold_titles": [], # Store titles of supporting facts for Hindsight
            "corpus": []       # The flat text for the Retriever
        }

        # 1. Parse Context (The Haystack)
        # We flatten "Title" + "Sentences" into single strings for BM25/Dense indexing
        titles = row["context"]["title"]
        sentences_lists = row["context"]["sentences"]
        
        for title, sentences in zip(titles, sentences_lists):
            # Join sentences into one block of text per document
            full_text = f"{title}: {' '.join(sentences)}"
            processed["corpus"].append(full_text)

        # 2. Parse Supporting Facts (The Cheat Sheet for HER)
        # row['supporting_facts'] is {'title': [], 'sent_id': []}
        # We just need the titles to know which docs were the "Gold" ones.
        # (We deduplicate because a doc might be cited multiple times)
        processed["gold_titles"] = list(set(row["supporting_facts"]["title"]))

        return processed

# --- Verification Block ---
if __name__ == "__main__":
    print("Testing HotpotQA Streamer...")
    
    # Load just 3 samples to test
    streamer = HotpotQAStreamer(split="train", limit=3)
    
    for i, sample in enumerate(streamer.stream()):
        print(f"\n--- Sample {i+1} ---")
        print(f"Q: {sample['question']}")
        print(f"A: {sample['answer']}")
        print(f"Corpus Size: {len(sample['corpus'])} docs")
        print(f"First Doc: {sample['corpus'][0][:100]}...")
        print(f"Gold Docs: {sample['gold_titles']}")