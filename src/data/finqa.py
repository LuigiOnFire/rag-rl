import json
import urllib.request
import logging
from typing import Dict, Generator, Any, Optional

# Adjust import path based on your repo structure
from .base import BaseStreamer

logger = logging.getLogger(__name__)

class FinQAStreamer(BaseStreamer):
    """
    Streams the FinQA dataset for GreenRAG.
    Parses the attached context into discrete documents (Pre-text, Table, Post-text)
    for use with EphemeralRetriever.
    """
    def __init__(self, split: str = "test", limit: Optional[int] = None):
        self.split = split
        self.limit = limit
        self.dataset = []
        
        try:
            url = f"https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
            logger.info(f"Loading FinQA split='{split}' from {url}...")
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                self.dataset = json.loads(response.read().decode('utf-8'))
            
            if self.limit:
                logger.info(f"Limiting FinQA dataset to {self.limit} samples.")
                self.dataset = self.dataset[:self.limit]
                
        except Exception as e:
            logger.error(f"Failed to load FinQA: {e}")
            raise e

    def stream(self, shuffle: bool = False) -> Generator[Dict[str, Any], None, None]:
        import random
        dataset_copy = self.dataset.copy()
        if shuffle:
            random.shuffle(dataset_copy)
            
        for row in dataset_copy:
            yield self._process_row(row)

    def _process_row(self, row: Any) -> Dict:
        """Converts FinQA format to GreenRAG format with discrete Ephemeral documents."""
        # Handle schema variances
        if "qa" in row and isinstance(row["qa"], dict):
            query = row["qa"].get("question", "")
            ans = str(row["qa"].get("exe_ans", ""))
            id_ = row.get("id", row["qa"].get("id", ""))
        else:
            query = row.get("question", "")
            ans = str(row.get("exe_ans", ""))
            id_ = row.get("id", "")

        pre_text = " ".join(row.get("pre_text", []))
        post_text = " ".join(row.get("post_text", []))
        
        # Format table to Markdown
        table_data = row.get("table", [])
        table_str = ""
        if isinstance(table_data, list):
            md_lines = []
            for i, r in enumerate(table_data):
                clean_row = [str(cell).strip() for cell in r]
                md_lines.append("| " + " | ".join(clean_row) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(clean_row)) + " |")
            table_str = "\n".join(md_lines)
        else:
            table_str = str(table_data)

        # Build Document list for EphemeralRetriever
        corpus = []
        if pre_text.strip():
            corpus.append({"title": "Pre-text", "text": pre_text})
        if table_str.strip():
            corpus.append({"title": "Table", "text": table_str})
        if post_text.strip():
            corpus.append({"title": "Post-text", "text": post_text})

        # Combined context for the Thrifty SLM check
        raw_context = f"Pre-text: {pre_text}\nTable:\n{table_str}\nPost-text: {post_text}"

        return {
            "id": id_,
            "question": query,
            "answer": ans,
            "gold_titles": [doc["title"] for doc in corpus],
            "corpus": corpus,          # Passed to EphemeralRetriever
            "raw_context": raw_context # Used by routers
        }