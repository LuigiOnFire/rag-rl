import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class EphemeralRetriever:
    """
    Simulates a massive vector DB by creating a temporary index for each sample.
    """
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Load on demand
            cls._model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        return cls._model

    def __init__(self, documents: List[str]):
        """
        Initialize with the specific corpus for this sample (Gold + Distractors).
        """
        self.documents = documents
        
        # 1. Build Sparse Index (BM25)
        tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Build Dense Index (Vector)
        model = self.get_model()
        self.doc_embeddings = model.encode(documents)

    def search_bm25(self, query: str, k: int = 3) -> List[str]:
        tokenized_query = query.split()
        return self.bm25.get_top_n(tokenized_query, self.documents, n=k)

    def search_dense(self, query: str, k: int = 3) -> List[str]:
        model = self.get_model()
        query_vec = model.encode(query)
        
        scores = np.dot(self.doc_embeddings, query_vec)
        # Top k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]
