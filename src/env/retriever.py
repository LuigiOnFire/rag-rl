import os
import pickle
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

class GlobalRetriever:
    """
    Simulates a massive vector DB by loading a pre-built index consisting of a 
    large corpus of documents (e.g. all original Wikipedia abstracts).
    Supports multiple corpus types: 'fullwiki' and 'squad_wiki'.
    """
    _instances = {}  # Cache instances per corpus type
    
    @classmethod
    def get_instance(cls, corpus_type: str = "fullwiki", use_dense: bool = True):
        """
        Get or create a GlobalRetriever instance for a specific corpus type.
        
        Args:
            corpus_type: 'fullwiki' (default, 5M Wikipedia) or 'squad_wiki' (DPR Wikipedia)
            use_dense: Whether to use dense retrieval (FAISS) if available
        """
        key = (corpus_type, use_dense)
        if key not in cls._instances:
            cls._instances[key] = cls(corpus_type=corpus_type, use_dense=use_dense)
        return cls._instances[key]

    def __init__(self,
                 corpus_type: str = "fullwiki",
                 use_dense: bool = True):
        """
        Loads the pre-built BM25 index and optionally the FAISS dense index.
        
        Args:
            corpus_type: 'fullwiki' (HotpotQA Wikipedia) or 'squad_wiki' (DPR Wikipedia)
            use_dense: Whether to use dense retrieval (FAISS) if available
        """
        self.corpus_type = corpus_type
        
        # Map corpus type to file paths
        corpus_paths = {
            "fullwiki": {
                "sparse": "data/meta/retriever_sparse_fullwiki.pkl",
                "dense": "data/meta/retriever_dense_fullwiki.faiss"
            },
            "squad_wiki": {
                "sparse": "data/meta/retriever_sparse_squad_wiki.pkl",
                "dense": "data/meta/retriever_dense_squad_wiki.faiss"
            }
        }
        
        if corpus_type not in corpus_paths:
            raise ValueError(f"Unknown corpus_type: {corpus_type}. Choose from {list(corpus_paths.keys())}")
        
        sparse_path = corpus_paths[corpus_type]["sparse"]
        dense_path = corpus_paths[corpus_type]["dense"]
        
        if not os.path.exists(sparse_path):
            raise FileNotFoundError(
                f"Sparse index not found at {sparse_path}. "
                f"Build it first with:\n"
                f"  python scripts/00_build_corpus.py --corpus-type {corpus_type}\n"
                f"  python scripts/01a_build_sparse_index.py --corpus-type {corpus_type}"
            )
            
        print(f"Loading Sparse Index ({corpus_type}) from {sparse_path}...")
        with open(sparse_path, "rb") as f:
            data = pickle.load(f)
            
        self.bm25 = data["bm25"]
        self.documents = data["documents"]

        self.use_dense = use_dense
        self.faiss_index = None
        self.encoder_model = None

        if self.use_dense:
            if not os.path.exists(dense_path):
                print(f"Warning: Dense index not found at {dense_path}. Falling back to BM25 for dense queries.")
                self.use_dense = False
            else:
                import faiss
                print(f"Loading Dense FAISS Index ({corpus_type}) from {dense_path}...")
                self.faiss_index = faiss.read_index(dense_path)
                print("Loading SentenceTransformer model BAAI/bge-base-en-v1.5...")
                self.encoder_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')

    def search_bm25(self, query: str, k: int = 3) -> List[str]:
        tokenized_query = query.split()
        return self.bm25.get_top_n(tokenized_query, self.documents, n=k)

    def search_dense(self, query: str, k: int = 3) -> List[str]:
        if not self.use_dense or self.faiss_index is None:
            # Fallback to BM25 search
            return self.search_bm25(query, k)
            
        # Encode with normalize_embeddings=True to match IndexFlatIP logic
        query_vec = self.encoder_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.faiss_index.search(query_vec, k)
        
        # indices[0] contains the top-k document indices
        return [self.documents[i] for i in indices[0] if i != -1]
