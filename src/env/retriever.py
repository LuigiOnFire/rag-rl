import os
import pickle
import numpy as np
import json
import logging
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import sys

os.environ["OPENAI_API_KEY"] = "sk-not-needed"


try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    logging.warning("Pyserini not installed. Please run `pip install pyserini`.")
    LuceneSearcher = None

class EphemeralRetriever:
    """
    Simulates a massive vector DB by creating a temporary index for each sample.
    (Used for the 'distractor' setting with tiny, 10-paragraph contexts)
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
        
        # 1. Build Sparse Index (BM25) - Safe to use rank_bm25 for tiny document lists
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
    Supports multiple corpus types ('fullwiki', 'dpr_wiki', 'squad_wiki') 
    and index topologies ('hnsw', 'ivf', 'flat').
    """
    _instances = {}

    @classmethod
    def get_instance(
        cls, 
        corpus_type: str = "fullwiki", 
        use_dense: bool = True, 
        index_type: str = "hnsw",
        ef_search: int = 64,
        nprobe: int = 64
    ):
        key = (corpus_type, use_dense, index_type)
        if key not in cls._instances:
            cls._instances[key] = cls(
                corpus_type=corpus_type, 
                use_dense=use_dense, 
                index_type=index_type,
                ef_search=ef_search,
                nprobe=nprobe
            )
        return cls._instances[key]

    def __init__(
        self, 
        corpus_type: str = "fullwiki", 
        use_dense: bool = True, 
        index_type: str = "hnsw",
        ef_search: int = 64,
        nprobe: int = 64
    ):
        self.corpus_type = corpus_type
        self.index_type = index_type
        
        # --- 1. PYSERINI SPARSE INITIALIZATION ---
        if corpus_type in ["dpr_wiki", "squad_wiki"]:
            self.bm25_searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr')
        elif corpus_type == "fullwiki":
            self.bm25_searcher = LuceneSearcher("data/indices/fullwiki_index")

        # --- 2. DENSE INITIALIZATION ---
        self.use_dense = use_dense
        self.faiss_index = None
        self.encoder_model = None
        self.documents = None

        if self.use_dense:
            # Check for topology-specific index first, then fall back to flat
            candidate_path = f"data/meta/retriever_dense_{corpus_type}_{index_type}.faiss"
            flat_fallback = f"data/meta/retriever_dense_{corpus_type}.faiss"
            
            if os.path.exists(candidate_path):
                dense_path = candidate_path
            elif os.path.exists(flat_fallback):
                logging.warning(f"{candidate_path} not found. Falling back to {flat_fallback}")
                dense_path = flat_fallback
            else:
                dense_path = candidate_path

            text_path = f"data/meta/{corpus_type}_corpus.pkl"

            if os.path.exists(dense_path) and os.path.exists(text_path):
                import faiss
                logging.info(f"Loading FAISS index from {dense_path}...")
                self.faiss_index = faiss.read_index(dense_path)
                
                # Configure ANN search depth based on loaded topology
                if hasattr(self.faiss_index, "hnsw"):
                    self.faiss_index.hnsw.efSearch = ef_search
                elif hasattr(self.faiss_index, "nprobe"):
                    self.faiss_index.nprobe = nprobe
                
                with open(text_path, "rb") as f:
                    self.documents = pickle.load(f)
                self.encoder_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')
            else:
                logging.warning(f"Dense index ({dense_path}) or text file ({text_path}) missing. Dense search disabled.")
                sys.exit(1)
                
    def _extract_title(self, text: str, existing_title: str = None) -> tuple[str, str]:
        """
        Attempts to scrape a title from the text if it's missing or invalid.
        Returns a tuple of (title, clean_text).
        """
        invalid_titles = [None, "Unknown Title", "[Unknown Title]", "[None]", "None", ""]
        
        # If we already have a clean title from the JSON schema, use it.
        if existing_title not in invalid_titles:
            return existing_title, text
            
        clean_text = text.strip()
            
        # Regex Pattern 1: Catch the DPR format: "Title" followed by space/newline and Content
        match_quote = re.match(r'^"([^"]+)"\s*(.*)', clean_text, flags=re.DOTALL)
        if match_quote:
            return match_quote.group(1).strip(), match_quote.group(2).strip()

        # Regex Pattern 2: Catch the Colon format: Title: Content
        # We limit the title length to ~100 chars to avoid accidentally splitting on a 
        # colon that happens deep inside the first paragraph.
        match_colon = re.match(r'^([^:]{1,100}):\s+(.*)', clean_text, flags=re.DOTALL)
        if match_colon:
            return match_colon.group(1).strip(), match_colon.group(2).strip()
            
        # Absolute fallback if it doesn't match either pattern
        # If you prefer to use the first sentence as a fallback title instead of Unknown,
        # you could slice `clean_text` here, but "Unknown Title" is safer.
        return "Unknown Title", clean_text

    def search_bm25(self, query: str, k: int = 3) -> List[str]:
        hits = self.bm25_searcher.search(query, k=k)
        formatted_results = []
        
        for hit in hits:
            try:
                doc = self.bm25_searcher.doc(hit.docid)
                if not doc:
                    continue
                    
                doc_dict = json.loads(doc.raw()) 
                raw_text = doc_dict.get('contents', doc_dict.get('text', ''))
                raw_title = doc_dict.get('title')
                
                # Route through the scraper
                title, clean_text = self._extract_title(raw_text, raw_title)
                formatted_results.append(f"{title}: {clean_text}")
                    
            except Exception as e:
                logging.warning(f"Error parsing Lucene hit: {e}")
                
        return formatted_results

    def search_dense(self, query: str, k: int = 3) -> List[str]:
        # Safety guard for missing dense components
        if not self.use_dense or self.faiss_index is None or self.documents is None:
            return self.search_bm25(query, k)
            
        query_vec = self.encoder_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.faiss_index.search(query_vec, k)
        
        results = []
        if indices is not None and len(indices) > 0:
            for i in indices[0]:
                if i != -1 and i < len(self.documents):
                    raw_doc = self.documents[i] 
                    
                    # Dense search just has the raw text blob from the pickle.
                    # We pass None for existing_title to force the regex scraper.
                    title, clean_text = self._extract_title(raw_doc, None)
                    results.append(f"{title}: {clean_text}")
                    
        return results