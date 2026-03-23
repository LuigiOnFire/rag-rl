import os
import pickle
from rank_bm25 import BM25Okapi
from tqdm import tqdm

IN_CORPUS = "data/meta/fullwiki_corpus.pkl"
OUT_SPARSE = "data/meta/retriever_sparse_fullwiki.pkl"

def main():
    if not os.path.exists(IN_CORPUS):
        raise FileNotFoundError(f"{IN_CORPUS} not found. Please run scripts/00_prepare_corpus.py first.")

    print(f"Loading corpus from {IN_CORPUS}...")
    with open(IN_CORPUS, "rb") as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents)} documents.")

    print("Tokenizing corpus...")
    tokenized_corpus = []
    for doc in tqdm(documents, desc="Tokenizing"):
        tokenized_corpus.append(doc.split())

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(OUT_SPARSE), exist_ok=True)
    
    print(f"Saving sparse index and documents to {OUT_SPARSE}...")
    with open(OUT_SPARSE, "wb") as f:
        # Saving both the index and the documents
        pickle.dump({"bm25": bm25, "documents": documents}, f)

    print("Sparse index generation complete!")

if __name__ == "__main__":
    main()
