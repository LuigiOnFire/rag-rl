import os
import pickle
import argparse
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Corpus-specific configurations
CORPUS_CONFIGS = {
    "fullwiki": {
        "in_corpus": "data/meta/fullwiki_corpus.pkl",
        "out_sparse": "data/meta/retriever_sparse_fullwiki.pkl"
    },
    "squad_wiki": {
        "in_corpus": "data/meta/squad_wiki_corpus.pkl",
        "out_sparse": "data/meta/retriever_sparse_squad_wiki.pkl"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Build sparse (BM25) index for retrieval.")
    parser.add_argument(
        "--corpus-type",
        default="fullwiki",
        choices=["fullwiki", "squad_wiki"],
        help="Corpus type: fullwiki (HotpotQA) or squad_wiki (DPR)"
    )
    args = parser.parse_args()

    config = CORPUS_CONFIGS[args.corpus_type]
    in_corpus = config["in_corpus"]
    out_sparse = config["out_sparse"]

    if not os.path.exists(in_corpus):
        raise FileNotFoundError(f"{in_corpus} not found. Please run scripts/00_build_corpus.py --corpus-type {args.corpus_type} first.")

    print(f"Loading corpus from {in_corpus}...")
    with open(in_corpus, "rb") as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents)} documents.")

    print("Tokenizing corpus...")
    tokenized_corpus = []
    for doc in tqdm(documents, desc="Tokenizing"):
        tokenized_corpus.append(doc.split())

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(out_sparse), exist_ok=True)
    
    print(f"Saving sparse index and documents to {out_sparse}...")
    with open(out_sparse, "wb") as f:
        # Saving both the index and the documents
        pickle.dump({"bm25": bm25, "documents": documents}, f)

    print("Sparse index generation complete!")

if __name__ == "__main__":
    main()
