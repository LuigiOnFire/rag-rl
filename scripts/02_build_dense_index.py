import os
import pickle
import argparse
import faiss
from sentence_transformers import SentenceTransformer

# Corpus-specific configurations
CORPUS_CONFIGS = {
    "fullwiki": {
        "in_corpus": "data/meta/fullwiki_corpus.pkl",
        "out_faiss": "data/meta/retriever_dense_fullwiki.faiss"
    },
    "squad_wiki": {
        "in_corpus": "data/meta/squad_wiki_corpus.pkl",
        "out_faiss": "data/meta/retriever_dense_squad_wiki.faiss"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Build dense (FAISS) index for retrieval.")
    parser.add_argument(
        "--corpus-type",
        default="fullwiki",
        choices=["fullwiki", "squad_wiki"],
        help="Corpus type: fullwiki (HotpotQA) or squad_wiki (DPR)"
    )
    args = parser.parse_args()

    config = CORPUS_CONFIGS[args.corpus_type]
    in_corpus = config["in_corpus"]
    out_faiss = config["out_faiss"]

    if not os.path.exists(in_corpus):
        raise FileNotFoundError(f"{in_corpus} not found. Please run scripts/00_build_corpus.py --corpus-type {args.corpus_type} first.")

    print(f"Loading corpus from {in_corpus}...")
    with open(in_corpus, "rb") as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents)} documents.")

    print("Loading embedding model BAAI/bge-base-en-v1.5...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')
    
    batch_size = 256
    print(f"Encoding vectors with batch_size={batch_size} (normalize_embeddings=True)...")
    
    embeddings = model.encode(
        documents, 
        batch_size=batch_size, 
        normalize_embeddings=True, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
            
    print("Building FAISS IndexFlatIP...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(out_faiss), exist_ok=True)
    
    print(f"Saving FAISS index to {out_faiss}...")
    faiss.write_index(index, out_faiss)

    print("Dense index generation complete!")

if __name__ == "__main__":
    main()
