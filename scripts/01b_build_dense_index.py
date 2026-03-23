import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

IN_CORPUS = "data/meta/fullwiki_corpus.pkl"
OUT_FAISS = "data/meta/retriever_dense_fullwiki.faiss"

def main():
    if not os.path.exists(IN_CORPUS):
        raise FileNotFoundError(f"{IN_CORPUS} not found. Please run scripts/00_prepare_corpus.py first.")

    print(f"Loading corpus from {IN_CORPUS}...")
    with open(IN_CORPUS, "rb") as f:
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
    
    os.makedirs(os.path.dirname(OUT_FAISS), exist_ok=True)
    
    print(f"Saving FAISS index to {OUT_FAISS}...")
    faiss.write_index(index, OUT_FAISS)

    print("Dense index generation complete!")

if __name__ == "__main__":
    main()
