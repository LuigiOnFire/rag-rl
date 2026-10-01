import os
import gc
import pickle
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_CONFIGS = {
    "fullwiki": {
        "in_corpus": "data/meta/fullwiki_corpus.pkl",
        "flat_faiss": "data/meta/retriever_dense_fullwiki.faiss",
    },
    "squad_wiki": {
        "in_corpus": "data/meta/squad_wiki_corpus.pkl",
        "flat_faiss": "data/meta/retriever_dense_squad_wiki.faiss",
    }
}


def load_or_compute_embeddings(corpus_type: str, force_reencode: bool = False, batch_size: int = 256) -> np.ndarray:
    """
    Retrieves normalized float32 embeddings. 
    Reuses existing IndexFlatIP vectors if available to bypass redundant GPU passes.
    """
    config = CORPUS_CONFIGS[corpus_type]
    flat_path = config["flat_faiss"]
    in_corpus = config["in_corpus"]

    # Path 1: Instant load from existing flat FAISS index
    if os.path.exists(flat_path) and not force_reencode:
        print(f"Found existing flat index at {flat_path}.")
        print("Extracting precomputed normalized embeddings (bypassing re-encoding)...")
        flat_index = faiss.read_index(flat_path)
        ntotal = flat_index.ntotal
        dimension = flat_index.d

        print(f"Reconstructing {ntotal:,} vectors (d={dimension})...")
        embeddings = flat_index.reconstruct_n(0, ntotal)
        
        # Explicit garbage collection to prevent memory ballooning
        del flat_index
        gc.collect()
        return embeddings

    # Path 2: Re-encode raw documents using SentenceTransformer
    if not os.path.exists(in_corpus):
        raise FileNotFoundError(
            f"{in_corpus} not found. Please run scripts/00_build_corpus.py --corpus-type {corpus_type} first."
        )

    print(f"Loading corpus from {in_corpus}...")
    with open(in_corpus, "rb") as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents):,} documents.")

    print("Loading embedding model BAAI/bge-base-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")

    print(f"Encoding vectors with batch_size={batch_size} (normalize_embeddings=True)...")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    print("Building faiss.IndexFlatIP...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def build_ivf_index(embeddings: np.ndarray, nlist: int = 4096, nprobe: int = 64) -> faiss.Index:
    dimension = embeddings.shape[1]
    n_vectors = len(embeddings)
    print(f"Building faiss.IndexIVFFlat (nlist={nlist}, metric=METRIC_INNER_PRODUCT)...")
    
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train coarse centroids using a representative sample (minimum 39 * nlist points)
    sample_size = min(max(nlist * 40, 100_000), n_vectors)
    print(f"Training IVF coarse centroids on {sample_size:,} sample vectors...")
    rng = np.random.default_rng(42)
    train_indices = rng.choice(n_vectors, size=sample_size, replace=False)
    index.train(embeddings[train_indices])

    print(f"Adding {n_vectors:,} vectors into Voronoi inverted lists...")
    index.add(embeddings)
    index.nprobe = nprobe
    return index


def build_hnsw_index(embeddings: np.ndarray, m: int = 32, ef_construction: int = 64, ef_search: int = 64) -> faiss.Index:
    dimension = embeddings.shape[1]
    n_vectors = len(embeddings)
    print(f"Building faiss.IndexHNSWFlat (M={m}, metric=METRIC_INNER_PRODUCT)...")
    
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    print(f"Building HNSW graph over {n_vectors:,} vectors (efConstruction={ef_construction})...")
    # HNSW does not require explicit training; is_trained is True by default
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser(description="Build dense (FAISS) index for retrieval.")
    parser.add_argument(
        "--corpus-type",
        default="fullwiki",
        choices=["fullwiki", "squad_wiki"],
        help="Corpus type: fullwiki (HotpotQA) or squad_wiki (DPR)"
    )
    parser.add_argument(
        "--index-type",
        default="ivf",
        choices=["flat", "ivf", "hnsw"],
        help="Index topology: flat (exact), ivf (inverted file), or hnsw (graph)"
    )
    parser.add_argument(
        "--out-faiss",
        default=None,
        help="Optional destination path for the serialized index"
    )
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="Force forward-pass encoding even if an existing flat index is present"
    )
    # IVF parameters
    parser.add_argument("--nlist", type=int, default=4096, help="Centroid count for IndexIVFFlat")
    parser.add_argument("--nprobe", type=int, default=64, help="Probe depth for IndexIVFFlat")
    # HNSW parameters
    parser.add_argument("--hnsw-m", type=int, default=32, help="Bi-directional links per node for HNSW")
    parser.add_argument("--ef-construction", type=int, default=64, help="efConstruction for HNSW")
    parser.add_argument("--ef-search", type=int, default=64, help="efSearch for HNSW")

    args = parser.parse_args()

    # Determine output path
    if args.out_faiss:
        out_faiss = args.out_faiss
    else:
        suffix = f"_{args.index_type}" if args.index_type != "flat" else ""
        out_faiss = f"data/meta/retriever_dense_{args.corpus_type}{suffix}.faiss"

    # Step 1: Load or compute embeddings
    embeddings = load_or_compute_embeddings(args.corpus_type, force_reencode=args.force_reencode)

    # Step 2: Build selected topology
    if args.index_type == "flat":
        index = build_flat_index(embeddings)
    elif args.index_type == "ivf":
        index = build_ivf_index(embeddings, nlist=args.nlist, nprobe=args.nprobe)
    elif args.index_type == "hnsw":
        index = build_hnsw_index(
            embeddings,
            m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search
        )

    # Step 3: Serialize
    os.makedirs(os.path.dirname(out_faiss), exist_ok=True)
    print(f"Saving {args.index_type.upper()} index ({index.ntotal:,} vectors) to {out_faiss}...")
    faiss.write_index(index, out_faiss)
    print("Dense index generation complete!")


if __name__ == "__main__":
    main()