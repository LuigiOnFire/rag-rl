import os
import json
import pickle
import argparse
from pyserini.index.lucene import LuceneIndexer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-type", choices=["fullwiki", "squad_wiki"])
    args = parser.parse_args()

    # 1. Load your existing corpus
    in_corpus = "data/meta/fullwiki_corpus.pkl" if args.corpus_type == "fullwiki" else "data/meta/squad_wiki_corpus.pkl"
    with open(in_corpus, "rb") as f:
        documents = pickle.load(f)

    # 2. Convert to Pyserini JSONL format (Required by Indexer)
    jsonl_dir = f"data/meta/{args.corpus_type}_jsonl"
    os.makedirs(jsonl_dir, exist_ok=True)
   
    with open(os.path.join(jsonl_dir, "corpus.jsonl"), "w") as f:
        for i, doc in enumerate(documents):
            # Pyserini needs 'id' and 'contents'
            f.write(json.dumps({"id": str(i), "contents": doc}) + "\n")

# 3. Build the Lucene Index
    index_dir = f"data/indices/{args.corpus_type}_index"
    os.makedirs(index_dir, exist_ok=True) 
    
    print(f"Indexing documents from {jsonl_dir} to {index_dir}...")
    
    # Use subprocess to call the CLI with explicit storage flags
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", jsonl_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storePositions", 
        "--storeDocvectors", 
        "--storeRaw" # <--- THIS IS THE CRITICAL FLAG
    ]
    
    subprocess.run(cmd, check=True)
    print("Lucene index complete!")
    
if __name__ == "__main__":
    main() 