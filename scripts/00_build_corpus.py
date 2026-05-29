import os
import json
import tarfile
import requests
import pickle
import gzip
import argparse
from tqdm import tqdm

# Corpus-specific configurations
CORPUS_CONFIGS = {
    "fullwiki": {
        "dump_url": "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
        "dump_path": "data/meta/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2",
        "out_corpus": "data/meta/fullwiki_corpus.pkl"
    },
    "squad_wiki": {
        "dump_url": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
        "dump_path": "data/meta/psgs_w100.tsv.gz",
        "out_corpus": "data/meta/squad_wiki_corpus.pkl"
    }
}

def download_file(url, outfile):
    if os.path.exists(outfile):
        print(f"{outfile} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {outfile}...")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(outfile, 'wb') as f, tqdm(
            desc=outfile,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

def stream_wikipedia_dump(tar_path):
    print(f"Streaming from {tar_path}...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".bz2"):
                f = tar.extractfile(member)
                if f is not None:
                    import bz2
                    with bz2.BZ2File(f, "r") as bz2_file:
                        for line in bz2_file:
                            if line.strip():
                                try:
                                    doc = json.loads(line.decode('utf-8'))
                                    yield doc
                                except Exception:
                                    continue

def stream_squad_wiki_tsv(gz_path):
    """Stream SQuAD Wikipedia (DPR) from TSV.gz format."""
    print(f"Streaming from {gz_path}...")
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                passage_id, text = parts[0], parts[1]
                title = parts[2] if len(parts) > 2 else ""
                # Format: "title: text"
                full_text = f"{title}: {text}" if title else text
                yield full_text

def main():
    parser = argparse.ArgumentParser(description="Build corpus for Oracle retrieval.")
    parser.add_argument(
        "--corpus-type",
        default="fullwiki",
        choices=["fullwiki", "squad_wiki"],
        help="Corpus type to build: fullwiki (HotpotQA Wikipedia) or squad_wiki (DPR Wikipedia)"
    )
    args = parser.parse_args()

    config = CORPUS_CONFIGS[args.corpus_type]
    dump_url = config["dump_url"]
    dump_path = config["dump_path"]
    out_corpus = config["out_corpus"]

    print(f"Building {args.corpus_type} corpus...")

    # If the file hasn't been downloaded or fails, we will catch it and use HF.
    try:
        download_file(dump_url, dump_path)
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download file: {e}")
        if args.corpus_type == "fullwiki":
            print("We will skip the direct tarball ingestion and proceed immediately with the HuggingFace dataset fallback.")
        else:
            raise

    documents = []
    
    try:
        if args.corpus_type == "fullwiki":
            for doc in tqdm(stream_wikipedia_dump(dump_path), desc="Extracting docs"):
                title = doc.get('title', '')
                text_lines = doc.get('text', [])
                if isinstance(text_lines, list):
                    body = ' '.join(text_lines)
                else:
                    body = str(text_lines)
                
                full_text = f"{title}: {body}"
                documents.append(full_text)
        elif args.corpus_type == "squad_wiki":
            for full_text in tqdm(stream_squad_wiki_tsv(dump_path), desc="Extracting docs"):
                documents.append(full_text)
    except Exception as e:
        print(f"Error during streaming: {e}")
        if args.corpus_type == "fullwiki":
            print("Fallback to HuggingFace dataset 'hotpot_qa' split...")
            from datasets import load_dataset
            ds = load_dataset("KomeijiForce/hotpotqa_wiki_abstract", split="train")
            for row in tqdm(ds, desc="Extracting from HF"):
                title = row.get("title", "")
                sentences = row.get("text", [])
                full_text = f"{title}: {' '.join(sentences)}"
                documents.append(full_text)
        else:
            raise

    print(f"Total documents extracted: {len(documents)}")
    
    os.makedirs(os.path.dirname(out_corpus), exist_ok=True)
    print(f"Saving corpus to {out_corpus}...")
    with open(out_corpus, "wb") as f:
        pickle.dump(documents, f)

    print("Corpus preparation complete!")

if __name__ == "__main__":
    main()
