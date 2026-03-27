import os
import json
import tarfile
import requests
import pickle
from tqdm import tqdm

# NOTE: Hotpot uses 2017 wikipedia, Musique wants to use 2020 wikipedia, NQopen also uses 2020 maybe?
DUMP_URL = "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
DUMP_PATH = "data/meta/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
OUT_CORPUS = "data/meta/fullwiki_corpus.pkl"

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

def main():
    # If the file hasn't been downloaded or fails, we will catch it and use HF.
    try:
        download_file(DUMP_URL, DUMP_PATH)
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download tar file: {e}")
        print("We will skip the direct tarball ingestion and proceed immediately with the HuggingFace dataset fallback.")

    documents = []
    
    try:
        for doc in tqdm(stream_wikipedia_dump(DUMP_PATH), desc="Extracting docs"):
            title = doc.get('title', '')
            text_lines = doc.get('text', [])
            if isinstance(text_lines, list):
                body = ' '.join(text_lines)
            else:
                body = str(text_lines)
            
            full_text = f"{title}: {body}"
            documents.append(full_text)
    except Exception as e:
        print(f"Error during streaming: {e}")
        print("Fallback to HuggingFace dataset 'hotpot_qa' split...")
        from datasets import load_dataset
        ds = load_dataset("KomeijiForce/hotpotqa_wiki_abstract", split="train")
        for row in tqdm(ds, desc="Extracting from HF"):
            title = row.get("title", "")
            sentences = row.get("text", [])
            full_text = f"{title}: {' '.join(sentences)}"
            documents.append(full_text)

    print(f"Total documents extracted: {len(documents)}")
    
    os.makedirs(os.path.dirname(OUT_CORPUS), exist_ok=True)
    print(f"Saving corpus to {OUT_CORPUS}...")
    with open(OUT_CORPUS, "wb") as f:
        pickle.dump(documents, f)

    print("Corpus preparation complete!")

if __name__ == "__main__":
    main()
