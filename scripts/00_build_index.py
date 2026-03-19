import os
import pickle
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def main():
    print("Loading HotpotQA distractor dataset...")
    # Load the train split of the distractor setting
    dataset = load_dataset("hotpot_qa", "distractor", split="train")

    print("Extracting unique paragraphs...")
    # Dictionary to keep track of unique documents: title -> text
    # Or just use the text itself. The index of the document in the list maps to BM25 index.
    # Actually, saving just a list of unique texts is enough. 
    # But retaining title is nice if we need it. Let's make documents a list of strings: "Title: Sentence1 Sentence2..."
    unique_docs = set()
    documents = []

    for row in tqdm(dataset, desc="Processing rows"):
        titles = row["context"]["title"]
        sentences_lists = row["context"]["sentences"]
        
        for title, sentences in zip(titles, sentences_lists):
            full_text = f"{title}: {' '.join(sentences)}"
            if full_text not in unique_docs:
                unique_docs.add(full_text)
                documents.append(full_text)

    print(f"Total unique documents: {len(documents)}")

    print("Tokenizing corpus...")
    # Tokenize corpus for BM25
    tokenized_corpus = [doc.split() for doc in documents]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    out_file = "data/meta/retriever_index.pkl"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    print(f"Serializing index and documents to {out_file}...")
    with open(out_file, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": documents}, f)

    print("Done!")

if __name__ == "__main__":
    main()
