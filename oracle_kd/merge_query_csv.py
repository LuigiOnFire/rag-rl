import argparse
import csv
import os

DATA_DIR = "oracle_kd/data/training"
SOURCE_FILES = {
    "hotpotqa": os.path.join(DATA_DIR, "hotpotqa_queries.csv"),
    "squad": os.path.join(DATA_DIR, "squad_queries.csv"),
    "nq": os.path.join(DATA_DIR, "nq_queries.csv"),
}
MASTER_FILE = os.path.join(DATA_DIR, "master_queries.csv")


def load_rows(file_path: str):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return [], []

    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if any((value or '').strip() for value in row.values())]
        return rows, reader.fieldnames or []


def write_rows(file_path: str, rows, fieldnames) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_fieldnames(rows, preferred_fieldnames=None, include_source: bool = False):
    fieldnames = []
    if preferred_fieldnames:
        for field in preferred_fieldnames:
            if field not in fieldnames:
                fieldnames.append(field)

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    if include_source and "source" not in fieldnames:
        fieldnames.append("source")

    return fieldnames


def get_source_file(file_path):
    if "hotpot" in file_path.lower():
        return "hotpotqa"
    elif "squad" in file_path.lower():
        return "squad"
    elif "nq" in file_path.lower():
        return "nq"
    else:
        raise ValueError("Unable to identify source from filename. Please ensure it contains 'hotpot' or 'squad'.")


def merge_into_source(input_file, source_name):
    source_file = SOURCE_FILES[source_name]
    merged_data = []
    seen_questions = set()

    if not os.path.exists(source_file) or os.path.getsize(source_file) == 0:
        write_rows(source_file, [], ["question"])
        print(f"Initialized empty CSV in: {source_file}")

    source_data, source_fieldnames = load_rows(source_file)
    input_data, input_fieldnames = load_rows(input_file)

    for entry in source_data:
        question = entry.get('question')
        if question and question not in seen_questions:
            merged_data.append(entry)
            seen_questions.add(question)

    source_count = len(seen_questions)
    print(f"Loaded {source_count} unique rows from existing source: {source_file}")

    added_from_input = 0
    for entry in input_data:
        question = entry.get('question')
        if question and question not in seen_questions:
            merged_data.append(entry)
            seen_questions.add(question)
            added_from_input += 1

    print(f"Added {added_from_input} new unique rows from input: {input_file}.")

    output_fieldnames = build_fieldnames(merged_data, preferred_fieldnames=input_fieldnames or source_fieldnames)
    write_rows(source_file, merged_data, output_fieldnames)

    print(f"Source merge complete. Output saved to: {source_file}")
    print(f"Total Unique Questions in source dataset: {len(seen_questions)}\n")


def merge_into_master(input_file, source_name):
    merged_data = []
    seen_questions = set()
    master_file = MASTER_FILE

    if not os.path.exists(master_file) or os.path.getsize(master_file) == 0:
        write_rows(master_file, [], ["question", "source"])
        print(f"Initialized empty CSV in: {master_file}")

    master_data, master_fieldnames = load_rows(master_file)
    input_data, input_fieldnames = load_rows(input_file)

    for entry in master_data:
        question = entry.get('question')
        if not question:
            continue
        if question not in seen_questions:
            merged_data.append(entry)
            seen_questions.add(question)

    master_count = len(seen_questions)

    added_from_input = 0
    for entry in input_data:
        question = entry.get('question')
        if not question:
            continue
        tagged_entry = dict(entry)
        tagged_entry['source'] = source_name
        if question not in seen_questions:
            merged_data.append(tagged_entry)
            seen_questions.add(question)
            added_from_input += 1

    output_fieldnames = build_fieldnames(
        merged_data,
        preferred_fieldnames=input_fieldnames or master_fieldnames,
        include_source=True,
    )
    write_rows(master_file, merged_data, output_fieldnames)

    print(f"Master merge complete. Output saved to: {master_file}")
    print(f"Kept {master_count} existing master entries; added {added_from_input} new rows from input.")
    print(f"Total Unique Questions in master dataset: {len(seen_questions)}")


def main():
    parser = argparse.ArgumentParser(description="Merge a single Oracle CSV into source-specific and master CSVs.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    args = parser.parse_args()

    source_name = get_source_file(args.input_file)
    merge_into_source(args.input_file, source_name)
    merge_into_master(args.input_file, source_name)


if __name__ == "__main__":
    main()