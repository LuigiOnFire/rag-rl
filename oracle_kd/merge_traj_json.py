import argparse
import json
import os

SOURCE_FILES = {
    "hotpotqa": "oracle_kd/data/trajectories/hotpot_qa_trajectories.json",
    "squad": "oracle_kd/data/trajectories/squad_trajectories.json",
    "nq": "oracle_kd/data/trajectories/nq_trajectories.json"
}
MASTER_FILE = "oracle_kd/data/trajectories/master_trajectories.json"


def load_records(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON array in {file_path}, got {type(data).__name__}.")
    except json.JSONDecodeError:
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def write_records(file_path, records):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)

def get_source_file(file_path):
    if "hotpot" in file_path.lower():
        return "hotpotqa"
    elif "squad" in file_path.lower():
        return "squad"
    elif "nq" in file_path.lower():
        return "nq"
    else:
        raise ValueError("Unable to identify source from filename. Please ensure it contains 'hotpot', 'squad', or 'nq'.")


def merge_into_source(input_file, source_name):
    source_file = SOURCE_FILES[source_name]
    merged_data = []
    seen_questions = set()
    
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(source_file), exist_ok=True)
    
    # Create or initialize the file if it doesn't exist OR if it is completely empty
    if not os.path.exists(source_file) or os.path.getsize(source_file) == 0:
        write_records(source_file, [])
        print(f"Initialized empty JSON array in: {source_file}")

    # Load both datasets cleanly
    source_data = load_records(source_file)
    input_data = load_records(input_file)

    # Existing source entries win; input only fills in missing questions.
    for entry in source_data:
        question = entry.get('question')
        if question and question not in seen_questions:
            merged_data.append(entry)
            seen_questions.add(question)
    source_count = len(seen_questions)
    print(f"Loaded {source_count} unique entries from existing source: {source_file}")

    added_from_input = 0
    for entry in input_data:
        question = entry.get('question')
        if question and question not in seen_questions:
            merged_data.append(entry)
            seen_questions.add(question)
            added_from_input += 1
            
    print(f"Added {added_from_input} new unique entries from input: {input_file}.")

    # Save merged data
    write_records(source_file, merged_data)

    print(f"Source merge complete. Output saved to: {source_file}")
    print(f"Total Unique Questions in source dataset: {len(seen_questions)}\n")


def merge_into_master(input_file, source_name):
    merged_data = []
    seen_questions = set()
    master_file = MASTER_FILE
    
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(master_file), exist_ok=True)
    
    # Create or initialize the file if it doesn't exist OR if it is completely empty
    if not os.path.exists(master_file) or os.path.getsize(master_file) == 0:
        write_records(master_file, [])
        print(f"Initialized empty JSON array in: {master_file}")

    master_data = load_records(master_file)
    input_data = load_records(input_file)

    # Existing master entries win; input only fills in missing questions.
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
            
    # Save master data
    write_records(master_file, merged_data)
        
    print(f"Master merge complete. Output saved to: {master_file}")
    print(f"Kept {master_count} historical master entries; added {added_from_input} new rows from input.")
    print(f"Total Unique Questions in master dataset: {len(seen_questions)}")


def main():
    parser = argparse.ArgumentParser(description="Merge an Oracle JSON file into source-specific files and master.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    args = parser.parse_args()

    source_name = get_source_file(args.input_file)
    merge_into_source(args.input_file, source_name)
    merge_into_master(args.input_file, source_name)


if __name__ == "__main__":
    main()