import argparse
import csv

def merge_csvs_streaming(file1_path: str, file2_path: str, output_path: str) -> None:
    seen_questions = set()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
        # 1. Read the first file (This one gets absolute precedence)
        with open(file1_path, 'r', encoding='utf-8') as f1:
            reader1 = csv.DictReader(f1)
            writer = csv.DictWriter(out_f, fieldnames=reader1.fieldnames)
            writer.writeheader()
            
            for row in reader1:
                question = row['question']
                if question not in seen_questions:
                    writer.writerow(row)
                    seen_questions.add(question)
            
            file1_count = len(seen_questions)
            print(f"Loaded {file1_count} unique rows from {file1_path} (Precedence Applied).")
        
        # 2. Read the second file (Only add if we haven't seen it in file 1)
        with open(file2_path, 'r', encoding='utf-8') as f2:
            reader2 = csv.DictReader(f2)
            
            added_from_f2 = 0
            for row in reader2:
                question = row['question']
                if question not in seen_questions:
                    writer.writerow(row)
                    seen_questions.add(question)
                    added_from_f2 += 1
            
            print(f"Added {added_from_f2} new unique rows from {file2_path}.")

    print(f"\nMerge complete. Output saved to: {output_path}")
    print(f"Total Unique Questions in merged dataset: {len(seen_questions)}")


def main():
    parser = argparse.ArgumentParser(description="Merge two Oracle CSVs, giving precedence to the first file.")
    
    parser.add_argument(
        "--file1", 
        required=True, 
        help="Path to the primary CSV (Duplicates in file2 will be dropped in favor of this one)."
    )
    parser.add_argument(
        "--file2", 
        required=True, 
        help="Path to the secondary CSV."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path where the merged CSV will be saved."
    )
    
    args = parser.parse_args()
    
    merge_csvs_streaming(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()