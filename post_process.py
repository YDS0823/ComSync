import os
import re
import json
import argparse
from tqdm import tqdm


def remove_trailing_patterns(input_path: str, temp_path: str):
    """Step 1: Remove trailing unwanted patterns (like END_OF_DEMO, assistant...)."""
    trailing_pattern = re.compile(r"END_OF_.*$|assistant.*$", flags=re.IGNORECASE)

    with open(input_path, "r", encoding="utf-8") as fin, open(temp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, list):
                    continue
                cleaned_list = [trailing_pattern.sub("", s).strip() for s in data if isinstance(s, str)]
                fout.write(json.dumps(cleaned_list, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON line: {line[:100]}")

def remove_prefix_patterns(input_path: str, output_path: str):
    """Step 2: Remove prefix phrases (model verbosity like 'Here is the rewritten docstring ...')."""
    prefixes_to_remove = [
            r"Here are the docstrings for the after-change code based on the provided before-change code and docstrings: ",
            r"The after-change docstring is:",
            r"```java //write a docstring for after-change code based on the given before-change code and before-change docstring",
            r"Here is the rewritten docstring for the after-change code based on the before-change code and before-change docstring",
            r"Here are the updated docstrings for each of the changed code snippets",
            r"Here are the docstrings for the after-change codes",
            r"Here are the updated docstrings for each of the after-change code snippets",
            r"Here's the updated docstring",
            r"Here's the rewritten docstring",
            r"Here is the rewritten docstring for the after-change code:",
            r"Here is the written docstring for the after-change code:",
            r"Here is the rewritten docstring:",
            r"Here is the written docstring for the after-change code based on the given before-change code and before-change docstring:",
            r"Here are the docstrings for the after-change code:",
            r"Based on the given before-change code and before-change docstring, I would write the following docstring for the after-change code:",
            r"Based on the given before-change code and before-change docstring, I will write the docstring for the after-change code. Here are the rewritten docstrings:",
            r"Here is the rewritten docstring based on the before-change code and before-change docstring:",
            r"Here is the docstring for the after-change code based on the given before-change code and before-change docstring:",
            r"Here is the docstring for the after-change code:",
            r"Based on the provided before-change code and docstrings, I will write a docstring for the after-change code.",
            r"Based on the before-change code and docstring, I will write a docstring for the after-change code.",
            r"Based on the given before-change code and before-change docstring, here is the written docstring for the after-change code:",
            r"Here is the rewritten docstring for the after-change code based on the given before-change code and before-change docstring:",
            r"Here is the rewritten docstring based on the before-change code and docstring:",
            r"Here are the rewritten docstrings for the after-change code:",
            r"Here are the rewritten docstrings for the after-change code based on the given before-change code and before-change docstring:",
            r"Here are the rewritten docstrings for the after-change code based on the given before-change code and before-change docstrings:",
            r"Here is the rewritten docstring based on the given before-change code and before-change docstring:",
            r"Here are the docstrings for the after-change code based on the given before-change code and before-change docstring:",
            r"Here are the docstrings for the after-change code based on the given before-change code and before-change docstrings:",
            r"Based on the before-change code and docstring, here is a rewritten docstring for the after-change code:",
            r"Based on the before-change code and before-change docstring,",
            r"Based on the provided before-change code and docstring, I will write a docstring for the after-change code.",
            r"Based on the provided before-change code and before-change docstrings, I will write a docstring for the after-change code.",
            r"Based on the given before-change code and before-change docstring,",
            r"Based on the before-change code and docstring, the after-change code is:",
            r"Based on the provided code snippets, I will write a docstring for each of the after-change code snippets based on the given before-change code and before-change docstrings.",
            r"Based on the provided before-change code and docstrings, I'll write a docstring for the after-change code. Here are the results:",
            r"Based on the before-change code and docstring, the after-change code and docstring are:",
            r"Based on the provided before-change code and before-change docstrings, I will write the docstrings for the after-change code.",
            r"Based on the before-change code and docstring, I will write a docstring for the after-change code:",
            r"Based on the provided before-change code and before-change docstring, I will write a docstring for the after-change code.",
            r"Based on the before-change code and docstring, here is a possible docstring for the after-change code:",
            r"Based on the provided before-change code and before-change docstring, here is the written docstring for the after-change code:",
            r"Based on the provided before-change code and before-change docstrings, I will write the after-change code and docstrings as follows:",
            r"I've written docstrings for the after-change code based on the given before-change code and before-change docstrings. Here are the results:",
            r"Based on the provided before-change code and docstring, I'll write a docstring for the after-change code.",
            r"Here is the after-change docstring:",
            r"Here is a possible docstring for the after-change code:",
            r"Here is the rewritten docstring based on the after-change code:",
            r"Here is the updated docstring for the after-change code:",
            r"The docstring for the after-change code would be:",
            r"Here is the rewritten docstring for the after-change code: ",
            r"Here is the written docstring for the after-change code: ",
            r"Here are the rewritten docstrings based on the given before-change code and before-change docstrings:",
            r"Here are the docstrings for each of the given examples:",
            r"Based on the given before-change code and before-change docstring, I will write a docstring for the after-change code.",
            r"Based on the given before-change code and before-change docstring, the after-change code is:",
            r"Based on the given before-change code and before-change docstring, I will write the docstring for the after-change code.",
            r"Here is the docstring:",
            r"Here is the after-change code:",
            r"The after-change code is:",
            r"The after-change code:",
            r"Here are the docstrings for each of the given code changes:",
            r"Based on the given before-change code and before-change docstring, here is the after-change docstring:",
            r"Based on the before-change code and docstring, here is the rewritten docstring for the after-change code:",
            r"The docstring for the after-change code is:",
            r"Based on the given before-change code and before-change docstring:",
            r"Based on the given before-change code and before-change docstring, the after-change docstring is:",
            r"Here is a docstring for the after-change code",
            r"Here is a potential docstring for the after-change code",
            r"Here is the revised docstring for the after-change code",
            r"Here is the updated docstring",
            r"Here is the updated docstring based on the given before-change code and before-change docstring",
            r"Here is a rewritten docstring for the after-change code",
            r"Here is the updated docstring based on the before-change code and the before-change docstring",
            r"Here is the after-change docstring based on the given before-change code and before-change docstring",
            r"Here is the updated docstring based on the provided before-change code and the changes made",
            r"Based on the given before-change code and before-change docstring, the after-change docstring for the after-change code is:",
        ]
    pattern = re.compile("|".join(prefixes_to_remove), flags=re.IGNORECASE)

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                sentences = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON line, skipping: {line[:80]}")
                continue

            if isinstance(sentences, list):
                processed = []
                for s in sentences:
                    if not isinstance(s, str):
                        continue
                    cleaned = pattern.sub("", s).strip()
                    processed.append(cleaned if cleaned else "")
                outfile.write(json.dumps(processed, ensure_ascii=False) + "\n")
            else:
                outfile.write(json.dumps(sentences, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Clean LLM generation outputs by removing trailing markers and redundant prefixes."
    )
    parser.add_argument("--input_dir", required=True, help="Input directory containing raw JSONL files.")
    parser.add_argument("--output_dir", required=True, help="Output directory for cleaned files.")
    parser.add_argument("--temp_dir", default="./temp_clean", help="Temporary directory for intermediate files.")
    parser.add_argument("--shots", nargs="+", type=int, default=[2, 4, 6, 8, 10], help="List of shot numbers to process.")
    parser.add_argument("--types", nargs="+", default=["dense", "hybrid", "expert"], help="Retrieval types to process.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    for shot in args.shots:
        for type_name in args.types:
            input_file = os.path.join(args.input_dir, f"{type_name}_shot{shot}.jsonl")
            temp_file = os.path.join(args.temp_dir, f"{type_name}_shot{shot}_temp.jsonl")
            output_file = os.path.join(args.output_dir, f"{type_name}_shot{shot}_clean.jsonl")

            if not os.path.exists(input_file):
                print(f"⚠️ Input not found, skipping: {input_file}")
                continue

            print(f"\n🔹 Cleaning file: {input_file}")
            print("  Step 1: Removing trailing patterns...")
            remove_trailing_patterns(input_file, temp_file)
            print("  Step 2: Removing prefix patterns...")
            remove_prefix_patterns(temp_file, output_file)
            print(f"✅ Done -> {output_file}")


if __name__ == "__main__":
    main()