import argparse
import json
import os
import re

INPUT_FILE_NAME = "thinking_4B_snips_end.json"
def parse_single_call(call_str: str):
    if not isinstance(call_str, str):
        return None
    call_str = call_str.strip()
    if not call_str:
        return None

    match = re.match(r'(\w+)\((.*)\)', call_str)
    if not match:
        return None
    func_name = match.group(1)
    args_str = match.group(2)

    args_list_raw = re.findall(r"(\w+)\s*=\s*('[^']*'|\bNone\b)", args_str)

    params = []
    for name, val_str in args_list_raw:
        if val_str == 'None':
            continue

        if val_str.startswith("'") and val_str.endswith("'"):
            actual_val = val_str[1:-1]
        else:
            actual_val = val_str
        params.append((name, actual_val))

    params.sort()
    return (func_name, tuple(params))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model output against expected results.")

    parser.add_argument(
        "--mismatch_dir",
        type=str,
        default="mismatches",
        help="Directory to save mismatch files."
    )
    parser.add_argument(
        "--correct_dir",
        type=str,
        default="correct",
        help="Directory to save correct prediction files."
    )
    args = parser.parse_args()

    file_name = INPUT_FILE_NAME

    os.makedirs(args.mismatch_dir, exist_ok=True)
    os.makedirs(args.correct_dir, exist_ok=True)

    base_input_filename = os.path.basename(file_name)
    mismatch_file_name = os.path.join(args.mismatch_dir, base_input_filename)
    correct_file_name = os.path.join(args.correct_dir, base_input_filename)

    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: Input file '{file_name}' not found. Please check the INPUT_FILE_NAME variable at the top of the script.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file: {file_name}. Ensure it's valid JSON.")
        return

    mismatches = []
    correct_entries = []
    correct_count = 0
    total_entries = len(data)

    for entry_index, entry in enumerate(data):
        expected_calls_list = entry.get('expected', [])
        actual_calls_list = entry.get('answer', [])

        if not isinstance(expected_calls_list, list):
            expected_calls_list = []

        if not isinstance(actual_calls_list, list):
            actual_calls_list = []

        exp_set = set()
        for call_str in expected_calls_list:
            parsed_call = parse_single_call(call_str)
            if parsed_call:
                exp_set.add(parsed_call)
            elif call_str:
                print(
                    f"Warning: Failed to parse expected call string for entry {entry_index + 1} (input: '{entry.get('input', 'N/A')}'): '{call_str}'")

        ans_set = set()
        for call_str in actual_calls_list:
            parsed_call = parse_single_call(call_str)
            if parsed_call:
                ans_set.add(parsed_call)
            elif call_str:
                print(
                    f"Warning: Failed to parse actual answer call string for entry {entry_index + 1} (input: '{entry.get('input', 'N/A')}'): '{call_str}'")

        if exp_set == ans_set:
            correct_count += 1
            correct_entries.append(entry)
        else:
            mismatches.append({
                "input": entry.get("input"),
                "expected_original": expected_calls_list,
                "actual_original": actual_calls_list,
                "parsed_expected_set": sorted([str(item) for item in exp_set]),
                "parsed_actual_set": sorted([str(item) for item in ans_set]),
                "extracted_items": entry.get("extracted_items", {})
            })

    accuracy = correct_count / total_entries if total_entries else 0.0

    try:
        with open(mismatch_file_name, 'w', encoding='utf-8') as mf:
            json.dump(mismatches, mf, indent=2)
        print(f"Mismatches saved to: {mismatch_file_name}")
    except IOError as e:
        print(f"Error writing mismatch file: {e}")

    try:
        with open(correct_file_name, 'w', encoding='utf-8') as cf:
            json.dump(correct_entries, cf, indent=2)
        print(f"Correct entries saved to: {correct_file_name}")
    except IOError as e:
        print(f"Error writing correct entries file: {e}")

    print(f"\nProcessed {total_entries} entries from '{file_name}'.")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_entries})")


if __name__ == '__main__':
    main()
