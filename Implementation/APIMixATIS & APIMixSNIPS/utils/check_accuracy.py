import json
import re

file_name_1 = "snips_4B_CDE.json"
file_name = "final/" + file_name_1
mismatch_file_name = f"mismatches/{file_name_1}"
correct_file_name = f"correct/{file_name_1}"

def parse_calls(calls_str):
    calls_str = calls_str.strip()
    if calls_str.startswith('[') and calls_str.endswith(']'):
        calls_str = calls_str[1:-1]
    parts = re.split(r'\)\s*,\s*', calls_str)
    results = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not part.endswith(')'):
            part += ')'
        m = re.match(r'(\w+)\((.*)\)', part)
        if not m:
             continue
        func, args = m.group(1), m.group(2)
        args_list = re.findall(r"(\w+)\s*=\s*('[^']*'|None)", args)
        params = sorted([(name, val.strip("'")) for name, val in args_list if val != 'None'])
        results.add((func, tuple(params)))

    return results


with open(file_name, 'r', encoding='utf-8') as f:
    data = json.load(f)

mismatches = []
correct_entries = []
correct = 0
total = len(data)

for entry in data:

    exp_set = set()
    for call in entry.get('expected', []):
        exp_set |= parse_calls(f"[{call}]")
    ans_set = parse_calls(entry.get('answer', ''))

    
    if exp_set == ans_set:
        correct += 1
        correct_entries.append(entry)
    else:
        mismatches.append({
            "input":            entry["input"],
            "expected":         entry.get("expected", []),
            "actual":           entry.get("answer", ""),
            "extracted_items": entry.get("extracted_items", {})
        })

accuracy = correct / total if total else 0.0


with open(mismatch_file_name, 'w', encoding='utf-8') as mf:
    json.dump(mismatches, mf, indent=2)

with open(correct_file_name, 'w', encoding='utf-8') as cf:
    json.dump(correct_entries, cf, indent=2)

print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
