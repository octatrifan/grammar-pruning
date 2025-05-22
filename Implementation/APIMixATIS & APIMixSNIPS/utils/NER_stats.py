import json
import re

def flatten_input_items(items_dict):
    flat = set()
    for slot, vals in items_dict.items():
        for v in vals:
            flat.add((slot, v))
    return flat

def flatten_gold_input(output_items):
    flat = set()
    for entry in output_items:
        for intent, slots in entry.items():
            for slot, val in slots.items():
                flat.add((slot, val))
    return flat

def extract_output_items(calls):
    results = []
    pat = re.compile(r"(\w+)\(([^)]*)\)")
    for call in calls:
        m = pat.match(call)
        if not m:
            continue
        intent, body = m.groups()
        params = {}
        for part in re.split(r",\s*", body):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = v.strip().strip("'")
        results.append({intent: params})
    return results

def flatten_output_items(item_list):
    flat = set()
    for entry in item_list:
        for intent, slots in entry.items():
            for slot, val in slots.items():
                flat.add((intent, slot, val))
    return flat

def compute_micro_metrics(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

with open("atis_data_augmented.json", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

ner_tp = ner_fp = ner_fn = 0
parse_tp = parse_fp = parse_fn = 0
ner_fn_items = {}
parse_fn_items = {}

for entry in data:
    pred_in = flatten_input_items(entry["input_items"])
    gold_in = flatten_gold_input(entry["output_items"])
    ner_tp += len(pred_in & gold_in)
    ner_fp += len(pred_in - gold_in)
    ner_fn += len(gold_in - pred_in)
    missed_in = gold_in - pred_in
    if missed_in:
        ner_fn_items[entry["input"]] = missed_in

    pred_calls = entry.get("output", [])
    pred_struct = extract_output_items(pred_calls)
    pred_out = flatten_output_items(pred_struct)
    gold_out = flatten_output_items(entry["output_items"])
    parse_tp += len(pred_out & gold_out)
    parse_fp += len(pred_out - gold_out)
    parse_fn += len(gold_out - pred_out)
    missed_out = gold_out - pred_out
    if missed_out:
        parse_fn_items[entry["input"]] = missed_out

ner_prec, ner_rec, ner_f1       = compute_micro_metrics(ner_tp, ner_fp, ner_fn)

# report
print("=== NER (input items) Metrics ===")
print(f"Precision: {ner_prec:.3f}")
print(f"Recall   : {ner_rec:.3f}")
print(f"F1       : {ner_f1:.3f}\n")

# print out the specific false negatives
print("=== NER Missing Items (False Negatives) ===")
for utterance, misses in ner_fn_items.items():
    print(f"- \"{utterance}\": missed {misses}")

