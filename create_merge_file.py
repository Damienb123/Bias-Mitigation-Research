import json

# Load your JSONL file (training/valid/test)
file_path = "mimic/mimic_disease_id_name_gpt_summary.jsonl"


data = {}
with open(file_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        entry = json.loads(line)
        # build a simple structure GPT expects
        data[idx] = {
            "name": entry["disease"]["name"][0] if entry["disease"]["name"] else "",
            # optionally include definitions or other info
        }

# Save as the merge JSON
with open("mimic_disease_id_name_merge.json", "w", encoding="utf-8") as f_out:
    json.dump(data, f_out, indent=2)

print("Created mimic_disease_id_name_merge.json")
