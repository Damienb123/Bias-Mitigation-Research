# ==============================
# Bias Investigation Notebook
# ==============================

import json
from collections import defaultdict
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util

# --- Load your GPT summaries ---
with open("gpt_summary/mimic_disease_id_name_gpt_summary.json") as f:
    gpt_data = json.load(f)

with open("mimic_disease_id.json") as f:
    original_data = json.load(f)

# --- Initialize sentence embeddings model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ==============================
# 1️ Consistency Check
# ==============================
print("=== Consistency Check ===")
seen_names = defaultdict(list)
for k, v in gpt_data.items():
    name = v["name"]
    seen_names[name].append((k, v["gpt_sum_all"]))

for name, entries in seen_names.items():
    if len(entries) > 1:
        print(f"\nDuplicate disease: {name}")
        for idx, summary in entries:
            print(f"  ID {idx}: {summary}")

# ==============================
# 2️ Sentiment / Subjectivity
# ==============================
print("\n=== Sentiment & Subjectivity ===")
for k, v in gpt_data.items():
    text = v["gpt_sum_all"]
    blob = TextBlob(text)
    print(f"ID {k} | {v['name']}")
    print(f"  Polarity: {blob.sentiment.polarity:.2f}, Subjectivity: {blob.sentiment.subjectivity:.2f}")

# ==============================
# 3 Knowledge Source Influence
# ==============================
print("\n=== Knowledge Source Influence ===")
sources = ["mesh_def", "pubmed_def", "wiki_def", "medical_text_def"]

for k, v in original_data.items():
    gpt_sum = gpt_data.get(k, {}).get("gpt_sum_all", "")
    if not gpt_sum:
        continue

    sims = {}
    for source in sources:
        src_text = v.get(source, "")
        if src_text:
            sims[source] = util.cos_sim(embed_model.encode(gpt_sum), embed_model.encode(src_text)).item()
        else:
            sims[source] = None
    print(f"ID {k} | {v['name']} | Source Similarity: {sims}")
