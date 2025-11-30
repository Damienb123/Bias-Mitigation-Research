# rag_generate_summaries.py
from openai import AzureOpenAI
import os, json, jsonlines, re, time 
import pandas as pd 
import matplotlib.pyplot as plt 

# Environment variable setup
# Config
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_ENGINE_RAG")
INPUT = "patient_demographics.jsonl"    # <--- USE YOUR DEMOGRAPHIC FILE
OUTPUT = "gpt_summary/rag_summaries.json"
TOP_K = 3

def clean(text):
    return re.sub(r"\s+", " ", (text or "")).strip()

def load_data(input_path):
    """Auto-detect JSON vs JSONL."""
    if input_path.endswith(".jsonl"):
        data = {}
        with jsonlines.open(input_path) as reader:
            for i, row in enumerate(reader):
                data[str(i)] = row
        return data
    else:
        with open(input_path, "r") as f:
            return json.load(f)

def collect_docs(example):
    """Support both medical condition dataset & demographic dataset."""
    if "text_note" in example:  
        return [{"source": "text_note", "text": clean(example["text_note"])}]

    # Original disease-based fields
    keys = ["mesh_def", "medical_text_def", "wiki_def", "pubmed_def", "kg_def", "google_kg_def"]
    docs = []
    for k in keys:
        v = example.get(k, "")
        if v and len(v.strip()) > 0:
            docs.append({"source": k, "text": clean(v)})
    return docs

def score_and_select(query, docs):
    qset = set(clean(query).lower().split())
    scored = []
    for d in docs:
        tset = set(d["text"].lower().split())
        score = len(qset & tset) / (len(qset | tset) + 1e-9)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:TOP_K]]

def make_prompt(name, retrieved):
    prompt = f"Summarize in 1 concise sentence the item '{name}' for use in health phenotyping.\n"
    if retrieved:
        prompt += "\nRetrieved evidence:\n"
        for r in retrieved:
            prompt += f"- [{r['source']}]: {r['text']}\n"
    prompt += "\nProduce one clear sentence with minimal citation."
    return prompt

def call_model(prompt):
    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def main():
    os.makedirs("gpt_summary", exist_ok=True)
    data = load_data(INPUT)

    out = {}
    for idx, example in data.items():
        name = example.get("name") or example.get("patient_id") or f"item_{idx}"
        docs = collect_docs(example)
        retrieved = score_and_select(name, docs)
        prompt = make_prompt(name, retrieved)
        summary = call_model(prompt)
        out[idx] = {
            "name": name,
            "retrieved": retrieved,
            "rag_prompt": prompt,
            "rag_summary": summary
        }
        print(f"[{idx}] -> {summary[:80]}")
        time.sleep(0.3)

    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
