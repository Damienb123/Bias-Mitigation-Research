import json
from collections import Counter

# Load the comparison JSON
input_file = "mimic/mimic_disease_id_compare.json"
with open(input_file, "r") as f:
    data = json.load(f)

# Define keywords to track
keywords = ["cancer", "infection", "heart", "kidney", "blood", "lung", "sepsis", "hypertension"]

results = {}

for key, item in data.items():
    text1 = item["model_1_output"].lower()
    text2 = item["model_2_output"].lower()

    # Word overlap similarity
    words1 = Counter(text1.split())
    words2 = Counter(text2.split())
    common_words = sum((words1 & words2).values())
    total_words = max(sum(words1.values()), sum(words2.values()))
    overlap_sim = common_words / total_words if total_words else 0

    # Jaccard similarity
    set1 = set(words1)
    set2 = set(words2)
    jaccard_sim = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) else 0

    # Keyword presence
    keywords_model1 = [k for k in keywords if k in text1]
    keywords_model2 = [k for k in keywords if k in text2]

    # Output length
    len1 = len(text1.split())
    len2 = len(text2.split())

    results[key] = {
        "name": item["name"],
        "model_1_output": item["model_1_output"],
        "model_2_output": item["model_2_output"],
        "word_overlap_similarity": round(overlap_sim, 2),
        "jaccard_similarity": round(jaccard_sim, 2),
        "model_1_keywords": keywords_model1,
        "model_2_keywords": keywords_model2,
        "model_1_length": len1,
        "model_2_length": len2
    }

# Save analysis results
output_file = "gpt_summary/mimic_disease_id_bias_analysis.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Bias analysis saved to {output_file}")
