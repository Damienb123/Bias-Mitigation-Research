# KNOWLEDGE_INDEX = disease in patient dataset
from .knowledge_index import KNOWLEDGE_INDEX

def retrieve_definitions(keywords):
    results = {}

    disease_db = KNOWLEDGE_INDEX["disease"]

    for kw in keywords:
        kw_norm = kw.lower().strip()

        # exact match
        if kw_norm in disease_db:
            results[kw] = disease_db[kw_norm]
            continue

        # token-based fuzzy match
        tokens = kw_norm.split()
        found = None
        for name, entry in disease_db.items():
            if any(t in name for t in tokens):
                found = entry
                break

        if found:
            results[kw] = found
        else:
            results[kw] = {
                "source": None,
                "definition": "No retrieved definition found."
            }

    return results
