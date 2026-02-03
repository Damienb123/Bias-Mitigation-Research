import json

with open("knowledge_definitions.json", "r", encoding="utf8") as f:
    KNOWLEDGE_INDEX = {
        "disease": json.load(f)
    }
