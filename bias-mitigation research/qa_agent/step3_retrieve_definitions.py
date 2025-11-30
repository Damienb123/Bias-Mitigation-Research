# imports
import json

# retrieve_definitions for getting best definiton to use
def retrieve_definitions(keywords):
    with open("data/definitions.json") as f:
        defs = json.load(f)

    results = {}
    for kw in keywords:
        results[kw] = defs.get(kw, "Definition not found.")
        # return results from given defintions found
        return results