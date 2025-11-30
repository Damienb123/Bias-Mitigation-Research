# ============================================================================================================================
# ============================================================================================================================
#
# QA agentic pipeline I/O walthrough
# ----------------------------------
# - The QA agent will get an input of patient demographic information (age group, gender, race, ZIP), symptoms
# - The QA agent will produce an output of Top K (K=5) medical procedures for your condition when you visit some hospitals
#
# QA Pipeline
# -----------
# QA agent pipeline step 1: fill out rows from the patient database that share the same demographic information
# QA agent pipeline step 2: Extract 5 medical keywords (e.g., diseases or procedures) from the output of step 1.By functions
# and LLM (give me the most frequently-represented q (q=3) keywords from all the rows).
# QA agent pipeline step 3: retrieve the definitions of these medical keywords from step 2 using JSON pipeline.
# Pick the best definition to use.
# QA agent pipeline step 4: Feed the outputs from the above steps to QA agent. Done by LLM with prompt engineering.
# 
# ============================================================================================================================
# ============================================================================================================================


import json 
import pandas as pd
from collections import defaultdict


# load JSONL function
def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for lin in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

# load mimic summary files
print("Loading mimic summary files...")

# loads relevant mimic files
disease_df = load_jsonl("mimic_disease_id_gpt_summary.jsonl")
procedure_df = load_jsonl("mimic_procedure_id_name_gpt_summary.jsonl")
precription_df = load_jsonl("mimic_prescription_name_gpt_summary.jsonl")

# displays to the command line that data is loaded
# shown  disease, procedure, and 
print("Loaded: ")
print(" Disease:", len(disease_df))
print(" Procedure:", len(procedure_df))
print(" Prescription")


# Load patient demographic data, in this case it'll be created as a table or
# another structure


# Merge JSONL files with demographic aligned with patients


# count per gender or list 