import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd

# loads the created the pkl file of summarized output
df = pd.read_pickle("pipeline_outputs.pkl")


# Flatten nested fields for patient_input
patient_df = pd.json_normalize(df["patient_input"])

# Merge flattened fields back into main DataFrame
df = pd.concat([df, patient_df], axis=1)

# Expands the keywords for count plots 
exp_keywords = df.explode("keywords")


# Analyzes the top diagnosis by race
def diagnoses_by_race():
    plt.figure(figsize=(12,8)) # adjusted size for readability
    sns.countplot(data=exp_keywords, x="RACE", hue="keywords")

    plt.title("Top Diagnoses by Race")
    plt.xlabel("Race")
    plt.ylabel("Diagnosis Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Overall cohort size by race
def cohort_size_by_race():
    plt.figure(figsize=(10,6)) # adjusted size for readability
    sns.barplot(x=df["RACE"], y=df["filtered_rows"])

    plt.title("Cohort Size by Race")
    plt.xlabel("Race")
    plt.ylabel("Filtered Cohort Size (# of similar patients)")
    plt.tight_layout()
    plt.show()


# Heatmap generation for types of procedures by sex to race
def extract_procedures(text):
    return re.findall(r"\*\*(.*?)\*\*", text)

df["procedures"] = df["llm_answer"].apply(extract_procedures)
exp_proc = df.explode("procedures")

def procedure_heatmap():
    pivot = exp_proc.pivot_table(
        index="SEX_CODE",
        columns="PAT_AGE",
        values="procedures",
        aggfunc="count",
        fill_value=0
    )

    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, cmap="Blues")

    plt.title("Procedure Frequency Heatmap (Gender × Age)")
    plt.xlabel("Age Group")
    plt.ylabel("Gender")
    plt.show()


# Common definitions found per zip code using a line plot
df["num_def_found"] = df["definitions"].apply(
    lambda d: sum(1 for x in d.values() if x["source"])
)

df_sorted = df.sort_values("PAT_ZIP")

def definition_retrieval_by_zip():
    plt.figure(figsize=(10,6))
    plt.plot(df_sorted["PAT_ZIP"], df_sorted["num_def_found"], marker="o")

    plt.title("RAG Definition Retrieval Success by ZIP Code")
    plt.xlabel("ZIP Code")
    plt.ylabel("# Definitions Retrieved")
    plt.grid(True)
    plt.show()


# A multidimensional analysis radar chart prioritizing bias recognized per patient
def make_radar(values, labels, title):
    angles = np.linspace(0, 2*np.pi, len(values), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, marker="o")
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    plt.show()

def bias_recognized_per_patient():
    for idx, row in df.iterrows():
        values = [
            row["filtered_rows"],
            len(row["keywords"]),
            sum(1 for v in row["definitions"].values() if v["source"]),
            len(row["procedures"])
        ]

        labels = ["Cohort", "Keywords", "Definitions", "Procedures"]
        make_radar(values, labels, f"Bias Radar – Patient {idx+1}")

# Runs all graphs to show all trends
# Can adjust the types of graphs to use and how to display data if needed
if __name__ == "__main__":
    print("\nGenerating graphical bias analysis...\n")

    diagnoses_by_race()
    cohort_size_by_race()
    procedure_heatmap()
    definition_retrieval_by_zip()
    bias_recognized_per_patient()

    print("\n Your graphical analysis has been successfully generated.\n")
