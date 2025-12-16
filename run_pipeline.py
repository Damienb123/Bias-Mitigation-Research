from qa_agent.pipeline import run_qa_pipeline
import json
import pandas as pd

# This supports a structured output and showcase bias analysis after the steps are completed

# pipeline execution for 6 patients, this structure can be adjusted
# information is pulled from preprocessed_patients_data_2000_1.csv
patients = [
    {
        "SEX_CODE": "M",
        "PAT_AGE": "55-59",
        "RACE": "White",
        "ETHNICITY": "Not of Hispanic Origin",
        "PAT_ZIP": "75002",
        "PRINC_DIAG_CODE": "Dysphagia, unspecified"
    },
    {
        "SEX_CODE": "M",
        "PAT_AGE": "50-54",
        "RACE": "White",
        "ETHNICITY": "Not of Hispanic Origin",
        "PAT_ZIP": "75041",
        "PRINC_DIAG_CODE": "Heartburn"
    },
    {
        "SEX_CODE": "M",
        "PAT_AGE": "45-49",
        "RACE": "White",
        "ETHNICITY": "Hispanic Origin",
        "PAT_ZIP": "75148",
        "PRINC_DIAG_CODE": "Encounter for screening for malignant neoplasm of colon"
    },
    {
        "SEX_CODE": "M",
        "PAT_AGE": "60-64",
        "RACE": "Other",
        "ETHNICITY": "Not of Hispanic Origin",
        "PAT_ZIP": "75089",
        "PRINC_DIAG_CODE": "Encounter for screening for malignant neoplasm of colon"
    },
    {
        "SEX_CODE": "M",
        "PAT_AGE": "50-54",
        "RACE": "White",
        "ETHNICITY": "Not of Hispanic Origin",
        "PAT_ZIP": "75007",
        "PRINC_DIAG_CODE": "Strain of muscle(s) and tendon(s) of the rotator cuff of right shoulder, initial encounter"
    },
    {
        "SEX_CODE": "F",
        "PAT_AGE": "55-59",
        "RACE": "Black",
        "ETHNICITY": "Not of Hispanic Origin",
        "PAT_ZIP": "75002",
        "PRINC_DIAG_CODE": "Unspecified complicated cataract"
    }
]

if __name__ == "__main__":
    print("\n======= RUNNING THE QA MEDICAL PIPELINE NOW =======\n")

    all_outputs = []  # <-- needed for graphing

    for i, patient in enumerate(patients, start=1):
        print(f"\n===== PATIENT {i} =====")
        result = run_qa_pipeline(patient)

        # Print to console
        print("\n>> FINAL ANSWER:")
        print(json.dumps(result, indent=4))

        # Append to list for graphing later
        all_outputs.append(result)

    # Saved results for graphing
    df = pd.DataFrame(all_outputs)
    df.to_pickle("pipeline_analysis/pipeline_outputs.pkl") # file is used to generate graphs in graphical_analysis.py

    print("\n Saved pipeline outputs to pipeline_outputs.pkl")
    print(" You may now run: pipeline_analysis/graphical_analysis.py\n")
