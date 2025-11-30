# Running the pipeline
from qa_agent.pipeline import run_qa_pipeline

if __name__ == "__main__":
    test_patients = [
        {
            "id": "",
            "gender": "",
            "race": "",
            "zip": "",
            "symptoms": ""
        },
        {
           "id": "",
            "gender": "",
            "race": "",
            "zip": "",
            "symptoms": ""  
        }
    ]

    for p in test_patients:
        print("\n===== RUNNING QA PIPELINE FOR PATIENTS =====")
        result = run_qa_pipeline(p)
        print(result)