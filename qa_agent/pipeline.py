import pandas as pd
from .step1_filter_patients import filter_patients
from .step2_extract_keywords import extract_keywords
from .step3_retrieve_definitions import retrieve_definitions
from .step4_generate_answer import generate_answer


def run_qa_pipeline(patient_info):

    df = pd.read_csv("preprocessed_patients_data_2000_1.csv")
    cohort = filter_patients(df, patient_info)
    filtered_rows = len(cohort)

    # Extract keywords from csv dataset 
    keywords = extract_keywords(cohort, top_q=3)

    # Ensure keywords list is valid
    if not keywords:
        keywords = ["no diagnosis found"]

    # Retrieve definitions from the source to assign to exrtacted keywords
    definitions = retrieve_definitions(keywords)

   # Generate a summarized LLM answer for the top most likely procedure for the diagnosis
    llm_answer = generate_answer(
        patient_info=patient_info,
        keywords=keywords,
        definitions=definitions
    )

    return {
        "patient_input": patient_info,
        "filtered_rows": filtered_rows,
        "keywords": keywords,
        "definitions": definitions,
        "llm_answer": llm_answer
    }
