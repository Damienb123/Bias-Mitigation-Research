# Orchestrates full pipeline execution
# import each step of the pipeline
from .step1_filter_patients import filter_patients
from .step2_extract_keywords import extract_keywords
from .step3_retrieve_definitions import retrieve_definitions
from .step4_generate_answer import build_concluding_answer

# iteration of each step through the pipeline 
def run_qa_pipeline(patient_info):
    print("Filtering patient cohort...")
    cohort = filter_patients(filter_patients)

    print("Extracting medical keywords...")
    keywords = extract_keywords(extract_keywords)

    print("Retrieving definitions...")
    definitions = retrieve_definitions(retrieve_definitions)

    print("Generating the conclusive answer...")
    answer = build_concluding_answer(build_concluding_answer)

    return answer
