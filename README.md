# Bias-Mitigation-Research ![Status](https://img.shields.io/badge/status-In%20Progress-yellow)

## Overview
This research extends components of the **RAM-EHR** framework (MIT License) as part of my Independent study with The University of Texas at San Antonio, which provides tools for retrieving and summarizing medical concepts from EHR-linked datasets. Building upon this foundation, the project investigates how **Retrieval-Augmented Generation (RAG)** can be integrated into machine learning workflows to:

- Reduce demographic bias  
- Improve the reliability of clinical predictions  
- Enhance interpretability through grounded model outputs  

By combining patient demographics with external biomedical knowledge sources, the pipeline evaluates how grounding affects fairness, accuracy, and transparency in downstream clinical tasks.

Additionally, this project compares **text-based LLM outputs** and **tabular/graphical representations** to study how each modality influences bias visibility and interpretability.

## Research Question
**How can Retrieval-Augmented Generation (RAG) be integrated into machine learning workflows using EHR data to mitigate bias, and how do text-based versus graphical output representations affect bias visibility and interpretability?**

## Technology Used
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![JSON](https://img.shields.io/badge/Data-JSONL-lightgrey)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-API-blue)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-green)

## Tasks Evaluated Throughout Research
You can find my work in the bias-mitigation research folder!

## Model-to-Model Comparison

### Purpose
Before extending the RAM-EHR framework, baseline behavior was evaluated by comparing GPT-3.5-Turbo and Grok-3 on medical summarization and reasoning tasks. This ensured pipeline stability before introducing bias-focused analysis.

### Evaluation Criteria
- Keyword overlap (Jaccard similarity)
- Summary length variance
- Reasoning consistency
- Clinical interpretability


## QA Agent Pipeline
#### Purpose
The purpose for developing a QA Agent pipeline is to design and evaluate a Large Language Model task completion using Electronic Health Records (EHR). Similarily from studying model to model comparison with RAM-EHR on their task completion strategies with RAG.

### Pipeline Steps
1. Filter patient cohorts based on demographic attributes
2. Extract top-k diagnosis keywords from EHR records
3. Retrieve grounded medical definitions using RAG
4. Generate LLM-based clinical procedure recommendations
5. Store structured outputs for downstream bias analysis
6. Visualize bias patterns using graphical analytics

#### Output (For 1 patient)
```
{
    "patient_input": {
        "SEX_CODE": "F",
        "PAT_AGE": "55-59",
        "RACE": "Black",
        "ETHNICITY": "Not of Hispanic Origin",
        "PAT_ZIP": "75002",
        "PRINC_DIAG_CODE": "Unspecified complicated cataract"
    },
    "filtered_rows": 1,
    "keywords": [
        "complicated cataract",
        "essential (primary) hypertension"
    ],
    "definitions": {
        "complicated cataract": {
            "source": "American Academy of Ophthalmology",
            "definition": "A complicated cataract is a lens opacity that develops as a result of another ocular disease or systemic condition, such as uveitis, glaucoma, trauma, or retinal degenerative disease."
        },
        "essential (primary) hypertension": {
            "source": "American Heart Association",
            "definition": "Essential hypertension is chronic high blood pressure with no identifiable secondary cause. It is influenced by genetic, lifestyle, and environmental factors."
        }
    },
    "llm_answer": "Based on the patient's demographics and the principal diagnosis of \"Unspecified complicated cataract,\" along with the common associated condition of essential (primary) hypertension, the top 5 medical procedures the patient is most likely to undergo are:\n\n1. **Cataract Extraction with Intraocular Lens Implantation (Phacoemulsification)**\n   - Explanation: The primary treatment for complicated cataracts is surgical removal of the opacified lens followed by implantation of an artificial intraocular lens. This procedure restores vision impaired by the cataract.\n\n2. **Preoperative Ocular Examination and Imaging**\n   - Explanation: Prior to cataract surgery, detailed eye examinations including slit-lamp evaluation, ocular coherence tomography (OCT), and biometry measurements are performed to assess the extent of lens opacity and plan surgery, especially important in complicated cases.\n\n3. **Management of Hypertension (Blood Pressure Monitoring and Medication Adjustment)**\n   - Explanation: Given the association with essential hypertension, careful monitoring and management of blood pressure is critical before and after surgery to reduce perioperative cardiovascular risks and promote healing.\n\n4. **YAG Laser Capsulotomy**\n   - Explanation: Post-cataract surgery, some patients develop posterior capsule opacification, especially in complicated cases. This outpatient laser procedure clears the visual axis to restore vision if opacification occurs.\n\n5. **Glaucoma Screening and Treatment**\n   - Explanation: Since complicated cataracts can be associated with ocular conditions like glaucoma, screening for elevated intraocular pressure and initiating treatment if necessary is important to preserve optic nerve function.\n\nThese procedures reflect the typical clinical pathway for a patient with complicated cataract and coexisting hypertension, aiming to restore vision while managing systemic and ocular comorbidities."
}

```

## How to Run
1. Locate the Folder bias-mitigation research that stores the qa_agent workflows, the dataset that is used for this QA agent is loadable from the bias-mitigation research folder in which it is stored labeled as "preprocessed_patients_data_2000_1.csv".
2. Set your environment variables (Azure OpenAI)
### Windows 
```
$env:AZURE_OPENAI_API_KEY="your-ai-model-key"
$env:AZURE_OPENAI_ENDPOINT="https://your-endpoint.azure.com/"
$env:AZURE_OPENAI_API_VERSION="YYYY-DD-MM-preview"
$env:AZURE_OPENAI_ENGINE="your-ai-model"
```

### MacOS / Linux
```
export AZURE_OPENAI_API_KEY="your-ai-model-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.azure.com/"
export AZURE_OPENAI_API_VERSION="YYYY-DD-MM-preview"
export AZURE_OPENAI_ENGINE="your-ai-model"

```
4. Run the QA Agent pipeline
```
python run_pipeline.py
```
- The output will be console printed structured QA results
At the end of the output, there will be a line that states a saved file:
```
pipeline_analysis/pipeline_outputs.pkl
```
- This file will show graphical bias related anaylsis
5. Run Graphical Bias Analysis
Generates visualizations used to analyze bias and interpretability.
```
cd pipleine_analysis
python graphical_anaylsis.py
```
6. run
```
python run_pipeline.py 
```

## Bias Analysis
After the completed RAG steps are executed, a more sophisticated approach for bias anaylsis can be visualized with graph representations.

### Types of Implemented Visualizations
- Top Diagnosis By Race
- Cohort Size Disparities by Race
- Procedure Frequency Heatmap (Gender x Age)
- RAG retrieval Quality by ZIP Code
- Patient-Level Bias Radar Charts

## Why This Matters
- Text-based outputs often obscure demographic bias
- Graphical analysis exposes overrepresentation instantly
- Visualizations enable clinicians and researchers to audit AI behavior without technical expertise


## Conclusion
This work contributes a reproducible framework for evaluating fairness, grounding, and interpretability in EHR-based AI systems by combining RAG-enabled QA pipelines with visual bias analysis.


## Reference
### RAM-EHR
```
@inproceedings{xu2024ram,
  title={RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records},
  author={Xu, Ran and Shi, Wenqi and Yu, Yue and Zhuang, Yuchen and Jin, Bowen and Wang, May D and Ho, Joyce C and Yang, Carl},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year={2024}
}
```
