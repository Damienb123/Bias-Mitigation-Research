# Bias-Mitigation-Research

## Overview
This research extends components of the **RAM-EHR** framework (MIT License) as part of my Independent study with The University of Texas at San Antonio, which provides tools for retrieving and summarizing medical concepts from EHR-linked datasets. Building upon this foundation, the project investigates how **Retrieval-Augmented Generation (RAG)** can be integrated into machine learning workflows to:

- Reduce demographic bias  
- Improve the reliability of clinical predictions  
- Enhance interpretability through grounded model outputs  

By combining patient demographics with external biomedical knowledge sources, the pipeline evaluates how grounding affects fairness, accuracy, and transparency in downstream clinical tasks.

Additionally, this project compares **text-based LLM outputs** and **tabular/graphical representations** to study how each modality influences bias visibility and interpretability.

## Research Question
**How can Retrieval-Augmented Generation (RAG) be integrated into machine learning workflows using EHR data to mitigate bias, and how do text-based versus graphical output representations affect bias visibility and interpretability?**

## Techology Used
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![JSON](https://img.shields.io/badge/Data-JSONL-lightgrey)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-API-blue)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-green)
![Status](https://img.shields.io/badge/status-In%20Progress-yellow)

## Tasks Evaluated Throughout Research
You can find my work in the bias-mitigation research folder!

### Model to Model Comparison
#### Purpose
The purpose for evaluating model to model comparison using RAM-EHR is to test how the their pipeline operates first before expanding on my research. Firstly, I updated the pipeline to use updated dependencies with OpenAI functions and to make working API calls with Azure OpenAI.

#### Models Being Used
Upon a successful link to my OpenAI Foundary project on Azure OpenAI, I continued RAM-EHR's integration of GPT-3.5-turbo against grok-3 (Newest Large Language Model created xAI) for its reasoning abilities.

#### Sub Tasks
The types of data being pulled from a dataset for training is from JSONL files and JSON test data
Keywords being analyzed for similarity purposes are:
- cancer
- infection
- heart
- kidney
- blood pressure
- lung disease
- sepsis
- hypertension

With this, word overlap similarity is studied in regards to predictions made generating medical summaries using knowledge sources. Jaccard similarity as well when the models output overlapping summaries explaining health conditions and diseases. Lastly, the output length of each summary the model produces as well.


### QA Agent Pipeline
#### Purpose
The purpose for developing a QA Agent pipeline is to design and evaluate a Large Language Model task completion using Electronic Health Records (EHR). Similarily from studying model to model comparison with RAM-EHR on their task completion strategies with RAG.

#### Sub Tasks
1. Predict or surface likely diseases, procedures, and prescriptions for a patient based on demographic information.
2. Identify and explain potential biases introduced by demographic variations (gender, race, age, ZIP).
3. Compare how text-based outputs vs graphical outputs affects the interpretability and visibility of bias.
4. Create a reproducible pipeline that allows researchers to analyze:
- Model outputs
- Retrieval behavior
- Demographic differences
- Mitigation strategies

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

### How to Run
1. Ensure you are using the appropriate environment variables are integrated in step 4 that are found when using Azure OpenAI
2. The steps will pull data from an existing dataset "preprocessed_patients_data_data_2000_1.csv"
3. run
```
python run_pipeline.py 
```

## Bias Analysis
After the completed RAG steps are executed, a more sophisticated approach for bias anaylsis can be visualized with graph representations.

### Types of Implemented Visualizations
Top Diagnosis By Race
Cohort Size Disparities by Race
Procedure Frequency Heatmap (Gender x Age)
RAG retrieval Quality by ZIP Code
Patient-Level Bias Radar Charts

### Why This Matters
Textual outputs hide bias, therefore difficult to spot overrepresentation
Graphs reveal demographic skew instantly
Visualizations help clinicians interpret model behavior without reading long text

### How to Run
1. After executing "run_pipeline.py" at the bottom of the of output it will display:

 **Saved pipeline outputs to pipeline_outputs.pkl
 You may now run: pipeline_analysis/graphical_analysis.py**

2. Locate the pipeline_analysis folder and ensure the "pipeline_outputs.pkl was loaded in that same folder
3. Run (in the same folder of pipeline_analysis folder):
```
python graphical_analysis.py
```
which will then output all graphical representations from the QA textual output.


## Conclusion
This research shows that:
RAG can mitigate hallucinations and improve factual grounding
But it does not automatically eliminate demographic bias.
	Graphical analysis substantially enhances bias visibility 
Compared to text-only summaries, visual representations expose systematic differences more clearly

The custom QA Agent pipeline provides a reproducible framework 

Allowing:
Retrieval behavior analysis
LLM output subgroup comparisons
Bias detection & mitigation research

Thus, this project successfully addresses the research question and contributes a novel combination of RAG grounding + LLM + graphical bias visualization for EHR based AI workflows.


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
