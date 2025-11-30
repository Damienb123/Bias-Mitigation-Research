# Bias-Mitigation-Research

## Overview
This research extends components of the **RAM-EHR** framework (MIT License), which provides tools for retrieving and summarizing medical concepts from EHR-linked datasets. Building upon this foundation, the project investigates how **Retrieval-Augmented Generation (RAG)** can be integrated into machine learning workflows to:

- Reduce demographic bias  
- Improve the reliability of clinical predictions  
- Enhance interpretability through grounded model outputs  

By combining patient demographics with external biomedical knowledge sources, the pipeline evaluates how grounding affects fairness, accuracy, and transparency in downstream clinical tasks.

Additionally, this project compares **text-based LLM outputs** and **tabular/graphical representations** to study how each modality influences bias visibility and interpretability.

## Research Question
**How can RAG be integrated into machine learning workflows using EHR data to mitigate bias, and how do text-based versus graphical output representations affect bias visibility and interpretability?**

## Techology Used
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![JSON](https://img.shields.io/badge/Data-JSONL-lightgrey)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-API-blue)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-green)
![Status](https://img.shields.io/badge/status-In%20Progress-yellow)

## Tasks Evaluated Throughout Research

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

#### Outcome (Will Provide after Completion)

### QA Agent Pipeline (On-going progress)
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

#### Outcome (Will Provide After Completion)

## Conclusion (Will Provide after completion)

## Reference
### RAM-EHR
Xu, R., Shi, W., Yu, Y., Zhuang, Y., Jin, B., Wang, M. D., Ho, J. C., & Yang, C. (2024). RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records. ArXiv.org. https://arxiv.org/abs/2403.00815
