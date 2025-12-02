# imports
from openai import AzureOpenAI
import os
# Environment variable setup located in AzureOpenAI
client = AzureOpenAI (
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_ENGINE")

def build_concluding_answer(patient_info, keywords, definitions):
    def_text = "\n".join([f"- {k}: {v}" for k,v in definitions.items()])
    # prompt the agent to list medical keys from the patient data as well as
    # generating top 5 medical procedures the patient will most likely have done
    prompt = f"""
    The patient has demographics: {patient_info}.

    The top medical keywords associated with similar patients are:
    {keywords}

    Definition:
    {def_text}

    Based on this information, generate the top 5 medical procedures the patient
    is most likely to undergo. Provide a short explanation for each.
    """

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role":"user","content":prompt}],
        max_tokens=350,
        temperature=0.2
    )

    return response.choice[0].message.content
