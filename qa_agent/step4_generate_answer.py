from openai import AzureOpenAI
import os
# Interaction with in this case - Azure OpenAI gpt-35-turbo for summarized answer
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_ENGINE") # for this research, gpt-35-turbo | this can change
# generate_answer gathers patient demographics, top q keywords, and definitions
def generate_answer(patient_info, keywords, definitions):

    defs_text = "\n".join([f"- {k}: {v}" for k,v in definitions.items()])

    prompt = f"""
    The patient has demographics: {patient_info}.

    The top medical keywords associated with similar patients are:
    {keywords}

    Definitions:
    {defs_text}

    Based on this information, generate the top 5 medical procedures the patient
    is most likely to undergo. Provide a short explanation for each.
    """

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role":"user","content":prompt}],
        max_tokens=350,
        temperature=0.2
    )

    return response.choices[0].message.content