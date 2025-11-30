from openai import AzureOpenAI
import asyncio
import os
import json
import time

# --- Create separate clients for each Azure endpoint ---
client_gpt35 = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_1"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_1"),  # GPT-3.5 endpoint
)

client_grok3 = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_2"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2"),  # Grok-3 endpoint
)

# --- Map each model to its client and deployment ---
deployments = {
    "model_1": {
        "name": os.getenv("AZURE_OPENAI_ENGINE_1"),  # GPT-3.5 deployment name
        "client": client_gpt35,
    },
    "model_2": {
        "name": os.getenv("AZURE_OPENAI_ENGINE_2"),  # Grok-3 deployment name
        "client": client_grok3,
    },
}


async def call_model(client, deployment_name, messages):
    """Send a request to the given Azure client/deployment."""
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=200,
        temperature=0.0,
    )
    return response.choices[0].message.content


async def compare_models(prompt):
    """Compare outputs from both models."""
    messages = [{"role": "user", "content": prompt}]
    outputs = {}
    for label, info in deployments.items():
        model_name = info["name"]
        client = info["client"]
        print(f"Querying {label}: {model_name}")
        try:
            outputs[label] = await call_model(client, model_name, messages)
        except Exception as e:
            outputs[label] = f"Error: {e}"
        time.sleep(1)
    return outputs


def main():
    dataset = "mimic"
    domain = "disease_id"
    input_file = f"{dataset}_{domain}_name_merge.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    results = {}
    for key, item in data.items():
        name = item.get("name", "")
        prompt = (
            f"Provide a concise, one-sentence medical summary for the disease '{name}', "
            f"to assist in health phenotyping."
        )
        print(f"\nComparing models for: {name}")
        outputs = asyncio.run(compare_models(prompt))
        results[key] = {
            "name": name,
            "model_1_output": outputs["model_1"],
            "model_2_output": outputs["model_2"],
        }

    os.makedirs("gpt_summary", exist_ok=True)
    output_path = "gpt_summary/mimic_disease_id_compare.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Results saved to {output_path}")


if __name__ == "__main__":
    main()
