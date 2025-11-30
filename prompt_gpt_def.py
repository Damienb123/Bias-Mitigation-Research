from openai import AzureOpenAI
import asyncio
from typing import List, Dict, Any
import argparse
import os
import re
import time
import json

# Environment variable setup

# --- Create separate Azure clients for each endpoint ---
client_1 = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_1"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_1"),
)

client_2 = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_2"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2"),
)

clients = {
    "model_1": client_1,  # GPT-3.5-Turbo
    "model_2": client_2,  # Grok-3
}

deployments = {
    "model_1": os.getenv("AZURE_OPENAI_ENGINE_1"),
    "model_2": os.getenv("AZURE_OPENAI_ENGINE_2"),
}


# --- Clean string helper ---
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


# --- Async request dispatcher ---
async def dispatch_openai_requests(
    messages_list: List[List[Dict[str, Any]]],
    temperature: float,
    max_tokens: int,
    model_name: str,
    model_key: str,
) -> List[str]:
    """Dispatch requests asynchronously to the correct Azure deployment."""

    async def make_request(messages):
        response = clients[model_key].chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content

    tasks = [make_request(x) for x in messages_list]
    return await asyncio.gather(*tasks)


def call_api_async(msg_lst, temperature, max_tokens, model_key, model_name):
    """Wrapper to run the async dispatcher."""
    print("===================================")
    print(f"Calling {model_name} ({model_key}), {len(msg_lst)} prompts, t={temperature}.")
    response = asyncio.run(
        dispatch_openai_requests(
            messages_list=msg_lst,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_name,
            model_key=model_key,
        )
    )
    print(f"{model_name} returned {len(response)} responses.")
    print("===================================")
    return response


# --- Prompt builder ---
def build_prompt(example, domain, remove):
    prompt = (
        f"Suppose you are working on a health-related phenotyping task and need to "
        f"get relevant information for the given {domain}. Here are some relevant information:\n"
        f"Disease name: {example['name']}"
    )
    name = f"{example['name']}"
    defs = []
    if remove != 'all':
        for k in example:
            if "def" in k and remove not in k and example[k]:
                defs.append(example[k])
    for i, definition in enumerate(defs):
        prompt += f"\nRelevant Information {i+1}: {definition}"
    prompt += (
        f"\nBased on the above information, could you generate 1 sentence summarizing "
        f"the knowledge for the {domain} '{name}' useful for a health phenotyping task?"
    )
    return prompt


# --- CLI setup ---
parser = argparse.ArgumentParser("")
parser.add_argument("--temperature", default=0.0, type=float)
parser.add_argument("--dataset", default="mimic", type=str)
parser.add_argument("--domain", default="disease_id", type=str)
parser.add_argument("--remove", default="", type=str)
parser.add_argument("--output_dir", default="gpt_summary_compare", type=str)
args = parser.parse_args()


# --- Load dataset ---
with open(f"{args.dataset}_{args.domain}_name_merge.json", "r") as f_out:
    data = json.load(f_out)

examples, idxs, names, prompt_lst = [], [], [], []
return_dict = {}
total_len = len(data)
length = 0



# --- Batch processing ---
for key in data:
    example = data[key]
    idxs.append(key)
    names.append(example["name"])
    examples.append(example)
    prompt_input = build_prompt(example, args.domain, args.remove)
    prompt_lst.append([{"role": "user", "content": prompt_input}])
    length += 1

    # Process every 5 prompts
    if length % 5 == 0:
        for model_key, model_name in deployments.items():
            if not model_name:
                print(f"Skipping {model_key} (no deployment set).")
                continue

            success = False
            while not success:
                try:
                    ans = call_api_async(
                        prompt_lst, args.temperature, max_tokens=200,
                        model_key=model_key, model_name=model_name
                    )
                    success = True
                except Exception as e:
                    print(f"Error from {model_key}: {e}")
                    time.sleep(5)

            for id, n, a, e in zip(idxs, names, ans, examples):
                e[f"{model_key}_summary"] = a
                return_dict[id] = e

        print(f"{len(return_dict)}/{total_len}")
        length = 0
        prompt_lst, idxs, names, examples = [], [], [], []

# --- Final batch (if leftover) ---
if prompt_lst:
    for model_key, model_name in deployments.items():
        if not model_name:
            continue

        success = False
        while not success:
            try:
                ans = call_api_async(
                    prompt_lst, args.temperature, max_tokens=200,
                    model_key=model_key, model_name=model_name
                )
                success = True
            except Exception as e:
                print(f"Error from {model_key}: {e}")
                time.sleep(5)

        for id, n, a, e in zip(idxs, names, ans, examples):
            e[f"{model_key}_summary"] = a
            return_dict[id] = e

# --- Save results ---
os.makedirs(args.output_dir, exist_ok=True)
file_name = f"{args.output_dir}/{args.dataset}_{args.domain}_compare_models.json"

with open(file_name, "w") as f_out:
    json.dump(return_dict, f_out, indent=2)

print(f"\n Results saved to {file_name}")
