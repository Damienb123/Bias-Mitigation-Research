from openai import AzureOpenAI
import os

# Create Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Read both deployment names from environment variables
deployments = {
    "model_1 (gpt-3.5)": os.getenv("AZURE_OPENAI_ENGINE_1"),
    "model_2 (gpt-4)": os.getenv("AZURE_OPENAI_ENGINE_2")
}

# Check that environment variables are set
for label, deployment in deployments.items():
    if not deployment:
        print(f"Missing deployment name for {label}. Please check your environment variables.")
    else:
        print(f"Found deployment for {label}: {deployment}")

print("\n--- Running test completions ---")

for label, deployment in deployments.items():
    if not deployment:
        continue
    try:
        print(f"\n🔹 Testing {label} ({deployment})...")
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Hello, can you briefly introduce yourself?"}],
            max_tokens=50
        )
        print(f"Response from {label}: {response.choices[0].message.content}\n")
    except Exception as e:
        print(f"Error testing {label}: {e}\n")
