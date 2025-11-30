import openai
import os
import sys

def print_env():
    print("--- Azure OpenAI environment (non-secret) ---")
    print("OPENAI_API_TYPE:", os.getenv("OPENAI_API_TYPE") or os.getenv("AZURE_OPENAI_TYPE"))
    print("AZURE_OPENAI_BASE:", os.getenv("AZURE_OPENAI_BASE") or os.getenv("OPENAI_API_BASE"))
    print("AZURE_OPENAI_VERSION:", os.getenv("AZURE_OPENAI_VERSION") or os.getenv("OPENAI_API_VERSION"))
    print("AZURE_OPENAI_ENGINE:", os.getenv("AZURE_OPENAI_ENGINE") or os.getenv("OPENAI_ENGINE"))
    print("API key present:", bool(os.getenv("AZURE_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")))
    print("---------------------------------------------\n")

def main():
    # Read environment variables (safe defaults left as placeholders)
    base = os.getenv("AZURE_OPENAI_BASE", os.getenv("OPENAI_API_BASE", ""))
    version = os.getenv("AZURE_OPENAI_VERSION", os.getenv("OPENAI_API_VERSION", ""))
    engine = os.getenv("AZURE_OPENAI_ENGINE", os.getenv("OPENAI_ENGINE", ""))
    key = os.getenv("AZURE_OPENAI_KEY", os.getenv("OPENAI_API_KEY", ""))

    # Configure SDK for Azure
    openai.api_type = "azure"
    if base:
        openai.api_base = base
    if version:
        openai.api_version = version
    if key:
        openai.api_key = key

    print_env()

    if not (base and version and engine and key):
        print("One or more required Azure environment variables are missing.\nPlease set AZURE_OPENAI_BASE, AZURE_OPENAI_VERSION, AZURE_OPENAI_ENGINE and AZURE_OPENAI_KEY (or their OPENAI_ equivalents).")
        sys.exit(2)

    # Show the endpoint parts the SDK will target (useful to confirm values)
    print("Attempting chat completion with:")
    print(f"  base: {openai.api_base}")
    print(f"  api_version: {openai.api_version}")
    print(f"  deployment/engine: {engine}\n")

    try:
        resp = openai.ChatCompletion.create(
            engine=engine,
            messages=[{"role": "user", "content": "Hello from test_azure_connection_verbose.py"}],
            max_tokens=16,
        )
        print("Call succeeded. Response (truncated):")
        print(resp['choices'][0]['message']['content'])
    except Exception as e:
        # Try to extract HTTP status / code if SDK provides it
        code = getattr(e, 'http_status', None)
        print("Call failed with exception:")
        if code:
            print(f"  HTTP status: {code}")
        print("  Exception type:", type(e))
        print("  Exception message:", e)
        print("\nCommon causes:\n - The deployment name (engine) does not exactly match the name shown in Azure Portal (case-sensitive).")
        print(" - The API base is incorrect (should be the full Azure OpenAI resource base, e.g. https://your-resource.openai.azure.com).")
        print(" - The api-version is not supported for the deployed model; try a supported version such as 2023-05-15 or the version recommended in Azure docs.")
        print(" - The deployment was just created; wait a few minutes and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main()
