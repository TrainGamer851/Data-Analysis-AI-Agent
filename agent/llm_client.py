import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")


def get_model_config():
    # Determine the LLM provider from the environment, defaulting to NRP if not specified.  
    provider = os.getenv("LLM_PROVIDER", "nrp").lower()
    # Ollama and NRP are the currently supported providers.
    if provider == "ollama":
        # Assemble the configuration for the Ollama provider based on environment variables.
        return {
            "provider": "ollama",
            "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            "base_url": os.getenv(
                "OLLAMA_API_BASE",
                "http://localhost:11434/v1",
            ),
            "api_key": "ollama",
        }
    # If the provider is Ollama, assemble the configuration from environment variables.
    if provider == "nrp":
        # Retrieve the API key from the environment to ensure it is available for the client configuration.
        api_key = os.getenv("NRP_API_KEY")
        # Ensure that the API key is available before proceeding with the configuration.
        if not api_key:
            raise ValueError("NRP_API_KEY is missing.")
        # Additional validation for other required environment variables can be added here if needed.
        return {
            "provider": "nrp",
            "model": os.getenv("NRP_MODEL", "qwen3"),
            "base_url": os.getenv(
                "NRP_API_BASE",
                "https://ellm.nrp-nautilus.io/v1",
            ),
            "api_key": api_key,
        }
    # If the provider is not recognized, raise an error to prevent misconfiguration.
    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_client():
    # Retrieve the model configuration to determine which client to initialize.
    config = get_model_config()
    # Currently, only the OpenAI-compatible client is supported.
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=120.0,
    )


def generate_response(messages):
    # Retrieve the model configuration and initialize the client.
    config = get_model_config()
    client = get_client()
    # Validate that the client and configuration are properly initialized.
    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
    )
    # Ensure the response contains the expected structure.
    return response.choices[0].message.content