"""Tests for Component 2: LLM Integration."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.llm_client import (
    generate_response,
    get_client,
    get_model_config,
)
from agent.prompts import (
    SYSTEM_PROMPT,
    BUSINESS_RECOMMENDATION_PROMPT,
    build_analysis_prompt,
    build_recommendation_prompt,
)


# Verify provider settings, credential validation, and response handling.
def test_model_configuration(monkeypatch):
    # Set up environment variables for the NRP provider to ensure consistent test behavior.
    """Check that NRP settings are loaded correctly."""
    # Specify the LLM provider to be NRP for this test.
    monkeypatch.setenv("LLM_PROVIDER", "nrp")
    # Provide explicit values so the test does not depend on the local environment.
    monkeypatch.setenv("NRP_API_KEY", "fake-test-key")
    monkeypatch.setenv("NRP_MODEL", "qwen3")
    monkeypatch.setenv(
        "NRP_API_BASE",
        "https://ellm.nrp-nautilus.io/v1",
    )
    # Read the assembled configuration and verify each configured value.
    config = get_model_config()
    # Verify that the returned configuration matches the expected values.
    assert config["model"] == "qwen3"
    assert config["api_key"] == "fake-test-key"
    assert config["base_url"] == "https://ellm.nrp-nautilus.io/v1"


def test_missing_api_key(monkeypatch):
    # Set up environment variables for the NRP provider to ensure consistent test behavior.
    """Missing credentials should raise an error."""
    # Specify the LLM provider to be NRP for this test.
    monkeypatch.setenv("LLM_PROVIDER", "nrp")
    # Ensure no developer or CI environment key can mask the missing-key case.
    monkeypatch.delenv("NRP_API_KEY", raising=False)
    # Configuration must fail before a client can make a remote request.
    with pytest.raises(ValueError, match="NRP_API_KEY"):
        get_model_config()


def test_client_creation(monkeypatch):
    # Set up environment variables for the NRP provider to ensure consistent test behavior.
    """Check that an OpenAI-compatible client is created."""
    # Specify the LLM provider to be NRP for this test.
    monkeypatch.setenv("LLM_PROVIDER", "nrp")
    # Supply the required NRP credential before constructing the client.
    monkeypatch.setenv("NRP_API_KEY", "fake-test-key")
    client = get_client()
    # A client object should be returned without making an API call.
    assert client is not None


def test_generate_response(monkeypatch):
    # Set up environment variables for the NRP provider to ensure consistent test behavior.
    """Check response handling without making an API request."""
    # Specify the LLM provider to be NRP for this test.
    monkeypatch.setenv("LLM_PROVIDER", "nrp")
    # Configure the provider and prepare a mocked completion response.
    monkeypatch.setenv("NRP_API_KEY", "fake-test-key")
    monkeypatch.setenv("NRP_MODEL", "qwen3")
    # Mock the client to return a predefined response without making an actual API call.
    mock_response = MagicMock()
    mock_response.choices[0].message.content = 'result = df["Sales"].sum()'
    # Patch the get_client function to return the mocked client.
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    # Replace the network-backed client so the test stays offline.
    with patch("agent.llm_client.get_client", return_value=mock_client):
        response = generate_response(
            [{"role": "user", "content": "Calculate total sales"}]
        )
    # Ensure the mocked client was used to generate the response.
    assert response == 'result = df["Sales"].sum()'
    assert mock_client.chat.completions.create.call_args.kwargs["model"] == "qwen3"


def test_business_analysis_prompt():
    # Test that the analysis prompt correctly incorporates the dataframe schema and the business question.
    """The prompt should include actual dataset columns."""
    # Use a small dataframe with representative business columns.
    df = pd.DataFrame(
        {
            "Category": ["Furniture", "Technology"],
            "Sales": [100, 200],
            "Profit": [10, 50],
        }
    )
    # Build the prompt and check that it includes schema and question context.
    prompt = build_analysis_prompt(
        "Which category has the highest profit?",
        df,
    )
    # The prompt should include the schema and the business question.
    assert "Category" in prompt
    assert "Sales" in prompt
    assert "Profit" in prompt
    assert "Which category has the highest profit?" in prompt
    assert "result" in prompt


def test_code_generation_instructions():
    # Verify that the system prompt contains essential instructions for code generation.
    """The system prompt should require dataset-based calculations."""
    # Confirm core constraints are included in the instructions.
    assert "pandas" in SYSTEM_PROMPT.lower()
    assert "result" in SYSTEM_PROMPT
    assert "Do not hardcode" in SYSTEM_PROMPT


def test_business_recommendation_prompt():
    # Build the recommendation prompt and check that it includes key instructions and context.
    """Recommendations should use evidence and acknowledge limitations."""
    # Ensure recommendations address both external evidence and citations.
    assert "recommendations" in BUSINESS_RECOMMENDATION_PROMPT.lower()
    assert "external" in BUSINESS_RECOMMENDATION_PROMPT.lower()
    assert "sources" in BUSINESS_RECOMMENDATION_PROMPT.lower()


def test_recommendation_uses_supplied_evidence():
    # Provide a question, calculation results, and research sources to the recommendation prompt.
    """Recommendation prompts should contain calculated evidence."""
    # Supply verified calculation output alongside a research source.
    prompt = build_recommendation_prompt(
        question="How can we improve profitability?",
        calculation_results="Furniture profit margin: 2.5%",
        research_sources=[
            {
                "title": "Example industry report",
                "url": "https://example.org/report",
            }
        ],
    )
    # The prompt should preserve both the measured result and its source.
    assert "2.5%" in prompt
    assert "Example industry report" in prompt
    assert "https://example.org/report" in prompt


def test_recommendation_without_research():
    # Test the behavior when no research sources are provided.
    """The prompt must disclose when research was not performed."""
    # Omit research sources to exercise the no-research fallback.
    prompt = build_recommendation_prompt(
        question="Which category needs improvement?",
        calculation_results="Furniture has the lowest profit margin.",
    )
    # The prompt should indicate that no external research was performed.
    assert "No external research was performed." in prompt


# ---------------------------------------------------------
# Ollama-specific unit tests
# ---------------------------------------------------------


def test_ollama_model_configuration(monkeypatch):
    # Set up environment variables for Ollama provider.
    """Ollama should use the local Qwen3 model."""
    from agent.llm_client import get_model_config

    # Select Ollama and set its model and local-compatible endpoint.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv(
        "OLLAMA_API_BASE",
        "http://localhost:11434/v1",
    )
    # Confirm the provider-specific values, including its placeholder API key.
    config = get_model_config()
    # Check that the configuration matches the expected Ollama settings.
    assert config["provider"] == "ollama"
    assert config["model"] == "qwen3:4b"
    assert config["base_url"] == "http://localhost:11434/v1"
    assert config["api_key"] == "ollama"


def test_ollama_does_not_require_nrp_key(monkeypatch):
    # Configure Ollama provider without setting NRP API key to test fallback behavior.
    """Ollama should work without an NRP API key."""
    from agent.llm_client import get_model_config

    # Remove NRP credentials to prove the local provider does not depend on them.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("NRP_API_KEY", raising=False)
    # Retrieve the configuration to verify that Ollama does not require the NRP API key.
    config = get_model_config()
    # The provider should still be Ollama and the API key should be the placeholder value.
    assert config["provider"] == "ollama"
    assert config["api_key"] == "ollama"


def test_ollama_client_creation(monkeypatch):
    # Set up environment variables for the Ollama provider to point to the local API.
    """Ollama client should connect to localhost."""
    from agent.llm_client import get_client

    # Point the client at the local Ollama-compatible API.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv(
        "OLLAMA_API_BASE",
        "http://localhost:11434/v1",
    )
    # Verify both the configured endpoint and placeholder key.
    client = get_client()
    # The client should have the correct base URL and API key as per the environment settings.
    assert str(client.base_url).rstrip("/") == ("http://localhost:11434/v1")
    assert client.api_key == "ollama"


def test_ollama_generate_response(monkeypatch):
    # Set up environment variables for the Ollama provider and mock the client.
    """The selected Ollama model should receive the messages."""
    from unittest.mock import MagicMock, patch
    from agent.llm_client import generate_response

    # Configure Ollama and create a mocked client to avoid network access.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    # Create a mock client to simulate the Ollama API response.
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Hello from Ollama"))
    ]
    # Keep the input messages so the mock call can be checked exactly.
    messages = [{"role": "user", "content": "Say hello"}]
    # Patch the get_client function to return the mocked client during the test.
    with patch("agent.llm_client.get_client", return_value=mock_client):
        result = generate_response(messages)
    # Ensure the result matches the mocked response content.
    assert result == "Hello from Ollama"
    # Verify that the mocked client's create method was called with the correct parameters.
    mock_client.chat.completions.create.assert_called_once_with(
        model="qwen3:4b",
        messages=messages,
    )


def test_switch_from_ollama_to_nrp(monkeypatch):
    # Test switching from Ollama to NRP provider and verify configurations.
    """Switching providers should select the correct configuration."""
    from agent.llm_client import get_model_config

    # Start with Ollama, then change environment settings to NRP.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    ollama_config = get_model_config()
    # Check that the Ollama configuration matches the expected values after initial setup.
    assert ollama_config["provider"] == "ollama"
    # Verify that the Ollama configuration matches the expected values.
    monkeypatch.setenv("LLM_PROVIDER", "nrp")
    monkeypatch.setenv("NRP_API_KEY", "fake-test-key")
    monkeypatch.setenv("NRP_MODEL", "qwen3")
    # Re-read configuration after switching to confirm changes take effect.
    nrp_config = get_model_config()
    # Check that the NRP configuration matches the expected values after switching providers.
    assert nrp_config["provider"] == "nrp"
    assert nrp_config["model"] == "qwen3"
    assert nrp_config["api_key"] == "fake-test-key"
