import os
import requests
import time

from dotenv import load_dotenv
from openai import OpenAI

from agent.debug_logger import debug_log, DebugTimer
from agent.time_budget import request_timeout

load_dotenv(".env")

GENERATION_PROFILES = {
    "default": {
        "max_tokens": 768,
        "temperature": 0.0,
    },
    "code": {
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "explanation": {
        "max_tokens": 512,
        "temperature": 0.0,
    },
}


# Return generation settings.
def get_generation_settings(purpose):
    # Raise `ValueError` when `purpose not in GENERATION_PROFILES`.
    if purpose not in GENERATION_PROFILES:
        raise ValueError(f"Unsupported generation purpose: {purpose}")

    return GENERATION_PROFILES[purpose].copy()


# Optional Qwen3 thinking-mode configuration.
def get_nrp_extra_body(model):

    disable_thinking = os.getenv("NRP_DISABLE_THINKING", "0") == "1"

    # Return the selected result when `disable_thinking and 'qwen3' in model.lower()`.
    if disable_thinking and "qwen3" in model.lower():
        return {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        }

    return {}


# Return model config.
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


# Return client.
def get_client():
    config = get_model_config()

    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=120.0,
        max_retries=0,
    )


# Build an analysis response from the model output and dataset context.
def generate_response(
    messages,
    *,
    timeout=120.0,
    purpose="default",
):
    timer = DebugTimer("LLM")

    settings = get_generation_settings(purpose)

    # Recheck the shared deadline before contacting the provider.
    timeout = request_timeout(
        timeout,
        reserve=5.0,
    )

    config = get_model_config()
    provider = config["provider"]

    debug_log(
        "LLM",
        f"Provider: {provider}",
    )

    debug_log(
        "LLM",
        f"Model: {config['model']}",
    )

    debug_log(
        "LLM",
        f"Messages: {len(messages)}",
    )

    prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)

    debug_log(
        "LLM",
        f"Total message characters: {prompt_chars:,}",
    )

    debug_log(
        "LLM",
        f"Configured request timeout: {timeout:.2f}s",
    )

    debug_log(
        "LLM SETTINGS",
        f"Generation purpose: {purpose}",
    )

    debug_log(
        "LLM SETTINGS",
        f"Maximum output tokens: {settings['max_tokens']}",
    )

    debug_log(
        "LLM SETTINGS",
        f"Temperature: {settings['temperature']}",
    )

    try:
        # Update `url` when `provider == 'ollama'`.
        if provider == "ollama":
            url = config["base_url"].rstrip("/").removesuffix("/v1")

            url += "/api/chat"

            debug_log(
                "OLLAMA",
                "Preparing native chat request",
            )

            debug_log(
                "OLLAMA",
                "Streaming disabled | Thinking disabled",
            )

            debug_log(
                "OLLAMA",
                f"Temperature: {settings['temperature']} | "
                f"Token limit: {settings['max_tokens']}",
            )

            debug_log(
                "OLLAMA",
                "Sending HTTP request",
            )

            response = requests.post(
                url,
                json={
                    "model": config["model"],
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "num_predict": settings["max_tokens"],
                        "temperature": settings["temperature"],
                    },
                },
                timeout=timeout,
            )

            debug_log(
                "OLLAMA",
                f"HTTP status: {response.status_code}",
            )

            response.raise_for_status()

            debug_log(
                "OLLAMA",
                "Decoding response JSON",
            )

            data = response.json()

            content = data.get("message", {}).get("content", "").strip()

            # Preserve existing debug statements.
            print(
                "[LLM] Finish reason:",
                data.get("done_reason"),
                flush=True,
            )

            print(
                "[LLM] Content:",
                repr(content),
                flush=True,
            )

            debug_log(
                "OLLAMA",
                f"Response characters: {len(content)}",
            )

            # Process each `field` from `('total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration')`.
            for field in (
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            ):
                value = data.get(field)

                # Call `debug_log` when `isinstance(value, (int, float))`.
                if isinstance(value, (int, float)):
                    debug_log(
                        "OLLAMA",
                        f"{field}: {value / 1e9:.3f}s",
                    )

            return content

        # NRP provider
        debug_log(
            "NRP",
            "Initializing OpenAI-compatible client",
        )

        client = get_client()

        extra_body = get_nrp_extra_body(config["model"])

        request_options = {
            "model": config["model"],
            "messages": messages,
            "timeout": timeout,
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
        }

        # Update `request_options['extra_body']` when `extra_body`.
        if extra_body:
            request_options["extra_body"] = extra_body

            debug_log(
                "NRP SETTINGS",
                "Requesting disabled Qwen3 thinking mode",
            )

        # Call `debug_log` when no earlier condition matches.
        else:
            debug_log(
                "NRP SETTINGS",
                "Using provider-default thinking mode",
            )

        debug_log(
            "NRP SETTINGS",
            f"Generation purpose: {purpose} | "
            f"Temperature: {settings['temperature']} | "
            f"Token limit: {settings['max_tokens']}",
        )

        start = time.monotonic()

        print("[NRP] Request started", flush=True)

        try:
            debug_log(
                "NRP",
                "Sending chat completion request",
            )

            response = client.chat.completions.create(**request_options)

            debug_log(
                "NRP",
                "Chat completion response received",
            )

            choice = response.choices[0]
            content = choice.message.content

            print(
                "[NRP] Content:",
                repr(content),
                flush=True,
            )

            debug_log(
                "NRP",
                f"Finish reason: " f"{getattr(choice, 'finish_reason', None)}",
            )

            debug_log(
                "NRP",
                f"Response characters: {len(content or '')}",
            )

            usage = getattr(response, "usage", None)

            # Process each item from `('prompt_tokens', 'completion_tokens', 'total_tokens')` when `usage is not None`.
            if usage is not None:
                # Process each `field` from `('prompt_tokens', 'completion_tokens', 'total_tokens')`.
                for field in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    value = getattr(usage, field, None)

                    # Call `debug_log` when `isinstance(value, int)`.
                    if isinstance(value, int):
                        debug_log(
                            "TOKENS",
                            f"{field}: {value:,}",
                        )

                completion_details = getattr(
                    usage,
                    "completion_tokens_details",
                    None,
                )

                reasoning_tokens = getattr(
                    completion_details,
                    "reasoning_tokens",
                    None,
                )

                # Call `debug_log` when `isinstance(reasoning_tokens, int)`.
                if isinstance(reasoning_tokens, int):
                    debug_log(
                        "TOKENS",
                        f"Reasoning tokens: {reasoning_tokens:,}",
                    )
                # Call `debug_log` when no earlier condition matches.
                else:
                    debug_log(
                        "TOKENS",
                        "Reasoning token count not reported",
                    )

            # Call `debug_log` when no earlier condition matches.
            else:
                debug_log(
                    "TOKENS",
                    "Provider did not report token usage",
                )

            reasoning_content = getattr(
                choice.message,
                "reasoning_content",
                None,
            )

            # Call `debug_log` when `isinstance(reasoning_content, str)`.
            if isinstance(reasoning_content, str):
                debug_log(
                    "NRP",
                    f"Separate reasoning characters: " f"{len(reasoning_content):,}",
                )

            # Call `debug_log` when `getattr(choice, 'finish_reason', None) == 'length'`.
            if getattr(choice, "finish_reason", None) == "length":
                debug_log(
                    "LLM WARNING",
                    "Generation reached the output token limit; "
                    "response may be incomplete",
                )

                # Call `debug_log` when `purpose == 'code'`.
                if purpose == "code":
                    # The runner already handles empty code by
                    # requesting a corrected response.
                    debug_log(
                        "LLM WARNING",
                        "Discarding potentially truncated Python code",
                    )

                    return ""

            return content

        finally:
            elapsed = time.monotonic() - start

            print(
                f"[NRP] Request duration: {elapsed:.2f}s",
                flush=True,
            )

    except Exception as exc:
        debug_log(
            "LLM ERROR",
            f"Provider: {provider} | " f"Exception: {type(exc).__name__}",
        )

        status = getattr(exc, "status_code", None)

        # Call `debug_log` when `status is not None`.
        if status is not None:
            debug_log(
                "LLM ERROR",
                f"HTTP status: {status}",
            )

        raise

    finally:
        timer.finish("LLM request lifecycle finished")
