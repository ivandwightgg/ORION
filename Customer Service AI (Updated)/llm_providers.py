from __future__ import annotations
import httpx
import yaml
from typing import Mapping, Any


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """
    Load the YAML configuration safely and validate the required structure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or "llm" not in config:
            raise ValueError("Invalid or missing 'llm' section in configuration file.")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


# Global configuration load
CONFIG = load_config()


async def call_llm(tier: str, prompt: Mapping[str, str]) -> str:
    """
    Calls a local Ollama LLM model based on the configured tier.
    Args:
        tier: Model tier ('tier1' or 'tier2')
        prompt: Dictionary containing 'system' and 'user' messages.
    Returns:
        The text output from the model as a string.
    """
    llm_cfg = CONFIG["llm"]

    if tier == "tier1":
        model = llm_cfg.get("tier1_model")
        max_tokens = llm_cfg.get("max_output_tokens_tier1", 512)
    elif tier == "tier2":
        model = llm_cfg.get("tier2_model")
        max_tokens = llm_cfg.get("max_output_tokens_tier2", 1024)
    else:
        raise ValueError(f"Invalid tier '{tier}'. Must be 'tier1' or 'tier2'.")

    if not model:
        raise ValueError(f"No model configured for {tier} in config.yaml.")

    return await _ollama_chat(model, prompt, max_tokens)


async def _ollama_chat(model: str, prompt: Mapping[str, str], max_tokens: int) -> str:
    """
    Internal helper function to call the Ollama chat API asynchronously.
    """
    url = "http://localhost:11434/api/chat"

    system_msg = prompt.get("system", "You are a helpful assistant.")
    user_msg = prompt.get("user", "")

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.2},
                },
            )
            response.raise_for_status()
            data = response.json()
    except httpx.RequestError as e:
        raise ConnectionError(f"Failed to reach Ollama API at {url}: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama API returned error {e.response.status_code}: {e.response.text}") from e

    # Extract message content if available
    message_content = data.get("message", {}).get("content")
    if message_content:
        return message_content

    # Fallback: return entire data for debugging if content missing
    return str(data)
