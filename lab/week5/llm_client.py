
import os
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai


load_dotenv()

CONFIG_FILE = Path(__file__).parent / "pricing.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    CONFIG = json.load(file)


def generate_gemini_response(prompt):
    """Generate a response using Google Gemini."""

    settings = CONFIG["gemini"]

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    model = settings["model"]

    client = genai.Client(api_key=api_key)

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    latency = time.perf_counter() - start_time

    usage = response.usage_metadata

    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    total_tokens = usage.total_token_count if usage else None

    if input_tokens is None or output_tokens is None:
        raise ValueError("Gemini did not return token usage metadata.")

    return {
        "provider": "Google Gemini",
        "model": model,
        "response": response.text or "",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency": latency
    }


def generate_ollama_response(prompt):
    """Generate a response using local Ollama."""

    settings = CONFIG["ollama"]

    model = settings["model"]

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    start_time = time.perf_counter()

    result = requests.post(
        url,
        json=payload,
        timeout=180
    )

    result.raise_for_status()

    latency = time.perf_counter() - start_time

    data = result.json()

    input_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")

    if input_tokens is None or output_tokens is None:
        raise ValueError("Ollama did not return token usage metadata.")

    return {
        "provider": "Ollama (Local LLM)",
        "model": model,
        "response": data.get("response", ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency": latency
    }


def generate_response(provider, prompt):
    """Route the request to the selected provider."""

    if provider == "gemini":
        return generate_gemini_response(prompt)

    if provider == "ollama":
        return generate_ollama_response(prompt)

    raise ValueError("Invalid provider selected.")
