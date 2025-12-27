from typing import Dict, List

from ollama import chat
from rich.console import Console

console = Console()

# LLM configuration - Using Ollama
DEFAULT_MODEL = "deepseek-r1:1.5b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000


def call_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Core LLM API call wrapped as a durable DBOS step.

    The @DBOS.step() decorator makes this function durable - if it fails,
    DBOS will automatically retry it. This is essential for building reliable
    agents that can recover from transient failures.
    """
    try:
        console.print(
            f"[dim]    🧠 Sending messages to LLM ({model}, temperature={temperature}, max_tokens={max_tokens})...[/dim]"
        )
        response = chat(
            model=model,
            messages=messages,
            format="json",  # Request JSON format from Ollama
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        console.print("[dim]    🧠 LLM response received. Cleaning and parsing...[/dim]")
        return response["message"]["content"]
    except Exception as e:
        console.print(f"[red]LLM API call failed: {str(e)}[/red]")
        raise Exception(f"LLM API call failed: {str(e)}")


def clean_json_response(response: str) -> str:
    """Clean LLM response to extract valid JSON.

    LLMs often return JSON wrapped in markdown code blocks.
    This utility function strips that formatting for reliable parsing.
    """
    console.print("[dim]    🧹 Cleaning LLM JSON response...[/dim]")
    cleaned = response.strip()

    # Remove markdown code blocks
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    console.print("[dim]    🧹 Cleaned JSON response:[/dim]")
    return cleaned.strip()
