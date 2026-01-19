"""
CLI runner for Fast Mode (no RAG by default).

Example:
    python fast_cli.py "Explain Bell state preparation in Qiskit" \\
        --model qwen/qwen3-235b-a22b-2507 \\
        --openrouter-key sk-or-... \\
        --no-rag
"""

import argparse
import os
import sys

from workflows.fast import run_fast_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fast Mode once from the command line (RAG off by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        help="User prompt/question to send to the Fast Mode agent.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model name. Supports OpenAI models or OpenRouter models like qwen/qwen3-235b-a22b-2507.",
    )
    parser.add_argument(
        "--openai-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (used when the model is an OpenAI model).",
    )
    parser.add_argument(
        "--openrouter-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (used when the model is an OpenRouter model such as qwen/*).",
    )
    parser.add_argument(
        "--use-rag",
        dest="use_rag",
        action="store_true",
        default=False,
        help="Enable RAG (requires a vector store; leave off to skip RAG).",
    )
    parser.add_argument(
        "--no-rag",
        dest="use_rag",
        action="store_false",
        help="Disable RAG (default).",
    )
    return parser.parse_args()


def _validate_keys(model: str, openai_key: str | None, openrouter_key: str | None) -> None:
    is_openrouter = "/" in model or model.startswith("openai/")
    if is_openrouter and not openrouter_key:
        sys.exit("Error: OpenRouter model selected but no OPENROUTER_API_KEY provided.")
    if not is_openrouter and not openai_key:
        sys.exit("Error: OpenAI model selected but no OPENAI_API_KEY provided.")


def main():
    args = parse_args()

    _validate_keys(args.model, args.openai_key, args.openrouter_key)

    result = run_fast_mode(
        user_input=args.prompt,
        vector_store=None,  # No vector store when RAG is off
        selected_model=args.model,
        api_key_openai=args.openai_key,
        api_key_openrouter=args.openrouter_key,
        use_rag=args.use_rag,
    )

    print(result)


if __name__ == "__main__":
    main()
