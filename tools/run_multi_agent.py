"""
CLI runner for the multi-agent orchestrator.

Usage:
    python -m tools.run_multi_agent --question "How do I use SamplerV2?" [--provider groq|openrouter] [--config config/multi_agent_config.yaml]
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from agents import (
    CodeAgentImpl,
    EvaluatorAgentImpl,
    ExecutorAgentImpl,
    LocalCodeSandbox,
    PlannerAgentImpl,
)
from clients.chat_factory import create_chat_client, ChatProvider
from config.loader import load_config
from orchestrator import MultiAgentOrchestrator, StateManager
from orchestrator.rag_adapter import MultiQueryRAGRetriever
from rag import RAGRetriever


def build_orchestrator(config_path: str | None, provider: str = "groq"):
    cfg = load_config(config_path or "config/multi_agent_config.yaml")

    # LLM clients (Groq or OpenRouter)
    planner_llm = create_chat_client(provider, model=cfg.planner_model)
    code_llm = create_chat_client(provider, model=cfg.code_model)
    evaluator_llm = create_chat_client(provider, model=cfg.evaluator_model)

    planner = PlannerAgentImpl(planner_llm)
    code_agent = CodeAgentImpl(code_llm)
    evaluator = EvaluatorAgentImpl(evaluator_llm)

    sandbox = LocalCodeSandbox()
    executor = ExecutorAgentImpl(sandbox=sandbox)

    rag_base = RAGRetriever()
    rag = MultiQueryRAGRetriever(rag_base)

    state_manager = StateManager()

    orchestrator = MultiAgentOrchestrator(
        planner=planner,
        code_agent=code_agent,
        executor=executor,
        evaluator=evaluator,
        rag_retriever=rag,
        state_manager=state_manager,
        config=cfg,
    )
    return orchestrator, cfg


async def main():
    # Load .env from project root so API keys are available.
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

    parser = argparse.ArgumentParser(description="Run multi-agent Qiskit assistant")
    parser.add_argument("--question", required=True, help="User question to answer")
    parser.add_argument(
        "--provider",
        choices=["groq", "openrouter"],
        default="groq",
        help="LLM provider to use (default: groq)",
    )
    parser.add_argument(
        "--config",
        default="config/multi_agent_config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    # Validate API key for selected provider
    if args.provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise SystemExit("GROQ_API_KEY environment variable is required")
    elif args.provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY environment variable is required")

    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY environment variable is required for RAG")

    print(f"Using provider: {args.provider}")
    orchestrator, cfg = build_orchestrator(args.config, args.provider)
    result = await orchestrator.run(args.question, config=cfg)

    print("=== RESULT ===")
    print(f"Success: {result.success}")
    print(f"Termination: {result.termination_reason}")
    print(f"Iterations: {result.iterations_used}")
    if result.final_answer:
        print("\nFinal Answer:\n", result.final_answer)
    if result.final_code:
        print("\nFinal Code:\n", result.final_code)


if __name__ == "__main__":
    asyncio.run(main())

