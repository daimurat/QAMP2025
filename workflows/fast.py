import os
from autogen import ConversableAgent, LLMConfig
from tools.retrieval_tool import RetrievalTool
from config.constants import (
    GPT_MODELS,
    OPENROUTER_MODELS,
    OPENROUTER_BASE_URL,
    OPENROUTER_PROVIDER_BODY,
)

def _read_prompt_from_file(path):
    with open(path, 'r') as f:
        return f.read()

def _resolve_llm_config(selected_model: str, api_key_openai: str | None, api_key_openrouter: str | None):
    """Build provider-aware llm_config."""
    use_openrouter = selected_model in OPENROUTER_MODELS
    if use_openrouter:
        api_key = api_key_openrouter or os.getenv("OPENROUTER_API_KEY")
        return {
            "api_type": "openai",
            "model": selected_model,
            "api_key": api_key,
            "base_url": OPENROUTER_BASE_URL,
            "extra_body": OPENROUTER_PROVIDER_BODY,
        }

    model_name = selected_model if selected_model in GPT_MODELS else "gpt-4.1-mini"
    api_key = api_key_openai or os.getenv("OPENAI_API_KEY")
    return {
        "api_type": "openai",
        "model": model_name,
        "api_key": api_key,
    }


def run_fast_mode(user_input: str, vector_store, selected_model: str, api_key_openai: str | None = None,
                 api_key_openrouter: str | None = None):
    # 1. Load Config
    Initial_Agent_Instructions = _read_prompt_from_file("prompts/qiskit_instructions.txt") # Reuse or adapt qiskit_instructions
    llm_settings = _resolve_llm_config(selected_model, api_key_openai, api_key_openrouter)

    # Define agent (LLM)
    llm_config = LLMConfig(
        config_list=llm_settings
    )

    qiskit_agent = ConversableAgent(
        name = "qiskit_agent",
        system_message=Initial_Agent_Instructions,
        llm_config=llm_config
    )

    # retrieve context
    retrieval_tool = RetrievalTool(vector_store)
    context = retrieval_tool.retrieve(user_input)

    response = qiskit_agent.run(
        max_turns=1,
        message=f"Context:\n{context}\n\nQuestion:\n{user_input}"
    )

    response.process()
    
    messages_list = list(response.messages)
    return messages_list[-1].get("content")
