"""
Application Constants
"""

# Model Lists
GPT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek/deepseek-chat-v3-0324",
    "qwen/qwen3-235b-a22b-2507",
]

# OpenRouter Settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_PROVIDER_BODY = {"provider": {"sort": "price"}}

# RAG Configuration
RAG_DB_PATH = "data/qamp.db"
RAG_EMBED_MODEL = "gemini-embedding-001"
RAG_DEFAULT_TOP_K = 5

# Response Modes
RESPONSE_MODES = {
    "FAST": "Fast Mode",
    "DEEP": "Deep Thought Mode"
}

# Prompt File Paths
PROMPT_PATHS = {
    "QISKIT_INSTRUCTIONS": "prompts/qiskit_instructions.txt",
    "REVIEW_INSTRUCTIONS": "prompts/review_instructions.txt",
    "FORMATTING_INSTRUCTIONS": "prompts/formatting_instructions.txt",
    "QISKIT_REFINEMENT": "prompts/qiskit_refinement.txt",
    "TYPO_INSTRUCTIONS": "prompts/typo_instructions.txt",
    "CODEEXECUTOR_INSTRUCTIONS": "prompts/codeexecutor_instructions.txt"
}

# API Key File Suffixes
API_KEY_SUFFIXES = {
    "OPENAI": "_encrypted_api_key",
    "OPENROUTER": "_or_encrypted_api_key",
    "GEMINI": "_gai_encrypted_api_key"
}
