"""
Application Constants
"""

# Model Lists
GPT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]

# RAG Configuration
INDEX_PATH = "faiss_index"
INDEX_NAME = "qiskit_docs_index"

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
    "GEMINI": "_gai_encrypted_api_key"
}
