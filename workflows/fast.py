import os
from autogen import ConversableAgent, LLMConfig
from tools.retrieval_tool import RetrievalTool

def _read_prompt_from_file(path):
    with open(path, 'r') as f:
        return f.read()

def run_fast_mode(user_input: str, vector_store):
    # 1. Load Config
    Initial_Agent_Instructions = _read_prompt_from_file("prompts/qiskit_instructions.txt") # Reuse or adapt qiskit_instructions

    # Define agent (LLM)
    llm_config = LLMConfig(
        config_list={
            "api_type": "openai",
            "model": "gpt-4.1-mini",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
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
