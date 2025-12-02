from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

def _read_prompt_from_file(path):
    with open(path, 'r') as f:
        return f.read()

def fast_mode_stream(user_input: str, context: str, message_history, stream_handler):
    # 1. Load Config
    Initial_Agent_Instructions = _read_prompt_from_file("prompts/qiskit_instructions.txt") # Reuse or adapt qiskit_instructions

    # Define agent (LLM)
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        streaming=True,
        callbacks=[stream_handler],
        temperature=0.2
    )

    system_msg = SystemMessage(content=Initial_Agent_Instructions)
    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{user_input}")
    messages = [system_msg] + message_history + [human_msg]

    return llm.stream(messages)