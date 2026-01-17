import time
from agents import create_assistant_agent
from tools.rag_tools import retrieve_qiskit_docs

def run_fast_mode(user_input: str):
    # Define agent
    qiskit_agent = create_assistant_agent("qiskit_developer")

    # retrieve context
    context = retrieve_qiskit_docs(user_input)

    # Start timing
    start_time = time.time()

    response = qiskit_agent.run(
        max_turns=1,
        message=f"Context:\n{context}\n\nQuestion:\n{user_input}"
    )

    response.process()
    
    # Calculate elapsed time
    latency = time.time() - start_time

    messages_list = list(response.messages)
    return messages_list[-1].get("content"), latency
