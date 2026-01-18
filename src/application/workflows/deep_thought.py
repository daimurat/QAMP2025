import time
import os
from autogen import LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.code_utils import extract_code
from src.application import agents, tools

def run_deep_thought_mode(user_input: str):
    """
    Run the deep thought mode with improved agent topology.
    
    Agent Topology:
    1. PlannerAgent: Receives user input, analyzes requirements, and creates implementation plan
    2. QiskitDeveloper: Receives plan from Planner, generates Qiskit code
    3. execute_code function: Executes code and returns results (called by QiskitDeveloper via Function Calling)
    
    Flow:
    User Input → Planner_proxy → PlannerAgent → Developer_proxy → QiskitDeveloper → execute_code → Result
    """

    llm_config = LLMConfig(config_list={
        "api_type": "openai",
        "model": "gpt-4.1-mini",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    })
   
    # Create function map for RAG tools
    function_map = {
        "retrieve_qiskit_docs": tools.retrieve_qiskit_docs,
    }
   
    # Create agents from external configuration files
    planner = agents.create_assistant_agent("planner", function_map=function_map)
    qiskit_developer = agents.create_assistant_agent("qiskit_developer")
    
    # Create proxy with function map for tool execution
    developer_proxy = agents.create_user_proxy_agent("qiskit_developer_proxy")
    
    # Define agents communication pattern
    pattern = AutoPattern(
        initial_agent=planner,
        agents=[planner, qiskit_developer],
        user_agent=developer_proxy,
        group_manager_args={"llm_config": llm_config},
    )

    # Start timing
    start_time = time.time()

    result, context_variables, last_agent = initiate_group_chat(
        pattern=pattern,
        messages=user_input
    )
    
    # Calculate elapsed time
    latency = time.time() - start_time
    
    # extract qiskit code
    lang = None
    qiskit_code = None

    for m in reversed(result.chat_history):
        if (m.get("name") == "QiskitDeveloper"):
            blocks = extract_code(m.get("content", ""))
            if blocks:
                lang, qiskit_code = blocks[-1]
                if qiskit_code=="TERMINATE":
                    continue
                break

    return qiskit_code, latency
