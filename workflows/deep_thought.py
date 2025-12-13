import os
import re
from autogen import AssistantAgent, LLMConfig, UserProxyAgent

def extract_qiskit_code_from_chat(user_proxy, assistant):
    """
    Extract Qiskit code from the entire conversation history.
    
    Args:
        user_proxy: UserProxyAgent instance
        assistant: AssistantAgent instance
    
    Returns:
        str: Extracted Qiskit code (longest one if multiple blocks exist)
    """
    # Get conversation history
    chat_messages = user_proxy.chat_messages.get(assistant, [])
    
    code_blocks = []
    
    for message in chat_messages:
        content = message.get("content", "")
        
        # Extract Markdown code blocks (```python ... ```)
        python_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
        code_blocks.extend(python_blocks)
        
        # Also detect code blocks without backticks
        if not python_blocks:
            # Look for lines containing "from qiskit" or "import qiskit"
            if 'qiskit' in content.lower():
                code_blocks.append(content)
    
    # Filter to only blocks containing Qiskit code
    qiskit_codes = [
        code for code in code_blocks
        if 'qiskit' in code.lower() or 'QuantumCircuit' in code
    ]
    
    if qiskit_codes:
        # Return the longest code block (usually the most complete code)
        return f"```python\n{max(qiskit_codes, key=len)}\n```"
    
    # Fallback: return the last message
    last_msg = assistant.last_message()
    if last_msg:
        return last_msg["content"]
    
    return "No Qiskit code found in conversation."

def run_deep_thought_mode(user_input: str):
    config_list = {
        "api_type": "openai",
        "model": "gpt-4.1-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    }

    llm_config = LLMConfig(config_list=config_list)

    planner = AssistantAgent(
        name="planner",
        description="Design quantum circuit, plan valification strategy, and determine acceptance criteria",
        llm_config=llm_config,
        # the default system message of the AssistantAgent is overwritten here
        system_message="You are a helpful AI assistant. You suggest coding and reasoning steps for another AI assistant to accomplish a task. Do not suggest concrete code. For any action beyond writing code or reasoning, convert it to a step that can be implemented by writing code. For example, browsing the web can be implemented by writing code that reads and prints the content of a web page. Finally, inspect the execution result. If the plan is not good, suggest a better plan. If the execution is wrong, analyze the error and suggest a fix.",
    )
    planner_user =  UserProxyAgent(
        name="planner_user",
        max_consecutive_auto_reply=0,  # terminate without auto-reply
        human_input_mode="NEVER",
        code_execution_config={
            "use_docker": False
        },
    )

    def ask_planner(message):
        """Ask planner agent for guidance (planner_user agent -> planner agent)"""
        planner_user.initiate_chat(planner, message=message)
        # Return the last message received from the planner
        last_msg = planner_user.last_message()
        if last_msg is None:
            return "No response from planner."
        return last_msg["content"]

    # create an AssistantAgent instance named "assistant"
    assistant = AssistantAgent(
        name="assistant",
        description="Generate quantum circuit",
        llm_config={
            "temperature": 0,
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
            "functions": [
                {
                    "name": "ask_planner",
                    "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest new plan.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "question to ask planner. Make sure the question include enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner.",
                            },
                        },
                        "required": ["message"],
                    },
                },
            ],
        },
    )

    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = UserProxyAgent(
        name="user_proxy",
        description="Execute and debug quantum circuit",
        human_input_mode="NEVER", # Never provide human feedback
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "planning",
            "use_docker": False,
        }, 
        function_map={"ask_planner": ask_planner},
    )

    # The assistant receives a message from the user, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message=user_input,
    )

    # Extract Qiskit code from the entire conversation history
    qiskit_code = extract_qiskit_code_from_chat(user_proxy, assistant)
    return qiskit_code
