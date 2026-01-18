import time
import os
import re
from typing import TYPE_CHECKING
from autogen import LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.code_utils import extract_code
from agents import create_assistant_agent, create_user_proxy_agent
from tools import retrieve_qiskit_docs
from tools.rag_tools import reset_rag_query_count
from utils.trace_logger import set_rag_trace_callback, clear_rag_trace_callback

if TYPE_CHECKING:
    from utils.trace_logger import TraceLogger
from config.constants import (
    GPT_MODELS,
    OPENROUTER_MODELS,
    OPENROUTER_BASE_URL,
    OPENROUTER_PROVIDER_BODY,
)

DEFAULT_MODEL = "gpt-4.1-mini"


def _build_llm_settings(selected_model: str | None, api_key_openai: str | None, api_key_openrouter: str | None):
    """Provider-aware llm settings for agents and manager."""
    use_openrouter = selected_model in OPENROUTER_MODELS
    model_name = selected_model or DEFAULT_MODEL

    if use_openrouter:
        return {
            "api_type": "openai",
            "model": model_name,
            "api_key": api_key_openrouter or os.environ.get("OPENROUTER_API_KEY"),
            "base_url": OPENROUTER_BASE_URL,
            "extra_body": OPENROUTER_PROVIDER_BODY,
        }

    model_name = model_name if model_name in GPT_MODELS else DEFAULT_MODEL
    return {
        "api_type": "openai",
        "model": model_name,
        "api_key": api_key_openai or os.environ.get("OPENAI_API_KEY"),
    }


def run_deep_thought_mode(
    user_input: str,
    selected_model: str | None = None,
    api_key_openai: str | None = None,
    api_key_openrouter: str | None = None,
    trace_logger: "TraceLogger | None" = None
):
    """
    Run the deep thought mode with improved agent topology.
    
    Agent Topology:
    1. PlannerAgent: Receives user input, analyzes requirements, and creates implementation plan
    2. QiskitDeveloper: Receives plan from Planner, generates Qiskit code
    3. execute_code function: Executes code and returns results (called by QiskitDeveloper via Function Calling)
    
    Flow:
    User Input → Planner_proxy → PlannerAgent → Developer_proxy → QiskitDeveloper → execute_code → Result
    
    Args:
        user_input: The prompt/task for the agents
        selected_model: Model to use (optional)
        api_key_openai: OpenAI API key (optional)
        api_key_openrouter: OpenRouter API key (optional)
        trace_logger: TraceLogger instance for observability (optional)
    
    Returns:
        tuple: (generated_code, latency_seconds)
    """
    # Reset RAG query counter for this task (limits to 3 queries)
    reset_rag_query_count()
    
    # Set up RAG trace callback if trace_logger is provided
    if trace_logger:
        def rag_callback(query: str, top_k: int, min_score: float, chunks: list):
            trace_logger.add_rag_query(query, top_k, min_score, chunks)
        set_rag_trace_callback(rag_callback)

    try:
        llm_settings = _build_llm_settings(selected_model, api_key_openai, api_key_openrouter)
        llm_config = LLMConfig(config_list=llm_settings)
       
        # Create function map for RAG tools
        function_map = {
            "retrieve_qiskit_docs": retrieve_qiskit_docs,
        }
       
        # Create agents from external configuration files
        planner = create_assistant_agent("planner", function_map=function_map, llm_overrides=llm_settings)
        qiskit_developer = create_assistant_agent("qiskit_developer", llm_overrides=llm_settings)
        
        # Create proxy with function map for tool execution
        developer_proxy = create_user_proxy_agent("qiskit_developer_proxy")
        
        # Define agents communication pattern
        # Simple termination: allow TERMINATE once code is generated
        class ExecutionAwareTermination:
            """Allow termination after code is generated or execution feedback is received."""
            
            def __init__(self):
                self.code_generated = False
            
            def __call__(self, msg):
                content = msg.get("content") or ""
                name = msg.get("name") or ""
                
                # Track if QiskitDeveloper generated code (code block present)
                if name == "QiskitDeveloper" and "```python" in content:
                    self.code_generated = True
                
                # Track if execution feedback received (also acceptable)
                if "exitcode:" in content.lower():
                    self.code_generated = True  # Execution means code was handled
                
                # Allow termination if TERMINATE is in message AND code was generated
                if "TERMINATE" in content:
                    return self.code_generated
                
                return False
        
        termination_checker = ExecutionAwareTermination()
        
        pattern = AutoPattern(
            initial_agent=planner,
            agents=[planner, qiskit_developer],
            user_agent=developer_proxy,
            group_manager_args={
                "llm_config": llm_config,
                "is_termination_msg": termination_checker,
            },
        )

        # Start timing
        start_time = time.time()

        result, context_variables, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=user_input
        )
        
        # Calculate elapsed time
        latency = time.time() - start_time
        
        # Capture chat history in trace logger if provided
        if trace_logger:
            trace_logger.capture_from_chat_history(result.chat_history)
        
        # Extract qiskit code with improved logic
        qiskit_code = _extract_best_code(result.chat_history)

        return qiskit_code, latency
    
    finally:
        # Always clear the RAG trace callback
        clear_rag_trace_callback()


def _extract_best_code(chat_history: list) -> str | None:
    """
    Extract the best code block from chat history.
    
    Priority:
    1. Code from QiskitDeveloper containing 'def ' (function definition)
    2. Any code from QiskitDeveloper (non-TERMINATE)
    3. Code from any agent containing 'def '
    4. None if no valid code found
    """
    developer_codes = []
    any_codes = []
    
    for m in reversed(chat_history):
        content = m.get("content", "") or ""
        is_developer = m.get("name") == "QiskitDeveloper"
        
        blocks = extract_code(content)
        if not blocks:
            continue
            
        for lang, code in blocks:
            if not code or code.strip() == "TERMINATE":
                continue
            
            # Skip conversational text (no code-like content)
            if _is_conversational(code):
                continue
                
            if is_developer:
                developer_codes.append(code)
            else:
                any_codes.append(code)
    
    # Priority 1: Developer code with function definition
    for code in developer_codes:
        if "def " in code:
            return code
    
    # Priority 2: Any developer code
    if developer_codes:
        return developer_codes[0]
    
    # Priority 3: Any code with function definition
    for code in any_codes:
        if "def " in code:
            return code
    
    # Priority 4: Any code at all
    if any_codes:
        return any_codes[0]
    
    return None


def _is_conversational(text: str) -> bool:
    """Check if text is conversational rather than code."""
    conversational_patterns = [
        r"^I'm glad",
        r"^I'd be happy",
        r"^Here's",
        r"^Let me",
        r"^Sure,",
        r"^Of course",
        r"^The implementation",
        r"^This code",
    ]
    text_stripped = text.strip()
    for pattern in conversational_patterns:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return True
    # If it has no Python keywords, likely not code
    code_indicators = ["import ", "def ", "class ", "return ", "=", "(", "from "]
    if not any(indicator in text for indicator in code_indicators):
        return True
    return False

