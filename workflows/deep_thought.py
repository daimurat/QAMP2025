import time
from agents import create_assistant_agent, create_user_proxy_agent

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
    # Start timing
    start_time = time.time()

    # ============================================================================
    # Phase 1: Planning - PlannerAgent analyzes user input and creates plan
    # ============================================================================
    
    # Create agents from external configuration files
    planner = create_assistant_agent("planner")
    planner_proxy = create_user_proxy_agent("planner_proxy")
    
    # Get implementation plan from Planner
    planner_proxy.initiate_chat(planner, message=user_input)
    last_msg = planner_proxy.last_message()
    if last_msg is None:
        raise RuntimeError("Failed to get response from PlannerAgent")
    plan = last_msg["content"]
    
    # ============================================================================
    # Phase 2: Implementation - QiskitDeveloper generates and verifies code
    # ============================================================================
    
    # Track execution results
    execute_results = []
    
    def execute_code(code: str) -> str:
        """
        Execute Qiskit code and return results or error messages.
        This function is called by QiskitDeveloper via Function Calling.
        
        Args:
            code: Python code string to execute
            
        Returns:
            Success message or error details
        """
        try:
            # Create execution environment
            exec_globals = {}
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Store successful execution
            execute_results.append({
                "code": code,
                "status": "success",
                "timestamp": time.time()
            })
            
            return "✓ Code executed successfully without errors."
            
        except Exception as e:
            # Store failed execution with error details
            error_msg = f"Error: {type(e).__name__}: {str(e)}"
            execute_results.append({
                "code": code,
                "status": "error",
                "error": error_msg,
                "timestamp": time.time()
            })
            
            return f"✗ Execution failed:\n{error_msg}\n\nPlease fix the code and try again."
    
    # Create agents from external configuration files
    qiskit_developer = create_assistant_agent("qiskit_developer")
    developer_proxy = create_user_proxy_agent(
        "developer_proxy",
        function_map={"execute_code": execute_code}
    )
    
    # Pass the plan from Planner to QiskitDeveloper
    developer_proxy.initiate_chat(
        qiskit_developer,
        message=f"""Based on the following implementation plan, generate Qiskit code:

{plan}

Generate complete, executable Python code with all necessary imports.
Use the execute_code function to verify your code works correctly.
Fix any errors until the code executes successfully.
""",
    )
    
    # Calculate elapsed time
    latency = time.time() - start_time
    
    # ============================================================================
    # Extract final Qiskit code
    # ============================================================================
    
    # Try to get the last successfully executed code
    successful_code = None
    for result in reversed(execute_results):
        if result["status"] == "success":
            successful_code = result["code"]
            qiskit_code = f"```python\n{successful_code}\n```"
            break
    
    if successful_code:
        # Return the verified working code
        qiskit_code = f"```python\n{successful_code}\n```"
    else:
        # Fallback: extract from conversation history
        qiskit_code = f"```python```"
    
    return qiskit_code, latency
