"""
Agent Configuration Loader

This module provides utilities to load agent configurations from YAML files
and create AutoGen agent instances (AssistantAgent and UserProxyAgent).

Usage:
    from agents import create_assistant_agent, create_user_proxy_agent
    
    config_list = {"api_type": "openai", "model": "gpt-4.1-mini"}
    planner = create_assistant_agent("planner", config_list)
    planner_proxy = create_user_proxy_agent("planner_proxy")
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from autogen import AssistantAgent, UserProxyAgent


# Get the directory where this file is located
AGENTS_DIR = Path(__file__).parent


def load_agent_config(agent_id: str) -> Dict[str, Any]:
    """
    Load agent configuration from YAML file.
    
    Args:
        agent_id: Agent identifier (e.g., "planner", "qiskit_developer")
        
    Returns:
        Dictionary containing agent configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = AGENTS_DIR / f"{agent_id}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Agent configuration file not found: {config_path}\n"
            f"Available agents: {list_available_agents()}"
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"Empty configuration file: {config_path}")
    
    # Validate required fields
    if 'agent_id' not in config:
        config['agent_id'] = agent_id
    
    if 'type' not in config:
        raise ValueError(f"Missing 'type' field in {config_path}")
    
    return config


def list_available_agents() -> list:
    """
    List all available agent configuration files.
    
    Returns:
        List of agent IDs (without .yaml extension)
    """
    yaml_files = AGENTS_DIR.glob("*.yaml")
    return [f.stem for f in yaml_files if f.stem != '__init__']


def create_assistant_agent(
    agent_id: str,
    function_map: Optional[Dict[str, Callable]] = None
) -> AssistantAgent:
    """
    Create an AssistantAgent from configuration file.
    
    Args:
        agent_id: Agent identifier (e.g., "planner", "qiskit_developer")
        config_list: LLM configuration (api_type, model, etc.)
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        
    Returns:
        Configured AssistantAgent instance
        
    Raises:
        ValueError: If agent type is not 'assistant'
    """
    config = load_agent_config(agent_id)
    
    if config['type'] != 'assistant':
        raise ValueError(
            f"Agent '{agent_id}' is type '{config['type']}', expected 'assistant'"
        )
    
    # Merge base config_list with agent-specific llm_config
    llm_config = config['llm_config']
    
    # Add API key if provided
    if 'OPENAI_API_KEY' in os.environ:
        llm_config['api_key'] = os.environ['OPENAI_API_KEY']
    
    # Create AssistantAgent
    return AssistantAgent(
        name=config.get('name', agent_id),
        description=config.get('description', ''),
        system_message=config.get('system_message', ''),
        llm_config=llm_config,
        function_map=function_map
    )


def create_user_proxy_agent(
    agent_id: str,
    function_map: Optional[Dict[str, Callable]] = None
) -> UserProxyAgent:
    """
    Create a UserProxyAgent from configuration file.
    
    Args:
        agent_id: Agent identifier (e.g., "planner_proxy", "developer_proxy")
        function_map: Optional dictionary mapping function names to implementations
        
    Returns:
        Configured UserProxyAgent instance
        
    Raises:
        ValueError: If agent type is not 'user_proxy'
    """
    config = load_agent_config(agent_id)
    
    if config['type'] != 'user_proxy':
        raise ValueError(
            f"Agent '{agent_id}' is type '{config['type']}', expected 'user_proxy'"
        )
    
    # Get proxy configuration
    proxy_config = config.get('proxy_config', {})
    
    # Prepare UserProxyAgent parameters
    params = {
        'name': config.get('name', agent_id),
        'description': config.get('description', ''),
        'human_input_mode': proxy_config.get('human_input_mode', 'NEVER'),
        'max_consecutive_auto_reply': proxy_config.get('max_consecutive_auto_reply', 0),
        'code_execution_config': proxy_config.get('code_execution_config', False),
    }
    
    # Add function_map if provided
    if function_map:
        params['function_map'] = function_map
    
    return UserProxyAgent(**params, is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),)


def load_all_agents(
    config_list: Dict[str, Any],
    function_map: Optional[Dict[str, Callable]] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load all available agents at once.
    
    Args:
        config_list: LLM configuration for assistant agents
        function_map: Optional function map for user proxy agents
        api_key: Optional API key
        
    Returns:
        Dictionary mapping agent_id to agent instance
    """
    agents = {}
    
    for agent_id in list_available_agents():
        try:
            config = load_agent_config(agent_id)
            
            if config['type'] == 'assistant':
                agents[agent_id] = create_assistant_agent(agent_id)
            elif config['type'] == 'user_proxy':
                agents[agent_id] = create_user_proxy_agent(agent_id, function_map)
            
        except Exception as e:
            print(f"Warning: Failed to load agent '{agent_id}': {e}")
    
    return agents


# Export main functions
__all__ = [
    'load_agent_config',
    'list_available_agents',
    'create_assistant_agent',
    'create_user_proxy_agent',
    'load_all_agents',
]

# Made with Bob
