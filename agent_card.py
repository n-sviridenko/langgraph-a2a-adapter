"""
Utilities for retrieving and exposing A2A agent cards.
"""

import json
from typing import Dict, List, Optional, Any, Union

from a2a_models import (
    AgentCard, AgentCapabilities, AgentProvider, AgentSkill, AgentAuthentication
)
from langgraph_sdk.schema import Assistant
from config import (
    AGENT_NAME, AGENT_DESCRIPTION, AGENT_VERSION,
    AGENT_DOCUMENTATION_URL, AGENT_PROVIDER_ORG, 
    AGENT_PROVIDER_URL, get_base_url, AGENT_SKILLS_JSON
)
from pydantic import ValidationError


# Default skills to use if AGENT_SKILLS is not set or invalid
DEFAULT_SKILLS = [
    {
        "id": "chat",
        "name": "Chat",
        "description": "General conversation ability",
        "examples": ["Hello, how are you?", "Tell me about yourself"]
    }
]


def get_skills() -> List[AgentSkill]:
    """
    Parse and validate agent skills from the AGENT_SKILLS_JSON environment variable.
    
    Returns:
        A list of validated AgentSkill objects
    """
    if not AGENT_SKILLS_JSON:
        # No skills JSON provided, use defaults
        return [AgentSkill.model_validate(skill) for skill in DEFAULT_SKILLS]
    
    try:
        # Try to parse the JSON string
        skills_data = json.loads(AGENT_SKILLS_JSON)
        
        # Validate the parsed data using Pydantic
        if not isinstance(skills_data, list):
            print(f"Warning: AGENT_SKILLS must be a JSON array. Using default skills.")
            return [AgentSkill.model_validate(skill) for skill in DEFAULT_SKILLS]
        
        # Convert each skill dict to an AgentSkill model with validation
        validated_skills = []
        for skill_data in skills_data:
            try:
                skill = AgentSkill.model_validate(skill_data)
                validated_skills.append(skill)
            except ValidationError as e:
                print(f"Warning: Skill validation error: {e}. Skipping this skill.")
        
        # If no valid skills were found, use defaults
        if not validated_skills:
            print(f"Warning: No valid skills found in AGENT_SKILLS. Using default skills.")
            return [AgentSkill.model_validate(skill) for skill in DEFAULT_SKILLS]
            
        return validated_skills
        
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse AGENT_SKILLS as JSON: {e}. Using default skills.")
        return [AgentSkill.model_validate(skill) for skill in DEFAULT_SKILLS]


def get_hardcoded_capabilities() -> AgentCapabilities:
    """
    Get hardcoded agent capabilities based on current implementation.
    """
    # Based on the current implementation, supporting streaming, 
    # push notifications, and state transitions
    return AgentCapabilities(
        streaming=True,
        pushNotifications=True,
        stateTransitionHistory=True
    )


def get_hardcoded_authentication() -> Optional[AgentAuthentication]:
    """
    Get hardcoded authentication schemes based on current implementation.
    """
    # Currently no authentication is required for the agent
    return None


def get_agent_card_from_env(
    assistant: Assistant, 
    base_url: str
) -> AgentCard:
    """
    Create an A2A AgentCard using environment variables for properties.
    
    Args:
        assistant: The LangGraph Assistant object (used only for ID)
        base_url: The base URL where the A2A server is running
        
    Returns:
        An A2A AgentCard
    """
    # Extract properties from configuration with defaults
    name = AGENT_NAME or assistant["name"] or "LangGraph Assistant"
    description = AGENT_DESCRIPTION or assistant.get("description") or "An AI assistant powered by LangGraph"
    version = AGENT_VERSION or str(assistant.get("version") or "1.0.0")
    documentation_url = AGENT_DOCUMENTATION_URL
    
    # Provider information (optional)
    provider = None
    if AGENT_PROVIDER_ORG:
        provider = AgentProvider(
            organization=AGENT_PROVIDER_ORG,
            url=AGENT_PROVIDER_URL
        )
    
    # Input/output modes - currently only text is fully supported
    default_input_modes = ["text"]
    default_output_modes = ["text"]
    
    # URL for the agent - point to the RPC endpoint
    agent_url = f"{base_url}/rpc"
    
    # Get hardcoded capabilities and authentication
    capabilities = get_hardcoded_capabilities()
    authentication = get_hardcoded_authentication()
    
    # Get skills
    skills = get_skills()
    
    # Create the agent card
    agent_card = AgentCard(
        name=name,
        description=description,
        url=agent_url,
        provider=provider,
        version=version,
        documentationUrl=documentation_url,
        capabilities=capabilities,
        authentication=authentication,
        defaultInputModes=default_input_modes,
        defaultOutputModes=default_output_modes,
        skills=skills
    )
    
    return agent_card 