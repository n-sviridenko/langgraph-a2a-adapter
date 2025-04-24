"""
Utilities for retrieving and exposing A2A agent cards.
"""

from typing import Dict, List, Optional, Any, Union

from a2a_models import (
    AgentCard, AgentCapabilities, AgentProvider, AgentSkill, AgentAuthentication
)
from langgraph_sdk.schema import Assistant
from config import (
    AGENT_NAME, AGENT_DESCRIPTION, AGENT_VERSION,
    AGENT_DOCUMENTATION_URL, AGENT_PROVIDER_ORG, 
    AGENT_PROVIDER_URL, get_base_url
)


def get_default_skills() -> List[AgentSkill]:
    """
    Get default agent skills.
    """
    return [
        AgentSkill(
            id="chat",
            name="Chat",
            description="General conversation ability",
            examples=["Hello, how are you?", "Tell me about yourself"]
        )
    ]


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
    name = AGENT_NAME or assistant.name or "LangGraph Assistant"
    description = AGENT_DESCRIPTION or assistant.description or "An AI assistant powered by LangGraph"
    version = AGENT_VERSION or str(assistant.version or "1.0.0")
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
    
    # URL for the agent
    agent_url = f"{base_url}/.well-known/agent.json"
    
    # Get hardcoded capabilities and authentication
    capabilities = get_hardcoded_capabilities()
    authentication = get_hardcoded_authentication()
    
    # Get default skills
    skills = get_default_skills()
    
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