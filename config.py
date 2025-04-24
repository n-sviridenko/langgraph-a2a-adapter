"""
Configuration settings for the LangGraph A2A Adapter.
All settings are loaded from environment variables with sensible defaults.
"""

import os

# LangGraph Connection Settings
LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://localhost:2024")
LANGGRAPH_API_KEY = os.getenv("LANGGRAPH_API_KEY", None)
LANGGRAPH_GRAPH_ID = os.getenv("LANGGRAPH_GRAPH_ID", None)

# A2A Server Configuration
A2A_HOST = os.getenv("A2A_HOST", "localhost")
A2A_PORT = os.getenv("A2A_PORT", "8000")
A2A_PROTOCOL = os.getenv("A2A_PROTOCOL", "http")
A2A_TASKS_SEND_WAIT_FOR_COMPLETION = os.getenv("A2A_TASKS_SEND_WAIT_FOR_COMPLETION", "true").lower() == "true"
A2A_PUSH_NOTIFICATION_TIMEOUT = float(os.getenv("A2A_PUSH_NOTIFICATION_TIMEOUT", "10.0"))

# Agent Card Configuration
AGENT_NAME = os.getenv("AGENT_NAME", None)
AGENT_DESCRIPTION = os.getenv("AGENT_DESCRIPTION", None)
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.0.0")
AGENT_DOCUMENTATION_URL = os.getenv("AGENT_DOCUMENTATION_URL", None)
AGENT_PROVIDER_ORG = os.getenv("AGENT_PROVIDER_ORG", None)
AGENT_PROVIDER_URL = os.getenv("AGENT_PROVIDER_URL", None)
AGENT_SKILLS_JSON = os.getenv("AGENT_SKILLS", None)

def get_base_url() -> str:
    """
    Get the base URL for the A2A server.
    """
    return f"{A2A_PROTOCOL}://{A2A_HOST}:{A2A_PORT}" 