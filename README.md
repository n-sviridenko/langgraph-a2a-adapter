# LangGraph A2A Proxy

This project implements an [A2A Protocol](https://google.github.io/A2A/) compliant FastAPI server that proxies requests to the LangGraph Server API.

## Overview

The LangGraph A2A Proxy allows A2A-compatible clients to interact with LangGraph assistants, providing:

1. Agent discovery through agent cards
2. Message exchange with assistants
3. Task management
4. Streaming responses
5. Push notifications for task updates

The proxy translates between the A2A protocol format and the LangGraph API format, allowing seamless integration.

## Architecture

The system consists of several key components:

1. **FastAPI Application** - Exposes A2A-compliant endpoints and handles JSON-RPC requests
2. **A2A Protocol Models** - Pydantic models representing A2A protocol structures
3. **LangGraph Client Wrapper** - Handles communication with the LangGraph API
4. **Agent Card Utilities** - Converts LangGraph assistants to A2A agent cards

## API Endpoints

### A2A Standard Endpoints

- `GET /.well-known/agent.json` - Returns the agent card for the default agent (A2A standard discovery endpoint)
- `POST /rpc` - Main JSON-RPC endpoint for A2A protocol requests
- `WebSocket /ws` - WebSocket endpoint for streaming A2A responses

#### Supported RPC Methods

- `tasks/send` - Send a message to an agent
- `tasks/get` - Get status of a task
- `tasks/cancel` - Cancel a running task
- `tasks/pushNotification/set` - Configure push notifications for a task
- `tasks/pushNotification/get` - Get push notification configuration for a task
- `tasks/sendSubscribe` (via WebSocket) - Send a message with streaming response
- `tasks/resubscribe` (via WebSocket) - Reconnect to an existing task stream

## Setup

### Prerequisites

- Python 3.9+
- A running LangGraph Server
- Poetry (optional, for dependency management)

### Installation

#### Using Poetry (Recommended)

1. Clone the repository
2. Install Poetry if you haven't already:
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Install dependencies:
   ```
   poetry install
   ```
4. Activate the virtual environment (Poetry 2.0.0+):
   ```
   # Option 1 (recommended)
   poetry env use python
   poetry env activate
   
   # Option 2: Install the shell plugin
   poetry plugin add poetry-plugin-shell
   poetry shell
   
   # Option 3: Activate directly
   source $(poetry env info --path)/bin/activate  # Unix/macOS
   ```

#### Using pip (Alternative)

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # OR
   .venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```
   pip install -e .
   ```

### Configuration

All configuration settings are centralized in `config.py` and loaded from environment variables with sensible defaults.

You can create a `.env` file in the project root based on the provided `.env.example`:

```
cp .env.example .env
# Then edit .env with your specific values
```

Set the following environment variables:

#### LangGraph Connection
- `LANGGRAPH_API_URL` - URL of the LangGraph Server (default: `http://localhost:2024`)
- `LANGGRAPH_API_KEY` - API key for LangGraph Server (if required)
- `LANGGRAPH_GRAPH_ID` - Optional graph ID to select a specific assistant (if not set, uses the first available assistant)

#### A2A Server Configuration
- `A2A_HOST` - Hostname for the A2A server (default: `localhost`)
- `A2A_PORT` - Port for the A2A server (default: `8000`)
- `A2A_PROTOCOL` - Protocol for the A2A server (default: `http`)
- `A2A_TASKS_SEND_WAIT_FOR_COMPLETION` - Whether to wait for task completion when handling `tasks/send` requests (default: `true`)
- `A2A_PUSH_NOTIFICATION_TIMEOUT` - Timeout in seconds for webhook forwarding (default: `10.0`)

#### Agent Card Configuration
- `AGENT_NAME` - Name of the agent (default: assistant name or "LangGraph Assistant")
- `AGENT_DESCRIPTION` - Description of the agent (default: assistant description or generic text)
- `AGENT_VERSION` - Version of the agent (default: assistant version or "1.0.0")
- `AGENT_DOCUMENTATION_URL` - URL to the agent's documentation (optional)
- `AGENT_PROVIDER_ORG` - Organization name for the agent provider (optional)
- `AGENT_PROVIDER_URL` - URL for the agent provider (optional)
- `AGENT_SKILLS` - JSON array of skills objects with format: `[{"id": "skill_id", "name": "Skill Name", "description": "Skill description", "examples": ["example1", "example2"]}]` (default: basic chat skill)

### Running the Server

#### Using Poetry

```
# If environment is activated
python main.py

# Or without activating
poetry run python main.py
```

#### Using standard Python

```
python main.py
```

This will start the server on `http://localhost:8000`.

## Mapping Details

### LangGraph to A2A Mapping

- LangGraph Assistant → A2A Agent Card
- LangGraph Thread → A2A Session
- LangGraph Run → A2A Task
- LangGraph Output → A2A Artifacts and Messages

### Status Mapping

| LangGraph Run Status | A2A Task State |
|----------------------|----------------|
| pending              | working        |
| error                | failed         |
| success              | completed      |
| timeout              | failed         |
| interrupted          | input-required |

## Extending the Proxy

### Adding New RPC Methods

1. Define the new request type in `a2a_models.py`
2. Add a handler function in `main.py`
3. Update the routing in the `handle_rpc_request` function

### Supporting New LangGraph Features

To support new LangGraph features, update the `LangGraphClientWrapper` class in `langgraph_client.py` with new methods that map between A2A and LangGraph formats.

## License

[MIT License](LICENSE)