"""
A2A-compliant FastAPI server that proxies to LangGraph Server API
"""

import os
import json
import uuid
import logging
import io
import asyncio
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Tuple, Awaitable
from datetime import datetime

import httpx
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, BackgroundTasks, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langgraph_sdk import get_client
from langgraph_sdk.schema import RunStatus

from a2a_models import (
    JSONRPCRequest, JSONRPCResponse, JSONRPCError,
    SendTaskRequest, GetTaskRequest, CancelTaskRequest,
    SendTaskStreamingRequest, TaskResubscriptionRequest,
    SetTaskPushNotificationRequest, GetTaskPushNotificationRequest,
    Task, TaskState, TaskStatus, Message, Artifact,
    TextPart, TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
    PushNotificationConfig, AgentCard
)
from langgraph_client import LangGraphClientWrapper
from agent_card import get_agent_card_from_env
from config import (
    LANGGRAPH_API_URL, LANGGRAPH_API_KEY, LANGGRAPH_GRAPH_ID,
    A2A_PUSH_NOTIFICATION_TIMEOUT, get_base_url, A2A_PORT
)

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Initialize connections and resources on startup
    global client_wrapper
    client_wrapper = LangGraphClientWrapper(
        api_url=LANGGRAPH_API_URL,
        api_key=LANGGRAPH_API_KEY
    )
    yield
    # Cleanup code would go here if needed

app = FastAPI(
    title="LangGraph A2A Adapter",
    description="A2A-compliant FastAPI server that uses LangGraph SDK",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LangGraph client wrapper
client_wrapper = None

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define errors
JSONRPC_PARSE_ERROR = {"code": -32700, "message": "Invalid JSON payload"}
JSONRPC_INVALID_REQUEST = {"code": -32600, "message": "Request payload validation error"}
JSONRPC_METHOD_NOT_FOUND = {"code": -32601, "message": "Method not found"}
JSONRPC_INVALID_PARAMS = {"code": -32602, "message": "Invalid parameters"}
JSONRPC_INTERNAL_ERROR = {"code": -32603, "message": "Internal error"}
TASK_NOT_FOUND_ERROR = {"code": -32001, "message": "Task not found"}
TASK_NOT_CANCELABLE_ERROR = {"code": -32002, "message": "Task cannot be canceled"}
PUSH_NOTIFICATION_NOT_SUPPORTED_ERROR = {"code": -32003, "message": "Push Notification is not supported"}
UNSUPPORTED_OPERATION_ERROR = {"code": -32004, "message": "This operation is not supported"}

# Define additional errors
SSE_METHOD_HTTP_ERROR = {
    "code": -32010, 
    "message": "Streaming method attempted via standard HTTP. For streaming, use SSE with content-type: text/event-stream"
}


def is_final_status(state: TaskState) -> bool:
    """Check if a task state represents a final state"""
    return state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]


def format_sse_error(request_id: str, error_code: dict, message: str = None) -> str:
    """Format error responses for SSE
    
    Args:
        request_id: The JSON-RPC request ID
        error_code: The error code dictionary (e.g. JSONRPC_INTERNAL_ERROR)
        message: Optional error message
        
    Returns:
        Formatted JSON string for error response
    """
    error_data = error_code.copy()
    if message:
        error_data["data"] = {"message": message}
    
    error_response = JSONRPCResponse(
        jsonrpc="2.0",
        id=request_id,
        error=JSONRPCError(**error_data)
    )
    return error_response.json()


def format_sse_update(request_id: str, task_id: str, update_content, final: bool = False) -> str:
    """Helper to format updates as JSON-RPC responses for SSE streaming
    
    Args:
        request_id: The JSON-RPC request ID
        task_id: The task ID
        update_content: Either a TaskStatus or Artifact object
        final: Whether this is the final update in the stream
        
    Returns:
        A JSON string for the SSE response
    """
    if isinstance(update_content, TaskStatus):
        # Create status update event
        event = TaskStatusUpdateEvent(
            id=task_id,
            status=update_content,
            final=final
        )
    elif isinstance(update_content, Artifact):
        # Create artifact update event
        event = TaskArtifactUpdateEvent(
            id=task_id,
            artifact=update_content
        )
    else:
        # Directly use the content if it's already an event
        event = update_content
    
    response = JSONRPCResponse(
        jsonrpc="2.0",
        id=request_id,
        result=event
    )
    
    return response.json()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "LangGraph A2A Adapter is running"}


# A2A WELL-KNOWN ENDPOINT - Standard endpoint as specified by A2A protocol
@app.get("/.well-known/agent.json")
async def get_default_agent():
    """
    Return the agent card for the default agent as per A2A protocol specification.
    This is the recommended well-known path for A2A agent discovery.
    """
    try:
        # Get assistant ID using the configured LANGGRAPH_GRAPH_ID or the first available
        assistant_id = await get_assistant_id()
        
        # Get the assistant
        assistant = await client_wrapper.get_assistant(assistant_id)
        
        base_url = get_base_url()
        
        # Convert to agent card using environment variables instead of metadata
        agent_card = get_agent_card_from_env(assistant, base_url)
        
        return agent_card
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving default agent: {str(e)}")


async def get_assistant_id() -> str:
    """Get the assistant ID to use for requests"""
    assistants = await client_wrapper.get_assistants()
    
    if not assistants:
        raise ValueError("No assistants available")
    
    # If LANGGRAPH_GRAPH_ID is set, try to find a matching assistant
    if LANGGRAPH_GRAPH_ID:
        for assistant in assistants:
            if assistant["graph_id"] == LANGGRAPH_GRAPH_ID:
                return assistant["assistant_id"]
                
        # If no match is found, log a warning
        print(f"Warning: No assistant found with graph_id {LANGGRAPH_GRAPH_ID}. Using first available assistant.")
        
    # Return the first assistant if no match or no filter
    return assistants[0]["assistant_id"]


@app.post("/rpc")
async def handle_rpc_request(request: Request):
    """Handle JSON-RPC requests according to A2A protocol"""
    # Check if this is an SSE request
    is_sse_request = request.headers.get("accept") == "text/event-stream"
    
    try:
        # Parse request body
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8')
        logger.info(f"Received RPC request: {body_str}")
        
        try:
            body = json.loads(body_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=None,
                error=JSONRPCError(**JSONRPC_PARSE_ERROR)
            )
        
        # Basic validation for JSON-RPC 2.0
        if not isinstance(body, dict):
            logger.error(f"Invalid request format, not a dictionary: {type(body)}")
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=None,
                error=JSONRPCError(**JSONRPC_INVALID_REQUEST)
            )
            
        # Create a request object
        try:
            rpc_request = JSONRPCRequest(**body)
            logger.info(f"Parsed RPC request - method: {rpc_request.method}, id: {rpc_request.id}")
        except Exception as e:
            logger.error(f"Failed to create RPC request object: {str(e)}")
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=body.get("id"),
                error=JSONRPCError(**JSONRPC_INVALID_REQUEST)
            )
        
        # Check if this is a streaming method 
        is_streaming_method = rpc_request.method in ["tasks/sendSubscribe", "tasks/resubscribe"]
        
        # Handle streaming methods with SSE
        if is_streaming_method:
            if not is_sse_request:
                # Client is trying to use a streaming method without SSE
                logger.warning(f"Streaming method {rpc_request.method} attempted via standard HTTP without SSE")
                return JSONRPCResponse(
                    jsonrpc="2.0",
                    id=rpc_request.id,
                    error=JSONRPCError(**SSE_METHOD_HTTP_ERROR)
                )
                
            # Set up SSE streaming response
            return StreamingResponse(
                handle_streaming_request(rpc_request, body),
                media_type="text/event-stream"
            )
            
        # Standard HTTP methods
        try:
            if rpc_request.method == "tasks/send":
                logger.info(f"Handling tasks/send request")
                return await handle_send_task(SendTaskRequest(**body))
            elif rpc_request.method == "tasks/get":
                logger.info(f"Handling tasks/get request")
                return await handle_get_task(GetTaskRequest(**body))
            elif rpc_request.method == "tasks/cancel":
                logger.info(f"Handling tasks/cancel request")
                return await handle_cancel_task(CancelTaskRequest(**body))
            elif rpc_request.method == "tasks/pushNotification/set":
                logger.info(f"Handling tasks/pushNotification/set request")
                return await handle_set_push_notification(SetTaskPushNotificationRequest(**body))
            elif rpc_request.method == "tasks/pushNotification/get":
                logger.info(f"Handling tasks/pushNotification/get request")
                return await handle_get_push_notification(GetTaskPushNotificationRequest(**body))
            else:
                logger.warning(f"Unknown method: {rpc_request.method}")
                return JSONRPCResponse(
                    jsonrpc="2.0",
                    id=rpc_request.id,
                    error=JSONRPCError(**JSONRPC_METHOD_NOT_FOUND)
                )
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}", exc_info=True)
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=rpc_request.id,
                error=JSONRPCError(**JSONRPC_INTERNAL_ERROR)
            )
    except Exception as e:
        logger.error(f"Unexpected error in RPC handler: {str(e)}", exc_info=True)
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=None,
            error=JSONRPCError(**JSONRPC_INTERNAL_ERROR)
        )


@app.post("/webhook-relay/{task_id}")
async def webhook_relay(task_id: str, request: Request):
    """Relay webhook notifications to the actual webhook URL stored in run metadata."""
    try:
        # Get the original request body
        body = await request.body()
        
        # Get headers (except host-specific ones)
        headers = {k: v for k, v in request.headers.items() 
                  if k.lower() not in ['host', 'content-length']}
        
        # Get the actual webhook URL from run metadata
        # Now _a2a_webhook_url directly contains the real webhook URL
        webhook_url = await client_wrapper.get_webhook_url_for_task(task_id)
        
        if webhook_url:
            # Forward the request to the actual webhook URL (await directly)
            result = await forward_webhook(
                webhook_url=webhook_url,
                body=body,
                headers=headers
            )
            return result
        else:
            return {"status": "warning", "message": "No webhook URL found for this task"}
    except Exception as e:
        print(f"Error processing webhook relay: {str(e)}")
        return {"status": "error", "message": str(e)}

async def forward_webhook(webhook_url: str, body: bytes, headers: Dict[str, str]):
    """Forward webhook payload to the actual webhook URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                content=body,
                headers=headers,
                timeout=A2A_PUSH_NOTIFICATION_TIMEOUT
            )
            print(f"Webhook forwarded to {webhook_url}, status: {response.status_code}")
            
            # Return the response status and details for LangGraph to know the result
            return {
                "status": "ok" if response.status_code < 400 else "error",
                "status_code": response.status_code,
                "message": f"Webhook forwarded with status {response.status_code}"
            }
    except Exception as e:
        print(f"Error forwarding webhook to {webhook_url}: {str(e)}")
        # Raise the exception to inform LangGraph of the failure
        raise


async def handle_send_task(request: SendTaskRequest) -> JSONRPCResponse:
    """Handle tasks/send method"""
    try:
        # Extract parameters
        task_id = request.params.id
        session_id = request.params.sessionId
        message = request.params.message
        
        # Check if session exists, if not create it
        try:
            await client_wrapper.get_session(session_id)
        except Exception:
            await client_wrapper.create_session(session_id)
        
        # Get assistant ID
        try:
            assistant_id = await get_assistant_id()
        except ValueError as e:
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
            )
        
        # Process webhook if provided
        webhook_url = None
        if request.params.pushNotification:
            push_config = request.params.pushNotification
            webhook_url = push_config.url
        
        # Send the message with webhook if provided
        task = await client_wrapper.send_message(
            task_id=task_id,
            session_id=session_id,
            assistant_id=assistant_id,
            message=message,
            webhook=webhook_url
        )
        
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            result=task
        )
        
    except Exception as e:
        print(f"Error in handle_send_task: {str(e)}")
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
        )


async def handle_get_task(request: GetTaskRequest) -> JSONRPCResponse:
    """Handle tasks/get method"""
    try:
        # Extract parameters
        task_id = request.params.id
        history_length = request.params.historyLength
        
        # Get task
        try:
            task = await client_wrapper.get_task(task_id, history_length)
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                result=task
            )
        except ValueError:
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error=JSONRPCError(**TASK_NOT_FOUND_ERROR)
            )
            
    except Exception as e:
        print(f"Error in handle_get_task: {str(e)}")
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
        )


async def handle_cancel_task(request: CancelTaskRequest) -> JSONRPCResponse:
    """Handle tasks/cancel method"""
    try:
        # Extract parameters
        task_id = request.params.id
        
        # Cancel task
        try:
            task = await client_wrapper.cancel_task(task_id)
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                result=task
            )
        except ValueError:
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error=JSONRPCError(**TASK_NOT_FOUND_ERROR)
            )
        except Exception:
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error=JSONRPCError(**TASK_NOT_CANCELABLE_ERROR)
            )
            
    except Exception as e:
        print(f"Error in handle_cancel_task: {str(e)}")
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
        )


async def handle_set_push_notification(request: SetTaskPushNotificationRequest) -> JSONRPCResponse:
    """Handle tasks/pushNotification/set method"""
    try:
        # Extract parameters
        task_id = request.params.id
        push_config = request.params.pushNotificationConfig
        
        # Get webhook URL from push config
        real_webhook_url = push_config.url
        
        # Update the webhook URL in run metadata
        success = await client_wrapper.update_webhook_url_for_task(task_id, real_webhook_url)
        
        if not success:
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error=JSONRPCError(**TASK_NOT_FOUND_ERROR)
            )
        
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            result=request.params
        )
            
    except Exception as e:
        print(f"Error in handle_set_push_notification: {str(e)}")
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
        )


async def handle_get_push_notification(request: GetTaskPushNotificationRequest) -> JSONRPCResponse:
    """Handle tasks/pushNotification/get method"""
    try:
        # Extract parameters
        task_id = request.params.id
        
        # Try to get webhook URL from run metadata (will get the real webhook URL)
        webhook_url = await client_wrapper.get_webhook_url_for_task(task_id)
        
        # If no webhook URL found, return error
        if webhook_url is None:
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error=JSONRPCError(**PUSH_NOTIFICATION_NOT_SUPPORTED_ERROR)
            )
        
        # Create a push notification config from the stored URL
        push_config = PushNotificationConfig(
            url=webhook_url,
            authentication=None,
            token=None
        )
        
        config = {
            "id": task_id,
            "pushNotificationConfig": push_config
        }
        
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            result=config
        )
            
    except Exception as e:
        print(f"Error in handle_get_push_notification: {str(e)}")
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
        )


async def handle_streaming_request(rpc_request: JSONRPCRequest, body: dict) -> AsyncIterator[str]:
    """Handle streaming requests (tasks/sendSubscribe and tasks/resubscribe) with SSE"""
    try:
        if rpc_request.method == "tasks/sendSubscribe":
            logger.info(f"Handling SSE tasks/sendSubscribe request")
            async for event in handle_streaming_task(SendTaskStreamingRequest(**body)):
                yield f"data: {event}\n\n"
        elif rpc_request.method == "tasks/resubscribe":
            logger.info(f"Handling SSE tasks/resubscribe request")
            async for event in handle_resubscribe_task(TaskResubscriptionRequest(**body)):
                yield f"data: {event}\n\n"
    except Exception as e:
        logger.error(f"Error in streaming handler: {str(e)}", exc_info=True)
        yield f"data: {format_sse_error(rpc_request.id, JSONRPC_INTERNAL_ERROR, str(e))}\n\n"


async def handle_streaming_task(request: SendTaskStreamingRequest) -> AsyncIterator[str]:
    """Handle tasks/sendSubscribe method with SSE streaming"""
    try:
        # Extract parameters
        task_id = request.params.id
        session_id = request.params.sessionId
        message = request.params.message
        
        logger.info(f"Processing streaming task - task_id: {task_id}, session_id: {session_id}")
        
        # Check if session exists, if not create it
        try:
            await client_wrapper.get_session(session_id)
            logger.info(f"Found existing session for streaming: {session_id}")
        except Exception:
            logger.info(f"Creating new session for streaming: {session_id}")
            await client_wrapper.create_session(session_id)
        
        # Get assistant ID
        try:
            assistant_id = await get_assistant_id()
            logger.info(f"Using assistant_id for streaming: {assistant_id}")
        except ValueError as e:
            logger.error(f"Failed to get assistant ID for streaming: {str(e)}")
            yield format_sse_error(request.id, JSONRPC_INTERNAL_ERROR, str(e))
            return
        
        # Process webhook if provided
        webhook_url = None
        if request.params.pushNotification:
            push_config = request.params.pushNotification
            webhook_url = push_config.url
            logger.info(f"Using webhook URL for streaming: {webhook_url}")
        
        # Send initial status
        initial_status = TaskStatus(
            state=TaskState.SUBMITTED,
            timestamp=datetime.now()
        )
        
        # Use the helper function to format the initial status
        yield format_sse_update(request.id, task_id, initial_status, False)
        
        # Start streaming using the simplified send_message method
        # Await the coroutine to get the async iterator
        message_stream = await client_wrapper.send_message(
            task_id=task_id,
            session_id=session_id,
            assistant_id=assistant_id,
            message=message,
            stream=True,
            webhook=webhook_url
        )
        
        # Now iterate over the async iterator
        async for update in message_stream:
            # Convert update to A2A event format using the helper
            if isinstance(update, TaskStatus):
                # Use the helper to check if this is a final status
                final = is_final_status(update.state)
                
                # Use the helper function to format the status update
                yield format_sse_update(request.id, task_id, update, final)
                
                # Break the stream if this is a final status
                if final:
                    break
            elif isinstance(update, Artifact):
                # Use the helper function to format the artifact update
                yield format_sse_update(request.id, task_id, update)
        
    except Exception as e:
        logger.error(f"Error in handle_streaming_task: {str(e)}", exc_info=True)
        yield format_sse_error(request.id, JSONRPC_INTERNAL_ERROR, str(e))


async def handle_resubscribe_task(request: TaskResubscriptionRequest) -> AsyncIterator[str]:
    """Handle tasks/resubscribe method with SSE streaming"""
    try:
        # Extract parameters
        task_id = request.params.id
        
        logger.info(f"Processing resubscribe request for task: {task_id}")
        
        # Try to find the thread and run
        try:
            # Find the thread and run using the task_id
            thread, run_id = await client_wrapper._find_thread_by_task_id(task_id)
            
            # Get the current run state
            run_info = await client_wrapper.client.runs.get(thread["thread_id"], run_id)
            thread_state = await client_wrapper.client.threads.get_state(thread["thread_id"])
            
            # Send the current task status
            task_state = client_wrapper._run_status_to_task_state(run_info["status"])
            current_status = TaskStatus(
                state=task_state,
                timestamp=run_info["updated_at"]
            )
            
            # Use the helper to check if this is a final status
            final = is_final_status(task_state)
            
            # Use the helper function to format the current status
            yield format_sse_update(request.id, task_id, current_status, final)
            
            # Check if the run is still active (not completed, failed, etc.)
            # Statuses like "pending", "running", "interrupted" would qualify
            if run_info["status"] not in ["success", "error", "timeout", "canceled"]:
                # Join the existing run using the join method
                # Don't await the join_stream call - it returns an async generator, not an awaitable
                stream = client_wrapper.client.runs.join_stream(
                    thread_id=thread["thread_id"],
                    run_id=run_id,
                    stream_mode=["values", "messages"]
                )
                
                async for chunk in stream:
                    # Use the shared helper method to process the chunk
                    update = client_wrapper._process_stream_chunk(chunk, task_id)
                    
                    if isinstance(update, TaskStatus):
                        # Use the helper to check if this is a final status
                        final = is_final_status(update.state)
                        
                        # Use the helper function to format the status update
                        yield format_sse_update(request.id, task_id, update, final)
                        
                        # Break if final status
                        if final:
                            break
                    elif isinstance(update, Artifact):
                        # Use the helper function to format the artifact update
                        yield format_sse_update(request.id, task_id, update)
            
        except ValueError:
            # Task not found
            yield format_sse_error(request.id, TASK_NOT_FOUND_ERROR)
            
    except Exception as e:
        logger.error(f"Error in handle_resubscribe_task: {str(e)}", exc_info=True)
        yield format_sse_error(request.id, JSONRPC_INTERNAL_ERROR, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(A2A_PORT), reload=True) 