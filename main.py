"""
A2A-compliant FastAPI server that proxies to LangGraph Server API
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Tuple, Awaitable
from datetime import datetime

import httpx
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, BackgroundTasks, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

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
    A2A_PUSH_NOTIFICATION_TIMEOUT, get_base_url
)

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph A2A Proxy",
    description="A2A-compliant FastAPI server that uses LangGraph SDK"
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

# Active WebSocket connections for streaming responses
websocket_connections: Dict[str, WebSocket] = {}

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


async def send_websocket_update(
    websocket: WebSocket, 
    task_id: str, 
    request_id: str, 
    update: Union[TaskStatus, Artifact]
) -> bool:
    """
    Send an update over WebSocket based on its type (TaskStatus or Artifact).
    Returns True if the connection should be closed (final status received).
    
    Parameters:
    - websocket: The WebSocket connection to send to
    - task_id: The task ID associated with the update
    - request_id: The JSON-RPC request ID for the response
    - update: Either a TaskStatus or Artifact update to send
    
    Returns:
    - should_close: True if this is a final status and the connection should close
    """
    if isinstance(update, TaskStatus):
        # Determine if this is a final status
        is_final = update.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]
        
        # Create and send status update event
        status_event = TaskStatusUpdateEvent(
            id=task_id,
            status=update,
            final=is_final
        )
        
        status_response = JSONRPCResponse(
            jsonrpc="2.0",
            id=request_id,
            result=status_event
        )
        
        await websocket.send_text(status_response.json())
        
        # Return True if this is a final status (to break the loop)
        return is_final
        
    elif isinstance(update, Artifact):
        # Create and send artifact update event
        artifact_event = TaskArtifactUpdateEvent(
            id=task_id,
            artifact=update
        )
        
        artifact_response = JSONRPCResponse(
            jsonrpc="2.0",
            id=request_id,
            result=artifact_event
        )
        
        await websocket.send_text(artifact_response.json())
        
        # Artifacts don't signal end of communication
        return False
    
    # Unknown update type
    return False


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "LangGraph A2A Proxy is running"}


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
            if assistant.graph_id == LANGGRAPH_GRAPH_ID:
                return assistant.assistant_id
                
        # If no match is found, log a warning
        print(f"Warning: No assistant found with graph_id {LANGGRAPH_GRAPH_ID}. Using first available assistant.")
        
    # Return the first assistant if no match or no filter
    return assistants[0].assistant_id


@app.on_event("startup")
async def startup_event():
    """Initialize connections and resources on startup"""
    global client_wrapper
    client_wrapper = LangGraphClientWrapper(
        api_url=LANGGRAPH_API_URL,
        api_key=LANGGRAPH_API_KEY
    )


@app.post("/rpc")
async def handle_rpc_request(request: Request):
    """Handle JSON-RPC requests according to A2A protocol"""
    try:
        # Parse request body
        body = await request.json()
        
        # Basic validation for JSON-RPC 2.0
        if not isinstance(body, dict):
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=None,
                error=JSONRPCError(**JSONRPC_INVALID_REQUEST)
            )
            
        # Create a request object
        rpc_request = JSONRPCRequest(**body)
        
        # Route to appropriate handler based on method
        try:
            if rpc_request.method == "tasks/send":
                return await handle_send_task(SendTaskRequest(**body))
            elif rpc_request.method == "tasks/get":
                return await handle_get_task(GetTaskRequest(**body))
            elif rpc_request.method == "tasks/cancel":
                return await handle_cancel_task(CancelTaskRequest(**body))
            elif rpc_request.method == "tasks/pushNotification/set":
                return await handle_set_push_notification(SetTaskPushNotificationRequest(**body))
            elif rpc_request.method == "tasks/pushNotification/get":
                return await handle_get_push_notification(GetTaskPushNotificationRequest(**body))
            else:
                return JSONRPCResponse(
                    jsonrpc="2.0",
                    id=rpc_request.id,
                    error=JSONRPCError(**JSONRPC_METHOD_NOT_FOUND)
                )
        except Exception as e:
            print(f"Error handling request: {str(e)}")
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=rpc_request.id,
                error=JSONRPCError(**JSONRPC_INTERNAL_ERROR)
            )
            
    except json.JSONDecodeError:
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=None,
            error=JSONRPCError(**JSONRPC_PARSE_ERROR)
        )
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for streaming tasks"""
    await websocket.accept()
    
    try:
        # Get the initial message (should be a streaming request)
        data = await websocket.receive_text()
        request_data = json.loads(data)
        
        # Validate the request
        if "method" not in request_data or request_data["method"] not in ["tasks/sendSubscribe", "tasks/resubscribe"]:
            error_response = JSONRPCResponse(
                jsonrpc="2.0",
                id=request_data.get("id"),
                error=JSONRPCError(**JSONRPC_METHOD_NOT_FOUND)
            )
            await websocket.send_text(error_response.json())
            await websocket.close()
            return
            
        if request_data["method"] == "tasks/sendSubscribe":
            await handle_streaming_task(websocket, SendTaskStreamingRequest(**request_data))
        elif request_data["method"] == "tasks/resubscribe":
            await handle_resubscribe_task(websocket, TaskResubscriptionRequest(**request_data))
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in websocket handler: {str(e)}")
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                error_response = JSONRPCResponse(
                    jsonrpc="2.0",
                    id=None,
                    error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
                )
                await websocket.send_text(error_response.json())
                await websocket.close()
        except Exception:
            pass


async def handle_streaming_task(websocket: WebSocket, request: SendTaskStreamingRequest):
    """Handle tasks/sendSubscribe method"""
    try:
        # Extract parameters
        task_id = request.params.id
        session_id = request.params.sessionId
        message = request.params.message
        
        # Store the websocket connection
        websocket_connections[task_id] = websocket
        
        # Check if session exists, if not create it
        try:
            await client_wrapper.get_session(session_id)
        except Exception:
            await client_wrapper.create_session(session_id)
        
        # Get assistant ID
        try:
            assistant_id = await get_assistant_id()
        except ValueError as e:
            error_response = JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
            )
            await websocket.send_text(error_response.json())
            await websocket.close()
            return
        
        # Send initial status
        initial_status = TaskStatus(
            state=TaskState.SUBMITTED,
            timestamp=datetime.now()
        )
        
        # Send initial status using the helper
        await send_websocket_update(websocket, task_id, request.id, initial_status)
        
        # Process webhook if provided
        webhook_url = None
        if request.params.pushNotification:
            push_config = request.params.pushNotification
            webhook_url = push_config.url
        
        # Start streaming using the simplified send_message method
        async for update in client_wrapper.send_message(
            task_id=task_id,
            session_id=session_id,
            assistant_id=assistant_id,
            message=message,
            stream=True,
            webhook=webhook_url
        ):
            # Send the update and check if we should close
            should_close = await send_websocket_update(websocket, task_id, request.id, update)
            if should_close:
                break
        
        # Remove from connections
        if task_id in websocket_connections:
            del websocket_connections[task_id]
            
        # Close websocket
        await websocket.close()
        
    except Exception as e:
        print(f"Error in handle_streaming_task: {str(e)}")
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                error_response = JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
                )
                await websocket.send_text(error_response.json())
                await websocket.close()
        except Exception:
            pass


async def handle_resubscribe_task(websocket: WebSocket, request: TaskResubscriptionRequest):
    """Handle tasks/resubscribe method"""
    try:
        # Extract parameters
        task_id = request.params.id
        
        # Store the websocket connection
        websocket_connections[task_id] = websocket
        
        # Try to find the thread and run
        try:
            # Find the thread and run using the task_id
            thread, run_id = await client_wrapper._find_thread_by_task_id(task_id)
            
            # Get the current run state
            run_info = await client_wrapper.client.runs.get(thread.thread_id, run_id)
            thread_state = await client_wrapper.client.threads.get_state(thread.thread_id)
            
            # Send the current task status
            task_state = client_wrapper._run_status_to_task_state(run_info.status)
            current_status = TaskStatus(
                state=task_state,
                timestamp=run_info.updated_at
            )
            
            # Send current status using the helper
            await send_websocket_update(websocket, task_id, request.id, current_status)
            
            # Check if the run is still active (not completed, failed, etc.)
            # Statuses like "pending", "running", "interrupted" would qualify
            if run_info.status not in ["success", "error", "timeout", "canceled"]:
                # Join the existing run using the join method
                stream = await client_wrapper.client.runs.join(
                    thread_id=thread.thread_id,
                    run_id=run_id,
                    stream=True,
                    stream_mode=["values", "messages"]
                )
                
                async for chunk in stream:
                    # Use the shared helper method to process the chunk
                    update = client_wrapper._process_stream_chunk(chunk, task_id)
                    
                    # Send the update and check if we should close
                    should_close = await send_websocket_update(websocket, task_id, request.id, update)
                    if should_close:
                        break
            
            # Run is already complete or not streaming, close connection
            await websocket.close()
            
        except ValueError:
            # Task not found
            error_response = JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error=JSONRPCError(**TASK_NOT_FOUND_ERROR)
            )
            await websocket.send_text(error_response.json())
            await websocket.close()
            
    except Exception as e:
        print(f"Error in handle_resubscribe_task: {str(e)}")
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                error_response = JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=JSONRPCError(**JSONRPC_INTERNAL_ERROR, data={"message": str(e)})
                )
                await websocket.send_text(error_response.json())
                await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 