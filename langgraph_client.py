"""
Wrapper for the LangGraph client SDK.
"""

import asyncio
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncIterator
import uuid

from langgraph_sdk import get_client
from langgraph_sdk.schema import (
    RunStatus, ThreadStatus, Assistant, Thread, Run,
    ThreadState, StreamPart, StreamMode
)

from a2a_models import (
    Message, Part, TaskState, TaskStatus, Task, Artifact,
    TextPart, FilePart, DataPart
)
from config import get_base_url, A2A_TASKS_SEND_WAIT_FOR_COMPLETION


class LangGraphClientWrapper:
    """Wrapper for the LangGraph client that maps A2A protocol to LangGraph API."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """Initialize with API URL and optional API key."""
        self.api_url = api_url
        self.api_key = api_key
        self.client = get_client(url=api_url, api_key=api_key)
        
    async def get_assistants(self) -> List[Assistant]:
        """Get all available assistants."""
        return await self.client.assistants.search()
    
    async def get_assistant(self, assistant_id: str) -> Assistant:
        """Get a specific assistant."""
        return await self.client.assistants.get(assistant_id)
    
    async def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> Thread:
        """Create a new session (thread)."""
        thread_metadata = metadata or {}
        
        # Store A2A session ID in thread metadata
        thread_metadata["_a2a_session_id"] = session_id
        
        # Initialize empty mapping for run IDs to task IDs
        thread_metadata["_a2a_map_task_id_by_run_id"] = {}
        
        return await self.client.threads.create(
            thread_id=session_id,
            metadata=thread_metadata
        )
    
    async def get_session(self, session_id: str) -> Thread:
        """Get an existing session (thread)."""
        return await self.client.threads.get(session_id)
    
    async def _update_thread_task_metadata(self, thread_id: str, task_id: str, run_id: str) -> None:
        """Update thread metadata with task and run ID mappings."""
        # Get current thread
        thread = await self.client.threads.get(thread_id)
        thread_metadata = thread.get("metadata", {}) or {}
        
        # Add or update the mapping
        task_map = thread_metadata.get("_a2a_map_task_id_by_run_id", {})
        task_map[run_id] = task_id
        thread_metadata["_a2a_map_task_id_by_run_id"] = task_map
        
        # Add task ID to thread metadata directly for easy searching
        thread_metadata[f"_a2a_has_task_{task_id}"] = True
        
        # Update thread metadata
        await self.client.threads.update(
            thread_id=thread_id,
            metadata=thread_metadata
        )
    
    async def send_message(
        self, 
        task_id: str,
        session_id: str,
        assistant_id: str,
        message: Message,
        stream: bool = False,
        webhook: Optional[str] = None
    ) -> Union[Task, AsyncIterator[Union[TaskStatus, Artifact]]]:
        """Send a message to an assistant and get a response."""
        # Convert A2A message to LangGraph input format
        base_message = self._message_to_langgraph_input(message)
        
        # Create input dict with messages array
        input_dict = {"messages": [base_message]}
        
        # Prepare run metadata
        run_metadata = {"_a2a_task_id": task_id}
        
        # Store the real webhook URL in metadata if provided
        if webhook:
            run_metadata["_a2a_webhook_url"] = webhook
            
            # Create internal relay webhook URL for LangGraph
            base_url = get_base_url()
            internal_webhook = f"{base_url}/webhook-relay/{task_id}"
        else:
            internal_webhook = None
        
        if message.metadata:
            # Merge message metadata with our tracking metadata
            run_metadata.update(message.metadata)
        
        if stream:
            return self._create_stream_run(task_id, session_id, assistant_id, input_dict, run_metadata, internal_webhook)
        else:
            # Check if we should wait for completion or return immediately
            wait_for_completion = A2A_TASKS_SEND_WAIT_FOR_COMPLETION
            
            # Create a Run with the input
            run = await self.client.runs.create(
                thread_id=session_id,
                assistant_id=assistant_id,
                input=input_dict,
                metadata=run_metadata,
                webhook=internal_webhook
            )
            
            # Update thread metadata
            await self._update_thread_task_metadata(session_id, task_id, run["run_id"])
            
            if wait_for_completion:
                # Wait for the run to complete using join
                run = await self.client.runs.join(
                    thread_id=session_id,
                    run_id=run["run_id"]
                )
            
            # Get thread state
            thread_state = await self.client.threads.get_state(session_id)
            
            # Convert to A2A task
            return self._create_task_from_run(task_id, session_id, run, thread_state)
    
    async def _find_thread_by_task_id(self, task_id: str) -> Tuple[Thread, str]:
        """Find thread and run ID by task ID, returns (thread, run_id) or raises ValueError if not found"""
        # Search directly for threads with this task ID in their metadata
        threads = await self.client.threads.search(
            metadata={f"_a2a_has_task_{task_id}": True}
        )
        
        if threads and len(threads) > 0:
            # Found a thread with this task ID
            thread = threads[0]
            # Get the run ID from the mapping
            task_map = thread.get("metadata", {}).get("_a2a_map_task_id_by_run_id", {})
            
            # Find the run ID for this task
            for run_id, mapped_task_id in task_map.items():
                if mapped_task_id == task_id:
                    return (thread, run_id)
        
        # If we get here, the task wasn't found
        raise ValueError(f"Task {task_id} not found")
    
    async def get_task(self, task_id: str, history_length: Optional[int] = None) -> Task:
        """Get a task by ID."""
        # Find the thread and run_id using task_id
        thread, run_id = await self._find_thread_by_task_id(task_id)
        
        # Get the run
        run_info = await self.client.runs.get(thread["thread_id"], run_id)
        thread_state = await self.client.threads.get_state(thread["thread_id"])
        
        # Get thread history if needed
        history = None
        if history_length and history_length > 0:
            thread_history = await self.client.threads.get_history(
                thread_id=thread["thread_id"],
                limit=history_length
            )
            history = self._create_history_from_thread(thread_history)
        
        # Convert to A2A task
        return self._create_task_from_run(task_id, thread["thread_id"], run_info, thread_state, history)
    
    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task."""
        # Find the thread and run_id using task_id
        thread, run_id = await self._find_thread_by_task_id(task_id)
        
        # Cancel the run
        await self.client.runs.cancel(thread["thread_id"], run_id)
        
        # Get the updated run
        run_info = await self.client.runs.get(thread["thread_id"], run_id)
        thread_state = await self.client.threads.get_state(thread["thread_id"])
        
        # Convert to A2A task
        return self._create_task_from_run(task_id, thread["thread_id"], run_info, thread_state)
    
    def _process_stream_chunk(self, chunk: StreamPart, task_id: str = None) -> Union[TaskStatus, Artifact]:
        """Process a stream chunk and convert it to either TaskStatus or Artifact."""
        if chunk.event == "values":
            # Values event - look for messages in the data
            if "messages" in chunk.data and isinstance(chunk.data["messages"], list):
                # Find the latest AI message in the messages array
                ai_messages = [m for m in chunk.data["messages"] if m.get("type") == "ai"]
                if ai_messages:
                    # Get the most recent AI message
                    latest_ai = ai_messages[-1]
                    # Extract content
                    content = latest_ai.get("content", "")
                    # Create artifact from this message
                    return Artifact(parts=[TextPart(text=str(content))], index=0)
            
            # Default working status
            return TaskStatus(state=TaskState.WORKING, timestamp=datetime.now())
        
        elif chunk.event == "messages":
            # Message event - convert to Artifact
            parts = []
            if "content" in chunk.data:
                content = chunk.data["content"]
                parts.append(TextPart(text=str(content)))
            return Artifact(parts=parts, index=0)
                
        elif chunk.event == "end":
            # End event - yield final status
            return TaskStatus(state=TaskState.COMPLETED, timestamp=datetime.now())
        
        # For unknown events, return working status
        return TaskStatus(state=TaskState.WORKING, timestamp=datetime.now())

    async def _create_stream_run(
        self, 
        task_id: str,
        thread_id: str,
        assistant_id: str,
        input_dict: Dict[str, Any],
        run_metadata: Dict[str, Any],
        webhook: Optional[str] = None
    ) -> AsyncIterator[Union[TaskStatus, Artifact]]:
        """Create a streaming run and yield results."""
        try:
            # Always yield an initial status
            yield TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=datetime.now()
            )
            
            # Create a run for streaming
            run = await self.client.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                input=input_dict,
                metadata=run_metadata,
                webhook=webhook
            )
            
            # Update thread metadata
            await self._update_thread_task_metadata(thread_id, task_id, run["run_id"])
            
            # Yield a status indicating we're now processing
            yield TaskStatus(
                state=TaskState.WORKING,
                timestamp=datetime.now()
            )
            
            # Now stream the run - don't await the join_stream call
            stream = self.client.runs.join_stream(
                thread_id=thread_id,
                run_id=run["run_id"],
                stream_mode=["values", "messages"]
            )
            
            # Process stream chunks
            async for chunk in stream:
                # Process and yield each chunk
                yield self._process_stream_chunk(chunk, task_id)
            
            # Always yield a final completion status
            yield TaskStatus(
                state=TaskState.COMPLETED,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in streaming run: {str(e)}")
            # Ensure we yield an error status
            yield TaskStatus(
                state=TaskState.FAILED,
                timestamp=datetime.now()
            )
    
    def _message_to_langgraph_input(self, message: Message) -> Dict[str, Any]:
        """Convert A2A message to LangGraph base message format."""
        base_message = {"role": message.role}
        
        # Process message parts into content
        content = []
        for part in message.parts:
            if isinstance(part, TextPart):
                content.append(part.text)
            elif isinstance(part, DataPart):
                content.append(part.data)
            elif isinstance(part, FilePart):
                file_info = part.file
                
                # Handle image files - only support for image/url currently
                if file_info.mimeType and file_info.mimeType.startswith("image/"):
                    # Only handle cases with bytes or URI - no fallbacks
                    if file_info.bytes:
                        b64_data = base64.b64encode(file_info.bytes).decode("utf-8")
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file_info.mimeType};base64,{b64_data}"
                            }
                        })
                    elif file_info.uri:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": file_info.uri
                            }
                        })
                # We don't handle other file types as they may not be compatible
        
        # If there's only one content item, simplify
        if len(content) == 1:
            base_message["content"] = content[0]
        else:
            base_message["content"] = content
            
        return base_message
    
    def _run_status_to_task_state(self, status: RunStatus) -> TaskState:
        """Map LangGraph run status to A2A task state."""
        status_map = {
            "pending": TaskState.WORKING,
            "error": TaskState.FAILED,
            "success": TaskState.COMPLETED,
            "timeout": TaskState.FAILED,
            "interrupted": TaskState.INPUT_REQUIRED
        }
        return status_map.get(status, TaskState.UNKNOWN)
    
    def _create_task_from_run(
        self, 
        task_id: str,
        session_id: str,
        run: Run,
        thread_state: ThreadState,
        history: Optional[List[Message]] = None
    ) -> Task:
        """Create an A2A task from a LangGraph run."""
        task_state = self._run_status_to_task_state(run["status"])
        
        # Create task status
        task_status = TaskStatus(
            state=task_state,
            timestamp=run["updated_at"]
        )
        
        # Create artifacts from thread state if available
        artifacts = []
        if thread_state and thread_state.get("values"):
            parts = []
            
            # Convert thread state values to parts
            if isinstance(thread_state["values"], dict):
                for key, value in thread_state["values"].items():
                    if isinstance(value, str):
                        parts.append(TextPart(text=value))
                    else:
                        parts.append(DataPart(data={key: value}))
            
            if parts:
                artifacts.append(Artifact(parts=parts))
        
        return Task(
            id=task_id,
            sessionId=session_id,
            status=task_status,
            artifacts=artifacts if artifacts else None,
            history=history,
            metadata=run["metadata"]
        )
    
    def _create_history_from_thread(self, thread_history: List[ThreadState]) -> List[Message]:
        """Create message history from thread history."""
        messages = []
        
        for state in thread_history:
            # Create a message for each state
            parts = []
            
            # Extract relevant content from thread state
            if isinstance(state["values"], dict):
                for key, value in state["values"].items():
                    if isinstance(value, str):
                        parts.append(TextPart(text=value))
                    else:
                        parts.append(DataPart(data={key: value}))
            
            # Determine the role based on metadata or other indicators
            # This is a simplification, and might need more complex logic
            role = "agent"  # Default to agent
            
            if parts:
                messages.append(Message(
                    role=role,
                    parts=parts,
                    metadata=state.get("metadata", {})
                ))
                
        return messages

    async def get_webhook_url_for_task(self, task_id: str) -> Optional[str]:
        """Get the webhook URL for a task if it exists."""
        try:
            # Find the thread and run_id using task_id
            thread, run_id = await self._find_thread_by_task_id(task_id)
            
            # Get the run
            run_info = await self.client.runs.get(thread["thread_id"], run_id)
            
            # Get the webhook URL from metadata
            if run_info.get("metadata") and "_a2a_webhook_url" in run_info["metadata"]:
                return run_info["metadata"]["_a2a_webhook_url"]
            
            return None
        except ValueError:
            # Task not found
            return None
        except Exception as e:
            print(f"Error retrieving webhook URL for task {task_id}: {str(e)}")
            return None

    async def update_webhook_url_for_task(self, task_id: str, webhook_url: str) -> bool:
        """Update the webhook URL for a task."""
        try:
            # Find the thread and run_id using task_id
            thread, run_id = await self._find_thread_by_task_id(task_id)
            
            # Get the run
            run_info = await self.client.runs.get(thread["thread_id"], run_id)
            
            # Get our relay webhook URL from base_url
            base_url = get_base_url()
            internal_webhook_url = f"{base_url}/webhook-relay/{task_id}"
            
            # Update run metadata with the real webhook URL
            metadata = run_info.get("metadata", {})
            metadata["_a2a_webhook_url"] = webhook_url
            
            # Try to update the run's webhook using the LangGraph API
            try:
                # This is a placeholder - the actual SDK might not have this method
                # You would need to check the LangGraph SDK documentation for the correct approach
                await self.client.runs.update(thread["thread_id"], run_id, webhook=internal_webhook_url)
            except Exception as e:
                print(f"Warning: Could not update run webhook: {str(e)}")
            
            return True
        except ValueError:
            # Task not found
            return False
        except Exception as e:
            print(f"Error updating webhook URL for task {task_id}: {str(e)}")
            return False 