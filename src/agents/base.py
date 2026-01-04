"""
Enterprise RAG System - Base Agent Class
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

from src.core.logging import LoggerMixin
from src.core.types import AgentMessage, AgentState, AgentType, MessageType


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str
    agent_type: AgentType
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: int = 60


class AgentResult(BaseModel):
    """Result from agent execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0


class BaseAgent(ABC, LoggerMixin):
    """
    Base class for all agents in the multi-agent system.
    
    Each agent is responsible for a specific task in the RAG pipeline.
    Agents communicate through structured messages and share state
    through the AgentState object.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.agent_type = config.agent_type
    
    @abstractmethod
    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """
        Execute the agent's primary task.
        
        Args:
            state: Shared agent state
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with output or error
        """
        pass
    
    def create_message(
        self,
        to_agent: AgentType,
        message_type: MessageType,
        payload: dict[str, Any],
        trace_id: str,
    ) -> AgentMessage:
        """Create a message to send to another agent."""
        from uuid import uuid4
        
        return AgentMessage(
            message_id=str(uuid4()),
            trace_id=trace_id,
            from_agent=self.agent_type,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
        )
    
    async def validate_input(self, state: AgentState) -> bool:
        """Validate that the agent has required input."""
        return True
    
    async def on_error(self, error: Exception, state: AgentState) -> None:
        """Handle errors during execution."""
        self.logger.error(
            f"Agent {self.name} error",
            error=str(error),
            trace_id=state.trace_id,
        )
