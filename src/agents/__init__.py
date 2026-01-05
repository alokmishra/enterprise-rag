"""
Enterprise RAG System - Multi-Agent Module

This module provides the multi-agent architecture for the RAG system.

Agent Types:
- Planner: Analyzes queries and creates execution plans
- Retriever: Retrieves relevant context from knowledge sources
- Synthesizer: Generates responses from context
- Verifier: Fact-checks claims against sources
- Critic: Evaluates response quality
- Citation: Manages source citations
- Formatter: Formats output for presentation

Orchestration:
- Orchestrator: Coordinates agent execution flow
- SimpleOrchestrator: Simplified flow for basic queries
- StreamingOrchestrator: Streaming response support
"""

# Base classes
from src.agents.base import BaseAgent, AgentConfig, AgentResult

# Individual agents
from src.agents.planner import PlannerAgent, ExecutionPlan
from src.agents.retriever import RetrieverAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.verifier import VerifierAgent
from src.agents.critic import CriticAgent, CriticDecision
from src.agents.citation import CitationAgent
from src.agents.formatter import FormatterAgent, OutputFormat

# Orchestration
from src.agents.workflows import (
    Orchestrator,
    OrchestratorConfig,
    SimpleOrchestrator,
    StreamingOrchestrator,
    ExecutionTrace,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    "AgentResult",
    # Agents
    "PlannerAgent",
    "ExecutionPlan",
    "RetrieverAgent",
    "SynthesizerAgent",
    "VerifierAgent",
    "CriticAgent",
    "CriticDecision",
    "CitationAgent",
    "FormatterAgent",
    "OutputFormat",
    # Orchestration
    "Orchestrator",
    "OrchestratorConfig",
    "SimpleOrchestrator",
    "StreamingOrchestrator",
    "ExecutionTrace",
]
