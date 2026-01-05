"""
Tests for src/agents/workflows/orchestrator.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOrchestrator:
    """Tests for the Orchestrator class."""

    def test_orchestrator_creation(self):
        """Test Orchestrator can be created."""
        from src.agents.workflows import Orchestrator

        orchestrator = Orchestrator()
        assert orchestrator is not None

    def test_orchestrator_with_config(self):
        """Test Orchestrator with custom config."""
        from src.agents.workflows import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            max_iterations=5,
            enable_verification=False,
            enable_critic=False,
        )
        orchestrator = Orchestrator(config)

        assert orchestrator.config.max_iterations == 5
        assert orchestrator.config.enable_verification is False

    def test_orchestrator_has_agents(self):
        """Test Orchestrator initializes all agents."""
        from src.agents.workflows import Orchestrator

        orchestrator = Orchestrator()

        assert orchestrator.planner is not None
        assert orchestrator.retriever is not None
        assert orchestrator.synthesizer is not None
        assert orchestrator.verifier is not None
        assert orchestrator.critic is not None
        assert orchestrator.citation is not None
        assert orchestrator.formatter is not None

    @pytest.mark.asyncio
    async def test_orchestrator_execute(self):
        """Test Orchestrator execute method."""
        from src.agents.workflows import Orchestrator

        # Mock all the agents
        with patch.multiple(
            'src.agents.workflows.orchestrator',
            PlannerAgent=MagicMock,
            RetrieverAgent=MagicMock,
            SynthesizerAgent=MagicMock,
            VerifierAgent=MagicMock,
            CriticAgent=MagicMock,
            CitationAgent=MagicMock,
            FormatterAgent=MagicMock,
        ):
            orchestrator = Orchestrator()

            # Mock agent execute methods
            orchestrator.planner.execute = AsyncMock(return_value=MagicMock(
                success=True,
                output=MagicMock(complexity="standard", strategy="hybrid"),
                latency_ms=50,
            ))
            orchestrator.retriever.execute = AsyncMock(return_value=MagicMock(
                success=True,
                output={"context": []},
                latency_ms=100,
            ))
            orchestrator.synthesizer.execute = AsyncMock(return_value=MagicMock(
                success=True,
                output={"response": "Generated response"},
                latency_ms=200,
            ))
            orchestrator.verifier.execute = AsyncMock(return_value=MagicMock(
                success=True,
                output={"results": [], "summary": {}},
                latency_ms=150,
            ))
            orchestrator.critic.execute = AsyncMock(return_value=MagicMock(
                success=True,
                output=MagicMock(decision="PASS", overall_score=0.9),
                latency_ms=100,
            ))
            orchestrator.citation.execute = AsyncMock(return_value=MagicMock(
                success=True,
                output={"citations": [], "citation_count": 0},
                latency_ms=50,
            ))
            orchestrator.formatter.execute = AsyncMock(return_value=MagicMock(
                success=True,
                output={"formatted_response": "Formatted response"},
                latency_ms=30,
            ))

            result = await orchestrator.execute("What is the policy?")

            assert "response" in result
            assert "trace_id" in result["metadata"]

    @pytest.mark.asyncio
    async def test_orchestrator_returns_trace(self):
        """Test Orchestrator returns execution trace."""
        from src.agents.workflows import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(enable_verification=False, enable_critic=False)

        with patch.multiple(
            'src.agents.workflows.orchestrator',
            PlannerAgent=MagicMock,
            RetrieverAgent=MagicMock,
            SynthesizerAgent=MagicMock,
            VerifierAgent=MagicMock,
            CriticAgent=MagicMock,
            CitationAgent=MagicMock,
            FormatterAgent=MagicMock,
        ):
            orchestrator = Orchestrator(config)

            # Setup mocks
            for agent in [orchestrator.planner, orchestrator.retriever,
                         orchestrator.synthesizer, orchestrator.citation,
                         orchestrator.formatter]:
                agent.execute = AsyncMock(return_value=MagicMock(
                    success=True,
                    output={},
                    latency_ms=50,
                ))

            result = await orchestrator.execute("Test query")

            assert "trace" in result
            assert result["trace"]["trace_id"] is not None


class TestSimpleOrchestrator:
    """Tests for the SimpleOrchestrator class."""

    def test_simple_orchestrator_creation(self):
        """Test SimpleOrchestrator can be created."""
        from src.agents.workflows import SimpleOrchestrator

        orchestrator = SimpleOrchestrator()
        assert orchestrator is not None

    def test_simple_orchestrator_disables_verification(self):
        """Test SimpleOrchestrator disables verification."""
        from src.agents.workflows import SimpleOrchestrator

        orchestrator = SimpleOrchestrator()
        assert orchestrator.config.enable_verification is False
        assert orchestrator.config.enable_critic is False

    def test_simple_orchestrator_single_iteration(self):
        """Test SimpleOrchestrator uses single iteration."""
        from src.agents.workflows import SimpleOrchestrator

        orchestrator = SimpleOrchestrator()
        assert orchestrator.config.max_iterations == 1


class TestStreamingOrchestrator:
    """Tests for the StreamingOrchestrator class."""

    def test_streaming_orchestrator_creation(self):
        """Test StreamingOrchestrator can be created."""
        from src.agents.workflows import StreamingOrchestrator

        orchestrator = StreamingOrchestrator()
        assert orchestrator is not None

    def test_streaming_orchestrator_has_streaming_method(self):
        """Test StreamingOrchestrator has execute_streaming method."""
        from src.agents.workflows import StreamingOrchestrator

        orchestrator = StreamingOrchestrator()
        assert hasattr(orchestrator, 'execute_streaming')


class TestOrchestratorConfig:
    """Tests for the OrchestratorConfig class."""

    def test_config_defaults(self):
        """Test OrchestratorConfig has sensible defaults."""
        from src.agents.workflows import OrchestratorConfig

        config = OrchestratorConfig()
        assert config.max_iterations == 3
        assert config.enable_verification is True
        assert config.enable_critic is True
        assert config.timeout_seconds == 120

    def test_config_custom_values(self):
        """Test OrchestratorConfig with custom values."""
        from src.agents.workflows import OrchestratorConfig
        from src.agents.formatter import OutputFormat

        config = OrchestratorConfig(
            max_iterations=5,
            enable_verification=False,
            output_format=OutputFormat.JSON,
        )
        assert config.max_iterations == 5
        assert config.output_format == OutputFormat.JSON


class TestExecutionTrace:
    """Tests for the ExecutionTrace class."""

    def test_execution_trace_creation(self):
        """Test ExecutionTrace can be created."""
        from src.agents.workflows import ExecutionTrace

        trace = ExecutionTrace(trace_id="test-trace-id")
        assert trace.trace_id == "test-trace-id"
        assert trace.steps == []
        assert trace.final_status == "pending"

    def test_execution_trace_add_step(self):
        """Test ExecutionTrace can add steps."""
        from src.agents.workflows import ExecutionTrace

        trace = ExecutionTrace(trace_id="test")
        trace.steps.append({
            "agent": "planner",
            "success": True,
            "latency_ms": 50,
        })

        assert len(trace.steps) == 1
        assert trace.steps[0]["agent"] == "planner"
