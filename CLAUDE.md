# CLAUDE.md - Enterprise RAG System

This file provides context for AI assistants working on this codebase.

## Project Overview

Enterprise RAG (Retrieval-Augmented Generation) system with multi-agent architecture. Built with Python 3.11+, FastAPI, and modern async patterns.

**Purpose**: Production-grade RAG system for enterprise document Q&A with fact verification, citations, and quality evaluation.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI (async) |
| Vector Store | Qdrant |
| Database | PostgreSQL + SQLAlchemy async |
| Cache | Redis |
| LLM Providers | Anthropic Claude, OpenAI GPT |
| Embeddings | OpenAI text-embedding-3-small |
| Search | Hybrid (vector + BM25) |

## Directory Structure

```
src/
├── api/                    # FastAPI application
│   ├── routes/             # API endpoints (health, query, documents)
│   ├── middleware/         # Auth, logging, error handling
│   └── services/           # Business logic (rag_pipeline, retrieval_service)
│
├── agents/                 # Multi-agent system (Phase 4)
│   ├── base.py             # BaseAgent, AgentConfig, AgentResult
│   ├── planner/            # Query analysis, execution planning
│   ├── retriever/          # Context retrieval agent
│   ├── synthesizer/        # Response generation agent
│   ├── verifier/           # Fact-checking agent
│   ├── critic/             # Quality evaluation agent
│   ├── citation/           # Source linking agent
│   ├── formatter/          # Output formatting agent
│   └── workflows/          # Orchestrator, execution flows
│
├── core/                   # Core utilities
│   ├── config.py           # Settings (Pydantic BaseSettings)
│   ├── logging.py          # Structured logging (structlog)
│   ├── exceptions.py       # Custom exceptions
│   └── types.py            # Shared Pydantic models, enums
│
├── storage/                # Storage backends
│   ├── base.py             # Abstract interfaces
│   ├── vector/qdrant.py    # Qdrant vector store
│   ├── cache/redis.py      # Redis cache
│   └── document/           # PostgreSQL (models, database, repository)
│
├── ingestion/              # Document processing (Phase 2)
│   ├── processors/         # PDF, DOCX, HTML, text, code parsers
│   ├── chunking/           # Recursive, sentence, paragraph chunkers
│   └── embeddings/         # OpenAI embeddings with batching
│
├── retrieval/              # Search & retrieval (Phase 3)
│   ├── search/             # Vector, sparse (BM25), hybrid search
│   ├── query/              # Query expansion, HyDE
│   ├── reranking/          # LLM and cross-encoder rerankers
│   ├── filters/            # Metadata filtering
│   └── context/            # Context assembly with token budgeting
│
└── generation/             # LLM integration (Phase 2)
    ├── llm/                # Anthropic, OpenAI clients + factory
    └── prompts/            # RAG prompt templates
```

## Implementation Status

| Phase | Status | Components |
|-------|--------|------------|
| 1. Foundation | Complete | core/, storage/, api/ basics |
| 2. Core RAG | Complete | ingestion/, retrieval/search, generation/ |
| 3. Advanced Retrieval | Complete | hybrid search, reranking, query expansion |
| 4. Multi-Agent | Complete | All 7 agents + orchestrator |
| 5. Production | Pending | Auth, monitoring, deployment |
| 6. Multi-Modal | Pending | Image, audio, video processing |
| 7. Knowledge Graph | Pending | knowledge_graph/, entity extraction |

## Key Patterns

### Agent Pattern
All agents extend `BaseAgent` and implement:
```python
async def execute(self, state: AgentState, **kwargs) -> AgentResult
```

### Shared State
Agents communicate via `AgentState` (defined in `src/core/types.py`):
- `trace_id`, `original_query`, `conversation_history`
- `execution_plan`, `retrieved_context`, `draft_responses`
- `verification_results`, `critic_feedback`, `iteration_count`

### Factory Pattern
Singletons via factory functions:
- `get_settings()` - Configuration
- `get_logger(name)` - Structured logger
- `get_rag_pipeline()` - RAG pipeline instance
- `get_llm_client(provider)` - LLM clients

### Repository Pattern
Database access via repositories in `src/storage/document/repository.py`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/query` | POST | Basic RAG query |
| `/query/stream` | POST | Streaming RAG query |
| `/query/agent` | POST | Multi-agent RAG query |
| `/query/agent/stream` | POST | Streaming multi-agent query |
| `/documents` | POST | Upload document |
| `/documents/{id}` | GET/DELETE | Document operations |

## Key Types (src/core/types.py)

- `AgentType` - Enum: PLANNER, RETRIEVER, SYNTHESIZER, VERIFIER, CRITIC, CITATION, FORMATTER
- `AgentState` - Shared state between agents
- `ContextItem` - Retrieved context chunk
- `Citation` - Source citation
- `VerificationResult` - Fact-check result
- `CriticFeedback` - Quality evaluation with scores

## Common Commands

```bash
# Run development server
uvicorn src.app:app --reload --port 8000

# Run tests
pytest tests/

# Type checking
mypy src/

# Linting
ruff check src/
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional (defaults shown)
RAG_ENV=development
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/rag
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
```

## Configuration

Settings in `src/core/config.py` via Pydantic BaseSettings. Override with environment variables or `.env` file.

## Documentation

- `docs/agents.md` - Multi-agent architecture documentation
- `README.md` - Project overview and setup

## Working on This Codebase

1. **Adding a new agent**: Create directory in `src/agents/`, extend `BaseAgent`, add to orchestrator
2. **Adding an endpoint**: Add route in `src/api/routes/`, register in `src/app.py`
3. **Adding storage**: Implement interface from `src/storage/base.py`
4. **Modifying types**: Update `src/core/types.py`, check agent compatibility

## Notes

- All I/O operations are async
- Structured logging with trace_id correlation
- Token budget management in context assembly
- Iterative refinement loop (max 3 iterations) in orchestrator
