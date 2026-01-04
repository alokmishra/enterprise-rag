# Enterprise RAG System - Directory Structure

## Design Principles

1. **Domain-Driven Organization**: Code organized by business domain, not technical layer
2. **Clear Boundaries**: Each module has well-defined interfaces for future service extraction
3. **Monorepo Approach**: Single repository for cohesive development, easier refactoring
4. **Configuration as Code**: All configs version-controlled and environment-aware
5. **Test Proximity**: Tests live alongside the code they test

---

## Root Directory Structure

```
enterprise-rag/
│
├── README.md                       # Project overview, quick start guide
├── ARCHITECTURE.md                 # Architecture decisions and diagrams
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE
├── Makefile                        # Common development commands
├── docker-compose.yml              # Local development stack
├── docker-compose.prod.yml         # Production-like local testing
├── pyproject.toml                  # Python project config (poetry/hatch)
├── poetry.lock                     # Dependency lock file
├── .env.example                    # Environment variable template
├── .gitignore
├── .pre-commit-config.yaml         # Pre-commit hooks (linting, formatting)
│
├── docs/                           # Documentation
├── src/                            # Main application source code
├── tests/                          # Test suites
├── scripts/                        # Utility scripts
├── configs/                        # Configuration files
├── deployments/                    # Deployment configurations
├── data/                           # Local data directory (git-ignored)
├── notebooks/                      # Jupyter notebooks for exploration
├── evaluation/                     # Evaluation datasets and scripts
└── tools/                          # Development tools and utilities
```

---

## Detailed Structure

### `/docs` - Documentation

```
docs/
├── architecture/
│   ├── overview.md                 # System architecture overview
│   ├── decisions/                  # Architecture Decision Records (ADRs)
│   │   ├── 001-monolith-first.md
│   │   ├── 002-vector-db-selection.md
│   │   ├── 003-llm-provider-strategy.md
│   │   └── ...
│   ├── diagrams/
│   │   ├── system-context.png
│   │   ├── container-diagram.png
│   │   ├── component-diagrams/
│   │   └── sequence-diagrams/
│   └── data-flow.md
│
├── api/
│   ├── openapi.yaml                # OpenAPI specification
│   ├── rest-api.md                 # REST API documentation
│   ├── websocket-api.md            # WebSocket API for streaming
│   └── sdk-usage.md                # SDK usage examples
│
├── guides/
│   ├── getting-started.md          # Developer onboarding
│   ├── local-development.md        # Local setup guide
│   ├── deployment.md               # Deployment guide
│   ├── configuration.md            # Configuration reference
│   ├── troubleshooting.md          # Common issues and solutions
│   └── runbooks/                   # Operational runbooks
│       ├── incident-response.md
│       ├── scaling.md
│       └── backup-restore.md
│
├── modules/                        # Module-specific documentation
│   ├── ingestion.md
│   ├── retrieval.md
│   ├── generation.md
│   └── ...
│
└── research/                       # Research notes and references
    ├── papers/                     # Relevant papers
    ├── benchmarks/                 # Benchmark results
    └── experiments/                # Experiment logs
```

---

### `/src` - Application Source Code

```
src/
├── __init__.py
├── main.py                         # Application entry point
├── app.py                          # FastAPI application factory
│
├── api/                            # API Layer
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── query.py                # Query endpoints
│   │   ├── documents.py            # Document management endpoints
│   │   ├── admin.py                # Admin endpoints
│   │   ├── health.py               # Health check endpoints
│   │   └── websocket.py            # WebSocket handlers for streaming
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py                 # Authentication middleware
│   │   ├── rate_limit.py           # Rate limiting
│   │   ├── logging.py              # Request logging
│   │   └── error_handler.py        # Global error handling
│   ├── dependencies.py             # FastAPI dependencies
│   └── schemas/                    # Pydantic request/response models
│       ├── __init__.py
│       ├── query.py
│       ├── document.py
│       ├── common.py
│       └── admin.py
│
├── core/                           # Core utilities and shared code
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── logging.py                  # Logging setup
│   ├── exceptions.py               # Custom exceptions
│   ├── events.py                   # Event definitions
│   └── types.py                    # Shared type definitions
│
├── ingestion/                      # Ingestion Domain
│   ├── __init__.py
│   ├── service.py                  # Ingestion service (main interface)
│   ├── connectors/                 # Source connectors
│   │   ├── __init__.py
│   │   ├── base.py                 # Base connector interface
│   │   ├── sharepoint.py
│   │   ├── confluence.py
│   │   ├── google_drive.py
│   │   ├── s3.py
│   │   ├── web_crawler.py
│   │   └── database.py
│   ├── processors/                 # Document processors
│   │   ├── __init__.py
│   │   ├── base.py                 # Base processor interface
│   │   ├── pdf.py
│   │   ├── docx.py
│   │   ├── html.py
│   │   ├── markdown.py
│   │   ├── code.py
│   │   ├── spreadsheet.py
│   │   └── email.py
│   ├── chunking/                   # Chunking strategies
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── semantic.py
│   │   ├── structural.py
│   │   ├── fixed_size.py
│   │   └── code_aware.py
│   ├── embeddings/                 # Embedding generation
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openai.py
│   │   ├── cohere.py
│   │   ├── voyage.py
│   │   └── local.py                # Local models (sentence-transformers)
│   ├── multimodal/                 # Multi-modal processing
│   │   ├── __init__.py
│   │   ├── image_processor.py
│   │   ├── audio_processor.py
│   │   ├── video_processor.py
│   │   ├── table_extractor.py
│   │   └── ocr.py
│   ├── pipeline.py                 # Ingestion pipeline orchestration
│   ├── tasks.py                    # Celery tasks for async ingestion
│   └── models.py                   # Ingestion domain models
│
├── retrieval/                      # Retrieval Domain
│   ├── __init__.py
│   ├── service.py                  # Retrieval service (main interface)
│   ├── query/                      # Query processing
│   │   ├── __init__.py
│   │   ├── analyzer.py             # Query analysis and classification
│   │   ├── rewriter.py             # Query rewriting (HyDE, expansion)
│   │   ├── decomposer.py           # Multi-part query decomposition
│   │   └── router.py               # Strategy routing
│   ├── search/                     # Search implementations
│   │   ├── __init__.py
│   │   ├── base.py                 # Base search interface
│   │   ├── vector.py               # Vector similarity search
│   │   ├── sparse.py               # BM25/keyword search
│   │   ├── hybrid.py               # Hybrid search fusion
│   │   └── graph.py                # Knowledge graph search
│   ├── reranking/                  # Reranking
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cross_encoder.py
│   │   ├── llm_reranker.py
│   │   └── diversity.py            # Diversity optimization
│   ├── context/                    # Context assembly
│   │   ├── __init__.py
│   │   ├── assembler.py            # Context assembly logic
│   │   ├── deduplication.py        # Chunk deduplication
│   │   └── compression.py          # Context compression
│   ├── filters/                    # Result filtering
│   │   ├── __init__.py
│   │   ├── acl.py                  # Access control filtering
│   │   ├── metadata.py             # Metadata filtering
│   │   └── temporal.py             # Time-based filtering
│   └── models.py                   # Retrieval domain models
│
├── generation/                     # Generation Domain
│   ├── __init__.py
│   ├── service.py                  # Generation service (main interface)
│   ├── llm/                        # LLM clients
│   │   ├── __init__.py
│   │   ├── base.py                 # Base LLM interface
│   │   ├── anthropic.py            # Claude client
│   │   ├── openai.py               # GPT client
│   │   ├── router.py               # Multi-provider routing
│   │   └── fallback.py             # Fallback logic
│   ├── prompts/                    # Prompt management
│   │   ├── __init__.py
│   │   ├── manager.py              # Prompt template manager
│   │   ├── templates/              # Prompt templates
│   │   │   ├── rag_default.yaml
│   │   │   ├── summarization.yaml
│   │   │   ├── comparison.yaml
│   │   │   └── ...
│   │   └── builder.py              # Dynamic prompt construction
│   ├── streaming/                  # Response streaming
│   │   ├── __init__.py
│   │   └── handler.py
│   └── models.py                   # Generation domain models
│
├── agents/                         # Multi-Agent System
│   ├── __init__.py
│   ├── orchestrator.py             # Main orchestrator
│   ├── base.py                     # Base agent class
│   ├── planner/                    # Planner Agent
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── strategies.py
│   ├── retriever/                  # Retriever Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── researcher/                 # Researcher Agent
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── exploration.py
│   ├── synthesizer/                # Synthesizer/Writer Agent
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── formats.py
│   ├── verifier/                   # Fact Verifier Agent
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── claim_extractor.py
│   │   └── evidence_matcher.py
│   ├── critic/                     # Critic Agent
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── quality_metrics.py
│   ├── citation/                   # Citation Manager Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── formatter/                  # Response Formatter Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── communication/              # Agent communication
│   │   ├── __init__.py
│   │   ├── messages.py             # Message definitions
│   │   ├── state.py                # Shared state management
│   │   └── bus.py                  # Message bus
│   └── workflows/                  # Predefined workflows
│       ├── __init__.py
│       ├── simple.py               # Fast path workflow
│       ├── standard.py             # Standard workflow
│       └── research.py             # Complex research workflow
│
├── knowledge_graph/                # Knowledge Graph Domain
│   ├── __init__.py
│   ├── service.py                  # KG service (main interface)
│   ├── extraction/                 # Entity & relationship extraction
│   │   ├── __init__.py
│   │   ├── entity_extractor.py
│   │   ├── relationship_extractor.py
│   │   ├── coreference.py
│   │   └── linker.py               # Entity linking
│   ├── storage/                    # Graph storage
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── neo4j.py
│   │   └── neptune.py
│   ├── query/                      # Graph querying
│   │   ├── __init__.py
│   │   ├── cypher_generator.py     # NL to Cypher
│   │   └── traversal.py
│   ├── ontology/                   # Ontology management
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   └── definitions/            # Entity/relationship definitions
│   │       ├── enterprise.yaml
│   │       └── custom.yaml
│   └── models.py                   # KG domain models
│
├── storage/                        # Storage Adapters
│   ├── __init__.py
│   ├── vector/                     # Vector stores
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── qdrant.py
│   │   ├── pinecone.py
│   │   ├── pgvector.py
│   │   └── weaviate.py
│   ├── document/                   # Document stores
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── s3.py
│   │   ├── gcs.py
│   │   └── local.py
│   ├── metadata/                   # Metadata stores
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── postgresql.py
│   ├── cache/                      # Caching
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── redis.py
│   │   └── memory.py
│   └── queue/                      # Message queues
│       ├── __init__.py
│       ├── base.py
│       ├── redis_queue.py
│       └── kafka.py
│
├── auth/                           # Authentication & Authorization
│   ├── __init__.py
│   ├── service.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── oauth.py
│   │   ├── saml.py
│   │   └── api_key.py
│   ├── permissions.py              # Permission management
│   └── models.py
│
├── observability/                  # Observability
│   ├── __init__.py
│   ├── metrics.py                  # Prometheus metrics
│   ├── tracing.py                  # Distributed tracing
│   ├── logging.py                  # Structured logging
│   └── health.py                   # Health checks
│
├── admin/                          # Admin functionality
│   ├── __init__.py
│   ├── service.py
│   ├── dashboard/                  # Admin dashboard
│   │   ├── __init__.py
│   │   └── routes.py
│   └── management/                 # Management commands
│       ├── __init__.py
│       ├── sources.py
│       ├── users.py
│       └── indexes.py
│
└── workers/                        # Background workers
    ├── __init__.py
    ├── celery_app.py               # Celery configuration
    ├── tasks/                      # Task definitions
    │   ├── __init__.py
    │   ├── ingestion.py
    │   ├── embedding.py
    │   ├── indexing.py
    │   └── maintenance.py
    └── schedules.py                # Scheduled tasks (Celery beat)
```

---

### `/tests` - Test Suites

```
tests/
├── __init__.py
├── conftest.py                     # Shared fixtures
├── factories/                      # Test data factories
│   ├── __init__.py
│   ├── document.py
│   ├── query.py
│   └── user.py
│
├── unit/                           # Unit tests
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── test_chunking.py
│   │   ├── test_processors.py
│   │   └── test_embeddings.py
│   ├── retrieval/
│   │   ├── test_query_analyzer.py
│   │   ├── test_search.py
│   │   └── test_reranking.py
│   ├── generation/
│   │   ├── test_prompts.py
│   │   └── test_llm_clients.py
│   ├── agents/
│   │   ├── test_orchestrator.py
│   │   ├── test_planner.py
│   │   ├── test_verifier.py
│   │   └── test_critic.py
│   └── knowledge_graph/
│       ├── test_extraction.py
│       └── test_linking.py
│
├── integration/                    # Integration tests
│   ├── __init__.py
│   ├── test_ingestion_pipeline.py
│   ├── test_retrieval_pipeline.py
│   ├── test_agent_workflows.py
│   ├── test_api_endpoints.py
│   └── test_storage_backends.py
│
├── e2e/                            # End-to-end tests
│   ├── __init__.py
│   ├── test_query_flows.py
│   ├── test_document_lifecycle.py
│   └── test_multi_agent.py
│
└── fixtures/                       # Test fixtures
    ├── documents/
    │   ├── sample.pdf
    │   ├── sample.docx
    │   └── sample.html
    ├── queries/
    │   └── test_queries.json
    └── responses/
        └── expected_responses.json
```

---

### `/configs` - Configuration Files

```
configs/
├── base.yaml                       # Base configuration
├── development.yaml                # Development overrides
├── staging.yaml                    # Staging overrides
├── production.yaml                 # Production overrides
├── test.yaml                       # Test configuration
│
├── models/                         # Model configurations
│   ├── embeddings.yaml             # Embedding model configs
│   ├── llm.yaml                    # LLM provider configs
│   └── rerankers.yaml              # Reranker configs
│
├── agents/                         # Agent configurations
│   ├── orchestrator.yaml
│   ├── planner.yaml
│   ├── verifier.yaml
│   └── critic.yaml
│
├── storage/                        # Storage configurations
│   ├── vector_stores.yaml
│   ├── databases.yaml
│   └── cache.yaml
│
├── ingestion/                      # Ingestion configurations
│   ├── connectors.yaml
│   ├── chunking.yaml
│   └── processing.yaml
│
└── ontology/                       # Knowledge graph ontology
    ├── entity_types.yaml
    └── relationship_types.yaml
```

---

### `/deployments` - Deployment Configurations

```
deployments/
├── docker/
│   ├── Dockerfile                  # Main application Dockerfile
│   ├── Dockerfile.worker           # Celery worker Dockerfile
│   └── nginx/
│       └── nginx.conf
│
├── kubernetes/
│   ├── base/                       # Base Kubernetes manifests
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml
│   │   ├── hpa.yaml
│   │   └── ingress.yaml
│   ├── overlays/                   # Environment-specific overlays
│   │   ├── development/
│   │   ├── staging/
│   │   └── production/
│   └── kustomization.yaml
│
├── terraform/                      # Infrastructure as Code
│   ├── modules/
│   │   ├── networking/
│   │   ├── database/
│   │   ├── kubernetes/
│   │   ├── vector_store/
│   │   └── monitoring/
│   ├── environments/
│   │   ├── dev/
│   │   ├── staging/
│   │   └── prod/
│   └── main.tf
│
├── helm/                           # Helm charts
│   └── enterprise-rag/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-staging.yaml
│       ├── values-production.yaml
│       └── templates/
│
└── scripts/
    ├── deploy.sh
    ├── rollback.sh
    └── migrate.sh
```

---

### `/scripts` - Utility Scripts

```
scripts/
├── setup/
│   ├── install_dependencies.sh
│   ├── setup_dev_environment.sh
│   └── init_databases.sh
│
├── data/
│   ├── seed_sample_data.py
│   ├── export_data.py
│   └── import_data.py
│
├── maintenance/
│   ├── cleanup_old_indexes.py
│   ├── reindex_documents.py
│   └── vacuum_databases.py
│
├── migration/
│   ├── migrate_vectors.py
│   └── migrate_metadata.py
│
└── utils/
    ├── generate_api_key.py
    ├── test_connections.py
    └── benchmark_retrieval.py
```

---

### `/evaluation` - Evaluation Framework

```
evaluation/
├── datasets/
│   ├── golden/                     # Golden Q&A datasets
│   │   ├── factual_qa.json
│   │   ├── multi_hop.json
│   │   └── summarization.json
│   ├── synthetic/                  # Synthetically generated
│   │   └── generated_qa.json
│   └── production/                 # Production samples
│       └── sampled_queries.json
│
├── metrics/
│   ├── __init__.py
│   ├── retrieval.py                # Recall, MRR, NDCG
│   ├── generation.py               # Faithfulness, relevance
│   ├── latency.py                  # Performance metrics
│   └── cost.py                     # Cost tracking
│
├── judges/
│   ├── __init__.py
│   ├── llm_judge.py                # LLM-as-judge evaluation
│   └── prompts/
│       ├── correctness.yaml
│       ├── faithfulness.yaml
│       └── relevance.yaml
│
├── benchmarks/
│   ├── __init__.py
│   ├── run_benchmark.py
│   └── results/
│
├── reports/
│   └── templates/
│       └── evaluation_report.html
│
└── scripts/
    ├── generate_synthetic_data.py
    ├── run_evaluation.py
    └── compare_experiments.py
```

---

### `/notebooks` - Jupyter Notebooks

```
notebooks/
├── exploration/
│   ├── 01_data_analysis.ipynb
│   ├── 02_embedding_comparison.ipynb
│   └── 03_chunking_experiments.ipynb
│
├── prototypes/
│   ├── agent_workflow_prototype.ipynb
│   ├── kg_extraction_prototype.ipynb
│   └── multimodal_prototype.ipynb
│
├── evaluation/
│   ├── retrieval_quality.ipynb
│   ├── generation_quality.ipynb
│   └── agent_performance.ipynb
│
└── demos/
    ├── end_to_end_demo.ipynb
    └── feature_showcase.ipynb
```

---

### `/tools` - Development Tools

```
tools/
├── cli/                            # CLI tools
│   ├── __init__.py
│   ├── main.py                     # CLI entry point
│   ├── commands/
│   │   ├── ingest.py               # rag ingest <source>
│   │   ├── query.py                # rag query "question"
│   │   ├── index.py                # rag index rebuild
│   │   └── admin.py                # rag admin <command>
│   └── utils.py
│
├── debug/
│   ├── agent_playground.py         # Test individual agents
│   ├── retrieval_inspector.py      # Inspect retrieval results
│   └── trace_viewer.py             # View agent traces
│
├── generators/
│   ├── generate_schemas.py         # Generate Pydantic schemas
│   ├── generate_sdk.py             # Generate client SDKs
│   └── generate_docs.py            # Generate documentation
│
└── migrations/
    ├── alembic/                    # Database migrations
    │   ├── alembic.ini
    │   ├── env.py
    │   └── versions/
    └── vector_migrations/          # Vector index migrations
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/main.py` | Application entry point |
| `src/app.py` | FastAPI app factory |
| `src/core/config.py` | Configuration management |
| `src/agents/orchestrator.py` | Multi-agent coordinator |
| `src/ingestion/pipeline.py` | Document ingestion orchestration |
| `src/retrieval/service.py` | Main retrieval interface |
| `configs/base.yaml` | Base configuration |
| `docker-compose.yml` | Local development stack |

---

## Module Boundaries (For Future Service Extraction)

Each major directory under `/src` is designed as a potential microservice:

| Module | Future Service | Dependencies |
|--------|---------------|--------------|
| `ingestion/` | Ingestion Service | storage, embeddings |
| `retrieval/` | Query Service | storage, reranking |
| `generation/` | Generation Service | llm providers |
| `agents/` | Orchestration Service | all other services |
| `knowledge_graph/` | Graph Service | storage |
| `auth/` | Auth Service | database |

---

## Getting Started Commands

```bash
# Clone and setup
git clone <repo>
cd enterprise-rag
make setup                          # Install dependencies, setup pre-commit

# Local development
make dev                            # Start docker-compose stack
make run                            # Run application
make test                           # Run tests

# Common tasks
make ingest path=./data/docs        # Ingest documents
make query q="What is..."           # Test query
make eval                           # Run evaluation suite
```

---

## Next Steps

1. Initialize the repository with this structure
2. Set up base configuration and dependencies
3. Implement core interfaces (base classes)
4. Build out modules incrementally per project plan phases
