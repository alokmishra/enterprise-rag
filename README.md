# Enterprise RAG System

A state-of-the-art Retrieval-Augmented Generation system with multi-modal ingestion, knowledge graph integration, and multi-agent orchestration.

## 🚀 Features

- **Multi-Modal Ingestion**: Process text, images, audio, video, tables, and code
- **Hybrid Retrieval**: Vector, sparse, and graph-based search with intelligent fusion
- **Knowledge Graph**: Entity extraction, relationship mapping, and graph-augmented retrieval
- **Multi-Agent Orchestration**: 9 specialized agents for planning, retrieval, synthesis, and verification
- **Enterprise-Grade**: Authentication, access control, audit logging, and multi-tenancy

## 📋 Architecture

This system follows a modular monolith architecture, designed for future microservice extraction.

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
├─────────────────────────────────────────────────────────────────┤
│                    Agent Orchestrator                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Planner │ │Retriever│ │Synthesiz│ │Verifier │ │ Critic  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Ingestion   │  │  Retrieval   │  │  Generation  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Vector Store │  │ Graph Store  │  │ Metadata DB  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | Python 3.11+, FastAPI |
| Vector Store | Qdrant / pgvector |
| Graph Database | Neo4j |
| Metadata Store | PostgreSQL |
| Cache | Redis |
| Task Queue | Celery |
| LLM Providers | Claude, OpenAI |
| Embeddings | OpenAI, Voyage, Cohere |

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Make

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/enterprise-rag.git
cd enterprise-rag

# Setup development environment
make setup

# Start infrastructure (databases, cache, etc.)
make infra-up

# Run the application
make run

# Run tests
make test
```

## 📁 Project Structure

```
enterprise-rag/
├── src/                    # Application source code
│   ├── api/                # FastAPI routes and middleware
│   ├── agents/             # Multi-agent system
│   ├── ingestion/          # Document processing pipeline
│   ├── retrieval/          # Search and retrieval
│   ├── generation/         # LLM integration
│   ├── knowledge_graph/    # Entity and relationship extraction
│   └── storage/            # Storage adapters
├── tests/                  # Test suites
├── configs/                # Configuration files
├── deployments/            # Docker, K8s, Terraform
├── evaluation/             # Evaluation framework
└── docs/                   # Documentation
```

## 🔧 Configuration

Configuration is managed through YAML files in `/configs`:

```bash
configs/
├── base.yaml              # Base configuration
├── development.yaml       # Development overrides
├── staging.yaml           # Staging overrides
└── production.yaml        # Production overrides
```

Set the environment with `RAG_ENV` environment variable:

```bash
export RAG_ENV=development
```

## 📖 Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Getting Started Guide](docs/guides/getting-started.md)
- [API Reference](docs/api/rest-api.md)
- [Deployment Guide](docs/guides/deployment.md)

## 🧪 Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run with coverage
make test-coverage
```

## 📊 Evaluation

```bash
# Run evaluation suite
make eval

# Generate evaluation report
make eval-report
```

## 🚢 Deployment

See [Deployment Guide](docs/guides/deployment.md) for detailed instructions.

```bash
# Build Docker image
make docker-build

# Deploy to Kubernetes
make deploy-k8s ENV=staging
```

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Architecture inspired by state-of-the-art RAG research
- Built with modern Python best practices
