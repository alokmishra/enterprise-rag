# Enterprise RAG System - Enhancement Roadmap

This document outlines recommended enhancements to improve the Enterprise RAG system's production readiness, security, scalability, and feature completeness.

---

## Table of Contents

1. [Immediate Fixes](#1-immediate-fixes)
2. [Architecture Improvements](#2-architecture-improvements)
3. [Performance Optimizations](#3-performance-optimizations)
4. [Enterprise Features](#4-enterprise-features)
5. [Security Hardening](#5-security-hardening)
6. [Scalability Improvements](#6-scalability-improvements)
7. [Observability Enhancements](#7-observability-enhancements)
8. [Testing Improvements](#8-testing-improvements)
9. [Production Readiness](#9-production-readiness)
10. [Developer Experience](#10-developer-experience)
11. [Priority Matrix](#11-priority-matrix)

---

## 1. Immediate Fixes

Critical issues that should be addressed before production deployment.

### 1.1 Fix Mutable Default in ExecutionTrace

**Location:** `src/agents/workflows/orchestrator.py:44`

**Issue:** `steps=[]` is a mutable default that will be shared across instances.

**Fix:**
```python
# Before
steps: list[AgentStep] = []

# After
steps: list[AgentStep] = Field(default_factory=list)
```

---

### 1.2 Fix Metrics Environment Label

**Location:** `src/observability/metrics.py`

**Issue:** Uses `settings.environment` which doesn't exist; should use `settings.RAG_ENV`.

**Fix:**
```python
# Before
environment = getattr(settings, 'environment', 'unknown')

# After
environment = settings.RAG_ENV
```

---

### 1.3 Fail-Fast on Critical Dependency Failures

**Location:** `src/app.py:37-55`

**Issue:** App starts even if database, vector store, or cache initialization fails.

**Recommendation:**
- Define critical vs optional dependencies
- Fail startup if critical dependencies are unavailable
- Or implement degraded mode with feature gating

```python
# Example implementation
critical_deps_ready = True

try:
    await init_database()
except Exception as e:
    logger.error("Failed to initialize database", error=str(e))
    critical_deps_ready = False

if not critical_deps_ready:
    raise RuntimeError("Critical dependencies unavailable")
```

---

### 1.4 Enforce SECRET_KEY in Production

**Location:** `src/core/config.py:32`

**Issue:** Default `SECRET_KEY` is `"change-me-in-production"`.

**Fix:**
```python
@field_validator("SECRET_KEY")
@classmethod
def validate_secret_key(cls, v, info):
    if info.data.get("RAG_ENV") == "production" and v == "change-me-in-production":
        raise ValueError("SECRET_KEY must be changed in production")
    return v
```

---

### 1.5 Tighten CORS Configuration

**Location:** `src/app.py:81-87`

**Issue:** Overly permissive CORS settings.

**Current:**
```python
allow_methods=["*"],
allow_headers=["*"],
allow_credentials=True,
```

**Recommendation:**
```python
allow_methods=["GET", "POST", "PUT", "DELETE"],
allow_headers=["Authorization", "Content-Type", "X-API-Key"],
allow_credentials=True,  # Only if needed
```

---

## 2. Architecture Improvements

### 2.1 Explicit Dependency Injection

**Current State:** Singleton patterns (`get_vector_store()`, `get_hybrid_searcher()`) can cause issues with multi-worker deployments, parallel tests, and per-tenant routing.

**Recommendation:**
- Inject dependencies via FastAPI `Depends()`
- Store clients in `app.state` during lifespan
- Use singletons only for process-scoped resources

```python
# Example
@router.post("/query")
async def query(
    request: QueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    cache: CacheStore = Depends(get_cache),
):
    ...
```

---

### 2.2 Durable Background Job Layer

**Current State:** Ingestion runs inline in API requests.

**Problem:** Multi-modal ingestion + chunking + embedding + graph updates are CPU/IO intensive and can timeout or block requests.

**Recommendation:**
- Add queue-based worker (Celery with Redis broker already configured)
- Support retries, backoff, and long-running tasks
- Track job status via API

**Endpoints to add:**
```
POST /api/v1/documents/ingest    → Returns job_id
GET  /api/v1/jobs/{job_id}       → Returns job status
```

---

### 2.3 Separate Data Plane vs Control Plane APIs

**Data Plane (Query/Inference):**
- `/api/v1/query`
- `/api/v1/query/stream`
- `/api/v1/query/agent`

**Control Plane (Admin/Config):**
- `/api/v1/admin/*`
- `/api/v1/documents/*`
- API key management
- Tenant management

**Recommendation:**
- Apply stricter auth/rate limits on control plane
- Enhanced audit logging for admin actions
- Consider separate deployments for high-security environments

---

## 3. Performance Optimizations

### 3.1 Implement Orchestrator Timeouts

**Location:** `src/agents/workflows/orchestrator.py`

**Issue:** `OrchestratorConfig.timeout_seconds` exists but is never used.

**Fix:**
```python
async def execute(self, query: str, ...) -> dict:
    try:
        result = await asyncio.wait_for(
            self._execute_pipeline(query, ...),
            timeout=self.config.timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.error("Query execution timed out", trace_id=trace_id)
        raise QueryTimeoutError(f"Query timed out after {self.config.timeout_seconds}s")
```

---

### 3.2 Parallelize Retrieval Operations

**Current:** Vector, graph, and keyword searches run sequentially.

**Recommendation:**
```python
async def retrieve(self, query: str) -> RetrievalResult:
    vector_task = asyncio.create_task(self.vector_search(query))
    graph_task = asyncio.create_task(self.graph_search(query))
    keyword_task = asyncio.create_task(self.keyword_search(query))
    
    vector_results, graph_results, keyword_results = await asyncio.gather(
        vector_task, graph_task, keyword_task,
        return_exceptions=True
    )
    
    return self.fuse_results(vector_results, graph_results, keyword_results)
```

---

### 3.3 Strategic Caching

| Cache Type | Key | TTL | Use Case |
|------------|-----|-----|----------|
| Embedding Cache | `hash(query + model)` | 24h | Avoid re-embedding identical queries |
| Retrieval Cache | `hash(embedding + filters + strategy)` | 1h | Cache search results |
| Response Cache | `hash(query + context_ids)` | 30m | Optional, for identical requests |

**Metrics to add:**
- `cache_hits_total{cache_type}`
- `cache_misses_total{cache_type}`

---

### 3.4 Optimize Qdrant Usage

1. **Batch upserts:** Avoid `wait=True` for large ingestions
2. **Payload indexes:** Add indexes for frequently-filtered fields:
   - `tenant_id`
   - `document_type`
   - `acl_groups`
3. **Smaller payloads:** Store large text in Postgres, keep only metadata in Qdrant

---

### 3.5 Offload Synchronous Work

**Location:** `src/retrieval/search/hybrid.py`

**Issue:** BM25 operations are synchronous and can block the event loop.

**Fix:**
```python
from anyio import to_thread

bm25_results = await to_thread.run_sync(self.bm25_index.search, query)
```

---

## 4. Enterprise Features

### 4.1 Multi-Tenancy with Hard Isolation

**Requirements:**
- `tenant_id` on every document, chunk, and query
- Enforced at DB, vector store, and graph queries
- Per-tenant encryption keys (advanced)

**Implementation points:**
- Add `tenant_id` to all models
- Filter all Qdrant queries by tenant
- Add tenant context to all graph traversals
- Validate tenant access in middleware

---

### 4.2 Document-Level Access Control (ACL)

**Current:** RBAC exists but doesn't enforce row/chunk-level access.

**Requirements:**
- ACL groups on documents/chunks
- Enforcement in:
  - Qdrant filters
  - Postgres queries
  - Neo4j traversals

**Example Qdrant filter:**
```python
filter = Filter(
    must=[
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
        FieldCondition(key="acl_groups", match=MatchAny(any=user_groups)),
    ]
)
```

---

### 4.3 Query History & Audit Logging

**Location:** `src/api/routes/query.py:195+, 210+` (currently TODOs)

**Requirements:**
- Persist query + response + sources + model + latency + user/tenant
- Tamper-evident audit trails for admin actions
- Retention policies (GDPR compliance)

**Schema:**
```python
class QueryLog(Base):
    id: UUID
    tenant_id: UUID
    user_id: UUID
    query: str
    response: str
    sources: list[str]
    model: str
    latency_ms: float
    tokens_used: int
    created_at: datetime
```

---

### 4.4 Feedback Loop Implementation

**Location:** `src/api/routes/query.py:190` (currently TODO)

**Requirements:**
- Store feedback tied to `query_id`
- Use for offline evaluation datasets
- Support regression testing ("golden Q/A")
- Enable prompt/retrieval tuning

---

### 4.5 Safety Controls

| Control | Description |
|---------|-------------|
| Prompt Injection Detection | Detect and block malicious prompts |
| Source Grounding | Refuse if insufficient evidence |
| PII/PHI Redaction | Pre-ingest and/or pre-answer redaction |
| Content Filtering | Toxicity/sensitive topic controls |

---

### 4.6 Model Routing & Fallbacks

**Requirements:**
- Automatic fallback on timeout/quota exhaustion
- Per-tenant model allowlist
- Cost controls / token budgets
- Load balancing across providers

```python
class ModelRouter:
    async def generate(self, prompt: str, tenant_id: str) -> str:
        allowed_models = await self.get_tenant_models(tenant_id)
        
        for model in allowed_models:
            try:
                return await self.call_with_timeout(model, prompt)
            except (TimeoutError, QuotaExceededError):
                continue
        
        raise AllModelsExhaustedError()
```

---

## 5. Security Hardening

### 5.1 Request Rate Limiting

**Implementation:** Redis-based token bucket per API key/user/tenant.

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@router.post("/query", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
async def query(request: QueryRequest):
    ...
```

---

### 5.2 Streaming Endpoint Hardening

| Control | Description |
|---------|-------------|
| Max concurrent streams | Limit per user/tenant |
| Heartbeat timeouts | Detect dead connections |
| Disconnect handling | Stop LLM generation on client disconnect |
| Token limits | Enforce max tokens per stream |

---

### 5.3 Secrets Management

**Requirements:**
- Load secrets from Vault/KMS (or Docker secrets)
- Support key rotation
- Include `kid` (key ID) for multiple active JWT signing keys

---

### 5.4 Supply Chain Security

| Control | Tool |
|---------|------|
| SBOM Generation | `syft`, `cyclonedx` |
| Dependency Pinning | `pip-tools`, `poetry.lock` |
| Vulnerability Scanning | `safety`, `snyk`, `dependabot` |
| Container Hardening | Non-root user, read-only filesystem, dropped capabilities |

---

## 6. Scalability Improvements

### 6.1 Concurrency Controls

Add per-provider async semaphores to prevent thundering herds:

```python
class LLMClient:
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate(self, prompt: str) -> str:
        async with self._semaphore:
            return await self._call_provider(prompt)
```

---

### 6.2 Separate Ingestion Workers

**Recommendation:** Deploy ingestion workers separately from query API.

- Ingestion: CPU/IO heavy, can be scaled based on queue depth
- Query API: Latency-sensitive, scale based on request rate

---

### 6.3 Tenant Sharding Strategy

**For large deployments:**

| Component | Sharding Strategy |
|-----------|-------------------|
| Qdrant | Per-tenant collections or partition keys |
| PostgreSQL | Schema-per-tenant or row-level partitioning |
| Neo4j | Tenant labels or separate databases |

---

## 7. Observability Enhancements

### 7.1 OpenTelemetry Spans per Agent

```python
async def _execute_planner(self, state: AgentState) -> AgentState:
    with tracer.start_as_current_span("agent.planner") as span:
        span.set_attribute("query", state.original_query)
        result = await self.planner.execute(state)
        span.set_attribute("strategy", result.execution_plan.retrieval_strategy)
        return result
```

**Spans to add:**
- Per agent (planner, retriever, synthesizer, verifier, critic, citation, formatter)
- External calls (Qdrant, PostgreSQL, Redis, Neo4j)
- LLM calls (with token counts)

---

### 7.2 Request Metrics Middleware

Ensure `/metrics` endpoint is exposed with proper auth:

```python
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

### 7.3 Log Redaction

Ensure logs never emit:
- API keys / JWTs
- Full document text
- User PII

```python
def redact_sensitive(data: dict) -> dict:
    sensitive_keys = {"api_key", "token", "password", "secret"}
    return {
        k: "[REDACTED]" if k.lower() in sensitive_keys else v
        for k, v in data.items()
    }
```

---

## 8. Testing Improvements

### 8.1 Integration Tests for Enterprise Invariants

| Test | Description |
|------|-------------|
| Tenant Isolation | Tenant A cannot retrieve Tenant B documents |
| ACL Enforcement | User without permission cannot access restricted docs |
| Query Persistence | Query history is correctly stored |
| Audit Logging | Admin actions create audit entries |

---

### 8.2 Load/Performance Tests

```python
# Example using locust
class RAGUser(HttpUser):
    @task
    def query(self):
        self.client.post("/api/v1/query", json={
            "query": "What is our vacation policy?"
        })
```

**Metrics to capture:**
- p50, p95, p99 latency
- Throughput (queries/second)
- Error rate under load

---

### 8.3 Chaos/Failure Mode Tests

| Scenario | Expected Behavior |
|----------|-------------------|
| Qdrant down | Graceful error, health check fails |
| Redis down | Degraded mode (no caching) |
| LLM timeout | Fallback to secondary provider |
| Database down | Fail readiness, reject queries |

---

### 8.4 Golden RAG Evaluation Tests

```python
GOLDEN_QA = [
    {
        "query": "What is the refund policy?",
        "expected_sources": ["policies/refund.md"],
        "must_contain": ["30 days", "full refund"],
        "must_not_contain": ["no refunds"],
    },
    ...
]

@pytest.mark.parametrize("qa", GOLDEN_QA)
async def test_golden_qa(qa):
    result = await rag_pipeline.query(qa["query"])
    assert any(s in result.sources for s in qa["expected_sources"])
    for phrase in qa["must_contain"]:
        assert phrase.lower() in result.answer.lower()
```

---

## 9. Production Readiness

### 9.1 Liveness vs Readiness Probes

| Probe | Checks |
|-------|--------|
| Liveness (`/health/live`) | Process is running |
| Readiness (`/health/ready`) | All dependencies reachable, migrations applied |

```python
@router.get("/health/live")
async def liveness():
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness():
    checks = {
        "database": await check_database(),
        "vector_store": await check_qdrant(),
        "cache": await check_redis(),
    }
    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    return JSONResponse({"ready": all_ready, "checks": checks}, status_code=status_code)
```

---

### 9.2 Graceful Shutdown

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_services()
    yield
    # Shutdown
    logger.info("Shutting down, draining requests...")
    await asyncio.sleep(5)  # Allow in-flight requests to complete
    await close_services()
```

---

### 9.3 Database Migrations

Ensure Alembic migrations run at startup or in CI:

```bash
# In entrypoint.sh
alembic upgrade head
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

### 9.4 Backup & Retention Policies

| Component | Backup Strategy | Retention |
|-----------|-----------------|-----------|
| PostgreSQL | Daily snapshots | 30 days |
| Qdrant | Collection snapshots | 7 days |
| Audit Logs | Append-only, archive to S3 | 1 year (compliance) |
| Query History | Postgres table | 90 days (configurable) |

---

## 10. Developer Experience

### 10.1 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
```

---

### 10.2 Local Development Documentation

Add to `docs/guides/local-development.md`:

```markdown
## Quick Start

1. Clone and install:
   ```bash
   git clone <repo>
   cd enterprise-rag
   make setup
   ```

2. Start infrastructure:
   ```bash
   make infra-up
   ```

3. Run the app:
   ```bash
   make dev
   ```

4. Test:
   ```bash
   curl http://localhost:8000/health
   ```

## Environment Variables

Copy `.env.example` to `.env` and configure:
- `OPENAI_API_KEY` - Required for embeddings
- `ANTHROPIC_API_KEY` - Required for LLM

## Troubleshooting

### Postgres won't start
Check if init script is a file (not directory):
```bash
ls -la scripts/setup/init_postgres.sql
```
```

---

## 11. Priority Matrix

### Effort vs Impact

```
                        HIGH IMPACT
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        │  • Fix mutable    │  • Multi-tenancy  │
        │    defaults       │  • ACL enforce-   │
        │  • Fail-fast      │    ment           │
        │    startup        │  • Durable        │
        │  • Timeouts       │    ingestion      │
        │  • Query persist  │  • Safety         │
   LOW  │                   │    controls       │  HIGH
  EFFORT├───────────────────┼───────────────────┤ EFFORT
        │                   │                   │
        │  • Tighten CORS   │  • Model routing  │
        │  • Metrics fix    │  • Golden eval    │
        │  • Liveness/      │  • Tenant         │
        │    readiness      │    sharding       │
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                        LOW IMPACT
```

### Recommended Implementation Order

#### Phase 1: Immediate (Week 1)
- [ ] Fix mutable default in ExecutionTrace
- [ ] Fix metrics environment label
- [ ] Enforce SECRET_KEY in production
- [ ] Tighten CORS configuration
- [ ] Add liveness/readiness probes

#### Phase 2: Short-term (Weeks 2-4)
- [ ] Implement orchestrator timeouts
- [ ] Complete query persistence endpoints
- [ ] Implement feedback storage
- [ ] Add rate limiting
- [ ] Parallelize retrieval operations

#### Phase 3: Medium-term (Months 2-3)
- [ ] Multi-tenancy implementation
- [ ] ACL enforcement in retrieval
- [ ] Durable ingestion job queue
- [ ] OpenTelemetry spans per agent
- [ ] Streaming endpoint hardening

#### Phase 4: Long-term (Months 4-6)
- [ ] Safety controls (prompt injection, PII)
- [ ] Model routing & fallbacks
- [ ] Golden RAG evaluation suite
- [ ] Tenant sharding strategy
- [ ] Advanced audit & compliance features

---

## Appendix: Code Locations Reference

| Component | File |
|-----------|------|
| App startup | `src/app.py` |
| Configuration | `src/core/config.py` |
| Orchestrator | `src/agents/workflows/orchestrator.py` |
| Query routes | `src/api/routes/query.py` |
| Hybrid search | `src/retrieval/search/hybrid.py` |
| Vector store | `src/storage/vector/qdrant.py` |
| RBAC | `src/auth/rbac.py` |
| Metrics | `src/observability/metrics.py` |
| Tracing | `src/observability/tracing.py` |

---

*Document generated: January 2026*
*Last updated: January 2026*
