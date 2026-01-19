# Deployment Models: Multi-Tenant SaaS vs Single-Tenant On-Premise

This document outlines the architecture and implementation strategy to support both multi-tenant SaaS and single-tenant on-premise deployments from a single codebase.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Deployment Modes](#2-deployment-modes)
3. [Architecture Strategy](#3-architecture-strategy)
4. [Configuration-Driven Behavior](#4-configuration-driven-behavior)
5. [Data Isolation Strategies](#5-data-isolation-strategies)
6. [Deployment Packaging](#6-deployment-packaging)
7. [Feature Matrix](#7-feature-matrix)
8. [Implementation Plan](#8-implementation-plan)
9. [Operational Considerations](#9-operational-considerations)

---

## 1. Overview

### The Challenge

Different enterprise clients have different requirements:

| Client Type | Requirement | Reason |
|-------------|-------------|--------|
| **Regulated Industries** (Finance, Healthcare) | On-premise / Private Cloud | Data sovereignty, compliance (HIPAA, SOX) |
| **Government / Defense** | Air-gapped deployment | Security clearance, no external connectivity |
| **SMB / Startups** | Multi-tenant SaaS | Cost efficiency, no infrastructure overhead |
| **Large Enterprises** | Dedicated SaaS tenant | Isolation without operational burden |

### The Solution

A **single codebase** with **configuration-driven deployment modes** that adapts behavior based on:
- `DEPLOYMENT_MODE`: `saas_multi_tenant` | `saas_dedicated` | `on_premise` | `air_gapped`
- `TENANT_ISOLATION`: `shared` | `schema` | `database` | `instance`

---

## 2. Deployment Modes

### 2.1 Multi-Tenant SaaS (Shared Infrastructure)

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG API Cluster                            │
│            (Shared across all tenants)                       │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  PostgreSQL │      │   Qdrant    │      │    Redis    │
│  (Shared +  │      │  (Shared +  │      │  (Shared)   │
│  Row-level  │      │  Filtered)  │      │             │
│  Isolation) │      │             │      │             │
└─────────────┘      └─────────────┘      └─────────────┘
```

**Characteristics:**
- Single deployment serves all tenants
- Row-level data isolation via `tenant_id`
- Shared compute, storage, and caching
- Cost-efficient for provider
- Suitable for: SMBs, startups, low-compliance industries

---

### 2.2 Dedicated SaaS Tenant (Isolated Infrastructure)

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
└─────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────────┐      ┌──────────────────────┐
│   Tenant A Cluster   │      │   Tenant B Cluster   │
│  (Dedicated pods)    │      │  (Dedicated pods)    │
└──────────────────────┘      └──────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────────┐      ┌──────────────────────┐
│  Tenant A Database   │      │  Tenant B Database   │
│  Tenant A Qdrant     │      │  Tenant B Qdrant     │
│  Tenant A Redis      │      │  Tenant B Redis      │
└──────────────────────┘      └──────────────────────┘
```

**Characteristics:**
- Separate infrastructure per tenant (managed by provider)
- Database-level or instance-level isolation
- Tenant-specific scaling and configuration
- Higher cost, premium offering
- Suitable for: Large enterprises, moderate compliance needs

---

### 2.3 On-Premise (Customer Data Center)

```
┌─────────────────────────────────────────────────────────────┐
│                 Customer Data Center                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 RAG System (Docker/K8s)               │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │  │
│  │  │   API   │  │ Workers │  │ Qdrant  │  │Postgres │  │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Customer's Existing Systems              │  │
│  │  (Active Directory, SSO, Document Storage, etc.)      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ (Outbound only, optional)
                   ┌─────────────────┐
                   │  LLM Provider   │
                   │  (OpenAI, etc.) │
                   └─────────────────┘
```

**Characteristics:**
- Deployed in customer's infrastructure
- Customer manages infrastructure (with our support)
- Can integrate with customer's existing systems
- Optional connectivity to cloud LLM providers
- Suitable for: Regulated industries, data sovereignty requirements

---

### 2.4 Air-Gapped (No External Connectivity)

```
┌─────────────────────────────────────────────────────────────┐
│           Customer Secure Environment (Air-Gapped)          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    RAG System                          │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │  │
│  │  │   API   │  │ Workers │  │ Qdrant  │  │Postgres │  │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │  │
│  │                                                        │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │           Local LLM (Llama, Mistral)            │  │  │
│  │  │           Local Embeddings (E5, BGE)            │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│                    ❌ NO INTERNET ACCESS                     │
└─────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Zero external network connectivity
- Local LLM (Llama, Mistral, etc.) and embeddings
- All dependencies bundled
- Manual updates via secure media transfer
- Suitable for: Government, defense, classified environments

---

## 3. Architecture Strategy

### 3.1 Single Codebase, Multiple Behaviors

```python
# src/core/deployment.py

from enum import Enum
from pydantic import BaseModel

class DeploymentMode(str, Enum):
    SAAS_MULTI_TENANT = "saas_multi_tenant"
    SAAS_DEDICATED = "saas_dedicated"
    ON_PREMISE = "on_premise"
    AIR_GAPPED = "air_gapped"

class TenantIsolation(str, Enum):
    SHARED = "shared"           # Row-level isolation (tenant_id filter)
    SCHEMA = "schema"           # Schema-per-tenant in same DB
    DATABASE = "database"       # Separate database per tenant
    INSTANCE = "instance"       # Completely separate deployment

class DeploymentConfig(BaseModel):
    mode: DeploymentMode
    tenant_isolation: TenantIsolation
    
    # Feature flags based on deployment
    enable_cloud_llm: bool = True
    enable_telemetry: bool = True
    enable_auto_updates: bool = True
    enable_usage_reporting: bool = True
    
    # Licensing
    license_key: str | None = None
    license_type: str = "trial"  # trial, standard, enterprise
    
    @classmethod
    def from_env(cls) -> "DeploymentConfig":
        mode = DeploymentMode(os.getenv("DEPLOYMENT_MODE", "saas_multi_tenant"))
        
        # Set defaults based on mode
        if mode == DeploymentMode.AIR_GAPPED:
            return cls(
                mode=mode,
                tenant_isolation=TenantIsolation.INSTANCE,
                enable_cloud_llm=False,
                enable_telemetry=False,
                enable_auto_updates=False,
                enable_usage_reporting=False,
            )
        elif mode == DeploymentMode.ON_PREMISE:
            return cls(
                mode=mode,
                tenant_isolation=TenantIsolation.INSTANCE,
                enable_cloud_llm=True,
                enable_telemetry=False,  # Customer opt-in
                enable_auto_updates=False,
                enable_usage_reporting=False,
            )
        # ... etc
```

---

### 3.2 Tenant Context Middleware

```python
# src/api/middleware/tenant.py

class TenantContextMiddleware:
    """Extract and validate tenant context for every request."""
    
    async def __call__(self, request: Request, call_next):
        deployment = get_deployment_config()
        
        if deployment.mode == DeploymentMode.SAAS_MULTI_TENANT:
            # Extract tenant from JWT, API key, or subdomain
            tenant_id = await self._extract_tenant(request)
            if not tenant_id:
                raise HTTPException(401, "Tenant context required")
        else:
            # Single-tenant: use default tenant
            tenant_id = "default"
        
        # Store in request state for downstream use
        request.state.tenant_id = tenant_id
        request.state.tenant_config = await self._load_tenant_config(tenant_id)
        
        return await call_next(request)
```

---

### 3.3 LLM Provider Abstraction

```python
# src/generation/llm/factory.py

class LLMFactory:
    """Factory that returns appropriate LLM client based on deployment mode."""
    
    @staticmethod
    def create(tenant_config: TenantConfig) -> BaseLLMClient:
        deployment = get_deployment_config()
        
        if deployment.mode == DeploymentMode.AIR_GAPPED:
            # Must use local LLM
            return LocalLLMClient(
                model_path=tenant_config.local_model_path,
                model_type=tenant_config.local_model_type,  # llama, mistral
            )
        
        # Check tenant's allowed providers
        provider = tenant_config.llm_provider
        
        if provider == "anthropic":
            return AnthropicClient(api_key=tenant_config.anthropic_key)
        elif provider == "openai":
            return OpenAIClient(api_key=tenant_config.openai_key)
        elif provider == "local":
            return LocalLLMClient(...)
        elif provider == "azure_openai":
            return AzureOpenAIClient(...)  # For enterprise Azure deployments
        else:
            raise ConfigurationError(f"Unknown LLM provider: {provider}")
```

---

## 4. Configuration-Driven Behavior

### 4.1 Environment Variables

```bash
# ============================================
# Deployment Mode Configuration
# ============================================

# Deployment mode: saas_multi_tenant | saas_dedicated | on_premise | air_gapped
DEPLOYMENT_MODE=saas_multi_tenant

# Tenant isolation: shared | schema | database | instance
TENANT_ISOLATION=shared

# ============================================
# Mode-Specific Settings
# ============================================

# For air-gapped deployments
LOCAL_LLM_ENABLED=true
LOCAL_LLM_MODEL_PATH=/models/llama-3-70b
LOCAL_EMBEDDING_MODEL_PATH=/models/bge-large

# For on-premise deployments
CUSTOMER_SSO_ENABLED=true
CUSTOMER_SSO_PROVIDER=saml  # saml | oidc | ldap
CUSTOMER_SSO_METADATA_URL=https://customer-idp.com/metadata

# Telemetry (disabled for on-premise by default)
TELEMETRY_ENABLED=false
USAGE_REPORTING_ENABLED=false

# Licensing
LICENSE_KEY=ent-xxxxx-xxxxx-xxxxx
LICENSE_ENDPOINT=https://license.ourcompany.com/validate  # Empty for air-gapped
```

---

### 4.2 Tenant Configuration Model

```python
# src/core/tenant.py

class TenantConfig(BaseModel):
    """Per-tenant configuration."""
    
    tenant_id: str
    tenant_name: str
    
    # Isolation
    isolation_level: TenantIsolation
    database_schema: str | None = None      # For schema isolation
    database_url: str | None = None         # For database isolation
    qdrant_collection: str | None = None    # For vector isolation
    
    # LLM Configuration
    llm_provider: str = "anthropic"         # anthropic | openai | azure | local
    llm_model: str = "claude-sonnet-4-20250514"
    anthropic_key: str | None = None
    openai_key: str | None = None
    
    # Limits
    max_documents: int = 10000
    max_queries_per_day: int = 1000
    max_storage_gb: float = 10.0
    token_budget_monthly: int = 1_000_000
    
    # Features
    features_enabled: list[str] = [
        "basic_rag",
        "multi_agent",
        "knowledge_graph",
    ]
    
    # Custom branding (SaaS)
    custom_domain: str | None = None
    logo_url: str | None = None
    
    @classmethod
    def default_single_tenant(cls) -> "TenantConfig":
        """Default config for single-tenant deployments."""
        return cls(
            tenant_id="default",
            tenant_name="Default",
            isolation_level=TenantIsolation.INSTANCE,
            max_documents=999_999_999,  # Unlimited
            max_queries_per_day=999_999_999,
            max_storage_gb=999_999.0,
            features_enabled=["*"],  # All features
        )
```

---

### 4.3 Feature Flags

```python
# src/core/features.py

class FeatureFlags:
    """Feature availability based on deployment and license."""
    
    def __init__(self, deployment: DeploymentConfig, tenant: TenantConfig):
        self.deployment = deployment
        self.tenant = tenant
    
    def is_enabled(self, feature: str) -> bool:
        # Check deployment mode restrictions
        if feature == "cloud_llm" and self.deployment.mode == DeploymentMode.AIR_GAPPED:
            return False
        
        if feature == "telemetry" and not self.deployment.enable_telemetry:
            return False
        
        # Check tenant's enabled features
        if "*" in self.tenant.features_enabled:
            return True
        
        return feature in self.tenant.features_enabled
    
    # Specific feature checks
    @property
    def multi_agent_enabled(self) -> bool:
        return self.is_enabled("multi_agent")
    
    @property
    def knowledge_graph_enabled(self) -> bool:
        return self.is_enabled("knowledge_graph")
    
    @property
    def multimodal_enabled(self) -> bool:
        return self.is_enabled("multimodal")
```

---

## 5. Data Isolation Strategies

### 5.1 Shared Database with Row-Level Isolation

```python
# All tables have tenant_id column
class Document(Base):
    __tablename__ = "documents"
    
    id: Mapped[UUID] = mapped_column(primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(index=True)  # Required
    title: Mapped[str]
    content: Mapped[str]
    
# Repository enforces tenant filter
class DocumentRepository:
    def __init__(self, session: AsyncSession, tenant_id: UUID):
        self.session = session
        self.tenant_id = tenant_id
    
    async def get(self, doc_id: UUID) -> Document | None:
        result = await self.session.execute(
            select(Document).where(
                Document.id == doc_id,
                Document.tenant_id == self.tenant_id  # Always filter
            )
        )
        return result.scalar_one_or_none()
    
    async def list(self) -> list[Document]:
        result = await self.session.execute(
            select(Document).where(
                Document.tenant_id == self.tenant_id  # Always filter
            )
        )
        return result.scalars().all()
```

---

### 5.2 Schema-Per-Tenant Isolation

```python
# Each tenant gets their own schema
async def create_tenant_schema(tenant_id: str):
    schema_name = f"tenant_{tenant_id}"
    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
        # Create tables in tenant schema
        await conn.run_sync(Base.metadata.create_all, schema=schema_name)

# Connection uses tenant's schema
def get_tenant_session(tenant_id: str) -> AsyncSession:
    schema_name = f"tenant_{tenant_id}"
    
    @event.listens_for(engine.sync_engine, "connect")
    def set_search_path(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute(f"SET search_path TO {schema_name}, public")
        cursor.close()
    
    return AsyncSession(engine)
```

---

### 5.3 Qdrant Collection Isolation

```python
# Shared collection with filtering (multi-tenant SaaS)
async def search_shared(query_vector: list[float], tenant_id: str, top_k: int):
    return await qdrant.search(
        collection_name="documents",  # Shared collection
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
            ]
        ),
        limit=top_k,
    )

# Separate collection per tenant (dedicated/on-premise)
async def search_dedicated(query_vector: list[float], tenant_id: str, top_k: int):
    collection_name = f"documents_{tenant_id}"  # Tenant-specific collection
    return await qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )
```

---

### 5.4 Isolation Strategy Selection

```python
# src/storage/isolation.py

class IsolationStrategy(ABC):
    @abstractmethod
    async def get_document_repository(self, tenant_id: str) -> DocumentRepository:
        pass
    
    @abstractmethod
    async def get_vector_store(self, tenant_id: str) -> VectorStore:
        pass

class SharedIsolation(IsolationStrategy):
    """Row-level isolation in shared resources."""
    
    async def get_document_repository(self, tenant_id: str) -> DocumentRepository:
        session = await get_session()
        return DocumentRepository(session, tenant_id)
    
    async def get_vector_store(self, tenant_id: str) -> VectorStore:
        return FilteredVectorStore(
            client=get_qdrant_client(),
            collection="documents",
            tenant_filter=tenant_id,
        )

class DatabaseIsolation(IsolationStrategy):
    """Separate database per tenant."""
    
    async def get_document_repository(self, tenant_id: str) -> DocumentRepository:
        tenant_config = await get_tenant_config(tenant_id)
        engine = create_async_engine(tenant_config.database_url)
        session = AsyncSession(engine)
        return DocumentRepository(session, tenant_id)
    
    async def get_vector_store(self, tenant_id: str) -> VectorStore:
        tenant_config = await get_tenant_config(tenant_id)
        return QdrantVectorStore(
            url=tenant_config.qdrant_url,
            collection=tenant_config.qdrant_collection,
        )

def get_isolation_strategy() -> IsolationStrategy:
    deployment = get_deployment_config()
    
    if deployment.tenant_isolation == TenantIsolation.SHARED:
        return SharedIsolation()
    elif deployment.tenant_isolation == TenantIsolation.SCHEMA:
        return SchemaIsolation()
    elif deployment.tenant_isolation == TenantIsolation.DATABASE:
        return DatabaseIsolation()
    else:
        return InstanceIsolation()
```

---

## 6. Deployment Packaging

### 6.1 Docker Images

```dockerfile
# Dockerfile.saas - Optimized for multi-tenant SaaS
FROM python:3.11-slim
# ... standard setup ...
ENV DEPLOYMENT_MODE=saas_multi_tenant

# Dockerfile.enterprise - For on-premise deployment
FROM python:3.11-slim
# Include local model support
RUN pip install llama-cpp-python sentence-transformers
# Pre-download common models
COPY models/ /app/models/
ENV DEPLOYMENT_MODE=on_premise
ENV LOCAL_LLM_ENABLED=true

# Dockerfile.airgapped - Everything bundled
FROM python:3.11-slim
# All dependencies bundled, no network calls
COPY wheels/ /wheels/
RUN pip install --no-index --find-links=/wheels -r requirements.txt
COPY models/ /app/models/
ENV DEPLOYMENT_MODE=air_gapped
ENV LOCAL_LLM_ENABLED=true
```

---

### 6.2 Helm Chart Values

```yaml
# values-saas.yaml
deployment:
  mode: saas_multi_tenant
  replicas: 3
  
tenancy:
  isolation: shared
  
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"

# Shared infrastructure
postgresql:
  enabled: true
  primary:
    persistence:
      size: 100Gi

qdrant:
  enabled: true
  persistence:
    size: 50Gi

# values-enterprise.yaml
deployment:
  mode: on_premise
  replicas: 2
  
tenancy:
  isolation: instance
  
# Customer provides their own infrastructure
postgresql:
  enabled: false
  external:
    host: customer-postgres.internal
    
qdrant:
  enabled: false
  external:
    host: customer-qdrant.internal

# Local LLM support
localLLM:
  enabled: true
  modelPath: /models/llama-3-70b
  gpuEnabled: true
```

---

### 6.3 Deployment Scripts

```bash
# scripts/deploy-enterprise.sh
#!/bin/bash
# Deploy to customer's Kubernetes cluster

set -e

CUSTOMER_NAME=$1
NAMESPACE="rag-${CUSTOMER_NAME}"

# Create namespace
kubectl create namespace $NAMESPACE || true

# Apply customer-specific secrets
kubectl apply -f "customers/${CUSTOMER_NAME}/secrets.yaml" -n $NAMESPACE

# Deploy with enterprise values
helm upgrade --install rag ./helm/enterprise-rag \
  --namespace $NAMESPACE \
  -f helm/values-enterprise.yaml \
  -f "customers/${CUSTOMER_NAME}/values.yaml" \
  --set license.key="${LICENSE_KEY}"

echo "Deployed to ${CUSTOMER_NAME}"
```

---

## 7. Feature Matrix

### By Deployment Mode

| Feature | Multi-Tenant SaaS | Dedicated SaaS | On-Premise | Air-Gapped |
|---------|-------------------|----------------|------------|------------|
| Cloud LLM (OpenAI, Anthropic) | ✅ | ✅ | ✅ (optional) | ❌ |
| Local LLM (Llama, Mistral) | ❌ | ✅ | ✅ | ✅ |
| Auto Updates | ✅ | ✅ | ❌ (manual) | ❌ (manual) |
| Telemetry | ✅ | ✅ (opt-out) | ❌ (opt-in) | ❌ |
| Multi-Agent | ✅ | ✅ | ✅ | ✅ |
| Knowledge Graph | ✅ | ✅ | ✅ | ✅ |
| SSO Integration | SAML/OIDC | SAML/OIDC | SAML/OIDC/LDAP/AD | LDAP/AD |
| Custom Domain | ✅ | ✅ | N/A | N/A |
| Data Residency | Provider region | Customer choice | Customer DC | Customer DC |

---

### By License Tier

| Feature | Trial | Standard | Enterprise |
|---------|-------|----------|------------|
| Documents | 100 | 10,000 | Unlimited |
| Queries/day | 50 | 1,000 | Unlimited |
| Users | 5 | 50 | Unlimited |
| Multi-Agent | ❌ | ✅ | ✅ |
| Knowledge Graph | ❌ | ❌ | ✅ |
| Priority Support | ❌ | ✅ | ✅ |
| On-Premise | ❌ | ❌ | ✅ |
| Air-Gapped | ❌ | ❌ | ✅ |
| Custom SLA | ❌ | ❌ | ✅ |

---

## 8. Implementation Plan

### Phase 1: Foundation (2-3 weeks)

- [ ] Add `DEPLOYMENT_MODE` and `TENANT_ISOLATION` configuration
- [ ] Implement `DeploymentConfig` and `TenantConfig` models
- [ ] Add `tenant_id` to all database models
- [ ] Implement tenant context middleware
- [ ] Add row-level filtering to all repositories

### Phase 2: Multi-Tenant SaaS (2-3 weeks)

- [ ] Implement shared database isolation
- [ ] Add Qdrant tenant filtering
- [ ] Tenant onboarding API
- [ ] Usage tracking and limits
- [ ] Multi-tenant admin dashboard

### Phase 3: Enterprise Packaging (3-4 weeks)

- [ ] Create enterprise Docker images
- [ ] Helm chart with enterprise values
- [ ] Local LLM integration (Llama, Mistral)
- [ ] Local embedding models (BGE, E5)
- [ ] SSO integration (SAML, LDAP)

### Phase 4: Air-Gapped Support (2-3 weeks)

- [ ] Bundle all dependencies (offline wheels)
- [ ] Offline license validation
- [ ] Manual update mechanism
- [ ] Air-gapped installation documentation
- [ ] Offline model packaging

### Phase 5: Operations (Ongoing)

- [ ] Customer deployment runbooks
- [ ] Monitoring and alerting templates
- [ ] Backup/restore procedures per deployment type
- [ ] Upgrade procedures

---

## 9. Operational Considerations

### 9.1 Licensing

```python
# src/licensing/validator.py

class LicenseValidator:
    async def validate(self) -> LicenseInfo:
        deployment = get_deployment_config()
        
        if deployment.mode == DeploymentMode.AIR_GAPPED:
            # Offline validation using signed license file
            return await self._validate_offline()
        else:
            # Online validation
            return await self._validate_online()
    
    async def _validate_offline(self) -> LicenseInfo:
        """Validate license without network access."""
        license_file = Path("/etc/rag/license.key")
        if not license_file.exists():
            raise LicenseError("License file not found")
        
        # Verify signature and expiration
        license_data = self._decrypt_and_verify(license_file.read_text())
        
        if license_data.expires_at < datetime.utcnow():
            raise LicenseError("License expired")
        
        return license_data
```

---

### 9.2 Updates

| Deployment | Update Mechanism |
|------------|------------------|
| SaaS | Automatic rolling updates |
| On-Premise | Customer-initiated via Helm upgrade |
| Air-Gapped | Manual transfer of Docker images + Helm charts |

```bash
# Air-gapped update package
tar -czvf rag-update-v1.2.0.tar.gz \
  docker-images/ \
  helm-charts/ \
  models/ \
  UPGRADE.md \
  checksums.sha256

# Customer runs
tar -xzvf rag-update-v1.2.0.tar.gz
./upgrade.sh
```

---

### 9.3 Support Tiers

| Tier | SaaS Multi-Tenant | Dedicated/On-Premise | Air-Gapped |
|------|-------------------|----------------------|------------|
| Self-Service | ✅ Docs, Community | ✅ Docs | ✅ Offline docs |
| Email Support | 48h response | 24h response | 24h response |
| Priority Support | — | 4h response | 4h response + on-site |
| Dedicated CSM | — | ✅ | ✅ |

---

### 9.4 Monitoring

```yaml
# For SaaS: Centralized monitoring
monitoring:
  provider: datadog
  api_key: ${DATADOG_API_KEY}
  
# For On-Premise: Customer's monitoring or bundled
monitoring:
  provider: prometheus
  scrape_interval: 15s
  alertmanager:
    enabled: true
    
# For Air-Gapped: Local-only monitoring
monitoring:
  provider: prometheus
  retention: 30d
  alerting:
    method: local  # No external webhooks
```

---

## Summary

By implementing configuration-driven deployment modes, the Enterprise RAG system can serve:

1. **Cost-conscious customers** → Multi-tenant SaaS with shared infrastructure
2. **Security-conscious enterprises** → Dedicated SaaS with isolated resources  
3. **Regulated industries** → On-premise deployment in customer data centers
4. **Government/Defense** → Air-gapped deployment with local LLMs

All from a **single codebase** with **behavior controlled by configuration**.

---

*Document created: January 2026*
