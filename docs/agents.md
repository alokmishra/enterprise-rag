# Multi-Agent RAG System

## Overview

The Enterprise RAG system uses a multi-agent architecture where specialized agents collaborate to process queries and generate high-quality, verified responses. Each agent has a specific responsibility in the pipeline, communicating through structured messages and sharing state.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                    │
│                    (Routes, coordinates, manages flow)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                PLANNER                                       │
│              (Analyzes query, creates execution plan)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌───────────┐    ┌───────────┐    ┌───────────┐
            │ RETRIEVER │    │RESEARCHER │    │   GRAPH   │
            │  (Vector) │    │  (Multi-  │    │  (Knowledge│
            │           │    │   hop)    │    │   Graph)  │
            └───────────┘    └───────────┘    └───────────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             SYNTHESIZER                                      │
│                  (Generates response from context)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌───────────┐    ┌───────────┐    ┌───────────┐
            │ VERIFIER  │    │  CITATION │    │  CRITIC   │
            │  (Fact-   │    │  (Source  │    │ (Quality  │
            │  check)   │    │  linking) │    │  review)  │
            └───────────┘    └───────────┘    └───────────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                              ┌───────────┐
                              │ FORMATTER │
                              │ (Output   │
                              │  format)  │
                              └───────────┘
```

## Agent Types

| Agent | Type Enum | Responsibility |
|-------|-----------|----------------|
| Orchestrator | `ORCHESTRATOR` | Routes requests, manages workflow, handles errors |
| Planner | `PLANNER` | Analyzes query complexity, creates execution plan |
| Retriever | `RETRIEVER` | Vector/hybrid search, retrieves relevant chunks |
| Researcher | `RESEARCHER` | Multi-hop reasoning, complex query decomposition |
| Synthesizer | `SYNTHESIZER` | Generates coherent response from context |
| Verifier | `VERIFIER` | Fact-checks claims against source documents |
| Critic | `CRITIC` | Evaluates response quality, requests revisions |
| Citation | `CITATION` | Links claims to sources, formats citations |
| Formatter | `FORMATTER` | Formats final output (markdown, JSON, etc.) |

---

## Agent Specifications

### 1. Orchestrator Agent

**Purpose**: Central coordinator that manages the entire query-response pipeline.

**Responsibilities**:
- Receive incoming queries
- Route to appropriate agents based on query type
- Manage execution flow and agent handoffs
- Handle timeouts and errors
- Aggregate results from parallel agent executions
- Enforce token budgets

**Inputs**:
- User query
- Conversation history
- Configuration (timeout, token budget)

**Outputs**:
- Final formatted response
- Execution trace/metadata

**Key Decisions**:
- Simple vs complex query routing
- Parallel vs sequential execution
- Retry logic on failures
- Early termination conditions

---

### 2. Planner Agent

**Purpose**: Analyzes query complexity and creates an execution plan.

**Responsibilities**:
- Classify query complexity (simple, standard, complex)
- Determine required retrieval strategies
- Decompose complex queries into sub-queries
- Estimate resource requirements
- Create step-by-step execution plan

**Inputs**:
- Original query
- Conversation history
- Available retrieval strategies

**Outputs**:
- `QueryComplexity` classification
- Execution plan with ordered steps
- Sub-queries (if decomposed)
- Recommended `RetrievalStrategy`

**Complexity Classification**:
| Level | Description | Example |
|-------|-------------|---------|
| `SIMPLE` | Direct lookup, single fact | "What is the capital of France?" |
| `STANDARD` | Multi-source, synthesis needed | "Compare our Q3 and Q4 revenue" |
| `COMPLEX` | Multi-hop, reasoning required | "How did market trends affect our product strategy?" |

---

### 3. Retriever Agent

**Purpose**: Retrieves relevant context from vector stores and search indices.

**Responsibilities**:
- Execute vector similarity search
- Apply metadata filters
- Perform hybrid search (vector + sparse)
- Deduplicate results
- Score and rank retrieved chunks

**Inputs**:
- Query (or sub-queries)
- Retrieval strategy
- Filters (date range, source, etc.)
- Top-k limit

**Outputs**:
- `RetrievalResult` with ranked `SearchResult` items
- Retrieval latency metrics

**Retrieval Strategies**:
| Strategy | Use Case |
|----------|----------|
| `VECTOR` | Semantic similarity search |
| `SPARSE` | Keyword/BM25 search |
| `HYBRID` | Combined vector + sparse |
| `GRAPH` | Knowledge graph traversal |
| `MULTI_QUERY` | Multiple query variations |
| `HYDE` | Hypothetical document embeddings |

---

### 4. Researcher Agent

**Purpose**: Handles complex queries requiring multi-hop reasoning.

**Responsibilities**:
- Decompose complex queries into reasoning steps
- Execute iterative retrieval-reasoning loops
- Synthesize intermediate findings
- Track reasoning chain
- Identify knowledge gaps

**Inputs**:
- Complex query
- Execution plan from Planner
- Initial context (if any)

**Outputs**:
- Enriched context with reasoning steps
- Intermediate findings
- Confidence scores per finding

**Reasoning Patterns**:
- **Bridge reasoning**: A→B, B→C, therefore A→C
- **Comparison**: Retrieve and compare multiple entities
- **Aggregation**: Collect and summarize across sources
- **Temporal**: Track changes over time

---

### 5. Synthesizer Agent

**Purpose**: Generates coherent responses from retrieved context.

**Responsibilities**:
- Assemble context into prompt
- Generate response using LLM
- Maintain conversation coherence
- Handle multi-turn context
- Apply response guidelines

**Inputs**:
- Original query
- Retrieved context (`ContextItem` list)
- Conversation history
- Response format preferences

**Outputs**:
- `GeneratedResponse` with content
- Token usage metrics
- Confidence score

**Generation Modes**:
| Mode | Description |
|------|-------------|
| Extractive | Pull exact quotes from sources |
| Abstractive | Synthesize new text from sources |
| Hybrid | Combine both approaches |

---

### 6. Verifier Agent

**Purpose**: Fact-checks claims in the generated response.

**Responsibilities**:
- Extract claims from response
- Match claims to source evidence
- Classify verification status
- Flag unsupported claims
- Calculate confidence scores

**Inputs**:
- Generated response
- Retrieved context
- Source documents

**Outputs**:
- List of `VerificationResult` items
- Overall verification score
- Flagged claims needing revision

**Verification Status**:
| Status | Meaning |
|--------|---------|
| `SUPPORTED` | Claim has direct evidence |
| `PARTIALLY_SUPPORTED` | Claim has some evidence |
| `NOT_FOUND` | No evidence found |
| `CONTRADICTED` | Evidence contradicts claim |
| `UNCERTAIN` | Ambiguous or conflicting evidence |

---

### 7. Critic Agent

**Purpose**: Evaluates response quality and determines if revisions are needed.

**Responsibilities**:
- Score response on multiple dimensions
- Identify quality issues
- Generate improvement suggestions
- Decide on pass/revise/reject
- Enforce quality thresholds

**Inputs**:
- Generated response
- Verification results
- Original query
- Quality thresholds

**Outputs**:
- `CriticFeedback` with scores and decision
- Specific suggestions for improvement

**Scoring Dimensions**:
| Dimension | Description |
|-----------|-------------|
| `relevance_score` | How well response addresses query |
| `completeness_score` | Coverage of query aspects |
| `accuracy_score` | Factual correctness |
| `coherence_score` | Logical flow and clarity |
| `citation_score` | Proper source attribution |
| `overall_score` | Weighted aggregate |

**Decisions**:
| Decision | Action |
|----------|--------|
| `PASS` | Response is ready for output |
| `MINOR_REVISION` | Small edits needed, no re-retrieval |
| `MAJOR_REVISION` | Significant rewrite needed |
| `RETRIEVAL_NEEDED` | Need more context, back to retriever |
| `REJECT` | Cannot generate acceptable response |

---

### 8. Citation Agent

**Purpose**: Links response claims to source documents.

**Responsibilities**:
- Match text spans to source chunks
- Generate citation references
- Format citation links
- Verify citation accuracy
- Handle multiple citation styles

**Inputs**:
- Generated response
- Retrieved chunks with metadata
- Citation format preference

**Outputs**:
- Response with inline citations
- List of `Citation` objects
- Citation index/bibliography

**Citation Formats**:
- Inline numeric: `[1]`, `[2]`
- Inline author-date: `(Smith, 2024)`
- Footnotes
- Hyperlinks

---

### 9. Formatter Agent

**Purpose**: Formats the final response for output.

**Responsibilities**:
- Apply output format (markdown, JSON, plain text)
- Structure response sections
- Format code blocks and tables
- Apply branding/styling
- Truncate if needed

**Inputs**:
- Verified response with citations
- Output format specification
- Length constraints

**Outputs**:
- Final formatted response
- Metadata (token count, sources used)

---

## Communication

### Message Types

Agents communicate via `AgentMessage`:

```python
class AgentMessage(BaseModel):
    message_id: str          # Unique message ID
    trace_id: str            # Request trace ID
    from_agent: AgentType    # Sender
    to_agent: AgentType      # Recipient
    message_type: MessageType  # REQUEST, RESPONSE, FEEDBACK, ERROR
    payload: dict[str, Any]  # Message content
    metadata: dict[str, Any] # Additional metadata
    timestamp: datetime      # When sent
```

### Message Flow Example

```
1. User Query → Orchestrator
2. Orchestrator → Planner (REQUEST: analyze query)
3. Planner → Orchestrator (RESPONSE: execution plan)
4. Orchestrator → Retriever (REQUEST: fetch context)
5. Retriever → Orchestrator (RESPONSE: search results)
6. Orchestrator → Synthesizer (REQUEST: generate response)
7. Synthesizer → Orchestrator (RESPONSE: draft response)
8. Orchestrator → Verifier (REQUEST: verify claims)
9. Verifier → Orchestrator (RESPONSE: verification results)
10. Orchestrator → Critic (REQUEST: evaluate quality)
11. Critic → Orchestrator (RESPONSE: feedback/decision)
    - If PASS: continue to Citation
    - If REVISION: back to Synthesizer with feedback
12. Orchestrator → Citation (REQUEST: add citations)
13. Citation → Orchestrator (RESPONSE: cited response)
14. Orchestrator → Formatter (REQUEST: format output)
15. Formatter → Orchestrator (RESPONSE: final response)
16. Orchestrator → User
```

---

## Shared State

Agents share context through `AgentState`:

```python
class AgentState(BaseModel):
    trace_id: str                              # Unique request ID
    original_query: str                        # User's query
    conversation_history: list[dict]           # Prior turns
    execution_plan: Optional[dict]             # From Planner
    retrieved_context: list[ContextItem]       # From Retriever
    draft_responses: list[str]                 # From Synthesizer
    verification_results: list[dict]           # From Verifier
    critic_feedback: list[dict]                # From Critic
    iteration_count: int                       # Revision iterations
    token_budget_remaining: int                # Token budget left
```

---

## Base Agent Implementation

All agents extend `BaseAgent`:

```python
class BaseAgent(ABC, LoggerMixin):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.agent_type = config.agent_type

    @abstractmethod
    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """Execute the agent's primary task."""
        pass

    def create_message(self, to_agent, message_type, payload, trace_id) -> AgentMessage:
        """Create a message to send to another agent."""
        pass

    async def validate_input(self, state: AgentState) -> bool:
        """Validate required input exists."""
        pass

    async def on_error(self, error: Exception, state: AgentState) -> None:
        """Handle execution errors."""
        pass
```

### Agent Configuration

```python
class AgentConfig(BaseModel):
    name: str                    # Agent instance name
    agent_type: AgentType        # Type enum
    model: Optional[str]         # LLM model to use
    temperature: float = 0.0     # LLM temperature
    max_tokens: int = 4096       # Max output tokens
    timeout_seconds: int = 60    # Execution timeout
```

### Agent Result

```python
class AgentResult(BaseModel):
    success: bool           # Whether execution succeeded
    output: Any             # Result payload
    error: Optional[str]    # Error message if failed
    tokens_used: int        # Tokens consumed
    latency_ms: float       # Execution time
```

---

## Workflows

### Standard Query Workflow

```
Planner → Retriever → Synthesizer → Verifier → Critic → Citation → Formatter
```

### Complex Query Workflow

```
Planner → Researcher ─┬→ Retriever (sub-query 1)
                      ├→ Retriever (sub-query 2)
                      └→ Graph Query
         ↓
    Synthesizer → Verifier → Critic ─┬→ PASS → Citation → Formatter
                                     └→ REVISE → Synthesizer (loop)
```

### Iterative Refinement

```
while iteration_count < max_iterations:
    Synthesizer → Verifier → Critic
    if decision == PASS:
        break
    elif decision in [MINOR_REVISION, MAJOR_REVISION]:
        incorporate feedback, continue
    elif decision == RETRIEVAL_NEEDED:
        Retriever → continue
    elif decision == REJECT:
        return error response
```

---

## Implementation Order (Phase 4)

```
Week 21-22: agents/base.py (✓), agents/planner/
Week 23-24: agents/retriever/, agents/researcher/
Week 25-26: agents/synthesizer/, agents/verifier/
Week 27-28: agents/critic/, agents/citation/, agents/formatter/
            agents/workflows/, agents/communication/
```

---

## Directory Structure

```
src/agents/
├── __init__.py
├── base.py                 # BaseAgent, AgentConfig, AgentResult
├── planner/
│   ├── __init__.py
│   └── planner_agent.py    # PlannerAgent implementation
├── retriever/
│   ├── __init__.py
│   └── retriever_agent.py  # RetrieverAgent implementation
├── researcher/
│   ├── __init__.py
│   └── researcher_agent.py # ResearcherAgent implementation
├── synthesizer/
│   ├── __init__.py
│   └── synthesizer_agent.py
├── verifier/
│   ├── __init__.py
│   └── verifier_agent.py
├── critic/
│   ├── __init__.py
│   └── critic_agent.py
├── citation/
│   ├── __init__.py
│   └── citation_agent.py
├── formatter/
│   ├── __init__.py
│   └── formatter_agent.py
├── communication/
│   ├── __init__.py
│   └── message_bus.py      # Inter-agent messaging
└── workflows/
    ├── __init__.py
    ├── standard.py         # Standard query workflow
    ├── complex.py          # Complex query workflow
    └── orchestrator.py     # Main orchestrator
```

---

## Configuration Example

```yaml
agents:
  planner:
    model: "gpt-4o-mini"
    temperature: 0.0
    timeout_seconds: 30

  retriever:
    model: null  # No LLM needed
    timeout_seconds: 10

  synthesizer:
    model: "gpt-4o"
    temperature: 0.3
    max_tokens: 4096

  verifier:
    model: "gpt-4o-mini"
    temperature: 0.0

  critic:
    model: "gpt-4o"
    temperature: 0.0
    thresholds:
      relevance: 0.7
      completeness: 0.6
      accuracy: 0.8
      overall: 0.7

  orchestrator:
    max_iterations: 3
    token_budget: 32000
    timeout_seconds: 120
```

---

## Error Handling

Each agent implements `on_error()` for graceful error handling:

| Error Type | Handling Strategy |
|------------|-------------------|
| Timeout | Return partial result, log warning |
| LLM Error | Retry with backoff, fallback model |
| Retrieval Empty | Expand query, try alternate strategy |
| Verification Failed | Flag response, add disclaimer |
| Token Budget Exceeded | Truncate context, summarize |

---

## Observability

All agents emit structured logs and metrics:

- **Logs**: Trace ID, agent name, action, duration, errors
- **Metrics**: Latency, token usage, success rate, quality scores
- **Traces**: Full execution path with timing

See `src/observability/` for implementation details.
