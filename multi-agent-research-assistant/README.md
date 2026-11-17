# Multi-Agent Research Assistant

Production-grade supervisor-worker research platform that produces citation-validated reports with hard per-run cost caps, Redis checkpoint recovery across process restarts, and 100% step-level execution tracing.

## Verification & Suite Metrics

*Note: All metrics below are measured directly from the deterministic automated test suite (`pytest -v`) and code-enforced guardrails; live API latency depends on Anthropic and Tavily upstream response times.*

| Capability / Guardrail | Enforcement Mechanism | Verification Status |
| :--- | :--- | :--- |
| **Hard Cost & Token Ceilings** | `BudgetService` evaluated prior to every LLM/search call (`MAX_COST_USD`, `MAX_TOTAL_TOKENS`) | Verified (`test_budget_enforces_cost_and_token_ceilings`) |
| **Infinite Loop Prevention** | Deterministic state counter (`revisions_by_sq`) capped at `MAX_REVISION_ATTEMPTS=1` | Verified (`test_supervisor_enforces_single_revision_limit`) |
| **Crash & Restart Recovery** | Full `RunCheckpoint` persisted to Redis after every LangGraph node + node-level idempotency cache | Verified (`test_crashed_run_resumes_from_redis_without_replaying_completed_nodes`) |
| **Source Provenance & Domain Caps** | URL canonicalization (tracking param stripping), `MAX_RESULTS_PER_DOMAIN=2`, full-page extraction | Verified (`test_url_canonicalization_and_domain_cap`, `test_researcher_handles_paywall_and_extracts_valid_source`) |
| **Zero-Hallucination Citations** | Regex post-processor strips any `[F-XXX]` tag absent from the verified `FindingRecord` registry | Verified (`test_citation_validator_strips_hallucinated_ids`) |
| **Step-Level Observability** | `GET /research/{id}/trace` exposes per-step inputs, outputs, tool calls, tokens, cost, and latency | Verified (`test_api_auth_rejection_and_end_to_end_run`) |

---

## System Architecture

```text
+-------------------+       POST /research (202)        +-----------------------+
|                   | --------------------------------> |                       |
|   API Client      |       GET /research/{id}          |     FastAPI Layer     |
|                   | <-------------------------------- |  (Auth, Rate Limit,   |
|                   |       GET /research/{id}/events   |   SSE Event Stream)   |
+-------------------+ <======== (SSE Stream) ========== +-----------+-----------+
                                                                    |
                                                          Schedule / Recover
                                                                    v
+-------------------------------------------------------------------+-----------+
|                        LangGraph State Machine (ResearchOrchestrator)          |
|                                                                               |
|  [Planner] ---> [Select Sub-Question] ---> [Researcher] ---> [Supervisor]     |
|      |                    |                     |                 |           |
|      |                    |                     |        (Approve / 1x Revise)|
|      |                    +--- (All Done / Budget Cap) ---> [Writer] ---> END |
+------+--------------------+---------------------+-----------------+-----------+
       |                    |                     |                 |
       +--------------------+---------------------+-----------------+
                            |  Persist Checkpoint & Node Cache after EVERY node
                            v
              +-------------+-------------+       +-----------------------------+
              |      Redis Persistence    |       |      External Services      |
              | - Run State Snapshots     |       | - Claude API (Tool Schemas) |
              | - Idempotent Node Outputs |       | - Tavily Search API         |
              | - Event List + Pub/Sub    |       | - Trafilatura Web Extractor |
              +---------------------------+       +-----------------------------+
```

---

## Key Design Decisions & Trade-Offs

1. **Forced Tool-Calling Schemas Over Persona Prompts**
   * **Decision:** Every specialist (`PlannerAgent`, `ResearcherAgent`, `SupervisorAgent`, `WriterAgent`) uses Anthropic's forced `tool_choice` mapped to strict Pydantic v2 models rather than free-form text or persona prompts.
   * **Trade-Off:** Slightly higher input token overhead per request due to JSON schema injection in exchange for deterministic hand-offs, zero regex parsing failures, and strict structural validation.
2. **Code-Enforced Budgets & Single-Revision Cap Over Prompt Instructions**
   * **Decision:** Token ceilings, USD cost caps, search quotas, wall-clock timers, and the 1-revision limit are enforced inside `BudgetService` and LangGraph conditional edges. When a budget cap triggers mid-research, the graph routes gracefully to `WriterAgent` rather than crashing so partial verified findings still produce a report.
   * **Trade-Off:** A run capped early yields narrower coverage, but runaway $12+ agent loops are mathematically impossible.
3. **Full-Page Extraction With Explicit Failure Enums Over Search Snippets**
   * **Decision:** `ResearcherAgent` fetches candidate URLs via `httpx` + `trafilatura` and maps HTTP 401/403/paywall markers, 429 rate limits, and timeouts into typed `SourceStatus` records instead of raising exceptions.
   * **Trade-Off:** Increases per-sub-question latency compared to trusting Tavily snippets, but prevents out-of-context snippet hallucinations and lets the Supervisor pivot queries when sources are paywalled.

---

## What Did Not Work & How It Was Fixed

1. **SSE Clients Missing Early Events on Fast Runs**
   * **Problem:** Using Redis Pub/Sub alone meant clients connecting to `GET /research/{id}/events` a few hundred milliseconds after `POST /research` missed `research_initialized` and `planning_completed` events.
   * **Fix:** Replaced pure Pub/Sub with a dual-write pattern in `RedisPersistenceService.emit_event` (`RPUSH` to a TTL-bounded Redis List + `PUBLISH` to a channel). `subscribe_events` replays the List history first before listening to live Pub/Sub messages.
2. **Subtle Duplicate URLs Bypassing Naive Set Checks**
   * **Problem:** Search engines frequently returned the same article with differing query parameters (`?utm_source=...`), `http` vs `https`, or `www.` prefixes, exhausting the per-domain budget on identical content.
   * **Fix:** Added `SourceService.canonicalize_url` to normalize scheme/host, strip `www.`, sort query parameters, and remove tracking keys (`utm_*`, `fbclid`, `gclid`) prior to deduplication and domain counting.
3. **LLM Hallucinating Plausible Citation Numbers (`[F-004]` when only 3 findings existed)**
   * **Problem:** Even with temperature `0.1` and strict instructions, the Writer occasionally synthesized two findings and invented an adjacent citation tag.
   * **Fix:** Added deterministic post-generation sanitization (`SourceService.validate_and_clean_citations`) that strips any `[F-XXX]` tag not present in the run's `FindingRecord` IDs and logs stripped IDs in `ResearchReport.invalid_citations_stripped` for trace inspection.

---

## Quickstart & Local Execution

### 1. Environment Setup
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and TAVILY_API_KEY
```

### 2. Run with Docker Compose
```bash
docker compose up --build -d
curl http://localhost:8000/health
```

### 3. Run Automated Test Suite
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

### 4. Submit a Research Run & Inspect Trace
```bash
# Start research
curl -X POST http://localhost:8000/research \
  -H "Authorization: Bearer dev-secret-token-change-in-prod" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the architectural trade-offs between Redis AOF and RDB persistence?"}'

# Stream real-time SSE progress
curl -N -H "Authorization: Bearer dev-secret-token-change-in-prod" \
  http://localhost:8000/research/<RUN_ID>/events

# Fetch completed report
curl -H "Authorization: Bearer dev-secret-token-change-in-prod" \
  http://localhost:8000/research/<RUN_ID>

# Inspect step-by-step execution trace (tokens, latency, tool calls, supervisor decisions)
curl -H "Authorization: Bearer dev-secret-token-change-in-prod" \
  http://localhost:8000/research/<RUN_ID>/trace
```
