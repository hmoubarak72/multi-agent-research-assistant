# Multi-Agent Research Assistant

Production-grade supervisor-worker research platform that produces citation-validated reports with hard per-run cost caps, Redis checkpoint recovery across process restarts, and 100% step-level execution tracing.

## Verification & Suite Metrics

*Note: All metrics below are measured directly from the deterministic automated test suite (`pytest -v`) and code-enforced guardrails; live API latency depends on Anthropic and Tavily upstream response times.*

| Capability / Guardrail | Enforcement Mechanism | Verification Status |
| :--- | :--- | :--- |
| **Hard Cost & Token Ceilings** | `BudgetService` evaluated prior to every LLM/search call (`MAX_COST_USD`, `MAX_TOTAL_TOKENS`) | Verified (`test_budget_enforces_cost_and_token_ceilings`)[cite: 2] |
| **Infinite Loop Prevention** | Deterministic state counter (`revisions_by_sq`) capped at `MAX_REVISION_ATTEMPTS=1` | Verified (`test_supervisor_enforces_single_revision_limit`)[cite: 2] |
| **Crash & Restart Recovery** | Full `RunCheckpoint` persisted to Redis after every LangGraph node + node-level idempotency cache | Verified (`test_crashed_run_resumes_from_redis_without_replaying_completed_nodes`)[cite: 2] |
| **Source Provenance & Domain Caps** | URL canonicalization (tracking param stripping), `MAX_RESULTS_PER_DOMAIN=2`, full-page extraction | Verified (`test_url_canonicalization_and_domain_cap`, `test_researcher_handles_paywall_and_extracts_valid_source`)[cite: 2] |
| **Zero-Hallucination Citations** | Regex post-processor strips any `[F-XXX]` tag absent from the verified `FindingRecord` registry | Verified (`test_citation_validator_strips_hallucinated_ids`)[cite: 2] |
| **Step-Level Observability** | `GET /research/{id}/trace` exposes per-step inputs, outputs, tool calls, tokens, cost, and latency | Verified (`test_api_auth_rejection_and_end_to_end_run`)[cite: 2] |

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