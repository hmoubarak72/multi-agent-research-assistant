import time
from app.models.findings import SourceMetadata, SourceStatus
from app.services.budget_service import BudgetService
from app.services.source_service import SourceService


def test_budget_enforces_cost_and_token_ceilings(budget_service: BudgetService):
    budget = budget_service.initialize_budget({"max_cost_usd": 0.05})
    allowed, reason = budget_service.check_execution_allowed(budget)
    assert allowed is True
    assert reason is None

    # Simulate heavy output token consumption exceeding $0.05 ($15/1M output tokens -> 4000 tokens = $0.06)
    step_cost = budget_service.record_llm_usage(budget, input_tokens=1000, output_tokens=4000)
    assert step_cost == 0.063
    allowed_after, reason_after = budget_service.check_execution_allowed(budget)
    assert allowed_after is False
    assert "Cost ceiling reached" in (reason_after or "")


def test_budget_enforces_search_caps(budget_service: BudgetService):
    budget = budget_service.initialize_budget({"max_searches_per_sub_question": 2})
    assert budget_service.can_execute_search(budget, "SQ-1")[0] is True
    budget_service.record_search(budget, "SQ-1")
    assert budget_service.can_execute_search(budget, "SQ-1")[0] is True
    budget_service.record_search(budget, "SQ-1")

    allowed, reason = budget_service.can_execute_search(budget, "SQ-1")
    assert allowed is False
    assert "Sub-question search cap reached" in (reason or "")


def test_budget_enforces_wall_clock_ceiling(budget_service: BudgetService):
    budget = budget_service.initialize_budget({"max_duration_seconds": 10})
    budget.started_at_epoch = time.time() - 15
    allowed, reason = budget_service.check_execution_allowed(budget)
    assert allowed is False
    assert "Wall-clock limit exceeded" in (reason or "")


def test_url_canonicalization_and_domain_cap(source_service: SourceService):
    u1 = "https://www.Example.com/article/?utm_source=twitter&b=2&a=1#section"
    u2 = "http://example.com/article?a=1&b=2"
    c1 = source_service.canonicalize_url(u1)
    c2 = source_service.canonicalize_url(u2)
    assert c1 == "https://example.com/article?a=1&b=2"
    assert c2 == "http://example.com/article?a=1&b=2"

    existing = [
        SourceMetadata(
            canonical_url="https://example.com/p1",
            original_url="https://example.com/p1",
            domain="example.com",
            status=SourceStatus.SUCCESS,
        ),
        SourceMetadata(
            canonical_url="https://example.com/p2",
            original_url="https://example.com/p2",
            domain="example.com",
            status=SourceStatus.SUCCESS,
        ),
    ]

    # Duplicate URL skipped
    ok_dup, _, _, status_dup = source_service.evaluate_candidate_url(
        "https://www.example.com/p1?utm_campaign=test", existing
    )
    assert ok_dup is False
    assert status_dup == SourceStatus.DUPLICATE_SKIPPED

    # 3rd page from same domain skipped due to max_results_per_domain=2
    ok_cap, _, _, status_cap = source_service.evaluate_candidate_url(
        "https://example.com/p3", existing
    )
    assert ok_cap is False
    assert status_cap == SourceStatus.DOMAIN_CAP_SKIPPED


def test_citation_validator_strips_hallucinated_ids(source_service: SourceService):
    raw_text = "Redis supports AOF persistence [F-001] and quantum teleportation [F-999]."
    cleaned, valid, stripped = source_service.validate_and_clean_citations(
        raw_text, valid_finding_ids={"F-001", "F-002"}
    )
    assert cleaned == "Redis supports AOF persistence [F-001] and quantum teleportation."
    assert valid == ["F-001"]
    assert stripped == ["F-999"]