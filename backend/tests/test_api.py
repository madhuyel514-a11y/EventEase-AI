from datetime import date, timedelta

from fastapi.testclient import TestClient

import json
import logging
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from google.genai._transformers import t_schema

from app.main import (
    AIPlan,
    BudgetLine,
    BudgetShockRequest,
    PlanRequest,
    TimelineRecoveryRequest,
    TradeOffRequest,
    _make_result,
    _parse_gemini_plan,
    _correct_timeline_dates,
    _demo_plan,
    _event_timeline,
    _tailor_event_timeline,
    _analyze_timeline_recovery,
    _cors_origins,
    _generate_with_groq,
    _strict_ai_plan_schema,
    _simulate_budget_shock,
    _compare_trade_offs,
    app,
    create_plan,
)

client = TestClient(app)


def payload() -> dict:
    return {
        "event_type": "Birthday",
        "description": "A relaxed garden birthday for my family",
        "event_date": (date.today() + timedelta(days=45)).isoformat(),
        "budget_inr": 100000,
        "guest_count": 40,
        "venue_preference": "Outdoor",
        "food_preferences": ["Vegetarian"],
        "dietary_preferences": ["No nuts"],
        "cultural_preferences": "",
        "accessibility_preferences": "Step-free access",
    }


def budget_shock_payload(shock_type: str = "category_percentage") -> dict:
    request = {
        "baseline_budget_inr": 100000,
        "baseline_guest_count": 40,
        "baseline_allocations": [
            {"category": "Food", "amount_inr": 40000, "percentage": 40, "rationale": "Meal plan."},
            {"category": "Decor", "amount_inr": 20000, "percentage": 20, "rationale": "Reusable decor."},
            {"category": "Entertainment", "amount_inr": 20000, "percentage": 20, "rationale": "Music."},
            {"category": "Venue", "amount_inr": 20000, "percentage": 20, "rationale": "Space."},
        ],
        "shock_type": shock_type,
    }
    if shock_type == "category_percentage":
        request.update({"category": "Decor", "increase_percent": 25})
    elif shock_type == "additional_guests":
        request["additional_guests"] = 8
    else:
        request.update({"expense_label": "Extra transport", "expense_amount_inr": 10000})
    return request


def trade_off_payload() -> dict:
    return {
        "budget_inr": 100000,
        "guest_count": 40,
        "baseline_total_inr": 80000,
        "allocations": [
            {"category": "Food & refreshments", "amount_inr": 40000},
            {"category": "Venue", "amount_inr": 20000},
            {"category": "Decor", "amount_inr": 20000},
        ],
    }


def test_budgetguard_caps_total_and_calculates_per_guest_cost() -> None:
    request = PlanRequest.model_validate(payload())
    plan = AIPlan(
        overview="A family celebration.",
        allocations=[
            BudgetLine(category="Food", amount_inr=80000, rationale="Seasonal menu."),
            BudgetLine(category="Decor", amount_inr=35000, rationale="Reusable flowers."),
        ],
        timeline=[
            {
                "title": "Confirm menu",
                "due_date": request.event_date,
                "detail": "Review dietary requirements.",
            }
        ],
        vendor_categories=[{"category": "Catering", "essential": True}],
    )
    result = _make_result(request, plan, "ai")
    assert result["budget"]["total_estimated_inr"] == request.budget_inr
    assert result["budget"]["over_budget"] is False
    assert result["budget"]["remaining_inr"] == 0
    assert result["budget"]["cost_per_guest_inr"] == 2500
    assert result["budget"]["savings_suggestions"]


def test_over_budget_ai_allocations_are_scaled_to_requested_budget() -> None:
    request = PlanRequest.model_validate(payload())
    plan = AIPlan(
        overview="A family celebration.",
        allocations=[
            BudgetLine(category="Food", amount_inr=80000, rationale="Seasonal menu."),
            BudgetLine(category="Decor", amount_inr=35000, rationale="Reusable flowers."),
            BudgetLine(category="Venue", amount_inr=15000, rationale="Outdoor space."),
        ],
        timeline=[
            {
                "title": "Confirm menu",
                "due_date": request.event_date,
                "detail": "Review dietary requirements.",
            }
        ],
        vendor_categories=[{"category": "Catering", "essential": True}],
    )
    result = _make_result(request, plan, "ai")
    amounts = [line["amount_inr"] for line in result["budget"]["allocations"]]

    assert sum(amounts) == request.budget_inr
    assert result["budget"]["total_estimated_inr"] == request.budget_inr
    assert result["budget"]["over_budget"] is False
    assert sum(line["percentage"] for line in result["budget"]["allocations"]) == pytest.approx(100, abs=0.2)


def test_plan_api_returns_explicit_demo_mode_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/plan", json=payload())
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "demo"
    assert result["budget"]["total_estimated_inr"] == sum(
        item["amount_inr"] for item in result["budget"]["allocations"]
    )
    assert result["vendor_notice"]


def test_plan_api_preserves_ai_mode_when_gemini_succeeds(monkeypatch) -> None:
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    generated = AIPlan(
        overview="A personalized celebration plan.",
        allocations=[BudgetLine(category="Decor", amount_inr=10000, rationale="A modest decor plan.")],
        timeline=[
            {
                "title": "Original model task",
                "due_date": payload()["event_date"],
                "detail": "A task from the mocked AI result.",
            }
        ],
        vendor_categories=[{"category": "Catering", "essential": True}],
    )
    called = []

    def generate(request):
        called.append(request)
        return generated

    monkeypatch.setattr("app.main._generate_with_gemini", generate)

    response = client.post("/api/plan", json=payload())

    assert response.status_code == 200
    result = response.json()
    assert len(called) == 1
    assert result["mode"] == "ai"
    assert result["overview"] == generated.overview
    assert "fallback_reason" not in result
    assert result["budget"]["total_estimated_inr"] == 10000


def test_groq_generation_uses_official_sdk_and_strict_ai_plan_schema(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    request = PlanRequest.model_validate(payload())
    generated_plan = AIPlan(
        overview="A Groq-generated family celebration.",
        allocations=[BudgetLine(category="Decor", amount_inr=10000, rationale="A modest decor plan.")],
        timeline=[
            {
                "title": "Confirm the garden layout",
                "due_date": request.event_date,
                "detail": "Review seating and step-free routes.",
            }
        ],
        vendor_categories=[{"category": "Catering", "essential": True}],
    )
    captured = {}

    class FakeGroq:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        def create_completion(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=generated_plan.model_dump_json())
                    )
                ]
            )

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = _generate_with_groq(request)

    assert result.overview == generated_plan.overview
    assert captured["api_key"] == "groq-test-key"
    assert captured["request"]["model"] == "openai/gpt-oss-20b"
    response_schema = captured["request"]["response_format"]["json_schema"]
    assert response_schema["strict"] is True
    assert response_schema["schema"] == _strict_ai_plan_schema()
    assert response_schema["schema"]["additionalProperties"] is False
    assert set(response_schema["schema"]["required"]) == {
        "overview", "allocations", "timeline", "vendor_categories",
    }


def test_plan_api_selects_groq_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    generated = AIPlan(
        overview="A Groq-generated celebration.",
        allocations=[BudgetLine(category="Decor", amount_inr=10000, rationale="A modest decor plan.")],
        timeline=[
            {
                "title": "Confirm the venue",
                "due_date": payload()["event_date"],
                "detail": "Review the event space.",
            }
        ],
        vendor_categories=[{"category": "Catering", "essential": True}],
    )
    called = []

    def generate(request):
        called.append(request)
        return generated

    monkeypatch.setattr("app.main._generate_with_groq", generate)
    monkeypatch.setattr(
        "app.main._generate_with_gemini",
        lambda _request: pytest.fail("Gemini must not run when Groq is selected."),
    )

    response = client.post("/api/plan", json=payload())

    assert response.status_code == 200
    assert response.json()["mode"] == "ai"
    assert response.json()["overview"] == generated.overview
    assert len(called) == 1


def test_plan_api_returns_labeled_demo_when_selected_groq_key_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        "app.main._generate_with_groq",
        lambda _request: pytest.fail("Generation must not run without the selected provider key."),
    )

    response = client.post("/api/plan", json=payload())

    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "demo"
    assert result["fallback_reason"] == (
        "No Groq API key is configured. This is a deterministic demo fallback, "
        "not an AI-generated plan. Configure GROQ_API_KEY to enable AI generation."
    )


def test_plan_api_does_not_expose_groq_auth_error_or_secret(monkeypatch, caplog) -> None:
    secret_detail = "provider private error and key marker"

    class ProviderAuthError(Exception):
        status_code = 401

    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setattr(
        "app.main._generate_with_groq",
        lambda _request: (_ for _ in ()).throw(ProviderAuthError(secret_detail)),
    )

    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = client.post("/api/plan", json=payload())

    assert response.status_code == 502
    assert "Groq credentials were rejected" in response.json()["detail"]
    assert secret_detail not in response.text
    assert "groq-test-key" not in response.text
    assert secret_detail not in caplog.text
    assert "groq-test-key" not in caplog.text


def test_plan_api_returns_labeled_demo_fallback_for_groq_upstream_error(monkeypatch, caplog) -> None:
    secret_detail = "private Groq upstream response"

    class ProviderServerError(Exception):
        status_code = 503

    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setattr(
        "app.main._generate_with_groq",
        lambda _request: (_ for _ in ()).throw(ProviderServerError(secret_detail)),
    )

    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = client.post("/api/plan", json=payload())

    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "demo"
    assert result["fallback_reason"] == (
        "Groq is temporarily unavailable. This is a deterministic demo fallback, "
        "not an AI-generated plan. You can retry later."
    )
    assert secret_detail not in response.text
    assert secret_detail not in caplog.text


def test_plan_api_returns_labeled_demo_fallback_for_gemini_rate_limit(monkeypatch, caplog) -> None:
    secret_detail = "quota provider internals and private error marker"

    class ProviderRateLimitError(Exception):
        status_code = 429

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    called = []

    def rate_limited(_request):
        called.append(True)
        raise ProviderRateLimitError(secret_detail)

    monkeypatch.setattr("app.main._generate_with_gemini", rate_limited)

    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = client.post("/api/plan", json=payload())

    assert response.status_code == 200
    result = response.json()
    assert called == [True]
    assert result["mode"] == "demo"
    assert "rate limited" in result["fallback_reason"]
    assert "deterministic demo fallback" in result["fallback_reason"]
    assert "not an AI-generated plan" in result["fallback_reason"]
    assert "retry later" in result["fallback_reason"]
    assert result["overview"]
    assert result["budget"]["allocations"]
    assert result["budget"]["total_estimated_inr"] == sum(
        item["amount_inr"] for item in result["budget"]["allocations"]
    )
    assert secret_detail not in response.text
    assert secret_detail not in caplog.text


@pytest.mark.parametrize(
    ("event_type", "required_phrases"),
    [
        ("Wedding", ("venue", "catering", "photography", "guest", "ceremony")),
        ("Birthday", ("venue", "cake", "invitations", "activities", "decorations")),
        ("Corporate event", ("venue", "agenda", "speakers", "av", "attendee")),
        ("Community gathering", ("community", "venue", "activities", "refreshments")),
        ("Anniversary", ("anniversary", "venue", "program")),
        ("Other", ("event needs", "venue", "invitations", "schedule")),
    ],
)
def test_demo_checklist_is_specific_to_supported_event_type(event_type, required_phrases) -> None:
    request_data = payload()
    request_data.update(
        {
            "event_type": event_type,
            "event_date": (date.today() + timedelta(days=120)).isoformat(),
            "budget_inr": 250000,
            "guest_count": 75,
            "food_preferences": ["Vegan"],
            "dietary_preferences": ["No nuts"],
            "cultural_preferences": "Include a family welcome ritual",
            "accessibility_preferences": "Step-free seating and quiet area",
        }
    )
    request = PlanRequest.model_validate(request_data)

    timeline = _demo_plan(request).timeline
    titles = [task.title.casefold() for task in timeline]
    details = " ".join(task.detail.casefold() for task in timeline)

    for phrase in required_phrases:
        assert any(phrase in title for title in titles)
    assert len(timeline) >= 5
    assert len({task.due_date for task in timeline}) == len(timeline)
    assert [task.due_date for task in timeline] == sorted(task.due_date for task in timeline)
    assert all(task.due_date < request.event_date for task in timeline)
    assert "75" in details
    assert "₹250,000" in details
    assert "vegan" in details
    assert "no nuts" in details
    assert "step-free seating and quiet area" in details
    assert "family welcome ritual" in details


def test_gemini_generated_checklist_is_replaced_with_event_specific_tasks(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    request_data = payload()
    request_data["event_type"] = "Wedding"
    request_data["event_date"] = (date.today() + timedelta(days=90)).isoformat()
    generic_plan = AIPlan(
        overview="A celebration.",
        allocations=[BudgetLine(category="Food", amount_inr=10000, rationale="Menu estimate.")],
        timeline=[{"title": "Generic preparation task", "due_date": request_data["event_date"], "detail": "Prepare."}],
        vendor_categories=[{"category": "Venue", "essential": True}],
    )
    monkeypatch.setattr("app.main._generate_with_gemini", lambda _request: generic_plan)

    response = client.post("/api/plan", json=request_data)

    assert response.status_code == 200
    titles = [task["title"].casefold() for task in response.json()["timeline"]]
    assert any("ceremony" in title for title in titles)
    assert any("photography" in title for title in titles)
    assert all("generic preparation" not in title for title in titles)
    assert generic_plan.timeline[0].title == "Generic preparation task"


def test_checklist_lead_times_are_event_relative_and_source_plan_is_unchanged() -> None:
    today = date(2026, 10, 20)
    event_date = today + timedelta(days=120)
    request_data = payload()
    request_data.update({"event_type": "Wedding", "event_date": event_date.isoformat()})
    request = PlanRequest.model_validate(request_data)
    source = AIPlan(
        overview="Wedding overview.",
        allocations=[BudgetLine(category="Decor", amount_inr=1000, rationale="Estimate.")],
        timeline=[
            {"title": "Original generated task", "due_date": event_date, "detail": "Do not mutate source."}
        ],
        vendor_categories=[{"category": "Venue", "essential": True}],
    )
    source_snapshot = source.model_dump(mode="json")

    tailored = _tailor_event_timeline(request, source, today)

    assert source.model_dump(mode="json") == source_snapshot
    assert tailored.timeline[0].due_date == event_date - timedelta(days=90)
    assert len({task.due_date for task in tailored.timeline}) == len(tailored.timeline)


def test_tomorrow_event_checklist_uses_today_but_recovery_warns_about_short_notice() -> None:
    today = date.today()
    request_data = payload()
    request_data.update({"event_type": "Wedding", "event_date": (today + timedelta(days=1)).isoformat()})
    request = PlanRequest.model_validate(request_data)

    timeline = _event_timeline(request, today)
    assert all(task.due_date == today for task in timeline)

    recovery_request = TimelineRecoveryRequest.model_validate(
        {
            "event_date": request.event_date.isoformat(),
            "tasks": [task.model_dump(mode="json") for task in timeline],
        }
    )
    result = _analyze_timeline_recovery(recovery_request, today)

    assert all(task.status == "unresolved" for task in result.tasks)
    assert all(task.suggested_due_date is None for task in result.tasks)
    assert all("No date remains" in task.explanation for task in result.tasks)
    assert result.has_unresolved_conflicts is True
    assert result.can_apply is False


def test_what_if_request_uses_updated_budget_and_guest_count(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    what_if = payload()
    what_if["budget_inr"] = 120000
    what_if["guest_count"] = 60
    response = client.post("/api/plan", json=what_if)

    assert response.status_code == 200
    result = response.json()
    assert result["event"]["guest_count"] == 60
    assert result["budget"]["budget_inr"] == 120000
    assert sum(line["amount_inr"] for line in result["budget"]["allocations"]) <= 120000


def test_budget_shock_percentage_increases_only_selected_category() -> None:
    response = client.post("/api/budget-shock", json=budget_shock_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["shock_cost_inr"] == 5000
    assert result["original_allocation_total_inr"] == 100000
    assert result["projected_total_inr"] == 105000
    assert result["projected_cost_per_guest_inr"] == 2625
    decor = next(line for line in result["category_comparison"] if line["category"] == "Decor")
    food = next(line for line in result["category_comparison"] if line["category"] == "Food")
    assert (decor["baseline_amount_inr"], decor["projected_amount_inr"]) == (20000, 25000)
    assert food["baseline_amount_inr"] == food["projected_amount_inr"]
    assert sum(line["projected_amount_inr"] for line in result["category_comparison"]) == result["projected_total_inr"]
    assert result["projected_gap_inr"] == max(0, result["projected_total_inr"] - result["available_budget_inr"])


@pytest.mark.parametrize(
    ("category", "expected_status"),
    [("Unknown category", 422), ("", 422)],
)
def test_budget_shock_rejects_invalid_or_unknown_categories(category, expected_status) -> None:
    request = budget_shock_payload()
    request["category"] = category
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == expected_status


def test_budget_shock_additional_guests_use_rounded_baseline_average() -> None:
    request = budget_shock_payload("additional_guests")
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 200
    result = response.json()
    assert result["shock_cost_inr"] == round(100000 / 40) * 8
    assert result["projected_guest_count"] == 48
    assert result["projected_cost_per_guest_inr"] == round(120000 / 48)
    assert result["projected_total_inr"] == result["original_allocation_total_inr"] + result["shock_cost_inr"]
    assert "approximately" in result["shock_description"]


def test_budget_shock_fixed_expense_is_a_separate_line() -> None:
    response = client.post("/api/budget-shock", json=budget_shock_payload("fixed_expense"))
    assert response.status_code == 200
    result = response.json()
    expense = next(line for line in result["category_comparison"] if line["category"] == "Extra transport")
    assert result["shock_cost_inr"] == 10000
    assert result["projected_total_inr"] == 110000
    assert result["projected_cost_per_guest_inr"] == 2750
    assert expense == {
        "category": "Extra transport",
        "baseline_amount_inr": 0,
        "projected_amount_inr": 10000,
        "change_inr": 10000,
    }


def test_budget_shock_within_budget_reports_no_gap_and_success() -> None:
    request = budget_shock_payload()
    request["baseline_budget_inr"] = 105000
    request["increase_percent"] = 10
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 200
    result = response.json()
    assert result["projected_total_inr"] == 102000
    assert result["projected_gap_inr"] == 0
    assert result["recovery"]["fits_within_budget"] is True


def test_budget_shock_recovery_uses_discretionary_caps_and_closes_gap() -> None:
    request = budget_shock_payload("fixed_expense")
    request["baseline_allocations"] = [
        {"category": "Food", "amount_inr": 20000},
        {"category": "Decor", "amount_inr": 40000},
        {"category": "Entertainment", "amount_inr": 20000},
        {"category": "Venue", "amount_inr": 20000},
    ]
    request["expense_amount_inr"] = 14000
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 200
    recovery = response.json()["recovery"]
    assert recovery["total_savings_inr"] == 14000
    assert recovery["proposed_total_inr"] == 100000
    assert recovery["remaining_shortfall_inr"] == 0
    assert recovery["fits_within_budget"] is True
    assert recovery["remaining_budget_inr"] == 0
    assert recovery["proposed_total_inr"] <= recovery["available_budget_inr"]
    assert {item["category"] for item in recovery["savings_breakdown"]} == {"Decor", "Entertainment"}
    assert recovery["total_savings_inr"] == sum(
        item["reduction_inr"] for item in recovery["savings_breakdown"]
    )
    assert recovery["proposed_total_inr"] == (
        response.json()["projected_total_inr"] - recovery["total_savings_inr"]
    )


def test_budget_shock_recovery_reports_unresolved_shortfall_and_respects_caps() -> None:
    request = budget_shock_payload("fixed_expense")
    request["baseline_allocations"] = [
        {"category": "Food", "amount_inr": 20000},
        {"category": "Decor", "amount_inr": 40000},
        {"category": "Entertainment", "amount_inr": 20000},
        {"category": "Venue", "amount_inr": 20000},
    ]
    request["expense_amount_inr"] = 30000
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 200
    result = response.json()
    recovery = result["recovery"]
    assert recovery["fits_within_budget"] is False
    assert recovery["remaining_shortfall_inr"] == 16000
    assert recovery["remaining_budget_inr"] == 0
    assert recovery["remaining_shortfall_inr"] == max(
        0, recovery["proposed_total_inr"] - recovery["available_budget_inr"]
    )
    for saving in recovery["savings_breakdown"]:
        assert saving["reduction_inr"] <= (
            saving["cap_basis_amount_inr"] * saving["maximum_reduction_percent"] // 100
        )
        assert saving["revised_amount_inr"] >= 0


def test_budget_shock_recovery_exactly_reports_requested_insufficient_shortfall() -> None:
    request = {
        "baseline_budget_inr": 150000,
        "baseline_plan_cost_inr": 150000,
        "baseline_guest_count": 60,
        "baseline_allocations": [
            {"category": "Venue Rental", "amount_inr": 40000},
            {"category": "Catering", "amount_inr": 55000},
            {"category": "Decor", "amount_inr": 20000},
            {"category": "Cake", "amount_inr": 8000},
            {"category": "Photography", "amount_inr": 15000},
            {"category": "Sound/DJ", "amount_inr": 12000},
        ],
        "shock_type": "category_percentage",
        "category": "Venue Rental",
        "increase_percent": 20,
    }
    original = json.loads(json.dumps(request))

    response = client.post("/api/budget-shock", json=request)

    assert response.status_code == 200
    result = response.json()
    recovery = result["recovery"]
    assert result["original_allocation_total_inr"] == 150000
    assert result["projected_total_inr"] == 158000
    assert result["projected_gap_inr"] == 8000
    assert recovery["fits_within_budget"] is False
    assert recovery["total_savings_inr"] == 7400
    assert recovery["proposed_total_inr"] == 150600
    assert recovery["remaining_shortfall_inr"] == 600
    assert recovery["remaining_budget_inr"] == 0
    assert recovery["proposed_total_inr"] > recovery["available_budget_inr"]
    savings = {item["category"]: item for item in recovery["savings_breakdown"]}
    assert set(savings) == {"Decor", "Sound/DJ"}
    assert savings["Decor"]["reduction_inr"] == 5000
    assert savings["Sound/DJ"]["reduction_inr"] == 2400
    for saving in savings.values():
        assert saving["reduction_inr"] <= (
            saving["cap_basis_amount_inr"] * saving["maximum_reduction_percent"] // 100
        )
    projected_categories = {item["category"]: item for item in result["category_comparison"]}
    assert projected_categories["Venue Rental"]["projected_amount_inr"] == 48000
    for protected in ("Catering", "Cake", "Photography"):
        assert projected_categories[protected]["projected_amount_inr"] == projected_categories[protected]["baseline_amount_inr"]
    assert request == original


def test_budget_shock_recovery_fits_when_permitted_savings_cover_gap() -> None:
    request = {
        "baseline_budget_inr": 150000,
        "baseline_plan_cost_inr": 150000,
        "baseline_guest_count": 60,
        "baseline_allocations": [
            {"category": "Venue Rental", "amount_inr": 40000},
            {"category": "Catering", "amount_inr": 55000},
            {"category": "Decor", "amount_inr": 20000},
            {"category": "Cake", "amount_inr": 8000},
            {"category": "Photography", "amount_inr": 15000},
            {"category": "Sound/DJ", "amount_inr": 12000},
        ],
        "shock_type": "category_percentage",
        "category": "Venue Rental",
        "increase_percent": 10,
    }

    response = client.post("/api/budget-shock", json=request)

    assert response.status_code == 200
    result = response.json()
    recovery = result["recovery"]
    assert result["projected_total_inr"] == 154000
    assert result["projected_gap_inr"] == 4000
    assert recovery["fits_within_budget"] is True
    assert recovery["total_savings_inr"] == 4000
    assert recovery["proposed_total_inr"] == 150000
    assert recovery["remaining_shortfall_inr"] == 0
    assert recovery["remaining_budget_inr"] == 0


def test_budget_shock_rounds_percentage_increase_to_nearest_rupee() -> None:
    request = budget_shock_payload()
    request["baseline_budget_inr"] = 1000
    request["baseline_allocations"] = [
        {"category": "Decor", "amount_inr": 5},
    ]
    request["category"] = "Decor"
    request["increase_percent"] = 10
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 200
    result = response.json()
    assert result["shock_cost_inr"] == 1
    assert result["projected_total_inr"] == 6
    assert result["available_budget_inr"] == 1000


def test_budget_shock_uses_half_up_rounding_for_per_guest_cost() -> None:
    request = budget_shock_payload("fixed_expense")
    request["baseline_guest_count"] = 32
    request["expense_amount_inr"] = 1
    request["baseline_allocations"] = [
        {"category": "Food", "amount_inr": 9998},
        {"category": "Venue", "amount_inr": 1},
    ]
    request["baseline_budget_inr"] = 100000
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 200
    result = response.json()
    assert result["projected_total_inr"] == 10000
    assert result["projected_cost_per_guest_inr"] == 313


def test_budget_shock_validates_baseline_plan_cost_match() -> None:
    request = budget_shock_payload()
    request["baseline_plan_cost_inr"] = 100000
    assert client.post("/api/budget-shock", json=request).status_code == 200

    request["baseline_plan_cost_inr"] = 99999
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 422
    assert "allowed rounding tolerance: ₹0" in response.text


def test_plan_percentages_match_simulator_baseline_rounding() -> None:
    request = PlanRequest.model_validate(payload())
    plan = AIPlan(
        overview="A small celebration.",
        allocations=[
            BudgetLine(category="Decor", amount_inr=5, rationale="Flowers."),
            BudgetLine(category="Food", amount_inr=395, rationale="Snacks."),
        ],
        timeline=[
            {"title": "Confirm menu", "due_date": request.event_date, "detail": "Check requirements."}
        ],
        vendor_categories=[{"category": "Catering", "essential": True}],
    )
    generated = _make_result(request, plan, "demo")
    assert generated["budget"]["allocations"][0]["percentage"] == 1.3
    simulator_request = {
        "baseline_budget_inr": generated["budget"]["budget_inr"],
        "baseline_plan_cost_inr": generated["budget"]["total_estimated_inr"],
        "baseline_guest_count": generated["event"]["guest_count"],
        "baseline_allocations": generated["budget"]["allocations"],
        "shock_type": "category_percentage",
        "category": "Decor",
        "increase_percent": 10,
    }
    assert client.post("/api/budget-shock", json=simulator_request).status_code == 200


def test_trade_off_balanced_exactly_matches_baseline_allocations() -> None:
    response = client.post("/api/trade-offs", json=trade_off_payload())

    assert response.status_code == 200
    result = response.json()
    balanced = next(option for option in result["options"] if option["name"] == "Balanced")
    assert [
        (line["category"], line["baseline_amount_inr"], line["proposed_amount_inr"], line["change_inr"])
        for line in balanced["allocations"]
    ] == [
        ("Food & refreshments", 40000, 40000, 0),
        ("Venue", 20000, 20000, 0),
        ("Decor", 20000, 20000, 0),
    ]
    assert balanced["estimated_total_inr"] == 80000
    assert balanced["remaining_budget_inr"] == 20000
    assert balanced["estimated_cost_per_guest_inr"] == round(80000 / 40)
    assert balanced["exceeds_budget"] is False


def test_trade_off_only_changes_exact_allowlisted_categories() -> None:
    request = trade_off_payload()
    request["allocations"].append({"category": "decor upgrade package", "amount_inr": 1000})
    request["baseline_total_inr"] = 81000

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    premium = next(option for option in result["options"] if option["name"] == "Premium")
    assert result["discretionary_categories_available"] == ["Decor"]
    for option in (saver, premium):
        changed = {line["category"] for line in option["allocations"] if line["change_inr"]}
        assert changed <= {"Decor"}
        unknown = next(line for line in option["allocations"] if line["category"] == "decor upgrade package")
        assert unknown["proposed_amount_inr"] == unknown["baseline_amount_inr"] == 1000


@pytest.mark.parametrize(
    ("category", "saver_change", "premium_change"),
    [
        ("Decor & Ambient Lighting", -2500, 2000),
        ("Audio & Entertainment", -2000, 1500),
        ("Decor & Lighting", -2500, 2000),
        ("Entertainment & Sound", -2000, 1500),
    ],
)
def test_trade_off_recognizes_explicit_composite_category_names(
    category: str,
    saver_change: int,
    premium_change: int,
) -> None:
    request = {
        "budget_inr": 12000,
        "guest_count": 10,
        "baseline_total_inr": 10000,
        "allocations": [{"category": category, "amount_inr": 10000}],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == [category]
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    premium = next(option for option in result["options"] if option["name"] == "Premium")
    assert saver["allocations"][0]["change_inr"] == saver_change
    assert premium["allocations"][0]["change_inr"] == premium_change
    assert premium["estimated_total_inr"] <= request["budget_inr"]


@pytest.mark.parametrize(
    "category",
    [
        "Decor & Floral Accents",
        "Floral Arrangements",
        "Event Decorations",
        "Table Decorations",
    ],
)
def test_trade_off_recognizes_explicit_decorative_aliases(category: str) -> None:
    request = {
        "budget_inr": 24000,
        "guest_count": 12,
        "baseline_total_inr": 20000,
        "allocations": [{"category": category, "amount_inr": 20000}],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    premium = next(option for option in result["options"] if option["name"] == "Premium")
    assert saver["allocations"][0]["change_inr"] == -5000
    assert premium["allocations"][0]["change_inr"] == 4000
    assert premium["estimated_total_inr"] == request["budget_inr"]


def test_trade_off_current_wedding_categories_only_change_approved_discretionary_lines() -> None:
    allocations = [
        ("Decor & Floral Accents", 20000),
        ("Attire & Styling", 10000),
        ("Venue Rental", 10000),
        ("Catering & Beverages", 15000),
        ("Cake & Desserts", 5000),
        ("Photography", 10000),
        ("Contingency Fund", 5000),
        ("Audio & Entertainment", 10000),
        ("Favors", 5000),
        ("Unclassified wedding expense", 5000),
    ]
    request = {
        "budget_inr": 100000,
        "guest_count": 80,
        "baseline_total_inr": sum(amount for _, amount in allocations),
        "allocations": [{"category": category, "amount_inr": amount} for category, amount in allocations],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    premium = next(option for option in result["options"] if option["name"] == "Premium")
    allowed = {"Decor & Floral Accents", "Audio & Entertainment", "Favors"}
    protected = {
        "Attire & Styling",
        "Venue Rental",
        "Catering & Beverages",
        "Cake & Desserts",
        "Photography",
        "Contingency Fund",
        "Unclassified wedding expense",
    }
    for option in (saver, premium):
        changed = {line["category"] for line in option["allocations"] if line["change_inr"]}
        assert changed <= allowed
        for line in option["allocations"]:
            if line["category"] in protected:
                assert line["change_inr"] == 0
    assert saver["allocations"][0]["change_inr"] < 0
    assert premium["allocations"][0]["change_inr"] > 0
    assert premium["estimated_total_inr"] <= request["budget_inr"]


def test_trade_off_premium_leaves_wedding_plan_unchanged_without_headroom() -> None:
    categories = [
        ("Decor & Floral Accents", 20000),
        ("Audio & Entertainment", 10000),
        ("Venue Rental", 10000),
        ("Contingency Fund", 5000),
    ]
    request = {
        "budget_inr": 45000,
        "guest_count": 40,
        "baseline_total_inr": 45000,
        "allocations": [{"category": name, "amount_inr": amount} for name, amount in categories],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    premium = next(option for option in response.json()["options"] if option["name"] == "Premium")
    assert premium["estimated_total_inr"] == request["budget_inr"]
    assert all(item["change_inr"] == 0 for item in premium["allocations"])
    assert "No budget headroom" in premium["explanation"]


def test_trade_off_recognizes_current_birthday_categories_and_preserves_protected_lines() -> None:
    allocations = [
        ("Gluten-Free Vegetarian Catering", 40000),
        ("Venue Rental", 25000),
        ("Milestone Cake & Dessert Bar", 7000),
        ("Decor, Lighting & Backdrop", 12000),
        ("Event Photography", 8000),
        ("DJ & Sound Equipment", 8000),
    ]
    request = {
        "budget_inr": 100000,
        "guest_count": 40,
        "baseline_total_inr": 100000,
        "allocations": [
            {"category": category, "amount_inr": amount}
            for category, amount in allocations
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == [
        "Decor, Lighting & Backdrop",
        "DJ & Sound Equipment",
    ]
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    balanced = next(option for option in result["options"] if option["name"] == "Balanced")
    premium = next(option for option in result["options"] if option["name"] == "Premium")

    baseline_by_category = dict(allocations)
    saver_changes = {line["category"]: line["change_inr"] for line in saver["allocations"]}
    assert saver_changes == {
        "Gluten-Free Vegetarian Catering": 0,
        "Venue Rental": 0,
        "Milestone Cake & Dessert Bar": 0,
        "Decor, Lighting & Backdrop": -3000,
        "Event Photography": 0,
        "DJ & Sound Equipment": -1600,
    }
    assert saver["estimated_total_inr"] == 95400
    assert sum(line["proposed_amount_inr"] for line in saver["allocations"]) == 95400

    assert balanced["estimated_total_inr"] == 100000
    assert {
        line["category"]: line["proposed_amount_inr"]
        for line in balanced["allocations"]
    } == baseline_by_category
    assert all(line["change_inr"] == 0 for line in balanced["allocations"])

    assert premium["estimated_total_inr"] == request["budget_inr"]
    assert premium["estimated_total_inr"] <= request["budget_inr"]
    assert all(line["change_inr"] == 0 for line in premium["allocations"])
    assert "No budget headroom" in premium["explanation"]
    for option in (saver, balanced, premium):
        assert all(
            line["change_inr"] == 0
            for line in option["allocations"]
            if line["category"] in {
                "Gluten-Free Vegetarian Catering",
                "Venue Rental",
                "Milestone Cake & Dessert Bar",
                "Event Photography",
            }
        )


def test_trade_off_recognizes_thematic_decor_and_sound_entertainment_aliases() -> None:
    allocations = [
        ("Thematic Decor & Lighting", 14000),
        ("Sound & Entertainment", 10000),
        ("Venue Rental", 25000),
        ("Gluten-Free Vegetarian Catering", 40000),
        ("Milestone Cake & Dessert Bar", 7000),
        ("Event Photography", 8000),
    ]
    request = {
        "budget_inr": 120000,
        "guest_count": 40,
        "baseline_total_inr": sum(amount for _, amount in allocations),
        "allocations": [
        {"category": category, "amount_inr": amount}
        for category, amount in allocations
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == [
        "Thematic Decor & Lighting",
        "Sound & Entertainment",
    ]
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    balanced = next(option for option in result["options"] if option["name"] == "Balanced")
    premium = next(option for option in result["options"] if option["name"] == "Premium")

    assert {
        line["category"]: line["change_inr"]
        for line in saver["allocations"]
    } == {
        "Thematic Decor & Lighting": -3500,
        "Sound & Entertainment": -2000,
        "Venue Rental": 0,
        "Gluten-Free Vegetarian Catering": 0,
        "Milestone Cake & Dessert Bar": 0,
        "Event Photography": 0,
    }
    assert {
        line["category"]: line["proposed_amount_inr"]
        for line in balanced["allocations"]
    } == dict(allocations)
    assert balanced["estimated_total_inr"] == request["baseline_total_inr"]
    assert all(line["change_inr"] == 0 for line in balanced["allocations"])

    assert {
        line["category"]: line["change_inr"]
        for line in premium["allocations"]
    } == {
        "Thematic Decor & Lighting": 2800,
        "Sound & Entertainment": 1500,
        "Venue Rental": 0,
        "Gluten-Free Vegetarian Catering": 0,
        "Milestone Cake & Dessert Bar": 0,
        "Event Photography": 0,
    }
    assert premium["estimated_total_inr"] <= request["budget_inr"]


def test_trade_off_current_plan_decor_styling_alias_preserves_protected_categories() -> None:
    allocations = [
        ("Venue Rental", 45000),
        ("Catering & Beverages", 50000),
        ("Specialty Cake & Desserts", 8000),
        ("Decor & Styling", 20000),
        ("Photography", 15000),
        ("Sound & Entertainment", 12000),
    ]
    request = {
        "budget_inr": 150000,
        "guest_count": 60,
        "baseline_total_inr": 150000,
        "allocations": [
            {"category": category, "amount_inr": amount}
            for category, amount in allocations
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == [
        "Decor & Styling",
        "Sound & Entertainment",
    ]
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    balanced = next(option for option in result["options"] if option["name"] == "Balanced")
    premium = next(option for option in result["options"] if option["name"] == "Premium")

    saver_by_category = {line["category"]: line for line in saver["allocations"]}
    assert {
        category: saver_by_category[category]["change_inr"]
        for category, _ in allocations
    } == {
        "Venue Rental": 0,
        "Catering & Beverages": 0,
        "Specialty Cake & Desserts": 0,
        "Decor & Styling": -5000,
        "Photography": 0,
        "Sound & Entertainment": -2400,
    }
    assert -sum(line["change_inr"] for line in saver["allocations"]) == 7400
    assert saver["estimated_total_inr"] == 142600
    assert saver["estimated_total_inr"] == sum(
        line["proposed_amount_inr"] for line in saver["allocations"]
    )
    assert saver["estimated_total_inr"] <= request["budget_inr"]

    assert {
        line["category"]: line["proposed_amount_inr"]
        for line in balanced["allocations"]
    } == dict(allocations)
    assert all(line["change_inr"] == 0 for line in balanced["allocations"])
    assert balanced["estimated_total_inr"] == request["baseline_total_inr"]

    assert premium["estimated_total_inr"] == request["budget_inr"]
    assert premium["estimated_total_inr"] <= request["budget_inr"]
    assert all(line["change_inr"] == 0 for line in premium["allocations"])
    assert "No budget headroom" in premium["explanation"]


def test_trade_off_exact_aliases_preserve_protected_and_unknown_allocations() -> None:
    allocations = [
        ("Decor & Ambient Lighting", 22000),
        ("Sound, DJ & Entertainment", 10000),
        ("Favors & Stationery", 5000),
        ("Venue Rental", 25000),
        ("Vegetarian & Gluten-Free Catering", 40000),
        ("Custom Gluten-Free Birthday Cake", 7000),
        ("Photography & Coverage", 8000),
        ("Unclassified event expense", 3000),
    ]
    request = {
        "budget_inr": 150000,
        "guest_count": 40,
        "baseline_total_inr": sum(amount for _, amount in allocations),
        "allocations": [
            {"category": category, "amount_inr": amount}
            for category, amount in allocations
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == [
        "Decor & Ambient Lighting",
        "Sound, DJ & Entertainment",
        "Favors & Stationery",
    ]
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    balanced = next(option for option in result["options"] if option["name"] == "Balanced")
    premium = next(option for option in result["options"] if option["name"] == "Premium")

    baseline_by_category = dict(allocations)
    saver_by_category = {line["category"]: line for line in saver["allocations"]}
    assert {
        category: saver_by_category[category]["change_inr"]
        for category, _ in allocations
    } == {
        "Decor & Ambient Lighting": -5500,
        "Sound, DJ & Entertainment": -2000,
        "Favors & Stationery": -2500,
        "Venue Rental": 0,
        "Vegetarian & Gluten-Free Catering": 0,
        "Custom Gluten-Free Birthday Cake": 0,
        "Photography & Coverage": 0,
        "Unclassified event expense": 0,
    }
    assert saver["estimated_total_inr"] == 110000
    assert saver["estimated_total_inr"] == sum(
        line["proposed_amount_inr"] for line in saver["allocations"]
    )
    assert saver["estimated_total_inr"] <= request["budget_inr"]

    balanced_by_category = {line["category"]: line for line in balanced["allocations"]}
    assert {
        category: balanced_by_category[category]["proposed_amount_inr"]
        for category, _ in allocations
    } == baseline_by_category
    assert all(line["change_inr"] == 0 for line in balanced["allocations"])
    assert balanced["estimated_total_inr"] == request["baseline_total_inr"]

    premium_by_category = {line["category"]: line for line in premium["allocations"]}
    assert {
        category: premium_by_category[category]["change_inr"]
        for category, _ in allocations
    } == {
        "Decor & Ambient Lighting": 4400,
        "Sound, DJ & Entertainment": 1500,
        "Favors & Stationery": 1250,
        "Venue Rental": 0,
        "Vegetarian & Gluten-Free Catering": 0,
        "Custom Gluten-Free Birthday Cake": 0,
        "Photography & Coverage": 0,
        "Unclassified event expense": 0,
    }
    assert premium["estimated_total_inr"] == 127150
    assert premium["estimated_total_inr"] == sum(
        line["proposed_amount_inr"] for line in premium["allocations"]
    )
    assert premium["estimated_total_inr"] <= request["budget_inr"]


@pytest.mark.parametrize(
    "category",
    [
        "Decor & Ambient Lighting Package",
        "Audio & Entertainment Services",
        "Decor & Lighting Design",
        "Entertainment & Sound System",
        "Catering & Beverages",
        "Venue Rental",
        "Custom Cake & Desserts",
        "Photography",
        "Contingency Fund",
    ],
)
def test_trade_off_leaves_unrecognized_and_non_discretionary_categories_unchanged(category: str) -> None:
    request = {
        "budget_inr": 12000,
        "guest_count": 10,
        "baseline_total_inr": 10000,
        "allocations": [{"category": category, "amount_inr": 10000}],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == []
    for option in result["options"]:
        allocation = option["allocations"][0]
        assert allocation["baseline_amount_inr"] == allocation["proposed_amount_inr"] == 10000
        assert allocation["change_inr"] == 0


def test_trade_off_budget_saver_respects_documented_category_caps() -> None:
    request = trade_off_payload()
    request["allocations"] = [
        {"category": "Decor", "amount_inr": 20000},
        {"category": "Entertainment", "amount_inr": 20000},
        {"category": "Favours", "amount_inr": 101},
        {"category": "Food", "amount_inr": 39899},
    ]
    request["baseline_total_inr"] = 80000

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    saver = next(option for option in response.json()["options"] if option["name"] == "Budget Saver")
    changes = {line["category"]: line["change_inr"] for line in saver["allocations"]}
    assert changes == {"Decor": -5000, "Entertainment": -4000, "Favours": -50, "Food": 0}
    assert saver["estimated_total_inr"] == sum(line["proposed_amount_inr"] for line in saver["allocations"])
    assert saver["estimated_total_inr"] == 70950
    assert saver["remaining_budget_inr"] == 29050
    assert saver["estimated_cost_per_guest_inr"] == round(70950 / 40)
    assert not saver["exceeds_budget"]


def test_trade_off_premium_uses_only_capped_partial_headroom() -> None:
    request = trade_off_payload()
    request["budget_inr"] = 81200
    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    premium = next(option for option in response.json()["options"] if option["name"] == "Premium")
    decor = next(line for line in premium["allocations"] if line["category"] == "Decor")
    assert decor["change_inr"] == 1200
    assert premium["estimated_total_inr"] == 81200
    assert premium["remaining_budget_inr"] == 0
    assert not premium["exceeds_budget"]


def test_trade_off_premium_leaves_budget_headroom_unspent_after_caps() -> None:
    request = trade_off_payload()

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    premium = next(option for option in response.json()["options"] if option["name"] == "Premium")
    decor = next(line for line in premium["allocations"] if line["category"] == "Decor")
    assert decor["change_inr"] == 4000
    assert premium["estimated_total_inr"] == 84000
    assert premium["remaining_budget_inr"] == 16000
    assert not premium["exceeds_budget"]
    assert "remaining headroom is left unspent" in premium["explanation"]


def test_trade_off_premium_with_no_headroom_is_unchanged() -> None:
    request = trade_off_payload()
    request["budget_inr"] = request["baseline_total_inr"]

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    premium = next(option for option in response.json()["options"] if option["name"] == "Premium")
    assert all(line["change_inr"] == 0 for line in premium["allocations"])
    assert premium["estimated_total_inr"] == request["baseline_total_inr"]
    assert premium["remaining_budget_inr"] == 0
    assert not premium["exceeds_budget"]
    assert "No budget headroom" in premium["explanation"]


def test_trade_off_with_no_allowlisted_categories_explains_unchanged_options() -> None:
    request = trade_off_payload()
    request["allocations"] = [
        {"category": "Food & refreshments", "amount_inr": 40000},
        {"category": "Venue", "amount_inr": 40000},
    ]
    request["baseline_total_inr"] = 80000

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == []
    for option in result["options"]:
        assert all(line["change_inr"] == 0 for line in option["allocations"])
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    premium = next(option for option in result["options"] if option["name"] == "Premium")
    assert "No supported discretionary categories" in saver["explanation"]
    assert "No supported discretionary categories" in premium["explanation"]


@pytest.mark.parametrize(
    "changes",
    [
        {"budget_inr": 0},
        {"guest_count": 0},
        {"allocations": [{"category": "Food", "amount_inr": -1}]},
        {"allocations": [{"category": "Decor", "amount_inr": 1}, {"category": " decor ", "amount_inr": 2}]},
        {"baseline_total_inr": 79999},
        {"budget_inr": 79999},
    ],
)
def test_trade_off_rejects_invalid_baseline_data(changes) -> None:
    request = trade_off_payload()
    request.update(changes)

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 422


def test_trade_off_does_not_mutate_baseline_and_never_calls_gemini(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.main._generate_with_gemini",
        lambda _request: pytest.fail("Trade-off comparison must not call Gemini."),
    )
    request_data = trade_off_payload()
    request = TradeOffRequest.model_validate(request_data)
    snapshot = request.model_dump(mode="json")

    result = _compare_trade_offs(request)

    assert request.model_dump(mode="json") == snapshot
    assert [option.name for option in result.options] == ["Budget Saver", "Balanced", "Premium"]
    assert all(not option.exceeds_budget for option in result.options)
    assert "not a live vendor-price search" in result.notice
    response = client.post("/api/trade-offs", json=request_data)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "changes",
    [
        {"baseline_budget_inr": -1},
        {"baseline_guest_count": 0},
        {"baseline_allocations": [{"category": "Food", "amount_inr": -1}]},
        {"baseline_allocations": [{"category": "Food", "amount_inr": 100001}]},
        {"baseline_allocations": [{"category": "Food", "amount_inr": 50000, "percentage": 101}]},
        {"increase_percent": 0},
        {"increase_percent": 501},
        {"baseline_budget_inr": 10_000_000, "baseline_allocations": [{"category": "Decor", "amount_inr": 10_000_000}], "increase_percent": 500},
        {"additional_guests": 0},
        {"baseline_budget_inr": 10_000_000, "baseline_guest_count": 1, "baseline_allocations": [{"category": "Food", "amount_inr": 10_000_000}], "additional_guests": 2},
        {"expense_amount_inr": -50},
    ],
)
def test_budget_shock_rejects_invalid_budget_allocations_and_parameters(changes) -> None:
    shock_type = "additional_guests" if "additional_guests" in changes else (
        "fixed_expense" if "expense_amount_inr" in changes else "category_percentage"
    )
    request = budget_shock_payload(shock_type)
    request.update(changes)
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 422


def test_budget_shock_does_not_mutate_baseline_request(monkeypatch) -> None:
    request = budget_shock_payload("fixed_expense")
    baseline = BudgetShockRequest.model_validate(request)
    baseline_snapshot = baseline.model_dump(mode="json")
    monkeypatch.setattr(
        "app.main._generate_with_gemini",
        lambda _request: pytest.fail("Budget shock simulation must not call Gemini."),
    )
    result = _simulate_budget_shock(baseline)
    assert result.projected_total_inr == 110000
    assert baseline.model_dump(mode="json") == baseline_snapshot


def test_budget_shock_never_reduces_essential_named_categories() -> None:
    request = budget_shock_payload("fixed_expense")
    request["baseline_allocations"] = [
        {"category": "Food", "amount_inr": 40000},
        {"category": "Essential Decor", "amount_inr": 20000},
        {"category": "Entertainment", "amount_inr": 20000},
        {"category": "Venue", "amount_inr": 20000},
    ]
    request["expense_amount_inr"] = 20000
    response = client.post("/api/budget-shock", json=request)
    assert response.status_code == 200
    result = response.json()
    savings = result["recovery"]["savings_breakdown"]
    assert all(item["category"] != "Essential Decor" for item in savings)
    assert {item["category"] for item in savings} <= {"Entertainment"}


def test_budget_shock_never_reduces_venue_catering_cake_or_photography() -> None:
    request = budget_shock_payload("fixed_expense")
    request["baseline_allocations"] = [
        {"category": "Catering & Beverages", "amount_inr": 20000},
        {"category": "Venue Rental", "amount_inr": 20000},
        {"category": "Custom Cake & Desserts", "amount_inr": 20000},
        {"category": "Photography", "amount_inr": 20000},
        {"category": "Decor", "amount_inr": 20000},
    ]
    request["expense_amount_inr"] = 20000

    response = client.post("/api/budget-shock", json=request)

    assert response.status_code == 200
    result = response.json()
    savings = result["recovery"]["savings_breakdown"]
    assert {item["category"] for item in savings} == {"Decor"}
    for item in result["category_comparison"]:
        if item["category"] in {
            "Catering & Beverages",
            "Venue Rental",
            "Custom Cake & Desserts",
            "Photography",
        }:
            assert item["projected_amount_inr"] == item["baseline_amount_inr"]


@pytest.mark.parametrize(
    ("category", "expected_change"),
    [
        ("  dEcOr   &   LiGhTiNg  ", -2500),
        ("Decor & Ambient Lighting", -2500),
        ("aUdIo & eNtErTaInMeNt", -2000),
        ("FAVORS", -5000),
    ],
)
def test_trade_off_case_and_whitespace_variants_use_existing_exact_allowlist(
    category: str,
    expected_change: int,
) -> None:
    request = {
        "budget_inr": 12000,
        "guest_count": 10,
        "baseline_total_inr": 10000,
        "allocations": [{"category": category, "amount_inr": 10000}],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == [category.strip()]
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    assert saver["allocations"][0]["change_inr"] == expected_change
    assert saver["estimated_total_inr"] == 10000 + expected_change
    assert saver["remaining_budget_inr"] == request["budget_inr"] - saver["estimated_total_inr"]
    assert saver["estimated_cost_per_guest_inr"] == round(saver["estimated_total_inr"] / request["guest_count"])


def test_trade_off_recognizes_ambient_styling_and_favors_stationery_aliases() -> None:
    allocations = [
        ("Decor & Ambient Styling", 22000),
        ("DJ & Sound Equipment", 10000),
        ("Favors & Stationery", 5000),
        ("Venue Rental", 25000),
        ("Gluten-Free Vegetarian Catering", 40000),
        ("Milestone Cake & Dessert Bar", 7000),
        ("Event Photography", 8000),
        ("Unclassified event expense", 6000),
    ]
    request = {
        "budget_inr": 130000,
        "guest_count": 40,
        "baseline_total_inr": sum(amount for _, amount in allocations),
        "allocations": [
    {"category": category, "amount_inr": amount}
    for category, amount in allocations
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == [
        "Decor & Ambient Styling",
        "DJ & Sound Equipment",
        "Favors & Stationery",
    ]
    saver = next(option for option in result["options"] if option["name"] == "Budget Saver")
    balanced = next(option for option in result["options"] if option["name"] == "Balanced")
    premium = next(option for option in result["options"] if option["name"] == "Premium")

    baseline_by_category = dict(allocations)
    saver_by_category = {line["category"]: line for line in saver["allocations"]}
    assert {
        category: saver_by_category[category]["change_inr"]
        for category, _ in allocations
    } == {
        "Decor & Ambient Styling": -5500,
        "DJ & Sound Equipment": -2000,
        "Favors & Stationery": -2500,
        "Venue Rental": 0,
        "Gluten-Free Vegetarian Catering": 0,
        "Milestone Cake & Dessert Bar": 0,
        "Event Photography": 0,
        "Unclassified event expense": 0,
    }
    assert saver["estimated_total_inr"] == 113000
    assert saver["estimated_total_inr"] == sum(
        line["proposed_amount_inr"] for line in saver["allocations"]
    )
    assert saver["estimated_total_inr"] <= request["budget_inr"]

    balanced_by_category = {line["category"]: line for line in balanced["allocations"]}
    assert {
        category: balanced_by_category[category]["proposed_amount_inr"]
        for category, _ in allocations
    } == baseline_by_category
    assert all(line["change_inr"] == 0 for line in balanced["allocations"])
    assert balanced["estimated_total_inr"] == request["baseline_total_inr"]

    premium_by_category = {line["category"]: line for line in premium["allocations"]}
    assert {
        category: premium_by_category[category]["change_inr"]
        for category, _ in allocations
    } == {
        "Decor & Ambient Styling": 4400,
        "DJ & Sound Equipment": 1500,
        "Favors & Stationery": 1100,
        "Venue Rental": 0,
        "Gluten-Free Vegetarian Catering": 0,
        "Milestone Cake & Dessert Bar": 0,
        "Event Photography": 0,
        "Unclassified event expense": 0,
    }
    assert premium["estimated_total_inr"] == request["budget_inr"]
    assert premium["estimated_total_inr"] == sum(
        line["proposed_amount_inr"] for line in premium["allocations"]
    )
    assert premium["estimated_total_inr"] <= request["budget_inr"]


@pytest.mark.parametrize(
    "category",
    [
        "Décor & Lighting",
        "Decor & Lighting!",
        "Themed Decor & Ambience",
        "Sound & Music Setup",
        "Favors & Digital Invites",
    ],
)
def test_trade_off_does_not_infer_unsupported_aliases_or_punctuation(category: str) -> None:
    request = {
        "budget_inr": 12000,
        "guest_count": 10,
        "baseline_total_inr": 10000,
        "allocations": [{"category": category, "amount_inr": 10000}],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    assert result["discretionary_categories_available"] == []
    for option in result["options"]:
        assert option["allocations"][0]["change_inr"] == 0


def test_trade_off_does_not_change_protected_event_categories() -> None:
    protected = [
        "Catering & Beverages",
        "Venue Rental",
        "Custom Cake & Desserts",
        "Photography",
        "Contingency Fund",
    ]
    request = {
        "budget_inr": 120000,
        "guest_count": 40,
        "baseline_total_inr": 100000,
        "allocations": [
            *({"category": category, "amount_inr": 10000} for category in protected),
            {"category": "Decor", "amount_inr": 50000},
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    for option in result["options"]:
        assert option["estimated_total_inr"] == sum(
            item["proposed_amount_inr"] for item in option["allocations"]
        )
        for line in option["allocations"]:
            if line["category"] in protected:
                assert line["proposed_amount_inr"] == line["baseline_amount_inr"]
                assert line["change_inr"] == 0


def test_trade_off_integer_cap_rounding_and_all_option_arithmetic() -> None:
    request = {
        "budget_inr": 12000,
        "guest_count": 16,
        "baseline_total_inr": 10001,
        "allocations": [
            {"category": "Decor", "amount_inr": 10001},
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    for option in response.json()["options"]:
        assert option["estimated_total_inr"] == sum(
            line["proposed_amount_inr"] for line in option["allocations"]
        )
        assert option["estimated_cost_per_guest_inr"] == round(option["estimated_total_inr"] / 16)
        assert option["remaining_budget_inr"] == 12000 - option["estimated_total_inr"]
        assert option["exceeds_budget"] is (option["estimated_total_inr"] > 12000)
        assert option["allocations"][0]["change_inr"] == (
            option["allocations"][0]["proposed_amount_inr"] - 10001
        )


def test_trade_off_and_budget_shock_endpoints_are_independent(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.main._generate_with_gemini",
        lambda _request: pytest.fail("Deterministic budget endpoints must not call Gemini."),
    )
    trade_request = trade_off_payload()
    shock_request = budget_shock_payload("fixed_expense")
    original_trade_request = json.loads(json.dumps(trade_request))
    original_shock_request = json.loads(json.dumps(shock_request))

    trade_response = client.post("/api/trade-offs", json=trade_request)
    shock_response = client.post("/api/budget-shock", json=shock_request)

    assert trade_response.status_code == 200
    assert shock_response.status_code == 200
    assert trade_response.json()["options"][1]["estimated_total_inr"] == 80000
    assert shock_response.json()["projected_total_inr"] == 110000
    assert trade_request == original_trade_request
    assert shock_request == original_shock_request


def test_cors_origins_are_configurable_and_trimmed() -> None:
    assert _cors_origins(" https://events.example.com/ , https://preview.example.com ") == [
        "https://events.example.com",
        "https://preview.example.com",
    ]
    assert _cors_origins("") == []


def test_invalid_inputs_are_rejected() -> None:
    invalid = payload()
    invalid["guest_count"] = 0
    response = client.post("/api/plan", json=invalid)
    assert response.status_code == 422


def test_unbounded_preference_entry_is_rejected() -> None:
    invalid = payload()
    invalid["food_preferences"] = ["x" * 41]
    response = client.post("/api/plan", json=invalid)
    assert response.status_code == 422


def test_past_event_date_is_rejected() -> None:
    invalid = payload()
    invalid["event_date"] = (date.today() - timedelta(days=1)).isoformat()
    response = client.post("/api/plan", json=invalid)
    assert response.status_code == 422


def test_invalid_generated_plan_fails_schema_validation() -> None:
    request = PlanRequest.model_validate(payload())
    try:
        AIPlan.model_validate({"overview": "", "allocations": []})
    except Exception as exc:
        assert exc.__class__.__name__ == "ValidationError"
    else:
        raise AssertionError("Malformed plan unexpectedly passed validation.")
    assert request.guest_count == 40


def test_gemini_response_schema_is_accepted_by_sdk() -> None:
    from google import genai

    client = genai.Client(api_key="test-key")
    schema = t_schema(client._api_client, AIPlan)
    assert schema is not None
    vendor_schema = AIPlan.model_json_schema()["$defs"]["VendorCategory"]
    assert "essential" in vendor_schema["required"]
    assert "default" not in vendor_schema["properties"]["essential"]


def test_gemini_parses_structured_response_and_json_text() -> None:
    response = {
        "overview": "A relaxed garden birthday.",
        "allocations": [
            {"category": "Food", "amount_inr": 25000, "rationale": "A seasonal menu."}
        ],
        "timeline": [
            {
                "title": "Confirm the menu",
                "due_date": "2026-12-01",
                "detail": "Check dietary requirements.",
            }
        ],
        "vendor_categories": [{"category": "Catering", "essential": True}],
    }
    event_date = date(2026, 12, 10)
    structured_plan = _parse_gemini_plan(response, None, event_date, date(2026, 10, 20))
    text_plan = _parse_gemini_plan(None, json.dumps(response), event_date, date(2026, 10, 20))
    assert structured_plan.vendor_categories[0].essential is True
    assert text_plan.allocations[0].amount_inr == 25000
    assert text_plan.timeline[0].due_date == date(2026, 12, 1)


def test_gemini_timeline_dates_are_clamped_to_pre_event_window() -> None:
    event_date = date(2026, 12, 10)
    current_date = date(2026, 10, 20)
    response = {
        "overview": "A relaxed garden birthday.",
        "allocations": [
            {"category": "Food", "amount_inr": 25000, "rationale": "A seasonal menu."}
        ],
        "timeline": [
            {"title": "Past deadline", "due_date": "2026-10-01", "detail": "Correct past date."},
            {"title": "Valid deadline", "due_date": "2026-11-15", "detail": "Keep valid date."},
            {"title": "After event", "due_date": "2026-12-20", "detail": "Correct late date."},
        ],
        "vendor_categories": [{"category": "Catering", "essential": True}],
    }
    plan = _parse_gemini_plan(response, None, event_date, current_date)
    assert [task.due_date for task in plan.timeline] == [
        current_date,
        date(2026, 11, 15),
        event_date - timedelta(days=1),
    ]


def test_short_notice_and_event_day_deadlines_remain_valid() -> None:
    event_date = date(2026, 10, 21)
    current_date = date(2026, 10, 20)
    plan = AIPlan(
        overview="A celebration planned at short notice.",
        allocations=[
            BudgetLine(category="Food", amount_inr=10000, rationale="Simple menu.")
        ],
        timeline=[
            {"title": "Book venue", "due_date": date(2026, 10, 1), "detail": "Confirm venue."},
            {"title": "Prepare essentials", "due_date": date(2026, 10, 22), "detail": "Pack supplies."},
        ],
        vendor_categories=[{"category": "Catering", "essential": True}],
    )
    short_notice_plan = _correct_timeline_dates(plan, event_date, current_date)
    assert [task.due_date for task in short_notice_plan.timeline] == [current_date, current_date]

    same_day_plan = _correct_timeline_dates(plan, current_date, current_date)
    assert all(task.due_date == current_date for task in same_day_plan.timeline)


def timeline_recovery_payload(event_date: date, due_date: date) -> dict:
    return {
        "event_date": event_date.isoformat(),
        "tasks": [
            {
                "title": "Confirm venue",
                "due_date": due_date.isoformat(),
                "detail": "Review access and booking details.",
            }
        ],
    }


def test_timeline_recovery_reschedules_overdue_task_and_preserves_original_dates() -> None:
    today = date(2026, 10, 20)
    request_data = timeline_recovery_payload(today + timedelta(days=45), today - timedelta(days=2))
    request = TimelineRecoveryRequest.model_validate(request_data)
    original_snapshot = request.model_dump(mode="json")

    result = _analyze_timeline_recovery(request, today)

    assert result.tasks[0].status == "needs_adjustment"
    assert result.tasks[0].is_overdue is True
    assert result.tasks[0].original_due_date == today - timedelta(days=2)
    assert result.tasks[0].suggested_due_date == today
    assert "deadline has passed" in result.tasks[0].explanation
    assert result.can_apply is True
    assert request.model_dump(mode="json") == original_snapshot


def test_timeline_recovery_does_not_flag_deadline_inside_window() -> None:
    today = date(2026, 10, 20)
    due_date = today + timedelta(days=2)
    request_data = timeline_recovery_payload(today + timedelta(days=5), due_date)
    request_data["tasks"][0]["title"] = "Review event supplies"
    request = TimelineRecoveryRequest.model_validate(request_data)

    result = _analyze_timeline_recovery(request, today)

    assert result.tasks[0].status == "on_schedule"
    assert result.tasks[0].is_overdue is False
    assert result.tasks[0].suggested_due_date == due_date
    assert result.can_apply is False


def test_timeline_recovery_flags_task_later_than_recommended_lead_time() -> None:
    today = date(2026, 10, 20)
    event_date = today + timedelta(days=60)
    request_data = timeline_recovery_payload(event_date, event_date - timedelta(days=5))
    original = json.loads(json.dumps(request_data))
    request = TimelineRecoveryRequest.model_validate(request_data)

    result = _analyze_timeline_recovery(request, today)

    assert result.tasks[0].status == "needs_adjustment"
    assert result.tasks[0].is_overdue is False
    assert result.tasks[0].suggested_due_date == event_date - timedelta(days=30)
    assert "30-day lead-time target" in result.tasks[0].explanation
    assert request.model_dump(mode="json") == original
    assert result.event_date == event_date


@pytest.mark.parametrize(
    ("category", "group", "amount_inr"),
    [
        ("Decor", "decor", 10001),
        ("Decor & Ambient Styling", "decor", 22000),
        ("Decor & Ambient Lighting", "decor", 14000),
        ("Decor & Styling", "decor", 20000),
        ("Decor & Lighting", "decor", 16000),
        ("Thematic Decor & Lighting", "decor", 14000),
        ("Decor, Lighting & Backdrop", "decor", 12000),
        ("Decor & Floral Accents", "decor", 10000),
        ("Floral Arrangements", "decor", 6000),
        ("Event Decorations", "decor", 5000),
        ("Table Decorations", "decor", 4000),
        ("Entertainment", "entertainment", 6000),
        ("Audio & Entertainment", "entertainment", 4000),
        ("Entertainment & Sound", "entertainment", 4000),
        ("Sound & Entertainment", "entertainment", 10000),
        ("Sound, DJ & Entertainment", "entertainment", 10000),
        ("Sound & DJ Setup", "entertainment", 5000),
        ("DJ & Sound Equipment", "dj", 8000),
        ("Music", "music", 2000),
        ("DJ", "dj", 2000),
        ("Favor", "favor", 1000),
        ("Favors", "favor", 2000),
        ("Favour", "favor", 3000),
        ("Favours", "favor", 4000),
        ("Favors & Stationery", "favor", 5000),
        ("Venue Rental & Setup", None, 2400),
        ("Venue Rental", None, 2700),
        ("Catering & Beverages", None, 3100),
        ("Catering (Vegetarian & Gluten-Free)", None, 3500),
        ("Custom Gluten-Free Birthday Cake", None, 3900),
        ("Specialty Cake & Desserts", None, 4300),
        ("Photography & Coverage", None, 4700),
        ("Photography Coverage", None, 5100),
        ("Contingency Fund", None, 5500),
        ("Unclassified expense", None, 5900),
    ],
)
def test_trade_off_canonical_aliases_reconcile_full_scenarios(
    category: str,
    group: str | None,
    amount_inr: int,
) -> None:
    rule_caps = {
        "decor": (25, 20),
        "entertainment": (20, 15),
        "music": (20, 15),
        "dj": (20, 15),
        "favor": (50, 25),
    }
    protected_allocations = [
        ("Venue Rental & Setup", 2400),
        ("Venue Rental", 2700),
        ("Catering & Beverages", 3100),
        ("Catering (Vegetarian & Gluten-Free)", 3500),
        ("Custom Gluten-Free Birthday Cake", 3900),
        ("Specialty Cake & Desserts", 4300),
        ("Photography & Coverage", 4700),
        ("Photography Coverage", 5100),
        ("Contingency Fund", 5500),
        ("Unclassified expense", 5900),
    ]
    allocations = [(category, amount_inr)] + [
        (name, amount) for name, amount in protected_allocations if name != category
    ]
    original_total = sum(amount for _, amount in allocations)
    budget = original_total + 100_000
    request = {
        "budget_inr": budget,
        "guest_count": 37,
        "baseline_total_inr": original_total,
        "allocations": [
            {"category": name, "amount_inr": amount}
            for name, amount in allocations
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200
    result = response.json()
    expected_discretionary = [category] if group is not None else []
    assert result["discretionary_categories_available"] == expected_discretionary
    assert len({line["category"] for line in result["options"][0]["allocations"]}) == len(allocations)

    saver_percent, premium_percent = rule_caps[group] if group is not None else (0, 0)
    expected_saver_change = -(amount_inr * saver_percent // 100)
    expected_premium_change = amount_inr * premium_percent // 100
    expected_changes = {
        name: (expected_saver_change if name == category else 0)
        for name, _ in allocations
    }
    options = {option["name"]: option for option in result["options"]}
    saver = options["Budget Saver"]
    balanced = options["Balanced"]
    premium = options["Premium"]

    for option in options.values():
        assert len(option["allocations"]) == len(allocations)
        assert option["estimated_total_inr"] == sum(
            line["proposed_amount_inr"] for line in option["allocations"]
        )
        assert option["estimated_cost_per_guest_inr"] == round(option["estimated_total_inr"] / 37)
        assert option["remaining_budget_inr"] == budget - option["estimated_total_inr"]
    assert {
        line["category"]: line["change_inr"] for line in saver["allocations"]
    } == expected_changes
    assert saver["estimated_total_inr"] == original_total + expected_saver_change
    assert -sum(line["change_inr"] for line in saver["allocations"]) == -expected_saver_change

    assert {
        line["category"]: line["proposed_amount_inr"]
        for line in balanced["allocations"]
    } == dict(allocations)
    assert all(line["change_inr"] == 0 for line in balanced["allocations"])
    assert balanced["estimated_total_inr"] == original_total

    expected_premium_changes = {
        name: (expected_premium_change if name == category else 0)
        for name, _ in allocations
    }
    assert {
        line["category"]: line["change_inr"] for line in premium["allocations"]
    } == expected_premium_changes
    assert premium["estimated_total_inr"] == original_total + expected_premium_change
    assert premium["estimated_total_inr"] <= budget
    for line in saver["allocations"]:
        if line["category"] != category:
            assert line["proposed_amount_inr"] == line["baseline_amount_inr"]
            assert line["change_inr"] == 0


@pytest.mark.parametrize(
    ("scenario", "allocations", "eligible_rules"),
    [
        (
            "birthday",
            [
                ("Themed Decor & Backdrop", 16000),
                ("Sound System & Music", 8000),
                ("Party Favors & Digital Invites", 4000),
                ("Venue Rental", 30000),
                ("Catering (Vegetarian & Gluten-Free)", 40000),
                ("Custom Gluten-Free Birthday Cake", 7000),
                ("Photography Coverage", 8000),
                ("Contingency Fund", 5000),
                ("Unclassified birthday expense", 2000),
            ],
            {
                "Themed Decor & Backdrop": ("decor", 25, 20),
                "Sound System & Music": ("music", 20, 15),
                "Party Favors & Digital Invites": ("favor", 50, 25),
            },
        ),
        (
            "wedding",
            [
                ("Floral & Mandap Decor", 25000),
                ("Audio & Digital Invitations", 12000),
                ("Venue Rental & Setup", 45000),
                ("Catering & Beverages", 55000),
                ("Specialty Cake & Desserts", 8000),
                ("Photography & Coverage", 18000),
                ("Ceremonial Essentials", 10000),
                ("Contingency Fund", 7000),
                ("Unclassified wedding expense", 3000),
            ],
            {
                "Floral & Mandap Decor": ("decor", 25, 20),
                "Audio & Digital Invitations": ("entertainment", 20, 15),
            },
        ),
    ],
)
def test_trade_off_birthday_and_wedding_alias_scenarios(
    scenario: str,
    allocations: list[tuple[str, int]],
    eligible_rules: dict[str, tuple[str, int, int]],
) -> None:
    original_total = sum(amount for _, amount in allocations)
    budget = original_total + 20000
    request = {
        "budget_inr": budget,
        "guest_count": 50,
        "baseline_total_inr": original_total,
        "allocations": [
            {"category": category, "amount_inr": amount}
            for category, amount in allocations
        ],
    }

    response = client.post("/api/trade-offs", json=request)

    assert response.status_code == 200, scenario
    result = response.json()
    expected_recognized = [category for category, _ in allocations if category in eligible_rules]
    assert result["discretionary_categories_available"] == expected_recognized
    options = {option["name"]: option for option in result["options"]}
    saver = options["Budget Saver"]
    balanced = options["Balanced"]
    premium = options["Premium"]
    baseline_by_category = dict(allocations)

    expected_saver_changes = {}
    expected_premium_changes = {}
    for category, amount in allocations:
        if category in eligible_rules:
            _group, saver_percent, premium_percent = eligible_rules[category]
            expected_saver_changes[category] = -(amount * saver_percent // 100)
            expected_premium_changes[category] = amount * premium_percent // 100
        else:
            expected_saver_changes[category] = 0
            expected_premium_changes[category] = 0

    assert {
        line["category"]: line["change_inr"] for line in saver["allocations"]
    } == expected_saver_changes
    total_savings = -sum(expected_saver_changes.values())
    assert total_savings > 0
    assert saver["estimated_total_inr"] == original_total - total_savings
    assert saver["estimated_total_inr"] == sum(
        line["proposed_amount_inr"] for line in saver["allocations"]
    )

    assert {
        line["category"]: line["proposed_amount_inr"]
        for line in balanced["allocations"]
    } == baseline_by_category
    assert all(line["change_inr"] == 0 for line in balanced["allocations"])
    assert balanced["estimated_total_inr"] == original_total

    assert {
        line["category"]: line["change_inr"] for line in premium["allocations"]
    } == expected_premium_changes
    assert premium["estimated_total_inr"] == original_total + sum(
        expected_premium_changes.values()
    )
    assert premium["estimated_total_inr"] <= budget
    for option in options.values():
        assert option["estimated_total_inr"] == sum(
            line["proposed_amount_inr"] for line in option["allocations"]
        )
        assert option["estimated_cost_per_guest_inr"] == round(
            option["estimated_total_inr"] / request["guest_count"]
        )
        assert option["remaining_budget_inr"] == budget - option["estimated_total_inr"]


def test_timeline_recovery_with_tomorrow_event_reports_unavailable_lead_time() -> None:
    today = date(2026, 10, 20)
    request = TimelineRecoveryRequest.model_validate(
        timeline_recovery_payload(today + timedelta(days=1), today)
    )

    result = _analyze_timeline_recovery(request, today)

    assert result.tasks[0].status == "unresolved"
    assert result.tasks[0].suggested_due_date is None
    assert "30-day lead time" in result.tasks[0].explanation
    assert "No task-duration, dependency, or operational feasibility assessment" in result.tasks[0].explanation
    assert result.has_unresolved_conflicts is True


@pytest.mark.parametrize(
    ("event_offset", "due_offset", "expected_suggestion"),
    [
        (0, 1, 0),
        (1, 2, 0),
    ],
)
def test_timeline_recovery_same_day_and_next_day_windows(
    event_offset: int,
    due_offset: int,
    expected_suggestion: int,
) -> None:
    today = date(2026, 10, 20)
    event_date = today + timedelta(days=event_offset)
    request = TimelineRecoveryRequest.model_validate(
        timeline_recovery_payload(event_date, today + timedelta(days=due_offset))
    )

    result = _analyze_timeline_recovery(request, today)

    assert result.window_available is True
    assert result.window_end == today + timedelta(days=expected_suggestion)
    assert result.tasks[0].status == "unresolved"
    assert result.tasks[0].suggested_due_date is None


@pytest.mark.parametrize(
    "changes",
    [
        {"event_date": "not-a-date"},
        {"tasks": [{"title": "Confirm venue", "due_date": "not-a-date", "detail": "Review access."}]},
        {"tasks": [{"title": "", "due_date": "2026-10-20", "detail": "Review access."}]},
        {"tasks": [{"title": "Confirm venue", "due_date": "2026-10-20", "detail": " "}]},
    ],
)
def test_timeline_recovery_rejects_malformed_input(changes) -> None:
    request = timeline_recovery_payload(date(2026, 10, 25), date(2026, 10, 20))
    request.update(changes)

    response = client.post("/api/timeline-recovery", json=request)

    assert response.status_code == 422


def test_timeline_recovery_reports_unresolved_when_event_is_past() -> None:
    today = date(2026, 10, 20)
    request = TimelineRecoveryRequest.model_validate(
        timeline_recovery_payload(today - timedelta(days=1), today - timedelta(days=2))
    )

    result = _analyze_timeline_recovery(request, today)

    assert result.window_available is False
    assert result.window_end is None
    assert result.has_unresolved_conflicts is True
    assert result.tasks[0].is_overdue is True
    assert result.tasks[0].status == "unresolved"
    assert result.tasks[0].suggested_due_date is None
    assert "no valid recovery date" in result.tasks[0].explanation
    assert result.can_apply is False
    assert result.event_date == today - timedelta(days=1)


def test_timeline_recovery_api_does_not_call_gemini(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "app.main._generate_with_gemini",
        lambda _request: pytest.fail("Timeline recovery must not call Gemini."),
    )
    today = date.today()
    request = timeline_recovery_payload(today + timedelta(days=3), today - timedelta(days=1))
    request["tasks"][0]["title"] = "Review event supplies"

    response = client.post("/api/timeline-recovery", json=request)

    assert response.status_code == 200
    assert response.json()["tasks"][0]["suggested_due_date"] == today.isoformat()


def test_gemini_validation_logs_only_safe_issue_metadata(monkeypatch, caplog) -> None:
    secret_marker = "test-api-key-must-not-be-logged"
    private_marker = "private event description must not be logged"
    monkeypatch.setenv("GEMINI_API_KEY", secret_marker)

    def invalid_plan(_request):
        return AIPlan.model_validate(
            {
                "overview": private_marker,
                "allocations": [],
                "timeline": [],
                "vendor_categories": [],
            }
        )

    monkeypatch.setattr("app.main._generate_with_gemini", invalid_plan)
    request_data = payload()
    request_data["description"] = private_marker
    request = PlanRequest.model_validate(request_data)
    with caplog.at_level(logging.WARNING, logger="app.main"):
        with pytest.raises(HTTPException) as error:
            create_plan(request)

    assert error.value.status_code == 502
    assert "Gemini plan validation failed" in caplog.text
    assert secret_marker not in caplog.text
    assert private_marker not in caplog.text


def test_unavailable_gemini_model_returns_actionable_error(monkeypatch) -> None:
    class MissingModelError(Exception):
        code = 404

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def missing_model(_request):
        raise MissingModelError()

    monkeypatch.setattr("app.main._generate_with_gemini", missing_model)
    with pytest.raises(HTTPException) as error:
        create_plan(PlanRequest.model_validate(payload()))
    assert error.value.status_code == 502
    assert "GEMINI_MODEL" in error.value.detail


def test_gemini_upstream_503_returns_labeled_demo_fallback(monkeypatch, caplog) -> None:
    secret_detail = "private provider response detail"

    class ProviderServerError(Exception):
        code = 503

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fail_upstream(_request):
        raise ProviderServerError(secret_detail)

    monkeypatch.setattr("app.main._generate_with_gemini", fail_upstream)
    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = client.post("/api/plan", json=payload())

    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "demo"
    assert result["fallback_reason"] == (
        "Gemini is temporarily unavailable. This is a deterministic demo fallback, "
        "not an AI-generated plan. You can retry later."
    )
    assert result["overview"]
    assert result["budget"]["total_estimated_inr"] == sum(
        item["amount_inr"] for item in result["budget"]["allocations"]
    )
    assert secret_detail not in response.text
    assert secret_detail not in caplog.text
