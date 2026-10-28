from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

load_dotenv()
logger = logging.getLogger(__name__)

EventType = Literal["Birthday", "Wedding", "Corporate event", "Community gathering", "Anniversary", "Other"]
VenuePreference = Literal["At home", "Outdoor", "Banquet hall", "Community space", "Flexible"]
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def _cors_origins(value: str | None = None) -> list[str]:
    configured = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS) if value is None else value
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


def _round_rupees(amount: Decimal) -> int:
    return int(amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


class PlanRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    event_type: EventType
    description: str = Field(min_length=8, max_length=500)
    event_date: date
    budget_inr: int = Field(ge=1_000, le=10_000_000)
    guest_count: int = Field(ge=1, le=10_000)
    venue_preference: VenuePreference
    food_preferences: list[str] = Field(default_factory=list, max_length=12)
    dietary_preferences: list[str] = Field(default_factory=list, max_length=12)
    cultural_preferences: str = Field(default="", max_length=300)
    accessibility_preferences: str = Field(default="", max_length=300)

    @field_validator("event_date")
    @classmethod
    def event_date_must_not_be_in_the_past(cls, value: date) -> date:
        if value < date.today():
            raise ValueError("Event date must be today or in the future.")
        return value

    @field_validator("food_preferences", "dietary_preferences")
    @classmethod
    def preference_values_must_be_bounded(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 40 for value in values):
            raise ValueError("Preference entries must contain between 1 and 40 characters.")
        return values


class BudgetLine(BaseModel):
    category: str = Field(min_length=1, max_length=80)
    amount_inr: int = Field(ge=0)
    rationale: str = Field(min_length=1, max_length=300)


class TimelineTask(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    due_date: date
    detail: str = Field(min_length=1, max_length=300)


class VendorCategory(BaseModel):
    category: str = Field(min_length=1, max_length=80)
    essential: bool


class AIPlan(BaseModel):
    overview: str = Field(min_length=1, max_length=800)
    allocations: list[BudgetLine] = Field(min_length=1, max_length=12)
    timeline: list[TimelineTask] = Field(min_length=1, max_length=12)
    vendor_categories: list[VendorCategory] = Field(min_length=1, max_length=10)


class TradeOffAllocationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: str = Field(min_length=1, max_length=80)
    amount_inr: int = Field(ge=0, le=10_000_000)


class TradeOffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    budget_inr: int = Field(ge=1_000, le=10_000_000)
    guest_count: int = Field(ge=1, le=10_000)
    allocations: list[TradeOffAllocationInput] = Field(min_length=1, max_length=12)
    baseline_total_inr: int | None = Field(default=None, ge=0, le=10_000_000)

    @field_validator("allocations")
    @classmethod
    def allocation_categories_are_unique(
        cls,
        values: list[TradeOffAllocationInput],
    ) -> list[TradeOffAllocationInput]:
        normalized = [" ".join(item.category.casefold().split()) for item in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Allocation categories must be unique after trimming and case normalization.")
        return values

    @model_validator(mode="after")
    def baseline_is_consistent(self) -> "TradeOffRequest":
        allocation_total = sum(item.amount_inr for item in self.allocations)
        if allocation_total > self.budget_inr:
            raise ValueError("Baseline allocations cannot exceed the event budget.")
        if self.baseline_total_inr is not None and self.baseline_total_inr != allocation_total:
            raise ValueError(
                "Baseline total must exactly match the allocation sum "
                "(allowed rounding tolerance: ₹0)."
            )
        return self


TradeOffOptionName = Literal["Budget Saver", "Balanced", "Premium"]


class TradeOffAllocation(BaseModel):
    category: str
    baseline_amount_inr: int = Field(ge=0)
    proposed_amount_inr: int = Field(ge=0)
    change_inr: int

    @model_validator(mode="after")
    def change_matches_amounts(self) -> "TradeOffAllocation":
        if self.change_inr != self.proposed_amount_inr - self.baseline_amount_inr:
            raise ValueError("Category change does not match the baseline and proposed amounts.")
        return self


class TradeOffOption(BaseModel):
    name: TradeOffOptionName
    explanation: str
    estimate_notice: str
    allocations: list[TradeOffAllocation]
    estimated_total_inr: int = Field(ge=0)
    estimated_cost_per_guest_inr: int = Field(ge=0)
    remaining_budget_inr: int
    exceeds_budget: bool

    @model_validator(mode="after")
    def option_arithmetic_is_consistent(self) -> "TradeOffOption":
        total = sum(item.proposed_amount_inr for item in self.allocations)
        if self.estimated_total_inr != total:
            raise ValueError("Option total must equal the sum of its category allocations.")
        return self


class TradeOffResult(BaseModel):
    budget_inr: int
    guest_count: int
    options: list[TradeOffOption] = Field(min_length=3, max_length=3)
    discretionary_categories_available: list[str]
    notice: str

    @model_validator(mode="after")
    def options_are_ordered_and_within_budget(self) -> "TradeOffResult":
        if [option.name for option in self.options] != ["Budget Saver", "Balanced", "Premium"]:
            raise ValueError("Trade-off options must be ordered Budget Saver, Balanced, Premium.")
        for option in self.options:
            if option.estimated_cost_per_guest_inr != round(option.estimated_total_inr / self.guest_count):
                raise ValueError("Option per-guest cost does not match its total and guest count.")
            if option.remaining_budget_inr != self.budget_inr - option.estimated_total_inr:
                raise ValueError("Option remaining budget does not match its total and event budget.")
            if option.exceeds_budget != (option.estimated_total_inr > self.budget_inr):
                raise ValueError("Option over-budget status does not match its total.")
            if option.name == "Premium" and option.exceeds_budget:
                raise ValueError("Premium option cannot exceed the event budget.")
        return self


class RecoveryTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=120)
    due_date: date
    detail: str = Field(min_length=1, max_length=300)


class TimelineRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_date: date
    tasks: list[RecoveryTaskInput] = Field(min_length=1, max_length=12)


RecoveryStatus = Literal["on_schedule", "needs_adjustment", "unresolved"]


class TimelineRecoveryTask(BaseModel):
    title: str
    detail: str
    original_due_date: date
    suggested_due_date: date | None
    is_overdue: bool
    status: RecoveryStatus
    explanation: str


class TimelineRecoveryResult(BaseModel):
    event_date: date
    current_date: date
    window_start: date
    window_end: date | None
    window_available: bool
    tasks: list[TimelineRecoveryTask]
    has_unresolved_conflicts: bool
    can_apply: bool
    notice: str

    @model_validator(mode="after")
    def recovery_dates_are_consistent(self) -> "TimelineRecoveryResult":
        expected_window_available = self.event_date >= self.current_date
        expected_window_end = (
            self.event_date
            if self.event_date == self.current_date
            else self.event_date - timedelta(days=1)
        ) if expected_window_available else None
        if self.window_start != self.current_date:
            raise ValueError("Recovery window must begin on the current date.")
        if self.window_available != expected_window_available or self.window_end != expected_window_end:
            raise ValueError("Recovery window does not match the event and current dates.")
        for task in self.tasks:
            if task.is_overdue != (task.original_due_date < self.current_date):
                raise ValueError("Overdue status does not match the original task deadline.")
            if (task.status == "unresolved") != (task.suggested_due_date is None):
                raise ValueError("Unresolved task status must match the availability of a suggested date.")
            if task.status == "on_schedule" and task.suggested_due_date != task.original_due_date:
                raise ValueError("On-schedule tasks must retain their original deadline.")
            if task.suggested_due_date is not None:
                if not self.window_available or not self.current_date <= task.suggested_due_date <= self.window_end:
                    raise ValueError("Suggested task date falls outside the valid recovery window.")
        expected_unresolved = not self.window_available or any(task.status == "unresolved" for task in self.tasks)
        if self.has_unresolved_conflicts != expected_unresolved:
            raise ValueError("Unresolved conflict status does not match the task suggestions.")
        expected_can_apply = (
            not expected_unresolved
            and any(task.suggested_due_date != task.original_due_date for task in self.tasks)
        )
        if self.can_apply != expected_can_apply:
            raise ValueError("Recovery application status does not match the proposed dates.")
        return self


ShockType = Literal["category_percentage", "additional_guests", "fixed_expense"]


class ShockAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: str = Field(min_length=1, max_length=80)
    amount_inr: int = Field(ge=0, le=10_000_000)
    percentage: Decimal | None = Field(default=None, ge=0, le=100, max_digits=5, decimal_places=1)
    rationale: str | None = Field(default=None, max_length=300)


class BudgetShockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    baseline_budget_inr: int = Field(ge=1_000, le=10_000_000)
    baseline_plan_cost_inr: int | None = Field(default=None, ge=0, le=10_000_000)
    baseline_guest_count: int = Field(ge=1, le=10_000)
    baseline_allocations: list[ShockAllocation] = Field(min_length=1, max_length=12)
    shock_type: ShockType
    category: str | None = Field(default=None, min_length=1, max_length=80)
    increase_percent: Decimal | None = Field(default=None, gt=0, le=500, max_digits=6, decimal_places=2)
    additional_guests: int | None = Field(default=None, ge=1, le=10_000)
    expense_label: str | None = Field(default=None, min_length=1, max_length=80)
    expense_amount_inr: int | None = Field(default=None, ge=1, le=10_000_000)

    @field_validator("baseline_allocations")
    @classmethod
    def allocation_categories_are_unique(cls, values: list[ShockAllocation]) -> list[ShockAllocation]:
        categories = [allocation.category.casefold() for allocation in values]
        if len(categories) != len(set(categories)):
            raise ValueError("Baseline allocation categories must be unique.")
        return values

    @model_validator(mode="after")
    def validate_baseline_and_shock_parameters(self) -> "BudgetShockRequest":
        allocation_total = sum(allocation.amount_inr for allocation in self.baseline_allocations)
        if allocation_total > self.baseline_budget_inr:
            raise ValueError("Baseline allocations cannot exceed the baseline budget.")
        if self.baseline_plan_cost_inr is not None and self.baseline_plan_cost_inr != allocation_total:
            raise ValueError(
                "Baseline plan cost must exactly match the allocation total "
                "(allowed rounding tolerance: ₹0)."
            )
        for allocation in self.baseline_allocations:
            if allocation.percentage is not None:
                expected_percentage = (
                    (Decimal(allocation.amount_inr * 100) / Decimal(allocation_total))
                    .quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                    if allocation_total
                    else Decimal("0.0")
                )
                if allocation.percentage != expected_percentage:
                    raise ValueError(
                        f"Baseline percentage for {allocation.category} must match its share "
                        "of the allocation total rounded to one decimal place."
                    )
        if self.baseline_guest_count + (self.additional_guests or 0) > 10_000:
            raise ValueError("Projected guest count cannot exceed 10,000.")

        parameters = {
            "category_percentage": {"category", "increase_percent"},
            "additional_guests": {"additional_guests"},
            "fixed_expense": {"expense_label", "expense_amount_inr"},
        }
        supplied = {
            name
            for name in ("category", "increase_percent", "additional_guests", "expense_label", "expense_amount_inr")
            if getattr(self, name) is not None
        }
        required = parameters[self.shock_type]
        if supplied != required:
            raise ValueError(f"{self.shock_type} requires only these parameters: {', '.join(sorted(required))}.")
        if self.shock_type == "category_percentage" and not any(
            item.category.casefold() == self.category.casefold() for item in self.baseline_allocations
        ):
            raise ValueError("Selected category must exist in the baseline allocations.")
        if self.shock_type == "category_percentage":
            selected = next(
                item for item in self.baseline_allocations
                if item.category.casefold() == self.category.casefold()
            )
            increase = int(
                (Decimal(selected.amount_inr) * self.increase_percent / Decimal(100))
                .quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )
            if increase > 10_000_000:
                raise ValueError("A simulated shock cannot exceed ₹10,000,000.")
        if self.shock_type == "additional_guests":
            baseline_total = sum(item.amount_inr for item in self.baseline_allocations)
            estimated_additional_cost = _round_rupees(
                Decimal(baseline_total) / Decimal(self.baseline_guest_count)
            ) * self.additional_guests
            if estimated_additional_cost > 10_000_000:
                raise ValueError("Estimated additional-guest costs cannot exceed ₹10,000,000.")
        if self.shock_type == "fixed_expense" and any(
            item.category.casefold() == self.expense_label.casefold() for item in self.baseline_allocations
        ):
            raise ValueError("Expense label must be different from every baseline category.")
        return self


class RecoverySaving(BaseModel):
    category: str
    before_reduction_amount_inr: int = Field(ge=0)
    cap_basis_amount_inr: int = Field(ge=0)
    reduction_inr: int = Field(ge=0)
    revised_amount_inr: int = Field(ge=0)
    maximum_reduction_percent: int = Field(ge=0, le=100)


class ShockCategoryComparison(BaseModel):
    category: str
    baseline_amount_inr: int = Field(ge=0)
    projected_amount_inr: int = Field(ge=0)
    change_inr: int


class RecoveryPlan(BaseModel):
    fits_within_budget: bool
    available_budget_inr: int = Field(ge=1_000)
    proposed_total_inr: int = Field(ge=0)
    remaining_budget_inr: int = Field(ge=0)
    savings_breakdown: list[RecoverySaving]
    total_savings_inr: int = Field(ge=0)
    remaining_shortfall_inr: int = Field(ge=0)
    notice: str

    @model_validator(mode="after")
    def calculated_totals_are_consistent(self) -> "RecoveryPlan":
        if self.total_savings_inr != sum(item.reduction_inr for item in self.savings_breakdown):
            raise ValueError("Recovery savings total does not match the breakdown.")
        if any(
            item.revised_amount_inr != item.before_reduction_amount_inr - item.reduction_inr
            or item.reduction_inr > item.cap_basis_amount_inr * item.maximum_reduction_percent // 100
            for item in self.savings_breakdown
        ):
            raise ValueError("Recovery reductions must respect category amounts and caps.")
        if self.fits_within_budget != (self.proposed_total_inr <= self.available_budget_inr):
            raise ValueError("Recovery success must match the proposed total and available budget.")
        if self.remaining_shortfall_inr != max(0, self.proposed_total_inr - self.available_budget_inr):
            raise ValueError("Recovery shortfall does not match the proposed total.")
        if self.remaining_budget_inr != max(0, self.available_budget_inr - self.proposed_total_inr):
            raise ValueError("Recovery remaining budget does not match the proposed total.")
        return self


class BudgetShockResult(BaseModel):
    shock_type: ShockType
    baseline_budget_inr: int
    original_allocation_total_inr: int
    projected_total_inr: int
    available_budget_inr: int
    projected_gap_inr: int
    projected_guest_count: int = Field(ge=1, le=10_000)
    projected_cost_per_guest_inr: int
    shock_cost_inr: int
    shock_description: str
    category_comparison: list[ShockCategoryComparison]
    recovery: RecoveryPlan
    estimate_notice: str

    @model_validator(mode="after")
    def projected_totals_are_consistent(self) -> "BudgetShockResult":
        if self.baseline_budget_inr != self.available_budget_inr:
            raise ValueError("Available budget does not match the baseline budget.")
        if self.projected_total_inr != self.original_allocation_total_inr + self.shock_cost_inr:
            raise ValueError("Projected total does not match the original total and shock.")
        if self.projected_gap_inr != max(0, self.projected_total_inr - self.available_budget_inr):
            raise ValueError("Projected budget gap does not match the available budget.")
        if sum(item.baseline_amount_inr for item in self.category_comparison) != self.original_allocation_total_inr:
            raise ValueError("Baseline category comparison does not match the original total.")
        if sum(item.projected_amount_inr for item in self.category_comparison) != self.projected_total_inr:
            raise ValueError("Projected category comparison does not match the projected total.")
        if self.recovery.proposed_total_inr != self.projected_total_inr - self.recovery.total_savings_inr:
            raise ValueError("Recovery total does not match the projected total and savings.")
        if self.recovery.available_budget_inr != self.available_budget_inr:
            raise ValueError("Recovery budget does not match the available budget.")
        if self.recovery.remaining_budget_inr != max(0, self.available_budget_inr - self.recovery.proposed_total_inr):
            raise ValueError("Recovery remaining budget does not match the available budget.")
        if self.projected_cost_per_guest_inr != _round_rupees(
            Decimal(self.projected_total_inr) / Decimal(self.projected_guest_count)
        ):
            raise ValueError("Projected cost per guest does not match the projected total and guest count.")
        return self


app = FastAPI(title="EventEase AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _savings_suggestions(lines: list[BudgetLine], budget: int) -> list[str]:
    total = sum(line.amount_inr for line in lines)
    if total <= budget:
        return ["Keep a 5–10% contingency aside for last-minute changes."]

    suggestions: list[str] = []
    flexible = {
        "decor": "Choose reusable or DIY decor and focus it around one photo-worthy area.",
        "photography": "Book a shorter photography slot or ask guests to share candid photos.",
        "entertainment": "Use a curated playlist instead of a full live-entertainment setup.",
        "favors": "Skip individual favors or choose one useful, locally sourced takeaway.",
        "venue": "Compare community spaces or host at home to lower venue costs.",
        "food": "Offer a focused seasonal menu and confirm headcount before finalizing catering.",
    }
    for line in sorted(lines, key=lambda item: item.amount_inr, reverse=True):
        key = line.category.lower()
        tip = next((value for name, value in flexible.items() if name in key), None)
        if tip and tip not in suggestions:
            suggestions.append(tip)
        if len(suggestions) == 3:
            break
    if not suggestions:
        suggestions.append("Reduce optional add-ons and keep a smaller contingency to bring the plan within budget.")
    return suggestions


def _fit_allocations_to_budget(plan: AIPlan, budget: int) -> AIPlan:
    total = sum(line.amount_inr for line in plan.allocations)
    if total <= budget:
        return plan

    scaled: list[tuple[BudgetLine, int, int]] = []
    for line in plan.allocations:
        amount, remainder = divmod(line.amount_inr * budget, total)
        scaled.append((line, amount, remainder))

    remaining = budget - sum(amount for _, amount, _ in scaled)
    for index in sorted(range(len(scaled)), key=lambda item: scaled[item][2], reverse=True)[:remaining]:
        line, amount, remainder = scaled[index]
        scaled[index] = (line, amount + 1, remainder)

    logger.warning("Generated allocations exceeded the requested budget; adjusted amounts to fit.")
    return plan.model_copy(
        update={
            "allocations": [
                line.model_copy(update={"amount_inr": amount})
                for line, amount, _ in scaled
            ]
        }
    )


def _make_result(request: PlanRequest, plan: AIPlan, mode: Literal["ai", "demo"]) -> dict:
    plan = _fit_allocations_to_budget(plan, request.budget_inr)
    total = sum(line.amount_inr for line in plan.allocations)
    budget = request.budget_inr
    return {
        "mode": mode,
        "overview": plan.overview,
        "event": {
            "type": request.event_type,
            "date": request.event_date.isoformat(),
            "guest_count": request.guest_count,
            "venue_preference": request.venue_preference,
        },
        "budget": {
            "budget_inr": budget,
            "total_estimated_inr": total,
            "remaining_inr": budget - total,
            "over_budget": total > budget,
            "cost_per_guest_inr": round(total / request.guest_count),
            "allocations": [
                {
                    **line.model_dump(),
                    "percentage": (
                        float(
                            (Decimal(line.amount_inr * 100) / Decimal(total))
                            .quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                        )
                        if total
                        else 0
                    ),
                }
                for line in plan.allocations
            ],
            "savings_suggestions": _savings_suggestions(plan.allocations, budget),
        },
        "timeline": [task.model_dump(mode="json") for task in plan.timeline],
        "vendor_categories": [vendor.model_dump() for vendor in plan.vendor_categories],
        "vendor_notice": "Vendor categories are planning prompts only. Availability and prices are not confirmed.",
    }


TRADE_OFF_CATEGORY_RULES = {
    "decor": {"saver_percent": 25, "premium_percent": 20, "priority": 0},
    "entertainment": {"saver_percent": 20, "premium_percent": 15, "priority": 1},
    "music": {"saver_percent": 20, "premium_percent": 15, "priority": 2},
    "dj": {"saver_percent": 20, "premium_percent": 15, "priority": 3},
    "favor": {"saver_percent": 50, "premium_percent": 25, "priority": 4},
}

TRADE_OFF_CATEGORY_GROUPS = {
    "decor": "decor",
    "decor & ambient lighting": "decor",
    "decor & ambient styling": "decor",
    "decor & styling": "decor",
    "decor & lighting": "decor",
    "decor, lighting & backdrop": "decor",
    "thematic decor & lighting": "decor",
    "themed decor & backdrop": "decor",
    "decor & floral accents": "decor",
    "floral & mandap decor": "decor",
    "floral arrangements": "decor",
    "event decorations": "decor",
    "table decorations": "decor",
    "entertainment": "entertainment",
    "audio & entertainment": "entertainment",
    "audio & digital invitations": "entertainment",
    "entertainment & sound": "entertainment",
    "sound & entertainment": "entertainment",
    "sound, dj & entertainment": "entertainment",
    "sound & dj setup": "entertainment",
    "dj & sound equipment": "dj",
    "music": "music",
    "sound system & music": "music",
    "dj": "dj",
    "favor": "favor",
    "favors": "favor",
    "favour": "favor",
    "favours": "favor",
    "favors & stationery": "favor",
    "party favors & digital invites": "favor",
}


def _trade_off_category_key(category: str) -> str:
    return " ".join(category.casefold().split())


def _trade_off_category_group(category: str) -> str | None:
    return TRADE_OFF_CATEGORY_GROUPS.get(_trade_off_category_key(category))


def _make_trade_off_option(
    name: TradeOffOptionName,
    explanation: str,
    baseline: list[TradeOffAllocationInput],
    proposed_amounts: dict[str, int],
    budget: int,
    guest_count: int,
) -> TradeOffOption:
    allocations = [
        TradeOffAllocation(
            category=item.category,
            baseline_amount_inr=item.amount_inr,
            proposed_amount_inr=proposed_amounts[item.category],
            change_inr=proposed_amounts[item.category] - item.amount_inr,
        )
        for item in baseline
    ]
    total = sum(item.proposed_amount_inr for item in allocations)
    return TradeOffOption(
        name=name,
        explanation=explanation,
        estimate_notice="All amounts are planning estimates, not verified vendor quotations.",
        allocations=allocations,
        estimated_total_inr=total,
        estimated_cost_per_guest_inr=round(total / guest_count),
        remaining_budget_inr=budget - total,
        exceeds_budget=total > budget,
    )


def _compare_trade_offs(request: TradeOffRequest) -> TradeOffResult:
    baseline = request.allocations
    baseline_amounts = {item.category: item.amount_inr for item in baseline}
    recognized = [
        item for item in baseline
        if _trade_off_category_group(item.category) is not None
    ]
    discretionary_categories = [item.category for item in recognized]

    saver_amounts = baseline_amounts.copy()
    saver_savings: list[tuple[str, int]] = []
    for item in recognized:
        rule = TRADE_OFF_CATEGORY_RULES[TRADE_OFF_CATEGORY_GROUPS[_trade_off_category_key(item.category)]]
        reduction = item.amount_inr * rule["saver_percent"] // 100
        if reduction:
            saver_amounts[item.category] -= reduction
            saver_savings.append((item.category, reduction))
    if saver_savings:
        saver_explanation = "Suggested discretionary reductions within the documented category caps: " + ", ".join(
            f"{category} −₹{amount:,}" for category, amount in saver_savings
        ) + ". These are trade-offs, not guaranteed harmless changes."
    else:
        saver_explanation = (
            "No supported discretionary categories with reducible amounts were found; "
            "the current allocations are unchanged."
        )

    balanced_explanation = "Reference option: reproduces the current validated plan allocations exactly."
    premium_amounts = baseline_amounts.copy()
    headroom = request.budget_inr - sum(baseline_amounts.values())
    premium_additions: list[tuple[str, int]] = []
    for item in sorted(
        recognized,
        key=lambda allocation: TRADE_OFF_CATEGORY_RULES[
            TRADE_OFF_CATEGORY_GROUPS[_trade_off_category_key(allocation.category)]
        ]["priority"],
    ):
        rule = TRADE_OFF_CATEGORY_RULES[TRADE_OFF_CATEGORY_GROUPS[_trade_off_category_key(item.category)]]
        cap = item.amount_inr * rule["premium_percent"] // 100
        increase = min(cap, headroom)
        if increase > 0:
            premium_amounts[item.category] += increase
            headroom -= increase
            premium_additions.append((item.category, increase))
        if headroom == 0:
            break
    if premium_additions:
        premium_explanation = "Uses available budget headroom for supported discretionary categories within their caps: " + ", ".join(
            f"{category} +₹{amount:,}" for category, amount in premium_additions
        ) + ". Any remaining headroom is left unspent because applicable category caps were reached."
    elif sum(baseline_amounts.values()) == request.budget_inr:
        premium_explanation = "No budget headroom is available; allocations remain unchanged."
    elif not recognized:
        premium_explanation = (
            "No supported discretionary categories were found; allocations remain unchanged "
            "and available headroom is left unspent."
        )
    else:
        premium_explanation = (
            "Supported categories have no increase capacity under their caps; allocations remain unchanged "
            "and available headroom is left unspent."
        )

    return TradeOffResult(
        budget_inr=request.budget_inr,
        guest_count=request.guest_count,
        options=[
            _make_trade_off_option(
                "Budget Saver", saver_explanation, baseline, saver_amounts,
                request.budget_inr, request.guest_count,
            ),
            _make_trade_off_option(
                "Balanced", balanced_explanation, baseline, baseline_amounts,
                request.budget_inr, request.guest_count,
            ),
            _make_trade_off_option(
                "Premium", premium_explanation, baseline, premium_amounts,
                request.budget_inr, request.guest_count,
            ),
        ],
        discretionary_categories_available=discretionary_categories,
        notice=(
            "This is a deterministic planning comparison, not a live vendor-price search. "
            "All figures are estimates, not verified vendor quotations."
        ),
    )


DISCRETIONARY_REDUCTION_RULES = (
    (("decor", "decoration", "flower"), 25),
    (("entertainment", "music", "dj"), 20),
    (("favor", "favour"), 50),
)
ESSENTIAL_CATEGORY_MARKERS = ("essential", "required", "mandatory")


def _simulate_budget_shock(request: BudgetShockRequest) -> BudgetShockResult:
    original_total = sum(item.amount_inr for item in request.baseline_allocations)
    projected_amounts = {item.category: item.amount_inr for item in request.baseline_allocations}
    projected_guest_count = request.baseline_guest_count
    added_line: tuple[str, int] | None = None
    shock_cost = 0

    if request.shock_type == "category_percentage":
        allocation = next(
            item for item in request.baseline_allocations
            if item.category.casefold() == request.category.casefold()
        )
        shock_cost = int(
            (Decimal(allocation.amount_inr) * request.increase_percent / Decimal(100))
            .quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        projected_amounts[allocation.category] += shock_cost
        description = f"{request.increase_percent}% increase in {allocation.category}"
    elif request.shock_type == "additional_guests":
        cost_per_guest = _round_rupees(Decimal(original_total) / Decimal(request.baseline_guest_count))
        shock_cost = cost_per_guest * request.additional_guests
        projected_guest_count += request.additional_guests
        added_line = ("Additional guests (estimated)", shock_cost)
        description = f"{request.additional_guests} additional guests at approximately {cost_per_guest:,} INR per guest"
    else:
        shock_cost = request.expense_amount_inr
        added_line = (request.expense_label, shock_cost)
        description = f"Unexpected expense: {request.expense_label}"

    projected_total = original_total + shock_cost
    available_budget = request.baseline_budget_inr
    projected_gap = max(0, projected_total - available_budget)
    comparison = [
        {
            "category": item.category,
            "baseline_amount_inr": item.amount_inr,
            "projected_amount_inr": projected_amounts[item.category],
            "change_inr": projected_amounts[item.category] - item.amount_inr,
        }
        for item in request.baseline_allocations
    ]
    if added_line:
        comparison.append(
            {
                "category": added_line[0],
                "baseline_amount_inr": 0,
                "projected_amount_inr": added_line[1],
                "change_inr": added_line[1],
            }
        )

    remaining_gap = projected_gap
    savings: list[RecoverySaving] = []
    reduced_categories: set[str] = set()
    for keywords, maximum_percent in DISCRETIONARY_REDUCTION_RULES:
        if remaining_gap == 0:
            break
        for allocation in request.baseline_allocations:
            normalized = allocation.category.casefold()
            category_key = normalized
            if category_key in reduced_categories:
                continue
            if any(marker in normalized for marker in ESSENTIAL_CATEGORY_MARKERS):
                continue
            if not any(keyword in normalized for keyword in keywords):
                continue
            reduced_categories.add(category_key)
            current_amount = projected_amounts[allocation.category]
            maximum_reduction = allocation.amount_inr * maximum_percent // 100
            reduction = min(maximum_reduction, current_amount, remaining_gap)
            if reduction <= 0:
                continue
            projected_amounts[allocation.category] -= reduction
            remaining_gap -= reduction
            savings.append(
                RecoverySaving(
                    category=allocation.category,
                    before_reduction_amount_inr=current_amount,
                    cap_basis_amount_inr=allocation.amount_inr,
                    reduction_inr=reduction,
                    revised_amount_inr=projected_amounts[allocation.category],
                    maximum_reduction_percent=maximum_percent,
                )
            )

    recovery_total = sum(projected_amounts.values()) + (added_line[1] if added_line else 0)
    total_savings = sum(saving.reduction_inr for saving in savings)
    recovery = RecoveryPlan(
        fits_within_budget=recovery_total <= available_budget,
        available_budget_inr=available_budget,
        proposed_total_inr=recovery_total,
        savings_breakdown=savings,
        total_savings_inr=total_savings,
        remaining_shortfall_inr=max(0, recovery_total - available_budget),
        notice=(
            "Only listed discretionary categories were considered; essential categories were not reduced. "
            "Each suggested reduction is a trade-off, not a guarantee that the change is harmless."
        ),
        remaining_budget_inr=max(0, available_budget - recovery_total),
    )
    return BudgetShockResult(
        shock_type=request.shock_type,
        baseline_budget_inr=request.baseline_budget_inr,
        original_allocation_total_inr=original_total,
        projected_total_inr=projected_total,
        available_budget_inr=available_budget,
        projected_gap_inr=projected_gap,
        projected_guest_count=projected_guest_count,
        projected_cost_per_guest_inr=_round_rupees(Decimal(projected_total) / Decimal(projected_guest_count)),
        shock_cost_inr=shock_cost,
        shock_description=description,
        category_comparison=[ShockCategoryComparison.model_validate(line) for line in comparison],
        recovery=recovery,
        estimate_notice="All amounts are planning estimates only, not vendor quotations or confirmed prices.",
    )


def _correct_timeline_dates(
    plan: AIPlan,
    event_date: date,
    today: date | None = None,
) -> AIPlan:
    current_date = today or date.today()
    latest_pre_event_date = event_date - timedelta(days=1)
    timeline = [
        task.model_copy(
            update={
                "due_date": (
                    current_date
                    if event_date <= current_date
                    else min(max(task.due_date, current_date), latest_pre_event_date)
                )
            }
        )
        for task in plan.timeline
    ]
    return plan.model_copy(update={"timeline": timeline})


def _analyze_timeline_recovery(
    request: TimelineRecoveryRequest,
    today: date | None = None,
) -> TimelineRecoveryResult:
    current_date = today or date.today()
    window_available = request.event_date >= current_date
    window_end = (
        request.event_date
        if request.event_date == current_date
        else request.event_date - timedelta(days=1)
    ) if window_available else None
    recovered_tasks: list[TimelineRecoveryTask] = []

    for task in request.tasks:
        overdue = task.due_date < current_date
        lead_days = _recommended_task_lead_days(task.title)
        recommended_latest = request.event_date - timedelta(days=lead_days)
        if not window_available:
            status: RecoveryStatus = "unresolved"
            suggested_date = None
            reason = (
                "The original deadline has passed and the event date is before today, "
                "so no valid recovery date is available."
                if overdue
                else "The event date is before today, so no valid pre-event recovery date is available."
            )
        elif recommended_latest < current_date:
            status = "unresolved"
            suggested_date = None
            reason = (
                f"No date remains in the planning window that meets the suggested {lead_days}-day "
                "lead time. The event date is unchanged; review priorities and confirm what is practical. "
                "No task-duration, dependency, or operational feasibility assessment is available."
            )
        else:
            latest_valid_date = max(current_date, min(recommended_latest, window_end))
            if not overdue and task.due_date <= latest_valid_date:
                status = "on_schedule"
                suggested_date = task.due_date
                reason = (
                    f"The deadline is within the planning window and allows the suggested "
                    f"{lead_days}-day lead time; no date change is proposed."
                )
            else:
                status = "needs_adjustment"
                suggested_date = current_date if overdue else latest_valid_date
                if overdue:
                    reason = (
                        f"The original deadline has passed. Today is the earliest date available and "
                        f"remains within the suggested {lead_days}-day lead-time window; confirm what "
                        "remains possible."
                    )
                else:
                    reason = (
                        f"The deadline is later than the suggested {lead_days}-day lead-time target. "
                        f"Consider the earlier date {suggested_date.isoformat()}."
                    )

        recovered_tasks.append(
            TimelineRecoveryTask(
                title=task.title,
                detail=task.detail,
                original_due_date=task.due_date,
                suggested_due_date=suggested_date,
                is_overdue=overdue,
                status=status,
                explanation=reason,
            )
        )

    has_unresolved_conflicts = not window_available or any(
        task.status == "unresolved" for task in recovered_tasks
    )
    can_apply = (
        not has_unresolved_conflicts
        and any(task.suggested_due_date != task.original_due_date for task in recovered_tasks)
    )
    return TimelineRecoveryResult(
        event_date=request.event_date,
        current_date=current_date,
        window_start=current_date,
        window_end=window_end,
        window_available=window_available,
        tasks=recovered_tasks,
        has_unresolved_conflicts=has_unresolved_conflicts,
        can_apply=can_apply,
        notice=(
            "Lead times are planning guidelines based on task titles, not duration or dependency data. "
            "Suggestions do not assess vendor availability or operational feasibility. Short-notice events "
            "may not allow the suggested lead time; confirm what remains practical."
        ),
    )


def _recommended_task_lead_days(title: str) -> int:
    normalized = title.casefold()
    lead_time_rules = (
        (("venue", "logistics"), 30),
        (("catering",), 21),
        (("photography",), 21),
        (("guest list", "invitations", "attendee", "guest coordination"), 14),
        (("ceremony", "cultural", "agenda", "speaker"), 14),
        (("av ", "av/", "audio visual"), 7),
        (("cake", "dessert", "activities", "entertainment", "decor"), 7),
        (("run-of-show", "schedule", "event-day", "event day"), 2),
    )
    return next(
        (days for terms, days in lead_time_rules if any(term in normalized for term in terms)),
        1,
    )


def _parse_gemini_plan(
    parsed: object,
    text: str | None,
    event_date: date,
    today: date | None = None,
) -> AIPlan:
    if parsed is not None:
        plan = AIPlan.model_validate(parsed)
    else:
        if not text:
            raise ValueError("Gemini returned an empty plan.")
        plan = AIPlan.model_validate(json.loads(text))
    return _correct_timeline_dates(plan, event_date, today)


def _safe_validation_issues(exc: ValidationError) -> list[dict]:
    known_fields = {
        "overview", "allocations", "category", "amount_inr", "rationale",
        "timeline", "title", "due_date", "detail", "vendor_categories", "essential",
    }
    return [
        {
            "location": [
                part if isinstance(part, int) or part in known_fields else "<unknown>"
                for part in error["loc"]
            ],
            "type": error["type"],
        }
        for error in exc.errors(include_input=False)
    ]


def _event_timeline(request: PlanRequest, today: date | None = None) -> list[TimelineTask]:
    current_date = today or date.today()
    venue = request.venue_preference.lower()
    guests = request.guest_count
    budget = f"₹{request.budget_inr:,}"
    menu_preferences = ", ".join(request.food_preferences + request.dietary_preferences)
    menu_note = (
        f"Account for selected food and dietary preferences ({menu_preferences})."
        if menu_preferences
        else "Collect dietary requirements before confirming the menu."
    )
    culture_note = (
        f"Include the requested cultural details: {request.cultural_preferences}."
        if request.cultural_preferences
        else "Confirm any ceremony or cultural details with the hosts."
    )
    access_note = (
        f"Plan around accessibility needs: {request.accessibility_preferences}."
        if request.accessibility_preferences
        else "Check step-free access, seating, and clear routes."
    )

    if request.event_type == "Wedding":
        tasks = [
            ("Shortlist venue and ceremony spaces", 90, f"Compare {venue} options for the wedding within the {budget} planning budget; verify availability and access directly."),
            ("Coordinate guest list and invitations", 60, f"Organize invitations and RSVPs for about {guests} guests; track responses and accessibility requests."),
            ("Plan catering and dietary requirements", 45, f"Compare menu approaches for about {guests} guests. {menu_note}"),
            ("Prepare photography brief", 40, "List ceremony and reception moments to discuss with photographers; request quotes without assuming availability."),
            ("Prepare ceremony and cultural details", 30, f"Draft the ceremony sequence and assign family or helper roles. {culture_note}"),
            ("Coordinate accessibility and guest logistics", 21, access_note),
            ("Review wedding-day schedule and essentials", 7, "Share a draft schedule, contact list, and item checklist with the people helping on the day."),
        ]
    elif request.event_type == "Birthday":
        tasks = [
            ("Choose birthday venue and layout", 30, f"Plan a layout for {venue} and about {guests} guests; compare options against the {budget} planning budget."),
            ("Prepare invitations and guest responses", 21, f"Send invitations with the event details and track RSVPs for about {guests} guests."),
            ("Plan cake and dessert requirements", 14, f"Choose a cake or dessert approach and confirm serving count and any dietary needs. {menu_note}"),
            ("Plan birthday activities and entertainment", 12, f"Choose age- and guest-appropriate activities and make a simple event-day sequence. {culture_note}"),
            ("Plan birthday decorations", 5, "List reusable decorations and setup materials that fit the chosen venue."),
            ("Confirm food and serving needs", 10, f"Estimate food and non-alcoholic drinks for about {guests} guests. {menu_note}"),
            ("Check guest accessibility and arrival details", 3, access_note),
        ]
    elif request.event_type == "Corporate event":
        tasks = [
            ("Plan corporate venue and event logistics", 45, f"Compare {venue} venues for about {guests} attendees within the {budget} planning budget; review access and setup requirements."),
            ("Draft the event agenda", 21, f"Set session timings, breaks, transitions, and owners for each agenda segment. {culture_note}"),
            ("Coordinate speakers and presentation materials", 18, "List speaker needs, presentation deadlines, and contact follow-ups; do not assume confirmations."),
            ("Plan AV and room setup", 10, "Inventory microphones, displays, connectivity, and a backup presentation plan with the venue team."),
            ("Coordinate attendee invitations and RSVPs", 14, f"Prepare registration details, attendee communications, and RSVP tracking for about {guests} people."),
            ("Review catering and attendee access needs", 12, f"Plan refreshments for about {guests} attendees. {menu_note} {access_note}"),
            ("Review corporate event run-of-show", 2, "Walk through agenda transitions, speaker hand-offs, AV checks, and attendee arrival instructions."),
        ]
    elif request.event_type == "Community gathering":
        tasks = [
            ("Coordinate community venue and logistics", 30, f"Review {venue} space, setup, and access for about {guests} attendees within the {budget} planning budget."),
            ("Prepare community invitations and RSVPs", 21, f"Share event details with the community and track responses for about {guests} people."),
            ("Plan refreshments and dietary requests", 14, f"Estimate servings and collect dietary needs. {menu_note}"),
            ("Organize activities and volunteer roles", 12, f"Create an activity sequence and assign setup, welcome, and cleanup roles. {culture_note}"),
            ("Check accessibility and arrival arrangements", 7, access_note),
            ("Prepare the gathering-day checklist", 2, "Review supplies, signage, contact details, and the volunteer arrival plan."),
        ]
    elif request.event_type == "Anniversary":
        tasks = [
            ("Choose anniversary venue and setting", 30, f"Compare {venue} options for about {guests} guests against the {budget} planning budget."),
            ("Prepare guest list and invitations", 21, f"Coordinate invitations and responses for about {guests} guests."),
            ("Plan anniversary menu and dietary needs", 14, f"Compare meal formats and serving estimates. {menu_note}"),
            ("Prepare a personal celebration program", 10, f"Plan speeches, memories, music, or other requested cultural details. {culture_note}"),
            ("Plan simple anniversary decor", 7, "List a restrained set of reusable decorations and setup needs."),
            ("Check guest accessibility and arrival details", 3, access_note),
        ]
    else:
        tasks = [
            ("Define event needs and planning priorities", 30, f"Summarize the event goals, guest needs, and key decisions within the {budget} planning budget."),
            ("Plan venue and event logistics", 21, f"Review {venue} options for about {guests} guests; verify actual availability, access, and setup needs."),
            ("Coordinate invitations and attendees", 14, f"Share event details and track responses for about {guests} guests."),
            ("Plan refreshments and preferences", 12, f"Estimate serving needs and collect dietary requests. {menu_note}"),
            ("Prepare the event schedule and activities", 7, f"Draft a simple schedule and include any requested cultural details. {culture_note}"),
            ("Check accessibility and day-of supplies", 3, access_note),
        ]

    timeline = [
        TimelineTask(
            title=title,
            due_date=max(current_date, request.event_date - timedelta(days=lead_days)),
            detail=detail,
        )
        for title, lead_days, detail in tasks
    ]
    timeline.sort(key=lambda task: task.due_date)
    return timeline


def _tailor_event_timeline(
    request: PlanRequest,
    plan: AIPlan,
    today: date | None = None,
) -> AIPlan:
    return plan.model_copy(update={"timeline": _event_timeline(request, today)})


def _demo_plan(request: PlanRequest) -> AIPlan:
    guests = request.guest_count
    budget = request.budget_inr
    food = round(budget * 0.38)
    venue = round(budget * (0.20 if request.venue_preference != "At home" else 0.05))
    decor = round(budget * 0.14)
    photo = round(budget * 0.12)
    reserve = max(0, budget - food - venue - decor - photo)
    return AIPlan(
        overview=(
            f"A thoughtful {request.event_type.lower()} for {guests} guests, planned around "
            f"your {request.venue_preference.lower()} preference and ₹{budget:,} budget. "
            "Treat each figure as an estimate and confirm details with local providers."
        ),
        allocations=[
            BudgetLine(category="Food & refreshments", amount_inr=food, rationale=f"A flexible menu for about {guests} guests, tailored to your food and dietary notes."),
            BudgetLine(category="Venue", amount_inr=venue, rationale="A planning allowance based on your selected venue style; no booking is implied."),
            BudgetLine(category="Decor", amount_inr=decor, rationale="Simple, reusable decor that suits the celebration."),
            BudgetLine(category="Photography", amount_inr=photo, rationale="A short event coverage allowance; compare local options."),
            BudgetLine(category="Contingency & essentials", amount_inr=reserve, rationale="A buffer for transport, supplies, and unexpected needs."),
        ],
        timeline=_event_timeline(request),
        vendor_categories=[
            VendorCategory(category="Venue", essential=request.venue_preference not in {"At home", "Flexible"}),
            VendorCategory(category="Catering", essential=True),
            VendorCategory(category="Decor & flowers", essential=False),
            VendorCategory(category="Photography", essential=False),
            VendorCategory(category="Accessibility & transport", essential=False),
        ],
    )


def _generate_with_gemini(request: PlanRequest) -> AIPlan:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = (
        "Create a practical celebration plan. All amounts are approximate INR estimates, never vendor quotes. "
        "Do not claim availability or confirmed bookings. Keep categories distinct and match the user's budget "
        "and guest count; it is acceptable for estimated allocations to exceed budget so BudgetGuard can flag it. "
        "Make overview, allocations, and vendor prompts specific to the event and preferences. "
        "For the timeline, use distinct practical tasks for the selected event type, not generic tasks with "
        "event-specific wording; the service applies a validated event-specific checklist after generation. "
        "Use preparation deadlines before the event date and include the user's stated preferences. "
        f"Event: {request.event_type}. Description: {request.description}. Date: {request.event_date}. "
        f"Budget INR: {request.budget_inr}. Guests: {request.guest_count}. Venue: {request.venue_preference}. "
        f"Food preferences: {request.food_preferences}. Dietary preferences: {request.dietary_preferences}. "
        f"Cultural preferences: {request.cultural_preferences}. Accessibility preferences: {request.accessibility_preferences}."
    )
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AIPlan,
            temperature=0.4,
        ),
    )
    return _parse_gemini_plan(response.parsed, response.text, request.event_date)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/plan")
def create_plan(request: PlanRequest) -> dict:
    if os.getenv("GEMINI_API_KEY", "").strip():
        try:
            plan = _tailor_event_timeline(request, _generate_with_gemini(request))
            return _make_result(request, plan, "ai")
        except ValidationError as exc:
            logger.warning("Gemini plan validation failed: %s", _safe_validation_issues(exc))
            raise HTTPException(status_code=502, detail="Gemini returned a plan that could not be validated. Please try again.") from exc
        except json.JSONDecodeError as exc:
            logger.warning("Gemini returned invalid JSON (line=%d, column=%d).", exc.lineno, exc.colno)
            raise HTTPException(status_code=502, detail="Gemini returned a malformed plan. Please try again.") from exc
        except Exception as exc:
            message = str(exc).lower()
            status_code = getattr(exc, "status_code", getattr(exc, "code", None))
            if status_code == 429 or "429" in message or "rate" in message or "resource_exhausted" in message:
                logger.warning("Gemini generation was rate limited (error_type=%s).", type(exc).__name__)
                result = _make_result(request, _demo_plan(request), "demo")
                result["fallback_reason"] = (
                    "Gemini is rate limited. This is a deterministic demo fallback, not an AI-generated plan. "
                    "You can retry later."
                )
                return result
            if status_code == 404:
                logger.warning("Configured Gemini model is unavailable (provider_status=404).")
                raise HTTPException(status_code=502, detail="The configured Gemini model is unavailable. Check GEMINI_MODEL and API access.") from exc
            if status_code in {401, 403}:
                logger.warning("Gemini credentials or permissions were rejected (provider_status=%s).", status_code)
                raise HTTPException(status_code=502, detail="Gemini credentials were rejected or lack access. Check backend configuration.") from exc
            if status_code in {500, 502, 503, 504}:
                logger.warning(
                    "Gemini service returned an upstream error (error_type=%s, provider_code=%s).",
                    type(exc).__name__,
                    status_code,
                )
                result = _make_result(request, _demo_plan(request), "demo")
                result["fallback_reason"] = (
                    "Gemini is temporarily unavailable. This is a deterministic demo fallback, "
                    "not an AI-generated plan. You can retry later."
                )
                return result
            safe_code = status_code if isinstance(status_code, int) else "unknown"
            logger.warning("Gemini generation failed (error_type=%s, provider_code=%s).", type(exc).__name__, safe_code)
            raise HTTPException(status_code=502, detail="Gemini could not be reached. Check your connection or try again later.") from exc

    return _make_result(request, _demo_plan(request), "demo")


@app.post("/api/budget-shock", response_model=BudgetShockResult)
def simulate_budget_shock(request: BudgetShockRequest) -> BudgetShockResult:
    return _simulate_budget_shock(request)


@app.post("/api/timeline-recovery", response_model=TimelineRecoveryResult)
def recover_timeline(request: TimelineRecoveryRequest) -> TimelineRecoveryResult:
    return _analyze_timeline_recovery(request)


@app.post("/api/trade-offs", response_model=TradeOffResult)
def explore_trade_offs(request: TradeOffRequest) -> TradeOffResult:
    return _compare_trade_offs(request)
