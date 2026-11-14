
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "pricing.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    CONFIG = json.load(file)


def calculate_economics(result):
    """Calculate cost, context usage, rate limits and budget."""

    provider = "gemini" if result["provider"] == "Google Gemini" else "ollama"

    settings = CONFIG[provider]
    budget = CONFIG["budget"]

    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    total_tokens = result["total_tokens"]

    # ----------------------------------
    # COST ANALYSIS
    # ----------------------------------

    input_rate = settings["input_per_million_tokens_usd"]
    output_rate = settings["output_per_million_tokens_usd"]

    input_cost = input_tokens / 1_000_000 * input_rate
    output_cost = output_tokens / 1_000_000 * output_rate

    total_cost = input_cost + output_cost

    # ----------------------------------
    # FEATURE BUDGET
    # ----------------------------------

    users = budget["users"]
    prompts_per_user = budget["prompts_per_user_per_day"]
    days = budget["days_per_month"]

    daily_requests = users * prompts_per_user
    monthly_requests = daily_requests * days

    daily_cost = total_cost * daily_requests
    monthly_cost = daily_cost * days

    monthly_budget = budget["monthly_budget_usd"]

    remaining_budget = monthly_budget - monthly_cost

    if monthly_cost <= monthly_budget:
        budget_status = "WITHIN BUDGET"
    else:
        budget_status = "BUDGET EXCEEDED"

    # ----------------------------------
    # CONTEXT WINDOW ANALYSIS
    # ----------------------------------

    context_limit = settings["context_window"]
    output_limit = settings["max_output_tokens"]

    if context_limit is not None:
        context_percentage = total_tokens / context_limit * 100
        context_status = (
            "WITHIN LIMIT"
            if total_tokens <= context_limit
            else "EXCEEDED"
        )
    else:
        context_percentage = None
        context_status = "NOT CONFIGURED"

    if output_limit is not None:
        output_status = (
            "WITHIN LIMIT"
            if output_tokens <= output_limit
            else "EXCEEDED"
        )
    else:
        output_status = "NOT CONFIGURED"

    # ----------------------------------
    # RATE LIMIT ANALYSIS
    # ----------------------------------

    limits = settings["rate_limits"]

    rpm_limit = limits["requests_per_minute"]
    tpm_limit = limits["tokens_per_minute"]
    rpd_limit = limits["requests_per_day"]

    if rpm_limit is not None:
        estimated_rpm = daily_requests / 1440
        rpm_status = (
            "WITHIN LIMIT"
            if estimated_rpm <= rpm_limit
            else "EXCEEDED"
        )
    else:
        estimated_rpm = None
        rpm_status = "NOT CONFIGURED"

    if tpm_limit is not None:
        estimated_tpm = input_tokens * daily_requests / 1440
        tpm_status = (
            "WITHIN LIMIT"
            if estimated_tpm <= tpm_limit
            else "EXCEEDED"
        )
    else:
        estimated_tpm = None
        tpm_status = "NOT CONFIGURED"

    if rpd_limit is not None:
        rpd_status = (
            "WITHIN LIMIT"
            if daily_requests <= rpd_limit
            else "EXCEEDED"
        )
    else:
        rpd_status = "NOT CONFIGURED"

    # Maximum daily request capacity based on configured limits
    capacities = []

    if rpd_limit is not None:
        capacities.append(rpd_limit)

    if rpm_limit is not None:
        capacities.append(rpm_limit * 1440)

    if tpm_limit is not None and input_tokens > 0:
        capacities.append(
            int(tpm_limit * 1440 / input_tokens)
        )

    max_daily_requests = min(capacities) if capacities else None

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,

        "daily_requests": daily_requests,
        "monthly_requests": monthly_requests,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,

        "monthly_budget": monthly_budget,
        "remaining_budget": remaining_budget,
        "budget_status": budget_status,

        "context_limit": context_limit,
        "context_percentage": context_percentage,
        "context_status": context_status,
        "output_limit": output_limit,
        "output_status": output_status,

        "rpm_limit": rpm_limit,
        "estimated_rpm": estimated_rpm,
        "rpm_status": rpm_status,

        "tpm_limit": tpm_limit,
        "estimated_tpm": estimated_tpm,
        "tpm_status": tpm_status,

        "rpd_limit": rpd_limit,
        "rpd_status": rpd_status,

        "max_daily_requests": max_daily_requests
    }
