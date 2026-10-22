"""Turn model output into a short brief for a marketing / CRM partner.

Default is a deterministic template (works offline, no API key). An LLM can be
plugged in through `llm_fn` (e.g. a function that calls OpenAI/Gemini with the
same facts); it only rewrites the wording, never the numbers.
"""
from __future__ import annotations

from typing import Callable, Optional

ACTIONS = {
    "network": "route to the network team's coverage fix list and offer a signal booster / Wi-Fi calling check",
    "billing": "proactive bill explanation call and correct missing credits before the next cycle",
    "price": "plan-fit review: move to a cheaper plan instead of losing the line",
    "device": "targeted upgrade or trade-in offer",
    "service": "callback from a senior agent with the open ticket history",
    "none": "light-touch check-in; no offer",
}


def summarise_segment(driver_mix: dict, reason_mix: dict, plan: dict,
                      llm_fn: Optional[Callable[[str], str]] = None) -> str:
    total = sum(driver_mix.values()) or 1
    lines = ["# Weekly churn watchlist brief (synthetic data)", "",
             f"Customers contacted under budget: {plan['contacted']}. "
             f"Churners reached: {plan['churners_reached_model']} with the model vs "
             f"{plan['churners_reached_random']} with random selection.",
             f"Estimated net value (illustrative assumptions {plan['assumptions']}): "
             f"{plan['net_value_model']:.0f} vs {plan['net_value_random']:.0f} for random.", "",
             "## What is driving risk in the top segment"]
    for d, c in driver_mix.items():
        lines.append(f"- {d}: {c} customers ({100 * c / total:.0f}%) -> {ACTIONS.get(d, 'review')}")
    lines += ["", "## Most common reason codes", *[f"- {k}: {v}" for k, v in reason_mix.items()]]
    text = "\n".join(lines)
    if llm_fn is not None:
        return llm_fn("Rewrite this for a marketing manager. Keep every number unchanged:\n" + text)
    return text
