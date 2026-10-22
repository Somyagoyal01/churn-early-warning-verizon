"""Labelled SYNTHETIC telecom customer data.

The column set follows the public IBM "Telco Customer Churn" sample schema
(tenure, contract, monthly charges, services, payment method) plus a few
behavioural signals (dropped calls, support tickets, recent bill shock) and a
short free-text note from the customer's last support contact.

No real or company data is used. Churn is drawn from a known logistic model so
the evaluation is reproducible and the "true" drivers are known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENTS = ["autopay-card", "autopay-bank", "manual-online", "paper-check"]
PLANS = ["unlimited-basic", "unlimited-plus", "unlimited-ultimate", "prepaid-switch"]

DRIVER_TEXT = {
    "network": [
        "calls keep dropping at home", "no signal in my building", "5g is slower than before",
        "data stops working in the evening", "coverage is poor where I work",
    ],
    "billing": [
        "bill went up without notice", "charged twice this month", "promo credit never applied",
        "unexpected fee on my statement", "autopay discount disappeared",
    ],
    "price": [
        "another carrier offered a cheaper plan", "plan is too expensive for what I use",
        "competitor has a better family deal", "looking for a lower monthly price",
    ],
    "device": [
        "phone upgrade offer was not available", "trade in value was too low",
        "new phone keeps overheating", "want the latest phone but no deal",
    ],
    "service": [
        "waited an hour on support", "agent transferred me three times",
        "store could not fix my issue", "nobody called back as promised",
    ],
    "none": [
        "asked about international roaming", "changed my billing address",
        "added a line for my daughter", "question about hotspot usage", "",
    ],
}
DRIVERS = list(DRIVER_TEXT)

# Hand-written notes with DIFFERENT wording from the generator, used only for
# evaluation, so the driver classifier is not scored on phrases it memorised.
HELDOUT_NOTES = [
    ("signal keeps cutting out during calls", "network"), ("internet is very slow at night", "network"),
    ("no bars inside my office", "network"), ("video calls freeze, data is unreliable", "network"),
    ("my monthly bill is higher than last month", "billing"), ("I was charged a fee I did not expect", "billing"),
    ("the discount on my statement is missing", "billing"), ("double charge on my card", "billing"),
    ("switching because the other carrier is cheaper", "price"), ("my plan costs too much", "price"),
    ("found a better deal elsewhere for my family", "price"), ("need a lower price plan", "price"),
    ("upgrade deal not offered to me", "device"), ("trade in quote for my old phone was low", "device"),
    ("phone battery heats up a lot", "device"), ("want a new phone deal", "device"),
    ("support kept me on hold for ages", "service"), ("got transferred again and again", "service"),
    ("the store staff could not help", "service"), ("promised callback never happened", "service"),
    ("how do I use roaming abroad", "none"), ("please update my address", "none"),
    ("adding a new line for my son", "none"), ("how much hotspot data do I get", "none"),
]


def generate(n: int = 8000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    tenure = np.where(
        contract == "month-to-month", rng.gamma(2.0, 9.0, n), rng.gamma(4.0, 10.0, n)
    ).clip(1, 84).round()
    plan = rng.choice(PLANS, n, p=[0.35, 0.35, 0.2, 0.1])
    base = {"unlimited-basic": 65, "unlimited-plus": 80, "unlimited-ultimate": 95, "prepaid-switch": 50}
    monthly = np.array([base[p] for p in plan]) + rng.normal(0, 8, n)
    lines = rng.choice([1, 2, 3, 4], n, p=[0.45, 0.3, 0.15, 0.1])
    payment = rng.choice(PAYMENTS, n, p=[0.35, 0.25, 0.3, 0.1])
    dropped = rng.poisson(np.where(rng.random(n) < 0.2, 6, 1.2))
    tickets = rng.poisson(0.6, n) + (dropped > 4)
    bill_shock = (rng.random(n) < 0.12).astype(int)
    device_age = rng.integers(1, 48, n)

    driver = np.full(n, "none", dtype=object)
    driver[dropped > 4] = "network"
    driver[(bill_shock == 1) & (driver == "none")] = "billing"
    rand = rng.random(n)
    driver[(driver == "none") & (rand < 0.10)] = "price"
    driver[(driver == "none") & (rand >= 0.10) & (rand < 0.16) & (device_age > 30)] = "device"
    driver[(driver == "none") & (tickets >= 2)] = "service"
    note = []
    for d in driver:
        text = rng.choice(DRIVER_TEXT[d])
        if rng.random() < 0.3:  # real notes often mention a second, weaker issue
            other = rng.choice([x for x in DRIVERS if x != d])
            text = text + " and " + rng.choice(DRIVER_TEXT[other])
        note.append(text.strip(" and "))

    z = (
        -2.6
        + 1.3 * (contract == "month-to-month")
        - 0.9 * (contract == "two-year")
        - 0.035 * tenure
        + 0.02 * (monthly - 75)
        + 0.22 * np.minimum(dropped, 10)
        + 0.35 * tickets
        + 1.1 * bill_shock
        + 0.6 * (payment == "paper-check")
        + 0.4 * (payment == "manual-online")
        - 0.25 * (lines - 1)
        + 0.9 * (driver == "price")
        + 0.5 * (driver == "device")
        + 0.02 * device_age
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "contract": contract,
            "tenure_months": tenure.astype(int),
            "plan": plan,
            "monthly_charge": monthly.round(2),
            "lines": lines,
            "payment_method": payment,
            "dropped_calls_30d": dropped,
            "support_tickets_90d": tickets,
            "bill_shock_flag": bill_shock,
            "device_age_months": device_age,
            "last_contact_note": note,
            "true_driver": driver,
            "churned_next_90d": churn,
        }
    )
