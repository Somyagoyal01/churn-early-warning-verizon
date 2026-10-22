"""End-to-end run: data -> models -> drivers -> outreach plan -> results.json + REPORT.md.

Usage:  python -m churnwatch.run [--n 8000] [--budget 300]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import HELDOUT_NOTES, generate
from .model import DriverModel, evaluate, plan_outreach, reason_codes, split, train_models, NUM, CAT
from .summarise import summarise_segment


def main(n: int = 8000, budget: int = 300, out: str = "results") -> dict:
    df = generate(n)
    train, test = split(df)
    logit, gbm = train_models(train)
    metrics = {"logistic": evaluate(logit, test), "gradient_boosting": evaluate(gbm, test)}

    drivers = DriverModel.fit(train["last_contact_note"], train["true_driver"])
    metrics["driver_classifier_macro_f1_synthetic_test"] = drivers.macro_f1(test["last_contact_note"], test["true_driver"])
    ho_text, ho_label = zip(*HELDOUT_NOTES)
    metrics["driver_classifier_macro_f1_heldout_wording"] = drivers.macro_f1(list(ho_text), list(ho_label))
    model_only = DriverModel(drivers.pipe, use_lexicon=False)
    metrics["driver_model_only_macro_f1_heldout_wording"] = model_only.macro_f1(list(ho_text), list(ho_label))

    p = gbm.predict_proba(test[NUM + CAT])[:, 1]
    plan = plan_outreach(test, p, budget)

    top = test.assign(p=p).sort_values("p", ascending=False).head(budget)
    top = top.assign(reasons=reason_codes(logit, top),
                     predicted_driver=drivers.predict(top["last_contact_note"]))
    driver_mix = dict(Counter(top["predicted_driver"]).most_common())
    reason_mix = dict(Counter(r for rs in top["reasons"] for r in rs).most_common(6))
    brief = summarise_segment(driver_mix, reason_mix, plan)

    results = {"rows": n, "test_rows": len(test), "metrics": metrics, "outreach": plan,
               "top_risk_driver_mix": driver_mix, "top_risk_reason_codes": reason_mix,
               "business_brief": brief}
    Path(out).mkdir(exist_ok=True)
    Path(out, "results.json").write_text(json.dumps(results, indent=2))
    top[["customer_id", "p", "reasons", "predicted_driver"]].head(25).to_csv(Path(out, "top25_watchlist.csv"), index=False)
    Path(out, "REPORT.md").write_text(brief + "\n")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--budget", type=int, default=300)
    a = ap.parse_args()
    print(json.dumps(main(a.n, a.budget), indent=2))
