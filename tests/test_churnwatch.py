import numpy as np

from churnwatch.data import DRIVERS, generate
from churnwatch.model import DriverModel, evaluate, plan_outreach, reason_codes, split, train_models, NUM, CAT
from churnwatch.run import main
from churnwatch.summarise import summarise_segment


def test_generate_schema_and_rate():
    df = generate(2000, seed=1)
    assert len(df) == 2000 and df["customer_id"].is_unique
    assert set(df["true_driver"]) <= set(DRIVERS)
    assert 0.1 < df["churned_next_90d"].mean() < 0.5


def test_models_beat_random():
    train, test = split(generate(3000, seed=2))
    logit, gbm = train_models(train)
    for m in (logit, gbm):
        res = evaluate(m, test)
        assert res["roc_auc"] > 0.7 and res["top_decile_lift"] > 1.5


def test_reason_codes_are_positive_contributors():
    train, test = split(generate(2000, seed=3))
    logit, _ = train_models(train)
    codes = reason_codes(logit, test.head(5), k=3)
    assert len(codes) == 5 and all(len(c) <= 3 for c in codes)


def test_driver_classifier():
    df = generate(3000, seed=4)
    train, test = split(df)
    dm = DriverModel.fit(train["last_contact_note"], train["true_driver"])
    assert dm.macro_f1(test["last_contact_note"], test["true_driver"]) > 0.7


def test_outreach_respects_budget_and_beats_random():
    train, test = split(generate(4000, seed=5))
    _, gbm = train_models(train)
    p = gbm.predict_proba(test[NUM + CAT])[:, 1]
    plan = plan_outreach(test, p, budget=100)
    assert plan["contacted"] <= 100
    assert plan["churners_reached_model"] >= plan["churners_reached_random"]


def test_summary_offline_and_llm_hook():
    plan = {"contacted": 3, "churners_reached_model": 2, "churners_reached_random": 1,
            "net_value_model": 10.0, "net_value_random": 1.0, "assumptions": {}}
    text = summarise_segment({"network": 2, "price": 1}, {"tenure_months": 2}, plan)
    assert "network" in text and "2 with the model" in text
    assert summarise_segment({}, {}, plan, llm_fn=lambda s: "LLM:" + s[:5]).startswith("LLM:")


def test_end_to_end(tmp_path):
    res = main(n=2500, budget=80, out=str(tmp_path))
    assert (tmp_path / "results.json").exists() and (tmp_path / "REPORT.md").exists()
    assert res["metrics"]["gradient_boosting"]["roc_auc"] > 0.7
