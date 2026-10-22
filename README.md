# Churn early-warning with reason codes and a budgeted retention list

> Personal project by Somya Goyal, built while applying to Verizon. Not affiliated with or endorsed by Verizon; uses public/synthetic data.

## Why this matters for Verizon

- **Problem:** retention teams can only contact a limited number of customers each week. A plain churn score doesn't say *why* someone is at risk, so the offer is often the wrong one (a discount for a network problem, say).
- **Who feels it:** marketing, CRM and care teams, the business partners named in the Engr II-Data Science JD ("build analytical models to predict customer behavior… present insights and recommendations to marketing, finance, or CRM").
- **What this shows:** a churn model, per-customer reason codes, a classifier that tags the "why" from the last support note, and a contact list chosen by expected value under a fixed budget. On synthetic data, the 300-customer list reaches **193 future churners vs 59 for random selection** (3.3x).

Public context: on the Q1 2026 earnings call, Verizon said consumer postpaid churn held at 0.9% and called retention the focus of its transformation work ([CX Dive, 28 Apr 2026](https://www.customerexperiencedive.com/news/verizon-better-customer-relationships-reduced-churn/818729/)). This project is a small, generic illustration of that kind of retention analytics. It says nothing about Verizon's internal data or methods.

## Results (from `python -m churnwatch.run`, 8,000 synthetic customers, 30% test split)

| Model | ROC-AUC | PR-AUC | Top-decile lift |
|---|---|---|---|
| Logistic regression (scaled + one-hot) | 0.828 | 0.619 | 2.78x |
| Gradient boosting (HistGB) | 0.815 | 0.596 | 2.79x |

The base churn rate in the test set is 24.7%. The simpler logistic model is as good as boosting here, so it also supplies the reason codes.

**Retention list (budget 300 contacts):** it reaches 193 churners vs 59 for random. Estimated net value is 41,336 vs 4,660 for random. ⚠️ That value is an **estimate** under illustrative assumptions: offer cost 30, a 25% save rate, and 12 months of the monthly charge as value. None of these are company figures.

**"Why" classifier on support notes:**

| Set | Macro-F1 |
|---|---|
| Synthetic test notes (same templates as training) | 0.953 |
| 24 hand-written notes with **new wording**, TF-IDF model only | 0.185 |
| Same 24 notes, model + lexicon fallback when wording is unfamiliar | 0.712 |

The honest lesson: a model trained on template text looks perfect in-distribution and collapses on new wording. The fallback (confidence + vocabulary-coverage gate, then a small telecom-care lexicon) recovers most of it. I wrote the lexicon and the held-out notes myself, so 0.712 is optimistic. Real call-centre notes would be the real test.

## How it works

1. `churnwatch/data.py`: a labelled synthetic generator. The schema follows the public IBM *Telco Customer Churn* sample (contract, tenure, charges, payment method), plus dropped calls, tickets, bill shock and a free-text last-contact note. Churn is drawn from a known logistic model, so the true drivers are known.
2. `churnwatch/model.py`: logistic and gradient-boosting models; metrics (ROC-AUC, PR-AUC, top-decile lift); reason codes (the top features pushing each customer's risk up, coef × scaled value); the driver classifier; and `plan_outreach`, which ranks customers by `p(churn) × save_rate × value − offer_cost` under a contact budget and compares against random.
3. `churnwatch/summarise.py`: a weekly brief for a marketing/CRM partner, mapping each driver to an action. It works offline with a template. An LLM can be plugged in via `llm_fn` to reword the brief; the numbers stay fixed.
4. `churnwatch/run.py`: runs end to end and writes `results/results.json`, `results/REPORT.md` and `results/top25_watchlist.csv`.

## Run it

```bash
pip install -r requirements.txt
python -m churnwatch.run --n 8000 --budget 300
pytest -q          # 7 tests
```

Optional LLM wording: pass any `llm_fn(prompt) -> str` (for example a wrapper around the OpenAI or Gemini API with your own key) to `summarise_segment`. No key is needed for the default path.

## Limitations / next steps

- The data is synthetic, so real churn is noisier and the AUCs here are not a claim about real customers. The next step is the public IBM Telco dataset, then real data with proper time-based splits.
- Reason codes come from a linear model. SHAP on the boosting model would be the next step.
- Uplift modelling (who is *persuadable*, not just at risk) needs A/B data from past offers.
- The save rate and offer cost are assumptions and should be replaced with measured campaign results.
