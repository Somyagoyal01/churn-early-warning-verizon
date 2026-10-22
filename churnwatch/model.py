"""Churn model, reason codes, complaint-driver classifier and retention targeting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["tenure_months", "monthly_charge", "lines", "dropped_calls_30d",
       "support_tickets_90d", "bill_shock_flag", "device_age_months"]
CAT = ["contract", "plan", "payment_method"]
TARGET = "churned_next_90d"


def _prep() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


def split(df: pd.DataFrame, seed: int = 0):
    return train_test_split(df, test_size=0.3, random_state=seed, stratify=df[TARGET])


def train_models(train: pd.DataFrame):
    X, y = train[NUM + CAT], train[TARGET]
    logit = Pipeline([("prep", _prep()), ("clf", LogisticRegression(max_iter=2000, C=0.5))]).fit(X, y)
    gbm = Pipeline([("prep", _prep()),
                    ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                                           max_iter=250, random_state=0))]).fit(X, y)
    return logit, gbm


def top_decile_lift(y: np.ndarray, p: np.ndarray) -> float:
    k = max(1, len(p) // 10)
    idx = np.argsort(-p)[:k]
    return float(y[idx].mean() / max(y.mean(), 1e-9))


def evaluate(model, test: pd.DataFrame) -> dict:
    p = model.predict_proba(test[NUM + CAT])[:, 1]
    y = test[TARGET].to_numpy()
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 3),
        "pr_auc": round(float(average_precision_score(y, p)), 3),
        "top_decile_lift": round(top_decile_lift(y, p), 2),
        "base_rate": round(float(y.mean()), 3),
    }


def reason_codes(logit: Pipeline, rows: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """Top-k features pushing each customer's churn risk UP (coef x scaled value)."""
    prep, clf = logit.named_steps["prep"], logit.named_steps["clf"]
    Z = prep.transform(rows[NUM + CAT])
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    contrib = Z * clf.coef_[0]
    out = []
    for r in contrib:
        order = [i for i in np.argsort(-r) if r[i] > 0][:k]
        out.append([names[i] for i in order])
    return out


# Domain lexicon (general telecom-care vocabulary). Used as a first pass because a
# model trained only on template notes does not generalise to new wording.
LEXICON = {
    "network": ["signal", "coverage", "drop", "slow", "data", "5g", "bars", "internet", "freez", "cutting out"],
    "billing": ["bill", "charged", "charge", "fee", "statement", "credit", "discount", "autopay"],
    "price": ["cheaper", "expensive", "price", "costs", "deal elsewhere", "competitor", "carrier", "better deal"],
    "device": ["phone", "upgrade", "trade in", "battery", "device", "overheat"],
    "service": ["support", "hold", "agent", "transferred", "store", "callback", "call back", "waited"],
    "none": ["roaming", "address", "new line", "adding", "hotspot"],
}
PRIORITY = ["price", "billing", "device", "network", "service", "none"]


def lexicon_label(text: str):
    t = text.lower()
    scores = {d: sum(w in t for w in ws) for d, ws in LEXICON.items()}
    best = max(scores.values())
    if best == 0:
        return None
    return next(d for d in PRIORITY if scores[d] == best)


@dataclass
class DriverModel:
    pipe: Pipeline
    use_lexicon: bool = True
    min_conf: float = 0.6
    min_known: float = 0.6

    @classmethod
    def fit(cls, texts, labels) -> "DriverModel":
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=4.0)),
        ]).fit(texts, labels)
        return cls(pipe)

    def predict(self, texts):
        texts = list(texts)
        proba = self.pipe.predict_proba(texts)
        model_pred = self.pipe.classes_[proba.argmax(1)]
        if not self.use_lexicon:
            return model_pred
        # trust the model only when it is confident AND most words are familiar;
        # otherwise fall back to the lexicon (unfamiliar wording = out of distribution)
        vocab = self.pipe.named_steps["tfidf"].vocabulary_
        out = []
        for t, m, conf in zip(texts, model_pred, proba.max(1)):
            words = [w for w in t.lower().split() if len(w) > 2]
            known = sum(w in vocab for w in words) / max(len(words), 1)
            ok = conf >= self.min_conf and known >= self.min_known
            out.append(m if ok else (lexicon_label(t) or m))
        return out

    def macro_f1(self, texts, labels) -> float:
        return round(float(f1_score(labels, self.predict(texts), average="macro")), 3)


def plan_outreach(test: pd.DataFrame, p: np.ndarray, budget: int, offer_cost: float = 30.0,
                  save_rate: float = 0.25, months_value: int = 12) -> dict:
    """Pick customers by expected net value of a retention offer, under a contact budget.

    expected value = p(churn) * save_rate * (monthly_charge * months_value) - offer_cost
    All money values are ILLUSTRATIVE assumptions, not company figures.
    """
    clv = test["monthly_charge"].to_numpy() * months_value
    ev = p * save_rate * clv - offer_cost
    y = test[TARGET].to_numpy()

    def realised(idx):
        # value realised on customers who really churn (saved fraction), minus offer cost on all contacted
        return float((y[idx] * save_rate * clv[idx]).sum() - offer_cost * len(idx))

    by_value = [i for i in np.argsort(-ev) if ev[i] > 0][:budget]
    rng = np.random.default_rng(0)
    random_pick = rng.choice(len(p), size=len(by_value), replace=False)
    return {
        "contacted": len(by_value),
        "churners_reached_model": int(y[by_value].sum()),
        "churners_reached_random": int(y[random_pick].sum()),
        "net_value_model": round(realised(np.array(by_value, dtype=int)), 0),
        "net_value_random": round(realised(random_pick), 0),
        "assumptions": {"offer_cost": offer_cost, "save_rate": save_rate, "months_value": months_value},
    }
