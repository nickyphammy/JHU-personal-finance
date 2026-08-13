# ClearLedger — Personal Finance Coach

ClearLedger is a subscription personal finance app facing high early churn because users must manually tag dozens of transactions before seeing value. This project implements a **Personal Finance Coach** that ingests bank CSVs, categorizes spending, renders an interactive dashboard, and (planned) predicts days-to-limit, ranks category reductions, and exposes grounded insights in a multi-turn AI coach.

V1 runs locally with Streamlit, keeps source data under `data/` read-only, and writes all derived outputs under `artifacts/` (gitignored; regenerable).

## Pipeline

1. Read source datasets from `data/` (transactions, cards, users, MCC lookup) **read-only**.
2. Validate schemas, clean/standardize types (currency/date), dedupe, and join into a unified dataset.
3. Categorize transactions — **current:** rule-based MCC mapping in analytics; **planned:** LLM batched + cached with MCC fallback.
4. Compute analytics (spend by category, budget utilization, trends) via `model/analytics_core.py`.
5. **Planned:** regression model for **days-to-limit** / projected month-end spend.
6. **Planned:** impact-ranked category-level recommendations.
7. **Planned:** notification events at 70% / 85% / 95% utilization.
8. Launch Streamlit dashboard (`ui/app.py`); **planned:** multi-turn coach grounded in artifacts + local memory.

**All source data is read from the existing `data/` folder only.** No source file is modified in place. Derived artifacts are written under `artifacts/` and are **not committed to git**.

## Project Structure

```
.
├── data/                         # Existing read-only inputs
│   ├── transactions.csv
│   ├── cards.csv
│   ├── users.csv
│   └── mcc_codes.json
├── artifacts/                    # Derived outputs (gitignored except .gitkeep)
├── data_processing/
│   ├── validate_schema.ipynb     # done
│   ├── clean.ipynb               # done
│   └── categorize.ipynb          # planned (LLM)
├── model/
│   ├── analytics_core.py         # done — analytics backend
│   ├── analytics.ipynb           # done — batch runner (default 1696)
│   ├── predict.ipynb             # planned
│   ├── recommend.ipynb           # planned
│   └── notify.ipynb              # planned
├── ui/
│   └── app.py                    # done — Streamlit frontend
├── tests/
│   ├── clean_tests.ipynb
│   ├── analytics_tests.ipynb
│   └── ui_tests.ipynb
├── requirements.txt
├── PROJECT_SPEC.md
├── README.md
└── codex_trail.txt
```

**Separation of concerns:** `model/analytics_core.py` owns computation and artifact writes. `ui/app.py` is frontend-only (select client → call backend → render).

## Execution Steps

### Prerequisites

- Python 3.11+
- Local access to the existing `data/` folder (already present)

### Virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### `.env` setup (needed when LLM categorize / coach land)

Create a `.env` file in the project root (never commit it):

```sh
LLM_PROVIDER=openai
LLM_API_KEY=...
LLM_MODEL=gpt-4o-mini
DATA_DIR=data
ARTIFACTS_DIR=artifacts
```

Optional: `AS_OF_DATE=YYYY-MM-DD` to override the default “today” (user’s max transaction date).

### Run the current pipeline

```sh
# From project root, with venv activated:

# 1) Notebooks (data prep)
#    Open and run: data_processing/validate_schema.ipynb
#    Open and run: data_processing/clean.ipynb
#       → artifacts/transactions_enriched.json
#       → artifacts/qa_report.json

# 2) Optional batch analytics for demo user 1696
#    Open and run: model/analytics.ipynb  (CLIENT_ID = 1696)

# 3) Launch Streamlit UI (computes analytics via backend for any client)
streamlit run ui/app.py
```

Demo user for screenshots/submission PDFs: **`client_id = 1696`** (UI default).  
The sidebar supports **any** `client_id` from `data/users.csv`.

### Tests

```sh
# Open and run notebooks under tests/:
#   tests/clean_tests.ipynb
#   tests/analytics_tests.ipynb
#   tests/ui_tests.ipynb
```

## Output Files

All outputs are under `artifacts/` (never under `data/`). Regenerable and gitignored:

| Artifact | Produced by | Notes |
|---|---|---|
| `artifacts/transactions_enriched.json` | `data_processing/clean.ipynb` | NDJSON for **all users** (large) |
| `artifacts/qa_report.json` | `data_processing/clean.ipynb` | QA counters + cleaning assumptions |
| `artifacts/spend_by_category_{client_id}.json` | `model/analytics_core.py` (UI or `analytics.ipynb`) | Category spend |
| `artifacts/spend_by_mcc_{client_id}.json` | `model/analytics_core.py` | MCC spend |
| `artifacts/budget_utilization_{client_id}.json` | `model/analytics_core.py` | Limit, MTD discretionary, utilization |
| `artifacts/spending_patterns_{client_id}.json` | `model/analytics_core.py` | Month / DOW / merchants |

Planned later: `categorized_*`, `runway_*` / `prediction_*`, `recommendations_*`, `events_*`, `session_*`.

## Guardrails

V1 guardrails enforced in prompts, pipeline config, and UI:

- **No investment, tax, or legal advice.** Refuse and redirect to budget/spend analysis.
- **Grounded answers only.** Coach responses restricted to the user’s own computed artifacts and summaries.
- **Refuse outside available data** (e.g. “Which stocks should I buy?”).
- **No fabricated numbers.** Every numeric claim must trace to values in `artifacts/`.
- **Spending limit definition.** `monthly_discretionary_limits` from `data/users.csv` after currency parsing.
- **Threshold alerts.** 70%, 85%, 95% of monthly discretionary limit based on MTD discretionary spend.
- **PII handling.** Do not expose `address`, `card_number`, or `cvv` in UI, logs, prompts, or derived artifacts. Use `client_id` as the primary identifier.
- **LLM cost and rate-limit caps.** Batch requests; cache results; enforce a configured daily cost cap; degrade gracefully when capped.
- **Fallback on API failure.** Categorization uses deterministic MCC mapping + `Other/Uncategorized`. Coach disables free-form chat and presents deterministic analytics + recommendations from artifacts.

## Notes

- Data is historical (2010s); “today” is `AS_OF_DATE` (default: user’s max transaction date).
- See [PROJECT_SPEC.md](PROJECT_SPEC.md) for full architecture, schemas, module responsibilities, and evaluation checklist.
- Next build priorities: `predict` → `recommend` → dashboard wiring → coach → LLM `categorize`.
