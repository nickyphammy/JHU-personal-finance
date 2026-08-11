# ClearLedger — Personal Finance Coach

ClearLedger is a subscription personal finance app facing high early churn because users must manually tag dozens of transactions before seeing value. This project implements a **Personal Finance Coach** that ingests a user’s bank CSV, auto-categorizes transactions with an LLM (no manual tagging), renders an interactive spending dashboard, predicts days remaining before the monthly discretionary limit is hit, ranks category-level spending reductions, and exposes those grounded insights in a multi-turn AI coach.

V1 runs locally with Streamlit, keeps source data under `data/` read-only, and writes all derived outputs under `artifacts/`.

## Pipeline

1. Read source datasets from `data/` (transactions, cards, users, MCC lookup) **read-only**.
2. Validate schemas, clean/standardize types (currency/date), dedupe, and join into a unified dataset.
3. Categorize transactions with an LLM (batched) with caching; fall back to MCC rules if needed.
4. Compute analytics (spend by category, budget utilization, trends).
5. Train/infer a regression model to predict **days-to-limit** for the monthly discretionary budget.
6. Generate impact-ranked category-level recommendations to extend runway.
7. Emit notification events at 70% / 85% / 95% utilization (v1: in-app display only).
8. Launch Streamlit dashboard + multi-turn coach grounded in computed artifacts + local memory.

**All source data is read from the existing `data/` folder only.** No source file is modified in place, and nothing is downloaded. All derived artifacts (cleaned datasets, caches, models, metrics, recommendations, chat state, logs) are written under `artifacts/`.

## Project Structure

Target implementation layout (domain folders instead of a nested `src/` package):

```
.
├── data/                         # Existing read-only inputs
│   ├── transactions.csv
│   ├── cards.csv
│   ├── users.csv
│   └── mcc_codes.json
├── artifacts/                    # Derived outputs only (written by pipeline, flat files)
├── data_processing/              # Schema validation, cleaning, joins, LLM categorization
│   ├── validate_schema.ipynb
│   ├── clean.ipynb
│   └── categorize.ipynb          # planned
├── model/                        # Analytics, prediction, recommendations, notifications
│   ├── analytics.ipynb           # planned
│   ├── predict.ipynb             # planned
│   ├── recommend.ipynb           # planned
│   └── notify.ipynb              # planned
├── ui/                           # Streamlit dashboard + coach
│   ├── app.ipynb                 # planned
│   ├── dashboard.ipynb           # planned
│   └── coach.ipynb               # planned
├── tests/                        # Unit / integration / validation hooks
├── config.yaml                   # Non-secret defaults
├── .env                          # Secrets (not committed)
├── requirements.txt
├── PROJECT_SPEC.md
├── README.md
└── codex_trail.txt
```

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

### `.env` setup

Create a `.env` file in the project root (never commit it) with at minimum:

```sh
LLM_PROVIDER=openai
LLM_API_KEY=...
LLM_MODEL=gpt-4o-mini
DATA_DIR=data
ARTIFACTS_DIR=artifacts
CLIENT_ID_DEFAULT=1696
```

Optional: set `AS_OF_DATE=YYYY-MM-DD` to override the default “today” (user’s max transaction date).

### Pipeline and UI commands

Run from the project root after activating the virtual environment. Pipeline modules are the intended entrypoints once implemented:

```sh
# Current workflow (notebook-first):
# 1) Open and run: data_processing/validate_schema.ipynb
# 2) Open and run: data_processing/clean.ipynb (set CLIENT_ID at the top)
#
# Planned: model/* and ui/* notebooks and runnable CLI entrypoints (see PROJECT_SPEC.md)
```

Demo user for screenshots and submission PDFs: **`client_id = 1696`**.

## Output Files

All outputs are derived artifacts under `artifacts/` (never under `data/`). Current clean-stage contract:

| Artifact | Produced by | Notes |
|---|---|---|
| `artifacts/transactions_enriched.json` | `data_processing/clean.ipynb` | NDJSON for **all users** (large; local / gitignored) |
| `artifacts/transactions_enriched_1696.json` | `data_processing/clean.ipynb` | Pretty JSON focus export for demo user 1696 |
| `artifacts/qa_report.json` | `data_processing/clean.ipynb` | QA counters + cleaning assumptions |

## Guardrails

V1 guardrails enforced in prompts, pipeline config, and UI:

- **No investment, tax, or legal advice.** Refuse and redirect to budget/spend analysis.
- **Grounded answers only.** Coach responses are restricted to the user’s own computed artifacts and summaries.
- **Refuse outside available data** (e.g. “Which stocks should I buy?”).
- **No fabricated numbers.** Every numeric claim must trace to values in `artifacts/`.
- **Spending limit definition.** `monthly_discretionary_limits` from `data/users.csv` after currency parsing.
- **Threshold alerts.** 70%, 85%, 95% of monthly discretionary limit based on MTD discretionary spend.
- **PII handling.** Do not expose `address`, `card_number`, or `cvv` in UI, logs, prompts, or derived artifacts. Use `client_id` as the primary identifier.
- **LLM cost and rate-limit caps.** Batch requests; cache results; enforce a configured daily cost cap; degrade gracefully when capped.
- **Fallback on API failure.** Categorization uses deterministic MCC mapping + `Other/Uncategorized`. Coach disables free-form chat and presents deterministic analytics + recommendations from artifacts.

## Notes

- Data is historical (2010s); “today” is defined relative to `AS_OF_DATE` (default: user’s max transaction date).
- See [PROJECT_SPEC.md](PROJECT_SPEC.md) for full architecture, schemas, module responsibilities, and evaluation checklist.
- Known implementation follow-ups: categorization evaluation harness (spot checks + MCC agreement) and cold-start policy details for prediction when history is thin.
