# ClearLedger — Personal Finance Coach v1

## Pipeline
1. Read source datasets from `data/` (transactions, cards, users, MCC lookup) **read-only**.
2. Clean + standardize types (currency/date), dedupe, and join into a unified dataset.
3. Categorize transactions with an LLM (batched) with caching; fall back to MCC rules if needed.
4. Compute analytics (spend by category, budget utilization, trends).
5. Train/infer a regression model to predict **days-to-limit** for the monthly discretionary budget.
6. Generate impact-ranked category-level recommendations to extend runway.
7. Emit notification events at 70% / 85% / 95% utilization (v1: in-app display only).
8. Launch Streamlit dashboard + multi-turn coach grounded in computed artifacts + local memory.

**All source data is read from the existing `data/` folder only.**
No source file is ever modified in place, and nothing is downloaded. All derived artifacts (cleaned datasets, caches, models, metrics, recommendations, chat state, logs) are written under `artifacts/` so runs are reproducible and the original dataset remains unchanged.

## Project Structure
(Proposed; will be created in implementation phase.)

```
data/                  # Existing read-only inputs
artifacts/             # Derived outputs only (written by pipeline)
src/                   # Pipeline modules + Streamlit app
tests/                 # Unit/integration tests
PROJECT_SPEC.md        # Detailed design spec
README.md              # This runbook
requirements.txt       # Python dependencies
```

## Run (Python)

### Prerequisites
- Python 3.11+
- Local access to the existing `data/` folder (already present)

### Virtual environment
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### `.env` setup (planned)
Create a `.env` file (never committed) with at minimum:
- `LLM_PROVIDER=openai` *(planned default)*
- `LLM_API_KEY=...`
- `LLM_MODEL=gpt-4o-mini` *(planned default; configurable)*
- `DATA_DIR=data`
- `ARTIFACTS_DIR=artifacts`
- `CLIENT_ID_DEFAULT=1696`

### Commands (Phase 0 note)
This repo is currently **Phase 0 (documentation only)**. The following commands are **planned** and do not exist yet:

```sh
# planned: run the full pipeline for a single client
python -m clearledger.pipeline run --client-id 1696

# planned: launch Streamlit UI (dashboard + coach)
streamlit run src/ui/app.py -- --client-id 1696
```

## Output Files
All outputs are derived artifacts written under `artifacts/` (never under `data/`). Exact filenames are part of the implementation contract:

- `artifacts/clean/transactions_enriched_{client_id}.parquet` (or `.csv`) — Processing
- `artifacts/clean/qa_report_{client_id}.json` — Processing
- `artifacts/cache/categorized_{client_id}.jsonl` — Categorization
- `artifacts/analytics/spend_by_category_{client_id}.json` — Analytics
- `artifacts/analytics/budget_utilization_{client_id}.json` — Analytics
- `artifacts/analytics/runway_{client_id}.json` — Prediction
- `artifacts/models/ridge_{client_id}.pkl` — Prediction
- `artifacts/metrics/prediction_{client_id}.json` — Prediction
- `artifacts/recommendations/recommendations_{client_id}.json` — Recommendations
- `artifacts/notifications/events_{client_id}.jsonl` — Notifications
- `artifacts/chat/session_{client_id}.json` — Coach (stateful memory)
- `artifacts/logs/pipeline_{client_id}.log` — Pipeline runner

## Current Guardrail Definition
V1 guardrails (must be enforced in prompts and UI):
- No personalized investment, tax, or legal advice. Refuse and redirect to budget/spend analysis.
- Coach answers restricted to the user’s own grounded data (computed artifacts and summaries).
- Refusal behavior when asked outside available data (e.g., “Which stocks should I buy?”).
- No fabricated numbers: every numeric claim must trace to computed values in artifacts.
- Spending limit definition: `monthly_discretionary_limits` from `data/users.csv` after currency parsing.
- Threshold alerts: 70%, 85%, 95% of monthly discretionary limit (based on MTD discretionary spend).
- PII handling:
  - Do not expose `address`, `card_number`, `cvv` in UI, logs, prompts, or derived artifacts.
  - Use `client_id` as the primary identifier.
- LLM cost and rate-limit caps:
  - Batch requests; cache results; enforce a configured daily cost cap; degrade gracefully when capped.
- Fallback behavior on API failure:
  - Categorization uses deterministic MCC mapping + `Other/Uncategorized` fallback.
  - Coach disables free-form chat and presents deterministic analytics + recommendations from artifacts.

## Notes
- Demo user for screenshots/examples: `client_id = 1696`.
- Data is historical (2010s); “today” is defined relative to `AS_OF_DATE` (default: user’s max transaction date).
- [VERIFY] Final category taxonomy and discretionary/non-discretionary mapping before implementation.
- Known gaps (to address in implementation phase):
  - Categorizations evaluation harness (spot checks + MCC agreement metrics).
  - Cold-start policy for prediction when insufficient history.

