# Personal Finance Coach & Spending Analyzer

## Introduction

ClearLedger’s Personal Finance Coach & Spending Analyzer is a Python application that turns raw bank transactions into actionable guidance without manual tagging. A user (or demo operator) supplies transaction history via CSV; the system auto-categorizes every row with an off-the-shelf LLM, renders an interactive spending dashboard, predicts how many days remain before the user’s monthly discretionary limit is breached, ranks category-level spending reductions to extend that runway, and surfaces those grounded numbers inside a multi-turn conversational coach.

V1 is scoped for a six-week build: local Streamlit deployment, no model fine-tuning, and a single coherent pipeline that demonstrates four AI capabilities together—LLM classification, interactive visualization, regression-based prediction, and stateful coach memory—rather than four disconnected exercises.

## Problem Statement

ClearLedger serves ~55,000 millennial and Gen Z subscribers, but users must manually tag 60–150 transactions before seeing insights. That friction drives **71% second-session churn**. Feedback shows users want actionable guidance, not static reports, while competitors prepare AI finance coaches. ClearLedger has roughly a six-week window to strengthen its Series A narrative.

**Target outcomes**

- Time-to-first-insight under **five minutes** (no manual tagging)
- Improve retention toward **50%** by delivering value in the first session

**Constraints that shape design**

- Limited in-house ML expertise → prefer off-the-shelf LLM APIs and a simple per-user regression model
- No labeled training data for categories → few-shot prompting + MCC fallback; evaluate via QA and consistency, not supervised accuracy
- Strict cost limits → batching, caching, daily cost caps, and graceful degradation when the API is unavailable
- Local deployment acceptable for v1 → Streamlit on a developer machine; source CSVs stay on disk and are never modified in place

## Architecture Overview

The system is a layered pipeline. Each stage adds value and writes durable artifacts that later stages (and the UI) can trust:

```
Data Ingestion → Data Processing → Analytics & Intelligence → Prediction
→ Recommendations → Notifications → UI (Dashboard + Coach) → Validation & Testing
```

Mapped to implementation folders:

```
data/ (read-only)
  → data_processing/   # validate, clean, join, LLM categorize
  → artifacts/clean + artifacts/cache
  → model/             # analytics, days-to-limit, recommendations, alerts
  → artifacts/analytics|models|metrics|recommendations|notifications
  → ui/                # Streamlit dashboard + grounded coach
  → artifacts/chat
```

**Grounded AI rule:** the coach may only cite numbers that exist in computed artifacts. If a figure is missing or the request is outside available data, it must say so—never invent spend totals, limits, or projections.

```mermaid
flowchart TD
  dataDir["data/ read-only"] --> dataProcessing["data_processing/"]
  dataProcessing --> artifactsClean["artifacts/clean + cache"]
  artifactsClean --> modelLayer["model/"]
  modelLayer --> artifactsIntel["artifacts/analytics models metrics recommendations notifications"]
  artifactsIntel --> uiLayer["ui/ dashboard + coach"]
  uiLayer --> chatState["artifacts/chat"]
```

## Folder Structure

Inputs are read only from `data/`. All derived outputs are written under `artifacts/` so source integrity is preserved and runs are reproducible. Application code lives in three domain folders plus tests and config.

**Target implementation layout**

```
.
├── data/                         # Existing source files only (read-only)
│   ├── transactions.csv
│   ├── cards.csv
│   ├── users.csv
│   └── mcc_codes.json
├── artifacts/                    # Derived outputs only (write-only, flat files)
├── data_processing/              # Schema validation, cleaning, joins, LLM categorization
│   ├── validate_schema.ipynb
│   ├── clean.ipynb               # planned
│   ├── categorize.ipynb          # planned
├── model/                        # Analytics, prediction, recommendations, notifications
│   ├── analytics.ipynb           # planned
│   ├── predict.ipynb             # planned
│   ├── recommend.ipynb           # planned
│   └── notify.ipynb              # planned
├── ui/                           # Streamlit dashboard + multi-turn coach
│   ├── app.ipynb                 # planned
│   ├── dashboard.ipynb           # planned
│   └── coach.ipynb               # planned
├── tests/                        # Unit, integration, and validation hooks
├── config.yaml                   # Non-secret defaults (paths, thresholds, taxonomy)
├── .env                          # Secrets (never committed)
├── requirements.txt
├── PROJECT_SPEC.md
├── README.md
└── codex_trail.txt
```

Placement rules (business-justified):

| Concern | Location | Why |
|---|---|---|
| Schema validation, cleaning, joins, MCC enrichment | `data_processing/` | Trustworthy foundation without mutating source CSVs |
| LLM auto-categorization + MCC fallback + cache | `data_processing/categorize.ipynb` | Categorization is data prep that unlocks time-to-first-insight; not the sklearn model |
| Spend analytics, runway prediction, ranked recommendations, 70/85/95 alerts | `model/` | Intelligence outputs that feed the UI/coach from cleaned data |
| Dashboard + grounded chat memory | `ui/` | Presentation and stateful coaching; reads artifacts only |

## Dataset Specification

**Source files (read-only)** under `data/`:

| File | Role | Approx. size |
|---|---|---|
| `transactions.csv` | Transaction records (amounts, timestamps, merchant, MCC) | ~699,938 rows |
| `cards.csv` | Card profiles linked to clients | 300 rows |
| `users.csv` | Demographics + `monthly_discretionary_limits` | 100 rows |
| `mcc_codes.json` | MCC code → merchant category description | 109 codes |

**Demo user:** `client_id = 1696` (used for screenshots, end-to-end demo, and submission PDFs).

**Join keys**

- `transactions.client_id` → `users.id`
- `transactions.card_id` → `cards.id`
- `transactions.mcc` → keys in `mcc_codes.json`

**Observed schemas**

`transactions.csv`:
`id`, `date`, `client_id`, `card_id`, `amount`, `use_chip`, `merchant_id`, `merchant_city`, `merchant_state`, `zip`, `mcc`, `errors`

`cards.csv`:
`id`, `client_id`, `card_brand`, `card_type`, `card_number`, `expires`, `cvv`, `has_chip`, `num_cards_issued`, `credit_limit`, `acct_open_date`, `year_pin_last_changed`, `card_on_dark_web`

`users.csv`:
`id`, `current_age`, `retirement_age`, `birth_year`, `birth_month`, `gender`, `address`, `latitude`, `longitude`, `per_capita_income`, `yearly_income`, `total_debt`, `credit_score`, `num_credit_cards`, `monthly_discretionary_limits`

`mcc_codes.json`:
`{ "<mcc_code>": "<description>", ... }`

**Quality risks (handled in processing + QA report + safe fallbacks)**

- Currency strings with `$` and commas (e.g. `amount`, `monthly_discretionary_limits`, incomes)
- Nullable / mixed location fields for online purchases
- Duplicate transactions
- MCC codes missing from the lookup → fallback category path
- Historical 2010s dates → “today” is `AS_OF_DATE` (default: the user’s max transaction date)

**PII policy:** `address`, `card_number`, and `cvv` must not appear in UI, logs, prompts, or derived artifacts. Use `client_id` as the primary identifier.

## Module Specification

| Stage | Module | Responsibility | Primary artifacts |
|---|---|---|---|
| Ingestion / validation | `data_processing/validate_schema.ipynb` | Load read-only sources; enforce expected columns/types; fail fast on schema drift | `artifacts/schema_validation.json` |
| Processing | `data_processing/clean.ipynb` | Parse currency/dates, dedupe, join users/cards/MCC, write enriched frame + QA report | `artifacts/transactions_enriched_{client_id}.*`, `artifacts/qa_report_{client_id}.json` |
| Categorization | `data_processing/categorize.ipynb` | Batched LLM categorization with disk cache; MCC-based fallback on outage/cap | `artifacts/categorized_{client_id}.jsonl` |
| Analytics | `model/analytics.ipynb` | Category spend, MTD discretionary, budget utilization, trends | `artifacts/spend_by_category_{client_id}.json`, `artifacts/budget_utilization_{client_id}.json` |
| Prediction | `model/predict.ipynb` | Per-user regression for days-to-limit / projected month-end spend; cold-start fallback | `artifacts/ridge_{client_id}.pkl`, `artifacts/runway_{client_id}.json`, `artifacts/prediction_{client_id}.json` |
| Recommendations | `model/recommend.ipynb` | Ranked category-level reductions with computed “days gained” impact | `artifacts/recommendations_{client_id}.json` |
| Notifications | `model/notify.ipynb` | Emit 70% / 85% / 95% utilization events (v1: in-app) | `artifacts/events_{client_id}.jsonl` |
| UI + coach | `ui/app.ipynb`, `ui/dashboard.ipynb`, `ui/coach.ipynb` | Streamlit dashboard + multi-turn grounded coach with local session memory | `artifacts/session_{client_id}.json` |

**Critical degradation path:** if the LLM API is unreachable or the daily cost cap is hit → categorize via deterministic MCC mapping (plus `Other/Uncategorized`) and switch the coach to deterministic “status” mode that only presents computed analytics and recommendations (no free-form invention).

Each module reads from `data/` or upstream artifacts and writes only under `artifacts/`.

## Environment & Configuration

- **Python:** 3.11+
- **Dependencies:** install from `requirements.txt` (pandas, pandera, openai, scikit-learn, streamlit, plotly, etc.)
- **Secrets (`.env`, never committed):** `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, optional overrides for `DATA_DIR`, `ARTIFACTS_DIR`, `CLIENT_ID_DEFAULT`, `AS_OF_DATE`
- **Non-secrets (`config.yaml`):** paths, category taxonomy, discretionary mapping, notification thresholds (70/85/95), cost/rate caps, cold-start minimum history
- **Runtime:** local Streamlit for v1; run pipeline stages from the project root so `data/` and `artifacts/` resolve correctly

When the LLM is unavailable, configuration must force the documented fallback path and surface a clear UI warning—not a silent failure.

## Known Constraints & Limitations

- **No labeled category ground truth:** evaluation uses QA metrics, spot checks, and MCC agreement where available—not supervised accuracy.
- **Historical data (2010s):** calendar “today” is not wall-clock now; use `AS_OF_DATE` (default: user’s latest transaction date).
- **Single-user prediction sensitivity:** cold start and behavioral shifts can degrade the days-to-limit model; an explicit fallback policy is required.
- **Cost and rate limits:** LLM calls must be batched and cached; exceeding caps triggers MCC fallback.
- **Advice scope:** no personalized investment, tax, or legal advice—budget and spend analysis only.
- **PII:** address and card secrets stay out of prompts, logs, and UI.

## Evaluation Checklist

- [ ] End-to-end demo for `client_id = 1696`: reads `data/` read-only, writes `artifacts/`, launches Streamlit, and the coach answers using only grounded computed values.
- [ ] Four AI capabilities shown as one system: LLM categorization (cache + fallback), interactive visualization, regression-based days-to-limit prediction, and stateful multi-turn coach memory.
- [ ] Dashboard (user 1696): Monthly Limit Meter (actual vs projected), Customer Controls, Predicted Month-End Spend, Month-to-Date Discretionary Spend, Category Trend Chart with category buckets, Recommendations section.
- [ ] Chat interface (user 1696): usable multi-turn coach; MTD Spend, Limit, and Projected values visible as grounded context.
- [ ] Validation hooks / unit tests cover schema checks, read-only `data/` guarantee, artifact contracts, and pipeline smoke for 1696.

## Implementation Checklist

- [ ] Confirm `data/` schemas match this spec; document any drift and update join keys/handling rules.
- [ ] Implement `data_processing/validate_schema.ipynb` and `data_processing/clean.ipynb` → flat artifact files under `artifacts/` + QA report.
- [ ] Implement `data_processing/categorize.ipynb` with batching, caching, and MCC fallback; verify 100% transaction coverage.
- [ ] Implement `model/analytics.ipynb` so all dashboard figures are reproducible from artifacts.
- [ ] Implement `model/predict.ipynb` (days-to-limit / projected month-end) with cold-start fallback; write metrics as flat files under `artifacts/`.
- [ ] Implement `model/recommend.ipynb` with deterministic impact ranking (“days gained”).
- [ ] Implement `model/notify.ipynb` for 70/85/95 events as flat files under `artifacts/`.
- [ ] Implement `ui/` (dashboard + coach) reading only `artifacts/`; persist chat as a flat file under `artifacts/`; show degraded mode on LLM outage.
- [ ] Add `tests/`: read-only guarantee for `data/`, schema checks, artifact contracts, end-to-end smoke for `client_id=1696`.
