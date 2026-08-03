# Personal Finance Coach & Spending Analyzer (High-Level Spec)

## Introduction (2 points)
- A Python-based spending analyzer + conversational finance coach that converts raw transactions into **actionable guidance** quickly.
- V1 is optimized for a 6-week build: local Streamlit demo, off-the-shelf APIs, and a single coherent pipeline (not disconnected mini-projects).

## Problem Statement (2 points)
- Today, users must manually tag **60–150 transactions** before seeing insights, driving **71% second-session churn**; target outcomes are **<5 minutes time-to-first-insight** and **50% retention**.
- Constraints shape all choices: limited in-house ML expertise, no labeled data, strict cost limits (no training/fine-tuning), and local deployment acceptable for v1.

## Architecture Overview (2 points)
- Layered pipeline: **Ingestion → Processing → Analytics → Prediction → Recommendations → Notifications → UI (Dashboard + Coach) → Validation**.
- “Grounded AI” rule: the coach can only cite numbers computed from the pipeline outputs (no fabricated figures); when data is missing, it must say so.

```
data/ (read-only) → pipeline modules → artifacts/ (derived, write-only) → Streamlit (dashboard + chat)
```

## Folder Structure (2 points)
- Inputs are read only from the existing `data/` folder; all derived outputs are written under `artifacts/` to preserve source integrity and enable reproducible runs.
- Proposed implementation layout: `src/` for modules, `tests/` for validation, plus `requirements.txt`, `.env` (secrets), and `config.yaml` (non-secrets).

Proposed tree (text only; do not create in Phase 0):
```
data/          # existing source files only (read-only)
artifacts/     # derived outputs only (write-only)
src/           # ingestion/processing/categorization/analytics/prediction/recommendations/notifications/coach/ui
tests/         # unit + integration tests
```

## Dataset Specification (2 points)
- Source files (read-only): `data/transactions.csv`, `data/cards.csv`, `data/users.csv`, `data/mcc_codes.json`; demo user for examples/screenshots is `client_id = 1696`.
- Joins + quality risks (handled via cleaning + QA report + safe fallbacks): join keys are `transactions.client_id → users.id`, `transactions.card_id → cards.id`, `transactions.mcc → mcc_codes.json`; risks include currency strings, mixed/nullable location fields for online purchases, duplicates, and MCC codes missing from the lookup.

Observed high-level schemas (for implementation validation):
- `transactions.csv`: `id,date,client_id,card_id,amount,use_chip,merchant_id,merchant_city,merchant_state,zip,mcc,errors`
- `cards.csv`: `id,client_id,card_brand,card_type,card_number,expires,cvv,has_chip,num_cards_issued,credit_limit,acct_open_date,year_pin_last_changed,card_on_dark_web`
- `users.csv`: `id,...,address,...,monthly_discretionary_limits`
- `mcc_codes.json`: `{ "<mcc_code>": "<description>", ... }`

## Module Specification (2 points)
- Modules and responsibilities (high level): `ingestion` (load files), `processing` (clean/join/QA), `categorization` (LLM batch + cache + MCC fallback), `analytics` (spend + budget utilization), `prediction` (days-to-limit), `recommendations` (impact-ranked actions), `notifications` (70/85/95 alerts), `coach` (grounded multi-turn chat + memory), `ui` (Streamlit dashboard/chat).
- Each module reads from `data/` or upstream artifacts and writes its own outputs under `artifacts/`; critical degradation path is LLM outage → MCC-based fallback categorization + deterministic “status” coach mode.

## Environment & Configuration (2 points)
- Python **3.11+**; dependencies installed via `requirements.txt`; run locally with Streamlit for v1.
- Configuration: `.env` for secrets (LLM API key, model name/provider), `config.yaml` for non-secrets (paths, taxonomy, thresholds); define explicit behavior when the LLM API is unreachable (fallback + warnings).

## Known Constraints & Limitations (2 points)
- No labeled ground truth for categorization: evaluation relies on QA metrics, spot checks, and consistency (e.g., MCC agreement where available), not “accuracy vs labels.”
- Historical data (2010s) means “today” is `AS_OF_DATE` (default: the user’s max transaction date), and single-user prediction is sensitive to cold start and behavioral changes.

## Evaluation Checklist (2 points)
- [ ] End-to-end demo for `client_id = 1696`: reads `data/` read-only, writes `artifacts/`, launches Streamlit, and the coach answers using only grounded computed values.
- [ ] The four AI capabilities are demonstrated as one system: LLM categorization (with caching/fallback), interactive visualization, regression-based days-to-limit prediction, and stateful multi-turn coach memory.

## Implementation Checklist
- [ ] Confirm `data/` schemas match observed columns; document any drift and update join keys/handling rules.
- [ ] Implement ingestion + processing to produce `artifacts/clean/` outputs and a QA report (parse failures, duplicates, join misses).
- [ ] Implement LLM categorization with batching + caching and an MCC-based fallback path; verify 100% transaction coverage.
- [ ] Implement analytics artifacts (spend by category, budget utilization, trends) and ensure all figures are reproducible from artifacts.
- [ ] Implement prediction (days-to-limit) with a simple per-user model and an explicit cold-start fallback; write metrics to `artifacts/metrics/`.
- [ ] Implement deterministic recommendation ranking with computed “days gained” impact; optionally add LLM phrasing constrained to computed numbers.
- [ ] Implement threshold notifications (70/85/95) as in-app events persisted to `artifacts/notifications/`.
- [ ] Implement Streamlit UI (dashboard + coach) that reads only `artifacts/` and stores chat state locally; add a clear degraded-mode UI for LLM outages.
- [ ] Add validation/tests: read-only guarantee for `data/`, schema checks, artifact contract checks, and an end-to-end demo script for `client_id=1696`.
