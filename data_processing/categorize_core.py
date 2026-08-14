"""LLM-based transaction categorization with MCC fallback + disk cache.

Design goals:
- 100% coverage: every transaction gets a category via LLM or deterministic MCC fallback.
- Cost/latency control: batch requests + disk cache keyed by stable transaction fingerprint.
- Safety: no PII; outputs are derived artifacts under `artifacts/` only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_exponential

from model.analytics_core import mcc_to_category


DEFAULT_TAXONOMY: tuple[str, ...] = (
    "Dining",
    "Groceries",
    "Utilities",
    "Transportation",
    "Entertainment",
    "Shopping",
    "Travel",
    "Housing",
    "Healthcare",
    "Education",
    "Income",
    "Transfers",
    "Subscriptions",
    "Fees & Interest",
    "Cash Withdrawal",
    "Other/Uncategorized",
)


def transaction_fingerprint(tx: dict[str, Any], *, version: str = "v1") -> str:
    """Stable cache key derived from non-PII transaction fields."""
    payload = {
        "v": version,
        "id": tx.get("id"),
        "client_id": tx.get("client_id"),
        "card_id": tx.get("card_id"),
        "transaction_dt": tx.get("transaction_dt") or tx.get("date"),
        "amount_usd": tx.get("amount_usd") or tx.get("amount"),
        "mcc_code": tx.get("mcc_code") or tx.get("mcc"),
        "mcc_description": tx.get("mcc_description"),
        "merchant_id": tx.get("merchant_id"),
        "merchant_city": tx.get("merchant_city"),
        "merchant_state": tx.get("merchant_state"),
        "use_chip": tx.get("use_chip"),
        "errors": tx.get("errors"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def merchant_fingerprint(tx: dict[str, Any], *, version: str = "v1") -> str:
    """Cache key for repeated merchants to minimize LLM spend.

    Uses only non-PII merchant/MCC context (no card numbers, address, etc.).
    """
    payload = {
        "v": version,
        "merchant_id": tx.get("merchant_id"),
        "merchant_city": tx.get("merchant_city"),
        "merchant_state": tx.get("merchant_state"),
        "mcc_code": tx.get("mcc_code") or tx.get("mcc"),
        "mcc_description": tx.get("mcc_description"),
        "use_chip": tx.get("use_chip"),
        "errors": tx.get("errors"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_llm_messages(
    tx_batch: list[dict[str, Any]],
    *,
    taxonomy: Iterable[str] = DEFAULT_TAXONOMY,
) -> list[dict[str, str]]:
    """Few-shot prompt with strict JSON output contract."""
    tax = list(taxonomy)
    system = (
        "You are a transaction categorization engine for a personal finance app.\n"
        "Return ONLY valid JSON. Do not include prose.\n"
        "Categories must be one of the provided taxonomy.\n"
        "If uncertain, choose 'Other/Uncategorized' with low confidence.\n"
        "Do not infer personal details; use only provided transaction fields."
    )

    # Few-shot examples grounded in MCC descriptions (deterministic fallback exists).
    examples = [
        {
            "id": "ex1",
            "amount_usd": 12.34,
            "mcc_code": "5812",
            "mcc_description": "Eating Places and Restaurants",
            "merchant_city": "Dallas",
            "merchant_state": "TX",
        },
        {
            "id": "ex2",
            "amount_usd": 86.45,
            "mcc_code": "5541",
            "mcc_description": "Service Stations",
            "merchant_city": "Merritt Island",
            "merchant_state": "FL",
        },
        {
            "id": "ex3",
            "amount_usd": 340.00,
            "mcc_code": "4722",
            "mcc_description": "Travel Agencies and Tour Operators",
            "merchant_city": "ONLINE",
            "merchant_state": None,
        },
    ]
    example_output = [
        {"id": "ex1", "category": "Dining", "confidence": 0.9},
        {"id": "ex2", "category": "Transportation", "confidence": 0.7},
        {"id": "ex3", "category": "Travel", "confidence": 0.7},
    ]

    user = {
        "taxonomy": tax,
        "instructions": (
            "Classify each transaction into exactly one category from taxonomy.\n"
            "Return JSON with shape: {\"results\": [{\"id\": <id>, \"category\": <str>, \"confidence\": <0..1>}, ...]}\n"
            "The number of results must equal the number of input transactions.\n"
            "Use id passthrough: results[i].id must match transactions[i].id."
        ),
        "examples": {"transactions": examples, "results": example_output},
        "transactions": [
            {
                "id": t.get("id"),
                "amount_usd": t.get("amount_usd"),
                "mcc_code": t.get("mcc_code") or t.get("mcc"),
                "mcc_description": t.get("mcc_description"),
                "merchant_city": t.get("merchant_city"),
                "merchant_state": t.get("merchant_state"),
                "use_chip": t.get("use_chip"),
                "errors": t.get("errors"),
            }
            for t in tx_batch
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


@dataclass(frozen=True)
class CategorizedResult:
    transaction_id: str
    category_final: str
    source: str  # "llm" | "mcc_fallback" | "low_confidence_fallback"
    confidence: float | None
    category_llm: str | None = None


LLMCall = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


def parse_llm_results(payload: str) -> list[dict[str, Any]]:
    obj = json.loads(payload)
    results = obj.get("results")
    if not isinstance(results, list):
        raise ValueError("LLM payload missing 'results' list")
    return results


def make_openai_llm_call(
    *,
    model: str,
    api_key_env: str = "LLM_API_KEY",
    base_url: str | None = None,
) -> LLMCall:
    """Create an OpenAI LLM call function using environment configuration.

    Base URL resolution order:
    1. explicit `base_url` argument
    2. `OPENAI_BASE_URL`
    3. `LLM_API_BASE`
    4. provider default (api.openai.com)
    """
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set `{api_key_env}` or `OPENAI_API_KEY` in environment/.env."
        )

    resolved_base = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_API_BASE")
        or None
    )

    from openai import OpenAI  # imported lazily for testability

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if resolved_base:
        client_kwargs["base_url"] = resolved_base
    client = OpenAI(**client_kwargs)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(4))
    def _call(tx_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = build_llm_messages(tx_batch)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )

        content = resp.choices[0].message.content or ""
        return parse_llm_results(content)

    return _call


def categorize_with_cache_and_fallback(
    transactions: list[dict[str, Any]],
    *,
    llm_call: LLMCall,
    cache: Cache,
    confidence_threshold: float = 0.6,
    taxonomy: Iterable[str] = DEFAULT_TAXONOMY,
    cache_version: str = "v1",
    cache_key_fn: Callable[[dict[str, Any]], str] | None = None,
) -> list[CategorizedResult]:
    """Categorize a batch using cache → LLM → MCC fallback with low-confidence handling."""
    tax = set(taxonomy)
    key_fn = cache_key_fn or (lambda tx: transaction_fingerprint(tx, version=cache_version))
    results: list[CategorizedResult] = []
    pending: list[dict[str, Any]] = []
    pending_keys: list[str] = []

    for tx in transactions:
        key = f"{cache_version}:{key_fn(tx)}"
        cached = cache.get(key)
        if isinstance(cached, dict) and "category_final" in cached:
            results.append(
                CategorizedResult(
                    transaction_id=str(tx.get("id")),
                    category_final=str(cached["category_final"]),
                    source=str(cached.get("source", "llm")),
                    confidence=cached.get("confidence"),
                    category_llm=cached.get("category_llm"),
                )
            )
        else:
            pending.append(tx)
            pending_keys.append(key)

    if pending:
        try:
            llm_rows = llm_call(pending)
            llm_by_id = {str(r.get("id")): r for r in llm_rows if "id" in r}
        except Exception:
            llm_by_id = {}

        for tx, key in zip(pending, pending_keys, strict=True):
            tx_id = str(tx.get("id"))
            llm_row = llm_by_id.get(tx_id)

            category_llm = None
            confidence = None
            category_final = None
            source = "mcc_fallback"

            if isinstance(llm_row, dict):
                category_llm = llm_row.get("category")
                confidence = llm_row.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidence = float(confidence)
                else:
                    confidence = None

                if isinstance(category_llm, str) and category_llm in tax:
                    if confidence is not None and confidence < confidence_threshold:
                        category_final = mcc_to_category(tx.get("mcc_code") or tx.get("mcc"), tx.get("mcc_description"))
                        source = "low_confidence_fallback"
                    else:
                        category_final = category_llm
                        source = "llm"

            if category_final is None:
                category_final = mcc_to_category(tx.get("mcc_code") or tx.get("mcc"), tx.get("mcc_description"))

            cached_obj = {
                "transaction_id": tx_id,
                "category_final": category_final,
                "category_llm": category_llm,
                "confidence": confidence,
                "source": source,
            }
            cache.set(key, cached_obj)
            results.append(
                CategorizedResult(
                    transaction_id=tx_id,
                    category_final=category_final,
                    category_llm=category_llm if isinstance(category_llm, str) else None,
                    confidence=confidence,
                    source=source,
                )
            )

    return results
