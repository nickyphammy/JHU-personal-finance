"""Chat coach backend (grounded, multi-turn, local memory).

The coach answers questions about spending using only computed context:
- MTD discretionary spend
- monthly discretionary limit
- utilization + projected utilization
- projected month-end discretionary spend
- as-of date

LLM is optional: deterministic affordabilty Q&A works even without API access.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd


AMOUNT_RE = re.compile(
    r"(?i)(?:\$\s*)?(-?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
)


def extract_first_amount_usd(text: str) -> float | None:
    """Extract the first currency-like amount from a user message."""
    if not text:
        return None
    m = AMOUNT_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def format_usd(x: float | None) -> str:
    return "N/A" if x is None else f"${x:,.2f}"


def build_grounded_context(
    *,
    client_id: int,
    as_of_date: str,
    mtd_spend_usd: float,
    limit_usd: float,
    utilization_pct: float | None,
    projected_month_end_usd: float | None,
    projected_utilization_pct: float | None,
) -> dict[str, Any]:
    remaining = float(limit_usd - mtd_spend_usd)
    return {
        "client_id": int(client_id),
        "as_of_date": str(as_of_date),
        "mtd_discretionary_spend_usd": float(mtd_spend_usd),
        "monthly_discretionary_limit_usd": float(limit_usd),
        "remaining_budget_usd": remaining,
        "utilization_pct": utilization_pct,
        "predicted_month_end_discretionary_spend_usd": projected_month_end_usd,
        "projected_utilization_pct": projected_utilization_pct,
    }


def affordability_reply(context: dict[str, Any], *, purchase_usd: float) -> str:
    """Deterministic affordability response (no LLM)."""
    remaining = float(context["remaining_budget_usd"])
    after = remaining - float(purchase_usd)
    if remaining < 0:
        return (
            f"No. You are already over your discretionary limit.\n\n"
            f"- Remaining budget: {format_usd(remaining)}\n"
            f"- If you spend {format_usd(purchase_usd)} now: {format_usd(after)}"
        )
    if after < 0:
        return (
            f"No. If you spend {format_usd(purchase_usd)} now, your remaining discretionary budget becomes "
            f"{format_usd(after)} (current remaining: {format_usd(remaining)})."
        )
    return (
        f"Yes. If you spend {format_usd(purchase_usd)} now, your remaining discretionary budget becomes "
        f"{format_usd(after)} (current remaining: {format_usd(remaining)})."
    )


def build_system_prompt() -> str:
    return (
        "You are ClearLedger's Personal Finance Coach.\n"
        "You help users understand discretionary spending using ONLY the provided grounded context.\n"
        "Guardrails:\n"
        "- No investment/tax/legal advice.\n"
        "- Do not fabricate numbers. Only use values present in the context.\n"
        "- If asked for data you don't have, say what you can compute from context and ask a clarifying question.\n"
        "- Keep answers concise and action-oriented.\n"
        "When answering with numbers, reference the exact context fields.\n"
    )


def build_user_prompt(*, context: dict[str, Any], question: str) -> str:
    payload = {
        "context": context,
        "question": question,
        "required_fields": [
            "as_of_date",
            "mtd_discretionary_spend_usd",
            "monthly_discretionary_limit_usd",
            "utilization_pct",
            "predicted_month_end_discretionary_spend_usd",
            "projected_utilization_pct",
            "remaining_budget_usd",
        ],
        "response_format": {
            "style": "plain text",
            "include": ["short_answer", "1-3 bullets of reasoning", "next step question if needed"],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


LLMCall = Callable[[list[dict[str, str]]], str]


def make_openai_chat_call(
    *,
    model: str,
    api_key_env: str = "LLM_API_KEY",
    base_url: str | None = None,
) -> LLMCall:
    """Create an OpenAI chat completion function using environment configuration.

    Base URL resolution order:
    1. explicit `base_url` argument
    2. `OPENAI_BASE_URL`
    3. `LLM_API_BASE`
    4. provider default
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

    from openai import OpenAI  # lazy import

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if resolved_base:
        client_kwargs["base_url"] = resolved_base
    client = OpenAI(**client_kwargs)

    def _call(messages: list[dict[str, str]]) -> str:
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=300,
                )
            except TypeError:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=300,
                )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()

    return _call


@dataclass(frozen=True)
class CoachReply:
    content: str
    used_llm: bool
    purchase_usd: float | None = None


def answer_question(
    *,
    context: dict[str, Any],
    question: str,
    llm_call: LLMCall | None,
    history: list[dict[str, str]] | None = None,
) -> CoachReply:
    q = (question or "").strip()
    if not q:
        return CoachReply(content="Ask a question about your discretionary spending.", used_llm=False)

    amt = extract_first_amount_usd(q)
    prev_assistant = ""
    if history:
        for m in reversed(history):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                prev_assistant = m["content"].lower()
                break
    amount_only_followup = amt is not None and q.strip().replace("$", "").replace(",", "").replace(".", "").isdigit()
    hinted = any(k in q.lower() for k in ["afford", "can i", "spend"]) or (
        amount_only_followup and any(k in prev_assistant for k in ["purchase amount", "afford", "how much"])
    )

    if amt is not None and hinted:
        return CoachReply(content=affordability_reply(context, purchase_usd=float(amt)), used_llm=False, purchase_usd=float(amt))

    if llm_call is None:
        # Deterministic fallback summary
        return CoachReply(
            content=(
                "LLM chat is unavailable (missing API key or disabled). Here’s what I can compute:\n\n"
                f"- As of {context['as_of_date']}, MTD discretionary spend is {format_usd(context['mtd_discretionary_spend_usd'])}\n"
                f"- Monthly limit is {format_usd(context['monthly_discretionary_limit_usd'])}\n"
                f"- Remaining budget is {format_usd(context['remaining_budget_usd'])}\n"
                "Ask: “Can I afford $X today?” for a deterministic answer."
            ),
            used_llm=False,
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": build_system_prompt()}]
    if history:
        # Keep recent history small (token budget)
        messages.extend(history[-10:])
    messages.append({"role": "user", "content": build_user_prompt(context=context, question=q)})
    content = llm_call(messages)
    return CoachReply(content=content, used_llm=True)


def session_path(root: Path, *, client_id: int) -> Path:
    return root / "artifacts" / f"session_{int(client_id)}.json"


def load_session(root: Path, *, client_id: int) -> list[dict[str, str]]:
    path = session_path(root, client_id=client_id)
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return []
    out: list[dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def write_session(root: Path, *, client_id: int, messages: list[dict[str, str]]) -> Path:
    path = session_path(root, client_id=client_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep file small
    trimmed = messages[-40:]
    payload = {
        "client_id": int(client_id),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "messages": trimmed,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
