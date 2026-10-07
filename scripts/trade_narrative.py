"""LLM trading narrative for the daily email, grounded in Gilman/Brody/execution artifacts."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import trade_store as store
from .trade_report import operational_summary

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRADING_NARRATIVE_MODEL = os.getenv(
    "TRADING_NARRATIVE_MODEL",
    os.getenv("TRENDS_MODEL", "gpt-5.6-luna"),
)

TRADING_NARRATIVE_REASONING = os.getenv(
    "TRADING_NARRATIVE_REASONING",
    os.getenv("TRENDS_REASONING", "low"),
)

TRADING_NARRATIVE_PROMPT = """
Write a concise trading briefing for the reader based ONLY on the supplied JSON.

Cover, in order:
1. What Gilman proposed (each candidate: ticker, direction, linked news theme, thesis in plain language).
2. How Brody evaluated those ideas (approve/reject per candidate and the main reasons).
3. Brody's portfolio-level decision and rationale.
4. What happened at execution (order submitted, filled, no fill, skipped, or error).

If research did not run (market closed or pipeline skipped research), say so clearly and do not invent proposals.

Use the operational_facts block for today's broker fills and open positions; do not contradict it.

The reader is not assumed to know internal codenames — refer to Gilman as the idea generator and Brody as the skeptical reviewer.

Write 160-260 words in plain text:
- no Markdown
- no headings
- no bullets
- no bold or italic markers

The FIRST line must stand alone as a short email subject suffix (under 120 characters) summarizing the day's outcome.

Return plain text only.
""".strip()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _decision_payload(db, run_id: str | None, source: str) -> dict | None:
    if not run_id:
        return None
    row = db.execute(
        "SELECT payload_json FROM decisions WHERE run_id=? AND source=? ORDER BY id DESC LIMIT 1",
        (run_id, source),
    ).fetchone()
    if not row:
        return None
    return json.loads(row["payload_json"])


def _run_context(db, run_id: str | None) -> dict[str, Any]:
    if not run_id:
        return {}
    row = db.execute(
        "SELECT status, details_json FROM runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if not row:
        return {}
    details = json.loads(row["details_json"] or "{}")
    return {
        "run_status": row["status"],
        "outcome": details.get("outcome"),
        "errors": details.get("errors") or [],
    }


def _execution_payload(run_id: str | None) -> dict | None:
    if not run_id:
        return None
    for path in (
        DATA_DIR / "alpaca_execution_latest.json",
        DATA_DIR / "runs" / f"{run_id}.execution.json",
    ):
        payload = _load_json(path)
        if payload and payload.get("run_id") == run_id:
            return payload
    return None


def _slim_gilman_trade(trade: dict) -> dict:
    return {
        "ticker": trade.get("ticker"),
        "direction": trade.get("direction"),
        "company_name": trade.get("company_name"),
        "investment_relevance": trade.get("investment_relevance"),
        "thesis": trade.get("thesis"),
        "confidence": trade.get("confidence"),
        "time_horizon": trade.get("time_horizon"),
        "risks": (trade.get("risks") or [])[:4],
    }


def _slim_brody_assessment(item: dict) -> dict:
    return {
        "ticker": item.get("ticker"),
        "direction": item.get("direction"),
        "verdict": item.get("verdict"),
        "confidence": item.get("confidence"),
        "reason": item.get("reason"),
        "red_flags": (item.get("red_flags") or [])[:4],
        "what_market_data_says": item.get("what_market_data_says"),
    }


def build_research_payload(
    db,
    run_id: str | None,
) -> dict[str, Any]:
    gilman = _decision_payload(db, run_id, "gilman")
    brody = _decision_payload(db, run_id, "brody")
    execution = _execution_payload(run_id)
    run_ctx = _run_context(db, run_id)

    proposals = []
    if gilman:
        for analysis in gilman.get("analyses") or []:
            if analysis.get("decision") != "TRADE" or not analysis.get("trade"):
                continue
            trade = {**analysis["trade"], "story_id": analysis.get("story_id")}
            proposals.append(_slim_gilman_trade(trade))

    brody_block = None
    if brody:
        brody_block = {
            "decision": brody.get("decision"),
            "confidence": brody.get("confidence"),
            "portfolio_reason": brody.get("portfolio_reason"),
            "research_only": brody.get("research_only"),
            "candidate_assessments": [
                _slim_brody_assessment(a)
                for a in (brody.get("candidate_assessments") or [])
            ],
            "selected_trade": (
                _slim_gilman_trade(brody["selected_trade"])
                if brody.get("selected_trade")
                else None
            ),
        }

    exec_block = None
    if execution:
        exec_block = {
            "status": execution.get("status"),
            "brody_decision": execution.get("brody_decision"),
            "paper": execution.get("paper"),
            "details": execution.get("details") or {},
        }

    return {
        "run_id": run_id,
        "pipeline": run_ctx,
        "gilman": {
            "trade_count": len(proposals),
            "proposals": proposals,
        }
        if proposals or gilman
        else None,
        "brody": brody_block,
        "execution": exec_block,
    }


def _research_ran(payload: dict[str, Any]) -> bool:
    return bool(payload.get("brody") or payload.get("gilman"))


def fallback_trading_narrative(
    research: dict[str, Any],
    ops_facts: str,
) -> str:
    pipeline = research.get("pipeline") or {}
    outcome = pipeline.get("outcome")
    errors = pipeline.get("errors") or []

    if outcome == "MARKET_CLOSED" and not _research_ran(research):
        headline = "Market closed — no new Gilman or Brody research today."
        body = (
            "The pipeline reconciled the paper account but skipped investment "
            "research and execution because the regular session was not open."
        )
    elif not _research_ran(research):
        headline = "No investment research recorded for this pipeline run."
        body = "Gilman and Brody did not produce decisions tied to this run."
        if errors:
            body += " Pipeline errors: " + "; ".join(errors[:3])
    else:
        brody = research.get("brody") or {}
        decision = brody.get("decision", "UNKNOWN")
        headline = f"Brody decision: {decision.replace('_', ' ')}."
        props = (research.get("gilman") or {}).get("proposals") or []
        if props:
            names = ", ".join(
                f"{p.get('ticker')} ({p.get('direction', '').lower()})"
                for p in props[:5]
            )
            body = f"Gilman proposed {len(props)} idea(s): {names}. "
        else:
            body = "Gilman did not advance any trade ideas to Brody. "
        body += brody.get("portfolio_reason") or ""
        execution = research.get("execution")
        if execution:
            body += f" Execution status: {execution.get('status')}."
        elif decision == "TRADE":
            body += " Execution outcome was not recorded for this run."

    ops = ops_facts.strip()
    if ops:
        return headline + "\n\n" + body.strip() + "\n\n—\n" + ops
    return headline + "\n\n" + body.strip()


def generate_trading_narrative(
    research: dict[str, Any],
    ops_facts: str,
) -> str:
    from openai import OpenAI

    logger.info(
        "Generating trading narrative with %s",
        TRADING_NARRATIVE_MODEL,
    )
    openai_client = OpenAI(max_retries=2, timeout=120.0)
    response = openai_client.responses.create(
        model=TRADING_NARRATIVE_MODEL,
        reasoning={"effort": TRADING_NARRATIVE_REASONING},
        instructions=TRADING_NARRATIVE_PROMPT,
        input=json.dumps(
            {
                "research": research,
                "operational_facts": ops_facts,
            },
            ensure_ascii=False,
        ),
        prompt_cache_key="daily-email-trading-narrative-v1",
        prompt_cache_retention="24h",
        max_output_tokens=1400,
        store=False,
    )
    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError("Trading narrative model returned no output.")
    return text


def compose_trading_section(
    db,
    run_id: str | None,
    now: datetime | None = None,
    zone: str = "America/Chicago",
) -> tuple[str, dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    ops = operational_summary(db, run_id, now, zone)
    research = build_research_payload(db, run_id)

    used_model = False
    try:
        if os.environ.get("OPENAI_API_KEY"):
            body = generate_trading_narrative(research, ops)
            used_model = True
        else:
            raise RuntimeError("OPENAI_API_KEY not set")
    except Exception as exc:
        logger.warning("Trading narrative unavailable: %s", exc)
        body = fallback_trading_narrative(research, ops)
        return body, {
            "generated_at": now.isoformat(),
            "run_id": run_id,
            "model": None,
            "fallback": True,
            "research": research,
            "operational_summary": ops,
            "narrative": body,
        }

    narrative = body.strip()
    if ops.strip() and ops.strip() not in narrative:
        narrative = narrative + "\n\n—\n" + ops.strip()

    meta = {
        "generated_at": now.isoformat(),
        "run_id": run_id,
        "model": TRADING_NARRATIVE_MODEL if used_model else None,
        "fallback": False,
        "research": research,
        "operational_summary": ops,
        "narrative": narrative,
    }
    return narrative, meta
