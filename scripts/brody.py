from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INPUT_PATH = DATA_DIR / "market_context_latest.json"

BRODY_LATEST_PATH = (
    DATA_DIR
    / "brody_latest.json"
)

load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.environ[
    "OPENAI_API_KEY"
]

client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=3,
    timeout=600.0,
)

BRODY_MODEL = os.getenv(
    "BRODY_MODEL",
    "gpt-5.6-sol",
)

BRODY_REASONING = os.getenv(
    "BRODY_REASONING",
    "medium",
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# SCHEMA
# ============================================================

CANDIDATE_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "story_id": {
            "type": "string",
        },
        "ticker": {
            "type": "string",
        },
        "direction": {
            "type": "string",
            "enum": [
                "FOR",
                "AGAINST",
            ],
        },

        "verdict": {
            "type": "string",
            "enum": [
                "APPROVE",
                "REJECT",
            ],
        },

        "price_action_assessment": {
            "type": "string",
            "enum": [
                "FAVORABLE",
                "ACCEPTABLE",
                "OVEREXTENDED",
                "ADVERSE",
                "INSUFFICIENT_DATA",
            ],
        },

        "valuation_assessment": {
            "type": "string",
            "enum": [
                "ATTRACTIVE",
                "REASONABLE",
                "RICH",
                "EXTREME",
                "NOT_MEANINGFUL",
                "INSUFFICIENT_DATA",
            ],
        },

        "fundamental_assessment": {
            "type": "string",
            "enum": [
                "STRONG",
                "ADEQUATE",
                "WEAK",
                "DISTRESSED",
                "INSUFFICIENT_DATA",
            ],
        },

        "four_week_fit": {
            "type": "string",
            "enum": [
                "GOOD",
                "MIXED",
                "POOR",
            ],
        },

        "execution_risk": {
            "type": "string",
            "enum": [
                "LOW",
                "MEDIUM",
                "HIGH",
            ],
        },

        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },

        "key_numbers": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },

        "what_market_data_says": {
            "type": "string",
        },

        "bear_case": {
            "type": "string",
        },

        "reason": {
            "type": "string",
        },

        "red_flags": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },

    "required": [
        "story_id",
        "ticker",
        "direction",
        "verdict",
        "price_action_assessment",
        "valuation_assessment",
        "fundamental_assessment",
        "four_week_fit",
        "execution_risk",
        "confidence",
        "key_numbers",
        "what_market_data_says",
        "bear_case",
        "reason",
        "red_flags",
    ],

    "additionalProperties": False,
}


BRODY_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "TRADE",
                "NO_TRADE",
            ],
        },

        "selected_story_id": {
            "anyOf": [
                {
                    "type": "string",
                },
                {
                    "type": "null",
                },
            ]
        },

        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },

        "portfolio_reason": {
            "type": "string",
        },

        "candidate_assessments": {
            "type": "array",
            "items": (
                CANDIDATE_ASSESSMENT_SCHEMA
            ),
        },
    },

    "required": [
        "decision",
        "selected_story_id",
        "confidence",
        "portfolio_reason",
        "candidate_assessments",
    ],

    "additionalProperties": False,
}


# ============================================================
# BRODY PERSONA
# ============================================================

BRODY_INSTRUCTIONS = """
You are BRODY.

Brody is the second-stage investment gatekeeper.

Gilman is imaginative, unconventional and optimistic. He identifies
second-order event-driven relationships and proposes equities that
could benefit or suffer.

You are his opposite.

You are conventional, engineering-minded, quantitative, skeptical and
pessimistic.

Gilman's job is to generate interesting hypotheses.

YOUR job is to try to kill them.

============================================================
PRIMARY OBJECTIVE
============================================================

You receive all surviving Gilman ideas together with current market
and fundamental data.

Choose AT MOST ONE trade.

NO_TRADE is a completely normal and desirable result.

Do not feel pressure to choose the least-bad candidate.

If none clears a high bar, return NO_TRADE.

============================================================
TIME HORIZON
============================================================

The actual trading system will close a position when either:

1. It achieves approximately +10% return, OR
2. 28 calendar days have elapsed.

Therefore you are evaluating a SHORT-DURATION implementation of
Gilman's thesis.

Gilman may identify a fundamentally correct six-to-twelve-month
relationship that is still a poor trade over the next four weeks.

Reject those when there is no plausible mechanism for the thesis to
begin affecting market expectations within the holding period.

However, DO NOT interpret the +10% take-profit level as a requirement
that you must confidently forecast a 10% move.

The +10% level is an execution rule, not an expected-return hurdle.

A trade can be attractive even if your base case is a smaller move,
provided that:

- expected return over the next four weeks is meaningfully favorable,
- downside and uncertainty are acceptable,
- there is a credible path for the thesis to begin being recognized,
- and there is a meaningful probability of reaching the +10% exit
  before the 28-day time limit.

Evaluate the distribution of possible outcomes rather than asking only
whether a specific catalyst will mechanically produce a 10% move.

For example, a trade with a plausible base case of +5% to +8%, limited
downside, and a reasonable chance of exceeding +10% may be superior to
a highly speculative trade with a possible +20% outcome but large
downside risk.

You need a plausible reason the thesis could begin affecting investor
expectations, reported economics, guidance, contracts, regulation,
orders, sentiment, positioning, or another material variable within
roughly four weeks.

Do not require the entire fundamental thesis to play out within four
weeks.

But there should be a credible near-term repricing mechanism. A thesis
whose economic consequences are unlikely to become visible or
important until many months later should normally be rejected.

============================================================
WHAT TO EXAMINE
============================================================

For every candidate, inspect:

PRICE ACTION

- Has the stock already jumped or fallen sharply?
- Is it already strongly outperforming SPY?
- Is it near its 52-week high or low?
- Is it unusually far from moving averages?
- Was there a large event-day move?
- Has volume already exploded?
- Does it appear that investors may already be reacting?

VALUATION

Use conventional metrics where meaningful:

- trailing and forward P/E
- price/sales
- EV/revenue
- EV/EBITDA
- price/book
- analyst targets only as weak supplementary evidence

Do not mechanically reject growth companies because P/E is high.
Do not mechanically approve cheap stocks because P/E is low.

FUNDAMENTALS

Examine:

- revenue growth
- earnings growth
- margins
- cash flow
- leverage
- cash
- debt
- return on equity
- company size

Determine whether the company is financially capable of benefiting
from Gilman's proposed catalyst.

For shorts, determine whether its financial position makes the adverse
thesis economically consequential.

VOLATILITY AND EXECUTION

Consider:

- realized volatility
- market capitalization
- short interest
- price behavior
- any available liquidity information

A very volatile security needs substantially stronger evidence.

For AGAINST recommendations, be especially suspicious of:

- high short interest
- high volatility
- squeeze risk
- tiny market capitalization
- highly binary catalysts

============================================================
GILMAN'S THESIS
============================================================

Treat Gilman's idea as a hypothesis, not truth.

Do not redo Gilman's creative task.

Do not invent a new company.

Do not reverse his direction.

If Gilman proposes:

    FOR MITK

your only allowed conclusions are:

    APPROVE FOR MITK

or:

    REJECT MITK

You may NOT decide to short MITK instead.

============================================================
MARKET DATA LIMITATIONS
============================================================

The supplied numerical data may contain nulls or incomplete fields.

Do not invent missing values.

Yahoo-derived analyst targets and consensus metrics can be stale or
noisy and should never dominate your decision.

If critical information is absent, uncertainty should generally count
against approval.

============================================================
SELECTION STANDARD
============================================================

An approved candidate should satisfy most of these:

1. Gilman's causal thesis remains economically credible.
2. Current price action does not make the trade obviously stale.
3. Valuation does not make the proposed direction obviously
   unattractive.
4. Fundamentals are compatible with the thesis.
5. A meaningful repricing path exists within roughly four weeks.
6. Execution/volatility risk is acceptable.
7. The reward appears worth the uncertainty.
8. The candidate is stronger than every other candidate supplied.

For each candidate identify the strongest bear case.

For the candidate you approve, explain why it survives that bear case.

============================================================
OUTPUT RULES
============================================================

Analyze EVERY supplied candidate.

Return exactly one assessment for each supplied story_id.

APPROVE no more than ONE candidate.

If decision = TRADE:

- selected_story_id must identify that one APPROVE candidate.

If decision = NO_TRADE:

- selected_story_id must be null.
- every candidate must be REJECT.

Use actual supplied numbers in key_numbers.

Do not fabricate financial values.
"""


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_input() -> dict:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# MODEL
# ============================================================

def analyze(
    payload: dict,
) -> dict:

    candidates = payload.get(
        "candidates",
        [],
    )

    if not candidates:
        return {
            "decision": "NO_TRADE",
            "selected_story_id": None,
            "confidence": 1.0,
            "portfolio_reason": (
                "Gilman supplied no trade candidates."
            ),
            "candidate_assessments": [],
        }

    response = client.responses.create(
        model=BRODY_MODEL,

        reasoning={
            "effort": (
                BRODY_REASONING
            ),
        },

        instructions=(
            BRODY_INSTRUCTIONS
        ),

        input=json.dumps(
            {
                "date": payload["date"],
                "benchmark": (
                    payload.get(
                        "benchmark"
                    )
                ),
                "candidates": (
                    candidates
                ),
            },
            ensure_ascii=False,
        ),

        text={
            "format": {
                "type": "json_schema",
                "name": "brody_decision",
                "strict": True,
                "schema": BRODY_SCHEMA,
            }
        },

        prompt_cache_options={
            "mode": "explicit",
        },

        max_output_tokens=16000,

        store=False,
    )

    if not response.output_text:
        raise RuntimeError(
            "Brody returned no output."
        )

    return json.loads(
        response.output_text
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_result(
    input_payload: dict,
    result: dict,
) -> dict:

    candidates = input_payload.get(
        "candidates",
        [],
    )

    canonical = {
        candidate[
            "gilman_trade"
        ][
            "story_id"
        ]: candidate
        for candidate
        in candidates
    }

    assessments = result[
        "candidate_assessments"
    ]

    returned_ids = {
        assessment[
            "story_id"
        ]
        for assessment
        in assessments
    }

    expected_ids = set(
        canonical
    )

    if returned_ids != expected_ids:
        raise RuntimeError(
            "Brody did not return exactly one "
            "assessment for every candidate."
        )

    approved = []

    for assessment in assessments:
        story_id = assessment[
            "story_id"
        ]

        original_trade = canonical[
            story_id
        ][
            "gilman_trade"
        ]

        if (
            assessment["ticker"]
            != original_trade["ticker"]
        ):
            raise RuntimeError(
                f"Brody changed ticker for {story_id}."
            )

        if (
            assessment["direction"]
            != original_trade["direction"]
        ):
            raise RuntimeError(
                f"Brody reversed direction for {story_id}."
            )

        if (
            assessment["verdict"]
            == "APPROVE"
        ):
            approved.append(
                assessment
            )

    if len(approved) > 1:
        raise RuntimeError(
            "Brody approved more than one trade."
        )

    if (
        result["decision"]
        == "NO_TRADE"
    ):
        if (
            result[
                "selected_story_id"
            ]
            is not None
        ):
            raise RuntimeError(
                "NO_TRADE has selected_story_id."
            )

        if approved:
            raise RuntimeError(
                "NO_TRADE contains an APPROVE candidate."
            )

        result[
            "selected_trade"
        ] = None

        return result

    # TRADE
    if len(approved) != 1:
        raise RuntimeError(
            "TRADE must contain exactly one APPROVE."
        )

    selected_story_id = result[
        "selected_story_id"
    ]

    if (
        selected_story_id
        != approved[0]["story_id"]
    ):
        raise RuntimeError(
            "selected_story_id does not match "
            "the approved candidate."
        )

    selected_candidate = canonical[
        selected_story_id
    ]

    # IMPORTANT:
    # Copy the canonical Gilman object from our own input rather
    # than trusting the LLM to regenerate ticker/direction/thesis.
    result["selected_trade"] = (
        selected_candidate[
            "gilman_trade"
        ]
    )

    result["selected_market_context"] = {
        key: value
        for key, value
        in selected_candidate.items()
        if key != "gilman_trade"
    }

    return result


# ============================================================
# OUTPUT
# ============================================================

def write_output(
    source_payload: dict,
    result: dict,
) -> None:

    date_str = source_payload[
        "date"
    ]

    output = {
        "date": date_str,
        "generated_at": utc_now(),
        "model": BRODY_MODEL,
        "reasoning_effort": (
            BRODY_REASONING
        ),
        "candidate_count": len(
            source_payload.get(
                "candidates",
                [],
            )
        ),
        **result,
    }

    dated_path = (
        DATA_DIR
        / f"{date_str}.brody.json"
    )

    dated_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    shutil.copyfile(
        dated_path,
        BRODY_LATEST_PATH,
    )

    logger.info(
        "Wrote %s",
        dated_path,
    )

    logger.info(
        "Wrote %s",
        BRODY_LATEST_PATH,
    )

    if (
        output["decision"]
        == "TRADE"
    ):
        trade = output[
            "selected_trade"
        ]

        logger.info(
            "BRODY APPROVED: %s %s | confidence=%.2f",
            trade["direction"],
            trade["ticker"],
            output["confidence"],
        )

    else:
        logger.info(
            "BRODY: NO_TRADE"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info(
        "Starting Brody"
    )

    payload = load_input()

    logger.info(
        "Evaluating %d Gilman candidates",
        len(
            payload.get(
                "candidates",
                [],
            )
        ),
    )

    result = analyze(
        payload
    )

    result = validate_result(
        payload,
        result,
    )

    write_output(
        payload,
        result,
    )

    logger.info(
        "Brody complete"
    )


if __name__ == "__main__":
    main()