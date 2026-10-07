# Trading pipeline upgrade

This patch was built against the nine Python scripts in the supplied `concat(1).txt` snapshot. It keeps trading paper-only. It does not include credentials or overwrite your `.env` with a redacted copy.

## Apply

Save `news-trading-upgrade.patch` in the repository root. Pause the scheduled pipeline while applying the code and migrating the database. Use your project's existing Python environment.

```bash
 git apply --check news-trading-upgrade.patch
 git apply news-trading-upgrade.patch
 python -m unittest discover -s tests -p test_trading_upgrade.py -v
 python -m scripts.upgrade_trading
```

The last command is local-only: it backs up `data/trading.db` using SQLite's backup API (including committed WAL data), saves a private `.env` backup, migrates tables, and sets:

```dotenv
MAX_HOLD_DAYS=180
GILMAN_MODEL=gpt-6-astra
BRODY_MODEL=gpt-6-astra
EVENT_MODEL=gpt-6-sol
```

Existing reasoning settings and unrelated configuration remain in place. Shell/launchd environment variables take precedence over `.env`; remove any conflicting model or holding-period overrides there. Dependencies were omitted from the source dump; this patch adds no new third-party package requirement beyond the original project's dependencies. Python 3.10+ on macOS/Linux is required; the process locks use `fcntl`.

To also extend existing open positions to at least 180 days from their recorded fill, use this command **instead of** the last command above:

```bash
python -m scripts.upgrade_trading --extend-open-positions
```

Positions with an existing time-exit attempt are skipped, because extending a deadline cannot undo an order already sent to the broker. Existing longer deadlines are never shortened. Without the flag, existing positions keep their persisted deadlines; newly opened positions use 180 days.

Backups are saved beneath `data/backups/` in a timestamped private directory. The migration performs no brokerage requests and sends no email.

## Populate and inspect trade history

This command reads broker records and updates local records, but never submits or cancels an order:

```bash
python -m scripts.position_manager --sync-only --full-history
python -m scripts.trade_history
python -m scripts.trade_history --ticker CSTE
python -m scripts.trade_history --view positions
python -m scripts.trade_history --view errors
python -m scripts.trade_history --view runs
python -m scripts.trade_history --view decisions --format json
python -m scripts.trade_history --since 2026-10-01 --format csv > trade-fills.csv
```

The default view lists broker-confirmed fills, including their transaction time, ticker, quantity, price, role, and order ID. Table times use America/Chicago; JSON/CSV preserve UTC. `--since` means midnight in the selected local timezone. Use `--timezone` to change it.

Fill ingestion is paginated and idempotent. The initial/full sync fetches all history available from Alpaca. Routine sync overlaps the last seven days to catch delayed records. A full sync can be rerun to catch older corrections.

Legacy or external fills without a known strategy order are labeled `unclassified`; they are not silently assigned to a new thesis. Historical closed rows retain their original data. Their previously calculated returns are hidden in the lookup until recomputed from known fills (`accounting_version=1`). New/reconciled closures use exact decimal fill arithmetic and show **gross** P&L before fees, borrow charges, dividends, or other adjustments.

An unresolved saved order ID, missing fill history, quantity mismatch, or missing take-profit remains visible as an error. Resolve the actual broker state before retrying execution; do not delete the row to force another trade. The application does not invent a replacement order after an ambiguous submission.

## Daily operation

The normal entrypoint remains:

```bash
python -m scripts.daily_pipeline
```

It now performs position maintenance first, runs research/trading where appropriate, reconciles again, and then sends the email. The first substantive block contains confirmed trade activity, pending/unresolved states, and the current run's Brody result. It includes exits as well as entries, and says “unconfirmed” when reconciliation fails. The reporting window is today through the displayed broker cutoff, not a prediction about the rest of the day.

A research, trends, or email failure no longer prevents the initial position-maintenance pass. Failures are persisted in `runs` and `stage_runs`; a failure report is attempted even if research fails. One problematic position does not prevent other positions from being processed. Broker-side writes are serialized across the executor and position manager.

`--no_news`/`--no-news` reuses enriched news up to 36 hours old and still reports trading outcomes. `--research-only` performs research and sends the report while prohibiting broker order submissions/cancellations; decisions generated in this mode cannot subsequently be passed to the executor. It is not a no-network/no-cost test mode. The unittest command above is the offline test mode.

For reliable deadline management, run this existing entrypoint separately on your desired maintenance schedule, in addition to the daily research job:

```bash
python -m scripts.position_manager
```

This command **can submit or cancel paper orders**. Your launchd utility/scheduler was excluded from the attachment, so the patch does not change its schedule. Market exits wait until the broker reports the market open. A daily/laptop-only schedule can execute a deadline later than its exact timestamp.

## Execution changes

- Shared 180-day policy, persisted per new position and included in Gilman/Brody input and decisions.
- One fixed marketable DAY limit per entry. Automatic price chasing/repricing has been removed; `ENTRY_REPRICE_MINUTES` is now unused. `MAX_ENTRY_WAIT_MINUTES`, `ENTRY_POLL_SECONDS`, `ENTRY_CUSHION_BPS`, `MAX_SPREAD_BPS`, and `TRADE_NOTIONAL_USD` still apply.
- Order intents and client IDs are saved before sending. Recovery uses the persisted ID across days, follows legacy replacement chains, and records broker order observations.
- Cancellation must be confirmed terminal before declaring no fill or sizing a take-profit for the final entry quantity.
- Partial entry fills are persisted during polling. Fills/deadline are persisted before take-profit submission.
- Partial take-profits and later exit attempts contribute to the same fill-based closure calculation, including after restart.
- Broker positions must match the quantity attributable to known strategy fills. Manual activity triggers reconciliation rather than an invented closure or an oversized exit.
- Fresh decision, matching run/policy, fresh quote, account, asset/borrow eligibility, spread, existing-position and open-order checks precede new entries. STRA remains prohibited.
- New execution identity uses persistent event (or story), ticker, and direction. A recorded attempt for the same identity is not automatically repeated, even after no fill or closure; a genuinely new event can produce a new candidate.
- Current execution rejects decisions older than 60 minutes by default (`DECISION_MAX_AGE_MINUTES`) and quotes older than 60 seconds (`MAX_QUOTE_AGE_SECONDS`).
- A database is bound to one paper account after its first broker connection; switching credentials to another account requires a separate database.

The small broker adapter uses Alpaca's paper REST API directly for orders, quotes and fill activity, avoiding dependence on changing SDK order models. Read/write calls have timeouts; order writes are never blindly retried. Existing other modules may still require `alpaca-py` until separately cleaned up.

## Research changes

Brody reads the Gilman handoff directly. The old `market_context.py` remains available for legacy/manual use but is no longer a daily pipeline stage. Brody has five read-only tools:

- `get_price_trend`
- `get_event_price_reaction`
- `get_fundamentals`
- `get_corporate_calendar`
- `get_execution_context`

The application validates tool arguments, limits tool rounds/calls, caches repeated requests, saves results to SQLite, and preserves the Responses continuation items/call IDs. Approval requires successful trend, fundamentals and broker-context calls; if the candidate has a verified event date, its specific event window must also have been retrieved. Missing data is reported as uncertainty, not fabricated numerical evidence.

Daily metrics conservatively exclude the current New York calendar date, even after close. Completed-session volume is compared with completed sessions; stock and SPY returns use aligned dates. This intentionally does not provide intraday relative volume or a causal “priced in” verdict. Intraday quote context comes from Alpaca. `ALPACA_DATA_FEED` can select an entitled quote feed; otherwise Alpaca's default is used.

Gilman now passes persistent event context and verified announcement timing to Brody. Unverified timing remains null/unknown. News source URLs/timestamps are retained in the public handoff. Exposure evidence must itself match the web-source audit. Meaningful URL query parameters are preserved during evidence normalization.

The model switches preserve existing reasoning efforts. Brody's `six_month_fit` field becomes `holding_period_fit`. `candidate_count` is added to Gilman's handoff while retaining `trade_count` for compatibility. Existing historical JSON is not rewritten.

Brody and Gilman decisions, including no-trade outputs, are stored in SQLite. Latest/date JSON exports remain for compatibility and are written atomically. This patch does not redesign the collector's clustering, Event Memory's within-batch matching, or every lower-priority item from the inspection.

## Email delivery

SMTP report claims are persisted before submission. A lost response after sending becomes `UNKNOWN`; the same report key is not automatically resent. Rendered `.eml` files are saved in `data/reports/` for diagnosis. An SMTP Message-ID supports tracing but does not guarantee exactly-once delivery. A new pipeline run intentionally produces a new report; rerunning the pipeline is not an email-only retry.

If the trends source is missing/stale or its model fails, a trade/news report is still sent with trends marked unavailable. The original Gmail recipient configuration is preserved.

## Validation and limits

36 offline regression tests cover 180-day deadlines, ambiguous entry/TP/exit submissions, next-day recovery, cancellation confirmation, partial entries/exits, legacy replacements, stale decisions/quotes, broker account isolation, fill pagination/deduplication, read-only closure reconciliation, incomplete-sync reporting, tool continuation/evidence gates, event timing, volume handling, config preservation, and ambiguous email delivery. No real broker, email, or model calls were made during patch development.

Patch application and Python syntax are checked against a fresh reconstruction of the supplied snapshot. Actual account access, installed OpenAI/yfinance versions, provider responses, scheduler behavior, and historical database contents require verification in your environment. Start with the offline tests and read-only sync above before resuming the scheduled paper pipeline.

Useful API references:

- [OpenAI Responses reasoning continuation](https://developers.openai.com/api/docs/guides/reasoning)
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Alpaca fill activities](https://docs.alpaca.markets/us/docs/account-activities)
- [Alpaca activity pagination](https://docs.alpaca.markets/us/reference/getaccountactivities-2)
- [Alpaca order replacement semantics](https://docs.alpaca.markets/us/reference/patchorderbyorderid-1)
- [yfinance Ticker reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html)
