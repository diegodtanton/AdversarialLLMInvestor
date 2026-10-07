# SQLite Quickstart for the News Project

Your project currently uses two SQLite databases:

- `data/news.db` — persistent news/event memory
- `data/trading.db` — Brody/Alpaca execution and position history

SQLite databases are just files. You can inspect them directly with SQL using the `sqlite3` command-line tool.

## 1. Open a database

From the repo root:

```bash
cd /Users/diegotanton/Desktop/Developer/News

sqlite3 data/news.db
```

or:

```bash
sqlite3 data/trading.db
```

You will then see a prompt like:

```text
sqlite>
```

Exit with:

```sql
.quit
```

---

## 2. Useful SQLite commands

Inside the `sqlite>` prompt:

```sql
.tables
```

Shows all tables.

```sql
.schema events
```

Shows how a table is defined.

```sql
.headers on
.mode column
```

Makes query output easier to read.

```sql
.quit
```

Exits SQLite.

These commands beginning with `.` are SQLite shell commands, not SQL.

---

## 3. Explore `news.db`

Open it:

```bash
sqlite3 data/news.db
```

Then:

```sql
.headers on
.mode column

.tables
```

See recent events:

```sql
SELECT
    event_id,
    canonical_title,
    first_seen,
    last_seen
FROM events
ORDER BY last_seen DESC
LIMIT 20;
```

See recent stories:

```sql
SELECT
    story_date,
    story_rank,
    title,
    update_type,
    novelty,
    event_id
FROM stories
ORDER BY story_date DESC, story_rank ASC
LIMIT 30;
```

See the history of one event:

```sql
SELECT
    story_date,
    title,
    update_type,
    novelty,
    historical_note
FROM stories
WHERE event_id = 'evt_YOUR_EVENT_ID'
ORDER BY story_date;
```

Find stories mentioning a company or word:

```sql
SELECT
    story_date,
    title,
    event_id
FROM stories
WHERE title LIKE '%Boeing%'
ORDER BY story_date DESC;
```

Count how much history you have:

```sql
SELECT COUNT(*) AS stories FROM stories;
SELECT COUNT(*) AS events FROM events;
```

---

## 4. Explore `trading.db`

Open it:

```bash
sqlite3 data/trading.db
```

Then:

```sql
.headers on
.mode column

.tables
```

See all executions:

```sql
SELECT
    trade_date,
    ticker,
    direction,
    status,
    avg_entry_price,
    take_profit_price,
    exit_deadline
FROM executions
ORDER BY created_at DESC;
```

See only currently active positions:

```sql
SELECT
    ticker,
    direction,
    avg_entry_price,
    filled_at,
    take_profit_price,
    exit_deadline,
    status
FROM executions
WHERE status IN (
    'TAKE_PROFIT_OPEN',
    'POSITION_OPEN',
    'TIME_EXIT_SUBMITTING',
    'TIME_EXIT_SUBMITTED'
)
ORDER BY filled_at;
```

See completed trades:

```sql
SELECT
    ticker,
    direction,
    avg_entry_price,
    exit_price,
    realized_return_pct,
    exit_reason,
    filled_at,
    closed_at
FROM executions
WHERE status = 'CLOSED'
ORDER BY closed_at DESC;
```

Quick performance summary:

```sql
SELECT
    COUNT(*) AS trades,
    ROUND(AVG(realized_return_pct), 2) AS avg_return_pct,
    ROUND(MIN(realized_return_pct), 2) AS worst_return_pct,
    ROUND(MAX(realized_return_pct), 2) AS best_return_pct
FROM executions
WHERE status = 'CLOSED';
```

---

## 5. Run a query without entering the SQLite prompt

This is handy from Terminal:

```bash
sqlite3 -header -column data/trading.db "
SELECT ticker, direction, status, avg_entry_price, exit_deadline
FROM executions
ORDER BY created_at DESC;
"
```

Likewise:

```bash
sqlite3 -header -column data/news.db "
SELECT story_date, story_rank, title
FROM stories
ORDER BY story_date DESC, story_rank ASC
LIMIT 20;
"
```

---

## 6. Query SQLite from Python

Python includes SQLite support automatically:

```python
import sqlite3
from pathlib import Path

ROOT = Path("/Users/diegotanton/Desktop/Developer/News")
DB_PATH = ROOT / "data" / "trading.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    '''
    SELECT
        ticker,
        direction,
        status,
        avg_entry_price,
        realized_return_pct
    FROM executions
    ORDER BY created_at DESC
    '''
).fetchall()

for row in rows:
    print(dict(row))

conn.close()
```

Run a Python file normally:

```bash
python your_script.py
```

---

## 7. Be careful with writes

Most exploration should use:

```sql
SELECT ...
```

Commands such as these modify data:

```sql
INSERT ...
UPDATE ...
DELETE ...
DROP TABLE ...
```

Avoid editing `news.db` or `trading.db` manually unless you know exactly what you are changing, because your automated scripts rely on their state.

Before experimenting with writes, make a backup:

```bash
cp data/news.db data/news.backup.db
cp data/trading.db data/trading.backup.db
```

For everyday inspection, `SELECT` queries are all you need.
