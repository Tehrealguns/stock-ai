"""
Database layer - SQLite persistence for portfolio, trades, and thoughts.
"""
import aiosqlite
import json
from datetime import datetime, timedelta
from pathlib import Path

import os
# Use /data for persistent volume on Railway, fallback to local dir
_data_dir = os.getenv("DATA_DIR", str(Path(__file__).parent))
DB_PATH = Path(_data_dir) / "stockmind.db"


async def get_db():
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def reset_db():
    """Delete and recreate the database from scratch."""
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"🗑️ Deleted old database at {DB_PATH}")
    await init_db()


async def init_db():
    """Initialize database tables."""
    # Check if we should force a fresh start
    if os.getenv("FRESH_START", "").lower() in ("true", "1", "yes"):
        if DB_PATH.exists():
            DB_PATH.unlink()
            print(f"🗑️ FRESH_START: Deleted old database at {DB_PATH}")

    db = await get_db()
    try:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                shares REAL NOT NULL DEFAULT 0,
                avg_cost REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                total REAL NOT NULL,
                reasoning TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS thoughts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL DEFAULT 'thinking',
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                added_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'lesson',
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                invested REAL NOT NULL,
                pnl REAL NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        await db.commit()

        # Set default config if not exists
        cursor = await db.execute("SELECT value FROM config WHERE key = 'cash_balance'")
        row = await cursor.fetchone()
        if not row:
            starting = os.getenv("STARTING_BALANCE", "100000")
            await db.execute(
                "INSERT INTO config (key, value) VALUES ('cash_balance', ?)",
                (starting,)
            )
            # Default watchlist — diverse across sectors and market caps
            default_symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",  # tech
                "JPM", "V", "GS",                                          # finance
                "JNJ", "UNH", "PFE",                                       # healthcare
                "XOM", "NEE",                                               # energy
                "CAT", "DE",                                                # industrials
                "COST", "NKE", "SBUX",                                      # consumer
                "AMD", "PLTR", "SOFI", "RKLB", "HOOD", "COIN", "SQ",       # growth / mid-cap
                "RIVN", "MARA", "SMCI",                                     # speculative / volatile
            ]
            for sym in default_symbols:
                await db.execute(
                    "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?, ?)",
                    (sym, datetime.now().isoformat())
                )
            await db.commit()
    finally:
        await db.close()


async def get_cash_balance() -> float:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT value FROM config WHERE key = 'cash_balance'")
        row = await cursor.fetchone()
        return float(row[0]) if row else 0.0
    finally:
        await db.close()


async def set_cash_balance(amount: float):
    db = await get_db()
    try:
        await db.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('cash_balance', ?)",
            (str(amount),)
        )
        await db.commit()
    finally:
        await db.close()


async def get_portfolio() -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM portfolio WHERE shares > 0 ORDER BY symbol")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def update_holding(symbol: str, shares: float, avg_cost: float):
    db = await get_db()
    try:
        if shares <= 0:
            await db.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
        else:
            await db.execute("""
                INSERT INTO portfolio (symbol, shares, avg_cost, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    shares = ?, avg_cost = ?, updated_at = ?
            """, (symbol, shares, avg_cost, datetime.now().isoformat(),
                  shares, avg_cost, datetime.now().isoformat()))
        await db.commit()
    finally:
        await db.close()


async def get_holding(symbol: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM portfolio WHERE symbol = ?", (symbol,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def add_trade(symbol: str, action: str, shares: float, price: float, total: float, reasoning: str = ""):
    db = await get_db()
    try:
        await db.execute("""
            INSERT INTO trades (symbol, action, shares, price, total, reasoning, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, action, shares, price, total, reasoning, datetime.now().isoformat()))
        await db.commit()
    finally:
        await db.close()


async def get_trades(limit: int = 50) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def add_thought(thought_type: str, content: str, metadata: dict | None = None):
    db = await get_db()
    try:
        await db.execute("""
            INSERT INTO thoughts (type, content, metadata, created_at)
            VALUES (?, ?, ?, ?)
        """, (thought_type, content, json.dumps(metadata) if metadata else None,
              datetime.now().isoformat()))
        await db.commit()
    finally:
        await db.close()


async def get_thoughts(limit: int = 100, after_id: int = 0) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM thoughts WHERE id > ? ORDER BY created_at DESC LIMIT ?",
            (after_id, limit)
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("metadata"):
                d["metadata"] = json.loads(d["metadata"])
            results.append(d)
        return results
    finally:
        await db.close()


async def get_watchlist() -> list[str]:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT symbol FROM watchlist ORDER BY symbol")
        rows = await cursor.fetchall()
        return [row[0] for row in rows]
    finally:
        await db.close()


async def add_to_watchlist(symbol: str):
    db = await get_db()
    try:
        await db.execute(
            "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?, ?)",
            (symbol, datetime.now().isoformat())
        )
        await db.commit()
    finally:
        await db.close()


async def get_total_trades_count() -> int:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) FROM trades")
        row = await cursor.fetchone()
        return row[0]
    finally:
        await db.close()


async def get_winning_trades_count() -> int:
    """Count trades where sell price * shares > buy cost (simplified)."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM trades WHERE action = 'sell'"
        )
        row = await cursor.fetchone()
        return row[0]  # simplified - we track this differently
    finally:
        await db.close()


# ─── Memory / Journal ────────────────────────────────────────

async def add_memory(category: str, content: str):
    """Store a persistent memory/lesson the AI learned."""
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO memory (category, content, created_at) VALUES (?, ?, ?)",
            (category, content, datetime.now().isoformat())
        )
        await db.commit()
    finally:
        await db.close()


async def get_memories(limit: int = 20) -> list[dict]:
    """Get the AI's stored memories/lessons."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM memory ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def save_portfolio_snapshot(total_value: float, cash: float, invested: float, pnl: float):
    """Save a point-in-time snapshot of portfolio value for charting."""
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO portfolio_snapshots (total_value, cash, invested, pnl, created_at) VALUES (?, ?, ?, ?, ?)",
            (round(total_value, 2), round(cash, 2), round(invested, 2), round(pnl, 2), datetime.now().isoformat())
        )
        await db.commit()
    finally:
        await db.close()


async def get_portfolio_snapshots(days: int = 30) -> list[dict]:
    """Get portfolio snapshots for the last N days."""
    db = await get_db()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = await db.execute(
            "SELECT * FROM portfolio_snapshots WHERE created_at >= ? ORDER BY created_at ASC",
            (cutoff,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def get_risk_profile() -> str:
    """Get the current risk profile (safe, moderate, aggressive)."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT value FROM config WHERE key = 'risk_profile'")
        row = await cursor.fetchone()
        return row[0] if row else "moderate"
    finally:
        await db.close()


async def set_risk_profile(profile: str):
    """Set the risk profile."""
    if profile not in ("safe", "moderate", "aggressive"):
        raise ValueError(f"Invalid risk profile: {profile}")
    db = await get_db()
    try:
        await db.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('risk_profile', ?)",
            (profile,)
        )
        await db.commit()
    finally:
        await db.close()


async def get_portfolio_history_summary() -> dict:
    """Get a summary of all-time trading performance for the AI's memory."""
    db = await get_db()
    try:
        # Total trades
        cursor = await db.execute("SELECT COUNT(*) FROM trades")
        total_trades = (await cursor.fetchone())[0]

        # Buys and sells
        cursor = await db.execute("SELECT COUNT(*) FROM trades WHERE action = 'buy'")
        total_buys = (await cursor.fetchone())[0]
        cursor = await db.execute("SELECT COUNT(*) FROM trades WHERE action = 'sell'")
        total_sells = (await cursor.fetchone())[0]

        # All unique symbols ever traded
        cursor = await db.execute("SELECT DISTINCT symbol FROM trades ORDER BY symbol")
        symbols_traded = [row[0] for row in await cursor.fetchall()]

        # First trade date
        cursor = await db.execute("SELECT MIN(created_at) FROM trades")
        first_trade = (await cursor.fetchone())[0]

        # Total money spent buying / received selling
        cursor = await db.execute("SELECT COALESCE(SUM(total), 0) FROM trades WHERE action = 'buy'")
        total_bought = (await cursor.fetchone())[0]
        cursor = await db.execute("SELECT COALESCE(SUM(total), 0) FROM trades WHERE action = 'sell'")
        total_sold = (await cursor.fetchone())[0]

        return {
            "total_trades": total_trades,
            "total_buys": total_buys,
            "total_sells": total_sells,
            "symbols_traded": symbols_traded,
            "first_trade_date": first_trade,
            "total_bought": round(total_bought, 2),
            "total_sold": round(total_sold, 2),
        }
    finally:
        await db.close()

