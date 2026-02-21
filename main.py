"""
StockMind - AI Stock Trading Agent
FastAPI backend serving the web app and AI agent.
"""
import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from database import (
    init_db, reset_db, get_thoughts, get_trades, get_watchlist,
    add_to_watchlist, get_cash_balance, get_portfolio_snapshots,
    get_risk_profile, set_risk_profile
)
from trading import get_portfolio_summary
from market_data import fetch_quotes, fetch_market_overview, is_market_hours
from agent import start_agent_loop, stop_agent, trigger_cycle

import json

# Background task reference
_agent_task = None


async def _delayed_agent_start():
    """Wait a bit after server starts before running the agent, so healthcheck passes."""
    await asyncio.sleep(10)  # Let the server be healthy first
    print("🧠 StockMind agent starting first cycle...")
    await start_agent_loop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    global _agent_task

    # Init database
    await init_db()
    print("✅ Database initialized")

    # Start agent loop in background AFTER a short delay
    _agent_task = asyncio.create_task(_delayed_agent_start())
    print(f"🧠 StockMind server ready — agent will start in ~10s")

    yield

    # Shutdown
    stop_agent()
    if _agent_task:
        _agent_task.cancel()
    print("🛑 StockMind agent stopped")


app = FastAPI(title="StockMind", lifespan=lifespan)

# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ─── API Routes ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web app."""
    index_path = static_dir / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/portfolio")
async def api_portfolio():
    """Get current portfolio summary with live prices."""
    try:
        summary = await get_portfolio_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/thoughts")
async def api_thoughts(after_id: int = 0, limit: int = 100):
    """Get recent thoughts from the AI."""
    try:
        thoughts = await get_thoughts(limit=limit, after_id=after_id)
        return JSONResponse(content={"thoughts": thoughts})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/trades")
async def api_trades(limit: int = 50):
    """Get trade history."""
    try:
        trades = await get_trades(limit=limit)
        return JSONResponse(content={"trades": trades})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/market")
async def api_market():
    """Get market overview."""
    try:
        overview = await fetch_market_overview()
        market_open = is_market_hours()
        return JSONResponse(content={"overview": overview, "market_open": market_open})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/watchlist")
async def api_watchlist():
    """Get watchlist with current prices."""
    try:
        symbols = await get_watchlist()
        quotes = await fetch_quotes(symbols) if symbols else {}
        return JSONResponse(content={"watchlist": symbols, "quotes": quotes})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/trigger")
async def api_trigger():
    """Manually trigger a thinking cycle."""
    try:
        asyncio.create_task(trigger_cycle())
        return JSONResponse(content={"status": "triggered"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/reset")
async def api_reset():
    """Reset the database — fresh start with $100k."""
    try:
        stop_agent()
        await reset_db()
        return JSONResponse(content={"status": "reset", "message": "Database wiped. Restart the service to begin fresh."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/watchlist/{symbol}")
async def api_add_watchlist(symbol: str):
    """Add a symbol to the watchlist."""
    try:
        await add_to_watchlist(symbol.upper())
        return JSONResponse(content={"status": "added", "symbol": symbol.upper()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/portfolio/history")
async def api_portfolio_history(days: int = 30):
    """Get portfolio value history for charts."""
    try:
        snapshots = await get_portfolio_snapshots(days=days)
        return JSONResponse(content={"snapshots": snapshots})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/settings")
async def api_get_settings():
    """Get current settings including risk profile and Twitter status."""
    try:
        profile = await get_risk_profile()
        from trading import RISK_PROFILES
        from notifications import is_enabled as twitter_is_enabled
        limits = RISK_PROFILES.get(profile, RISK_PROFILES["moderate"])
        return JSONResponse(content={
            "risk_profile": profile,
            "limits": limits,
            "twitter_enabled": twitter_is_enabled(),
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/settings")
async def api_update_settings(request: Request):
    """Update settings (risk profile)."""
    try:
        body = await request.json()
        profile = body.get("risk_profile")
        if profile:
            await set_risk_profile(profile)
        current = await get_risk_profile()
        from trading import RISK_PROFILES
        limits = RISK_PROFILES.get(current, RISK_PROFILES["moderate"])
        return JSONResponse(content={
            "status": "updated",
            "risk_profile": current,
            "limits": limits,
        })
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/stream")
async def api_stream(request: Request):
    """SSE stream for real-time thought updates."""
    async def event_generator():
        last_id = 0
        while True:
            if await request.is_disconnected():
                break
            thoughts = await get_thoughts(limit=10, after_id=last_id)
            if thoughts:
                # thoughts come back in DESC order, reverse for chronological
                for thought in reversed(thoughts):
                    if thought["id"] > last_id:
                        last_id = thought["id"]
                        yield {
                            "event": "thought",
                            "data": json.dumps(thought),
                            "id": str(thought["id"]),
                        }
            await asyncio.sleep(2)  # Poll every 2 seconds

    return EventSourceResponse(event_generator())


@app.get("/api/status")
async def api_status():
    """Get agent status."""
    from agent import _running, _last_cycle_time, _next_check_time, _current_session, SESSIONS
    cash = await get_cash_balance()

    next_session_name = None
    next_check_iso = None
    if _next_check_time:
        next_check_iso = _next_check_time.isoformat()
        # Figure out which session is next
        for sid, s in SESSIONS.items():
            h = _next_check_time.hour
            if s["hours"][0] <= h < s["hours"][1]:
                next_session_name = s["name"]
                break

    return JSONResponse(content={
        "running": _running,
        "last_cycle": _last_cycle_time.isoformat() if _last_cycle_time else None,
        "current_session": SESSIONS.get(_current_session, {}).get("name") if _current_session else None,
        "next_check": next_check_iso,
        "next_session": next_session_name,
        "cash": cash,
        "market_open": is_market_hours(),
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8888"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

