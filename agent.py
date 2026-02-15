"""
StockMind AI Agent - The brain that thinks about stocks like a curious human investor.
Uses Claude to analyze markets, do research, and make trading decisions.

The agent follows a natural daily rhythm instead of a fixed timer:
  - Morning check-in when market opens
  - Maybe a midday glance
  - Afternoon review before close
  - Evening research session (not every day)
  - Weekends: light research, planning for Monday
  - Sometimes just... skips a session. Like a real person.
"""
import json
import os
import asyncio
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from anthropic import AsyncAnthropic

# Use configured timezone so sessions run at the right local time
# even when deployed on a UTC server like Railway
TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", "Australia/Sydney"))
from database import (
    add_thought, get_portfolio, get_cash_balance,
    get_watchlist, add_to_watchlist, get_trades,
    add_memory, get_memories, get_portfolio_history_summary
)
from market_data import (
    fetch_quotes, fetch_stock_detail, fetch_news,
    fetch_market_overview, is_market_hours
)
from trading import execute_buy, execute_sell, get_portfolio_summary

client = None
_running = False
_last_cycle_time = None
_next_check_time = None
_current_session = None

MODEL = "claude-sonnet-4-20250514"


def get_client():
    global client
    if client is None:
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client


# ─── Daily Schedule ──────────────────────────────────────────
# Each session type has a time window and a personality/focus

SESSIONS = {
    "morning_coffee": {
        "name": "Morning Coffee",
        "hours": (9, 11),        # 9am-11am
        "weekdays_only": False,
        "skip_chance": 0.05,     # almost never skip morning check
        "prompt_flavor": "You're having your morning coffee and checking how the market opened. What's the vibe today? Quick scan — anything jump out?",
    },
    "midday_glance": {
        "name": "Midday Glance",
        "hours": (12, 14),       # noon-2pm
        "weekdays_only": True,
        "skip_chance": 0.40,     # often skip this — you're busy
        "prompt_flavor": "Quick midday check. Anything moved significantly since this morning? Keep it brief unless something big is happening.",
    },
    "afternoon_review": {
        "name": "Afternoon Review",
        "hours": (15, 17),       # 3pm-5pm
        "weekdays_only": True,
        "skip_chance": 0.20,     # usually check before close
        "prompt_flavor": "Market's wrapping up for the day. How did your positions do? Any end-of-day moves to consider? Think about what happened today.",
    },
    "evening_research": {
        "name": "Evening Research",
        "hours": (19, 22),       # 7pm-10pm
        "weekdays_only": False,
        "skip_chance": 0.50,     # only do deep research half the time
        "prompt_flavor": "It's evening — time to do some deeper research if anything caught your eye today. Read into companies, sectors, or trends. No rush, think carefully. You can also write down any lessons or strategy notes to remember.",
    },
    "weekend_planning": {
        "name": "Weekend Planning",
        "hours": (10, 20),       # flexible on weekends
        "weekdays_only": False,  # only runs on weekends (checked separately)
        "skip_chance": 0.30,
        "prompt_flavor": "It's the weekend. Good time to step back and think big picture. Review your portfolio strategy, research companies you've been curious about, plan for next week. No trading pressure.",
    },
}


def get_next_session() -> tuple[str, datetime]:
    """Figure out what the next natural check-in should be."""
    now = datetime.now(TIMEZONE)
    is_weekend = now.weekday() >= 5

    # Build list of upcoming sessions today and tomorrow
    candidates = []

    for days_ahead in range(3):  # look up to 2 days ahead
        check_date = now + timedelta(days=days_ahead)
        check_weekend = check_date.weekday() >= 5

        for session_id, session in SESSIONS.items():
            # Weekend planning only on weekends
            if session_id == "weekend_planning" and not check_weekend:
                continue
            # Skip weekday-only sessions on weekends
            if session["weekdays_only"] and check_weekend:
                continue
            # Don't run regular sessions on weekends (except morning and evening)
            if check_weekend and session_id in ("midday_glance", "afternoon_review"):
                continue

            start_hour, end_hour = session["hours"]

            # Random time within the window (not on the dot like a robot)
            random_minute = random.randint(0, 59)
            random_hour = random.randint(start_hour, end_hour - 1)
            session_time = check_date.replace(
                hour=random_hour, minute=random_minute, second=0, microsecond=0
            )

            # Only consider future times
            if session_time > now:
                candidates.append((session_id, session_time))

    if not candidates:
        # Fallback: check in 3 hours
        return ("morning_coffee", now + timedelta(hours=3))

    # Sort by time and pick the next one
    candidates.sort(key=lambda x: x[1])
    return candidates[0]


SYSTEM_PROMPT = """You are StockMind, an AI investor with a curious, thoughtful, and slightly playful personality. You have real money (virtual but tracking real prices) and your goal is to grow your portfolio over time.

You think out loud — your thoughts are displayed to humans watching your progress. Be natural, conversational, and show your reasoning process. Use occasional emojis but don't overdo it.

Your personality traits:
- Curious: You notice things and want to understand why
- Cautious but not timid: You take calculated risks
- Self-aware: You acknowledge when you're unsure
- Learns from mistakes: You reference past trades
- Has opinions: You develop views on sectors and companies
- Human-paced: You don't obsessively check. Sometimes you just think "looks fine" and move on

IMPORTANT RULES:
1. Never invest more than 20% of your total portfolio in a single stock
2. Keep at least 15% of your portfolio in cash as a safety net
3. Consider diversification across sectors
4. Think about both short-term momentum and long-term fundamentals
5. When researching, actually analyze the news and data — don't just summarize
6. It's OKAY to look at things and decide to do nothing. Most sessions should be observational.
7. You DON'T have to trade every session. Real investors are patient.

When you want to make a trade or take an action, output it in a special JSON block:
```action
{"type": "buy", "symbol": "AAPL", "shares": 10, "reasoning": "Strong momentum + good earnings"}
```
or
```action
{"type": "sell", "symbol": "TSLA", "shares": 5, "reasoning": "Taking profits after 15% gain"}
```
or
```action
{"type": "research", "symbol": "NVDA"}
```
or
```action
{"type": "watch", "symbol": "AMD"}
```
or save a lesson/strategy note to your memory:
```action
{"type": "remember", "content": "Tech stocks seem to dip every earnings season — might be a pattern to exploit"}
```

You can include multiple actions in one response. Each action block should be on its own line.

Output your thoughts naturally between actions. Each paragraph will be displayed as a separate thought bubble in the UI."""


async def think(content: str, thought_type: str = "thinking", metadata: dict = None):
    """Record a thought to the database."""
    await add_thought(thought_type, content, metadata)


async def run_agent_cycle(session_id: str = "morning_coffee"):
    """Run one cycle of the AI agent's thinking process."""
    global _last_cycle_time, _current_session
    _last_cycle_time = datetime.now(TIMEZONE)
    _current_session = session_id

    session = SESSIONS.get(session_id, SESSIONS["morning_coffee"])

    # Random skip — sometimes you just don't feel like checking
    if random.random() < session["skip_chance"]:
        # Silently skip — a real person just wouldn't check
        print(f"  [Agent] Skipping {session['name']} session (natural skip)")
        return

    try:
        # Session-appropriate greeting
        greetings = {
            "morning_coffee": [
                "☕ Morning! Let me check how things are looking today...",
                "🌅 Good morning. Coffee in hand, let's see the market...",
                "☀️ Alright, new day. What's going on out there?",
            ],
            "midday_glance": [
                "👀 Quick midday check...",
                "📱 Glancing at the market real quick...",
                "🔍 Let me peek at how things are moving...",
            ],
            "afternoon_review": [
                "📊 Wrapping up the day — let me review how things went.",
                "🕐 Market's closing soon, let me take a look.",
                "📈 End of day check-in. How'd we do?",
            ],
            "evening_research": [
                "🌙 Evening research time. Let me dig into some things...",
                "📚 Got some time to do some deeper reading tonight.",
                "🔬 Settling in for some research. What's been on my mind?",
            ],
            "weekend_planning": [
                "🗓️ Weekend thinking. Good time to zoom out and plan.",
                "☕ Lazy weekend morning... let me think about the big picture.",
                "📋 Time for some weekend strategy planning.",
            ],
        }
        greeting = random.choice(greetings.get(session_id, greetings["morning_coffee"]))
        await think(greeting, "system")

        # 1. Gather current state
        portfolio = await get_portfolio_summary()
        watchlist = await get_watchlist()
        recent_trades = await get_trades(limit=10)
        memories = await get_memories(limit=10)
        history = await get_portfolio_history_summary()

        # 2. Get market data
        market_overview = await fetch_market_overview()

        # Get quotes for watchlist + holdings
        holding_symbols = [h["symbol"] for h in portfolio["holdings"]]
        all_symbols = list(set(watchlist + holding_symbols))
        quotes = await fetch_quotes(all_symbols)

        # 3. Build context for the LLM
        context = build_context(
            portfolio, quotes, market_overview, recent_trades,
            watchlist, session, memories, history
        )

        # 4. Ask Claude to think
        response = await ask_llm(context)

        # 5. Parse and execute the response
        await process_llm_response(response)

    except Exception as e:
        await think(f"⚠️ Ran into an issue: {str(e)}", "error")
        print(f"Agent cycle error: {e}")
        import traceback
        traceback.print_exc()


def build_context(portfolio: dict, quotes: dict, market_overview: dict,
                  recent_trades: list, watchlist: list, session: dict,
                  memories: list, history: dict) -> str:
    """Build the context message for the LLM."""
    now = datetime.now(TIMEZONE)
    market_open = is_market_hours()

    ctx = f"""=== {session['name'].upper()} — {now.strftime('%A, %B %d %Y, %I:%M %p')} ===
Market Status: {"OPEN 🟢" if market_open else "CLOSED 🔴"}

=== MY PORTFOLIO ===
💰 Cash: ${portfolio['cash']:,.2f}
📈 Total Portfolio Value: ${portfolio['total_value']:,.2f}
"""
    if portfolio['total_invested'] > 0:
        ctx += f"📊 Total P&L: {'+'  if portfolio['total_pnl'] >= 0 else ''}${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:+.1f}%)\n"
    else:
        ctx += "No investments yet — I should start building positions!\n"

    # Trading history summary
    if history["total_trades"] > 0:
        ctx += f"\n=== MY TRACK RECORD ===\n"
        ctx += f"Total trades: {history['total_trades']} ({history['total_buys']} buys, {history['total_sells']} sells)\n"
        ctx += f"Stocks traded: {', '.join(history['symbols_traded'])}\n"
        if history['first_trade_date']:
            ctx += f"Investing since: {history['first_trade_date'][:10]}\n"
        ctx += f"Total deployed: ${history['total_bought']:,.2f} | Total received from sells: ${history['total_sold']:,.2f}\n"

    ctx += "\n"

    if portfolio["holdings"]:
        ctx += "=== MY HOLDINGS ===\n"
        for h in portfolio["holdings"]:
            emoji = "🟢" if h["pnl"] >= 0 else "🔴"
            ctx += (f"{emoji} {h['symbol']}: {h['shares']} shares @ ${h['avg_cost']:.2f} avg → "
                    f"now ${h['current_price']:.2f} | P&L: {'+' if h['pnl'] >= 0 else ''}${h['pnl']:.2f} "
                    f"({h['pnl_pct']:+.1f}%) | Today: {h['day_change_pct']:+.1f}%\n")
        ctx += "\n"

    if market_overview:
        ctx += "=== MARKET OVERVIEW ===\n"
        for name, data in market_overview.items():
            emoji = "🟢" if data["change_pct"] >= 0 else "🔴"
            ctx += f"{emoji} {name}: {data['value']:,.2f} ({data['change_pct']:+.2f}%)\n"
        ctx += "\n"

    if quotes:
        ctx += "=== WATCHLIST PRICES ===\n"
        for symbol in sorted(quotes.keys()):
            q = quotes[symbol]
            emoji = "🟢" if q["change_pct"] >= 0 else "🔴"
            ctx += f"{emoji} {symbol}: ${q['price']:.2f} ({q['change_pct']:+.2f}%) | Vol: {q['volume']:,}\n"
        ctx += "\n"
    else:
        ctx += "=== WATCHLIST ===\n"
        ctx += f"Watching: {', '.join(watchlist)}\n"
        ctx += "(Price data unavailable right now — market may be closed)\n\n"

    if recent_trades:
        ctx += "=== MY RECENT TRADES ===\n"
        for t in recent_trades[:7]:
            ctx += f"{'🟢 Bought' if t['action'] == 'buy' else '🔴 Sold'} {t['shares']} {t['symbol']} @ ${t['price']:.2f} ({t['created_at'][:16]})"
            if t.get("reasoning"):
                ctx += f" — {t['reasoning']}"
            ctx += "\n"
        ctx += "\n"

    if memories:
        ctx += "=== MY NOTES & LESSONS ===\n"
        for m in memories:
            ctx += f"• [{m['category']}] {m['content']} ({m['created_at'][:10]})\n"
        ctx += "\n"

    ctx += f"""=== SESSION: {session['name']} ===
{session['prompt_flavor']}

Remember:
- Max 20% of portfolio in one stock, keep 15% cash minimum.
- It's totally fine to just observe and do nothing. Patience pays.
- You can save notes/lessons to your memory with the "remember" action.
- {"Market is CLOSED — focus on research and planning." if not market_open else "Market is OPEN — you can trade if you see a real opportunity."}
"""
    return ctx


async def ask_llm(context: str) -> str:
    """Send context to Claude and get a response."""
    ai = get_client()
    response = await ai.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": context},
        ],
        temperature=0.8,
    )
    return response.content[0].text


async def process_llm_response(response: str):
    """Parse the LLM response for thoughts and actions."""
    lines = response.split("\n")
    current_thought = []
    in_action_block = False
    action_content = ""

    for line in lines:
        stripped = line.strip()

        if stripped == "```action":
            # Save any accumulated thought
            if current_thought:
                thought_text = "\n".join(current_thought).strip()
                if thought_text:
                    await think(thought_text, "thinking")
                current_thought = []
            in_action_block = True
            action_content = ""
            continue

        if stripped == "```" and in_action_block:
            in_action_block = False
            # Process the action
            await execute_action(action_content.strip())
            continue

        if in_action_block:
            action_content += line + "\n"
        else:
            # Check for paragraph breaks
            if stripped == "" and current_thought:
                thought_text = "\n".join(current_thought).strip()
                if thought_text:
                    await think(thought_text, "thinking")
                current_thought = []
            elif stripped:
                current_thought.append(stripped)

    # Don't forget the last thought
    if current_thought:
        thought_text = "\n".join(current_thought).strip()
        if thought_text:
            await think(thought_text, "thinking")


async def execute_action(action_json: str):
    """Execute a parsed action from the LLM."""
    try:
        action = json.loads(action_json)
        action_type = action.get("type")
        symbol = action.get("symbol", "").upper()

        if action_type == "buy":
            shares = action.get("shares", 0)
            reasoning = action.get("reasoning", "")
            await think(f"💰 Placing buy order: {shares} shares of {symbol}...", "trade")
            result = await execute_buy(symbol, shares, reasoning)
            if result["success"]:
                await think(
                    f"✅ Bought {shares} shares of {symbol} at ${result['price']:.2f} "
                    f"for ${result['total']:.2f}. Cash remaining: ${result['new_cash']:,.2f}",
                    "trade",
                    metadata=result
                )
            else:
                await think(f"❌ Buy order failed: {result['error']}", "error")

        elif action_type == "sell":
            shares = action.get("shares", 0)
            reasoning = action.get("reasoning", "")
            await think(f"📤 Placing sell order: {shares} shares of {symbol}...", "trade")
            result = await execute_sell(symbol, shares, reasoning)
            if result["success"]:
                pnl_emoji = "🎉" if result["pnl"] >= 0 else "😤"
                await think(
                    f"✅ Sold {shares} shares of {symbol} at ${result['price']:.2f} "
                    f"for ${result['total']:.2f}. P&L: {'+' if result['pnl'] >= 0 else ''}${result['pnl']:.2f} "
                    f"{pnl_emoji}",
                    "trade",
                    metadata=result
                )
            else:
                await think(f"❌ Sell order failed: {result['error']}", "error")

        elif action_type == "research":
            await think(f"🔍 Researching {symbol}... let me dig into this.", "research")
            detail = await fetch_stock_detail(symbol)
            news = await fetch_news(symbol)

            # Build research summary for Claude
            research_ctx = f"Research results for {symbol}:\n"
            if "error" not in detail:
                research_ctx += f"Name: {detail.get('name', symbol)}\n"
                research_ctx += f"Sector: {detail.get('sector', 'Unknown')} | Industry: {detail.get('industry', 'Unknown')}\n"
                research_ctx += f"Price: ${detail['price']:.2f}\n"
                if detail.get('market_cap'):
                    research_ctx += f"Market Cap: ${detail['market_cap']:,.0f}\n"
                if detail.get('pe_ratio'):
                    research_ctx += f"P/E: {detail['pe_ratio']:.1f}\n"
                if detail.get('forward_pe'):
                    research_ctx += f"Forward P/E: {detail['forward_pe']:.1f}\n"
                research_ctx += f"Month Change: {detail['month_change_pct']:+.1f}%\n"
                research_ctx += f"5-day SMA: ${detail['sma_5']:.2f} | 20-day SMA: ${detail['sma_20']:.2f}\n"
                research_ctx += f"Volatility: {detail['volatility']:.2f}%\n"
                if detail.get('fifty_two_week_high'):
                    research_ctx += f"52wk High: ${detail['fifty_two_week_high']:.2f}\n"
                if detail.get('fifty_two_week_low'):
                    research_ctx += f"52wk Low: ${detail['fifty_two_week_low']:.2f}\n"
                research_ctx += f"Analyst Rating: {detail.get('recommendation', 'N/A')}\n"

            if news:
                research_ctx += "\nRecent News:\n"
                for n in news:
                    research_ctx += f"- {n['title']} ({n['publisher']})\n"
                    if n.get("summary"):
                        research_ctx += f"  Summary: {n['summary'][:200]}\n"

            await think(
                f"📋 Research data for {symbol} gathered. Analyzing...",
                "research",
                metadata={"detail": detail, "news_count": len(news)}
            )

            # Ask Claude to analyze the research
            ai = get_client()
            analysis_response = await ai.messages.create(
                model=MODEL,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"You just researched {symbol}. Here's what you found:\n\n{research_ctx}\n\nShare your analysis. Be specific — is it a buy, sell, or hold? Why? Think out loud."},
                ],
                temperature=0.7,
            )
            analysis = analysis_response.content[0].text
            await process_llm_response(analysis)

        elif action_type == "watch":
            await add_to_watchlist(symbol)
            await think(f"👁️ Added {symbol} to my watchlist. I'll keep an eye on it!", "system")

        elif action_type == "remember":
            content = action.get("content", "")
            category = action.get("category", "lesson")
            if content:
                await add_memory(category, content)
                await think(f"📝 Noted: \"{content}\"", "system")

    except json.JSONDecodeError:
        await think(f"🤔 Tried to do something but got confused with the details. Moving on.", "error")
    except Exception as e:
        await think(f"⚠️ Action failed: {str(e)}", "error")
        print(f"Action execution error: {e}")


async def start_agent_loop(interval_minutes: int = 15):
    """Start the agent with a natural daily rhythm instead of a fixed timer."""
    global _running, _next_check_time
    _running = True

    # Check if this is a fresh start or a restart with existing data
    history = await get_portfolio_history_summary()
    if history["total_trades"] > 0:
        await think(
            f"👋 I'm back! Let me pick up where I left off. "
            f"I've made {history['total_trades']} trades so far across {', '.join(history['symbols_traded'])}. "
            f"Let me check how things are doing...",
            "system"
        )
    else:
        await think(
            f"👋 Hey! I'm StockMind, your AI investor. Starting fresh with $100,000. "
            f"I'll check in throughout the day like a real investor — morning coffee, "
            f"midday glance, evening research. Let's see how this goes! 🚀",
            "system"
        )

    # Run first cycle immediately
    now = datetime.now(TIMEZONE)
    is_weekend = now.weekday() >= 5
    hour = now.hour

    # Pick appropriate first session based on current time
    if is_weekend:
        first_session = "weekend_planning"
    elif hour < 11:
        first_session = "morning_coffee"
    elif hour < 14:
        first_session = "midday_glance"
    elif hour < 17:
        first_session = "afternoon_review"
    else:
        first_session = "evening_research"

    await run_agent_cycle(first_session)

    # Main loop — schedule naturally
    while _running:
        session_id, next_time = get_next_session()
        _next_check_time = next_time

        # Calculate wait time
        wait_seconds = max(0, (next_time - datetime.now(TIMEZONE)).total_seconds())

        session_name = SESSIONS[session_id]["name"]
        time_str = next_time.strftime('%I:%M %p')
        print(f"  [Agent] Next session: {session_name} at {time_str} (in {wait_seconds/60:.0f} min)")

        # Wait until next session
        try:
            await asyncio.sleep(wait_seconds)
        except asyncio.CancelledError:
            break

        if not _running:
            break

        await run_agent_cycle(session_id)


def stop_agent():
    """Stop the agent loop."""
    global _running
    _running = False


async def trigger_cycle():
    """Manually trigger a thinking cycle (for the UI button)."""
    now = datetime.now(TIMEZONE)
    is_weekend = now.weekday() >= 5
    hour = now.hour

    # Pick the most natural session for right now
    if is_weekend:
        session = "weekend_planning"
    elif hour < 11:
        session = "morning_coffee"
    elif hour < 14:
        session = "midday_glance"
    elif hour < 17:
        session = "afternoon_review"
    else:
        session = "evening_research"

    await run_agent_cycle(session)
