"""
Notification system -- posts trades and research to Twitter/X.
Uses tweepy with Twitter API v2 and OAuth 1.0a user authentication.
All functions fail gracefully so Twitter issues never crash the agent.
"""
import os
import tweepy
from concurrent.futures import ThreadPoolExecutor
import asyncio

_executor = ThreadPoolExecutor(max_workers=1)
_client = None

DASHBOARD_URL = "https://stock-ai-production-ca5f.up.railway.app/"


def _get_client() -> tweepy.Client | None:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("TWITTER_API_KEY")
    api_secret = os.getenv("TWITTER_API_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

    if not all([api_key, api_secret, access_token, access_secret]):
        return None

    _client = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_secret,
    )
    return _client


def is_enabled() -> bool:
    enabled_flag = os.getenv("TWITTER_ENABLED", "false").lower()
    if enabled_flag not in ("true", "1", "yes"):
        return False
    return _get_client() is not None


def _truncate(text: str, max_len: int = 280) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _tweet_sync(text: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    text = _truncate(text)
    client.create_tweet(text=text)
    return True


async def tweet(text: str) -> bool:
    if not is_enabled():
        return False
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _tweet_sync, text)
    except Exception as e:
        print(f"[Twitter] Tweet failed: {e}")
        return False


def _fmt_money(n: float) -> str:
    return f"${n:,.2f}"


async def tweet_trade(
    action: str,
    symbol: str,
    shares: float,
    price: float,
    total: float,
    pnl: float | None = None,
    reasoning: str = "",
) -> bool:
    link = f"\n\n{DASHBOARD_URL}"

    if action == "buy":
        msg = f"StockMind bought {shares} shares of ${symbol} at {_fmt_money(price)} for {_fmt_money(total)}"
    else:
        pnl_str = ""
        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            pnl_str = f" | P&L: {sign}{_fmt_money(pnl)}"
        msg = f"StockMind sold {shares} shares of ${symbol} at {_fmt_money(price)}{pnl_str}"

    if reasoning:
        # Twitter counts t.co links as 23 chars
        available = 280 - len(msg) - 4 - 25  # " -- " + newlines + link
        if available > 10:
            reason_text = reasoning[:available] if len(reasoning) <= available else reasoning[: available - 3] + "..."
            msg += f" -- {reason_text}"

    msg += link
    return await tweet(msg)


async def tweet_research(symbol: str, analysis: str) -> bool:
    link = f"\n\n{DASHBOARD_URL}"
    prefix = f"StockMind analyzed ${symbol}: "
    available = 280 - len(prefix) - 25  # t.co link + newlines
    clean = " ".join(analysis.split())
    if len(clean) > available:
        clean = clean[: available - 3].rstrip() + "..."
    return await tweet(prefix + clean + link)
