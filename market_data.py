"""
Market data fetcher - real stock prices via yfinance.
"""
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)


def _fetch_quotes_sync(symbols: list[str]) -> dict:
    """Fetch current quotes for a list of symbols (sync, runs in thread)."""
    result = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")

            if hist.empty:
                print(f"  {symbol}: no history data")
                continue

            current_price = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0

            result[symbol] = {
                "symbol": symbol,
                "price": round(current_price, 2),
                "prev_close": round(prev_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0,
                "high": round(float(hist["High"].iloc[-1]), 2),
                "low": round(float(hist["Low"].iloc[-1]), 2),
                "open": round(float(hist["Open"].iloc[-1]), 2),
            }
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            continue
    return result


async def fetch_quotes(symbols: list[str]) -> dict:
    """Async wrapper to fetch quotes."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_quotes_sync, symbols)


def _fetch_stock_detail_sync(symbol: str) -> dict:
    """Fetch detailed info for a single stock."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        hist = ticker.history(period="1mo")

        # Calculate some analytics
        if not hist.empty:
            prices = hist["Close"]
            current = float(prices.iloc[-1])
            month_ago = float(prices.iloc[0])
            month_change = ((current - month_ago) / month_ago * 100) if month_ago else 0

            # Simple moving averages
            sma_5 = float(prices.tail(5).mean()) if len(prices) >= 5 else current
            sma_20 = float(prices.tail(20).mean()) if len(prices) >= 20 else current

            # Volatility (std of daily returns)
            returns = prices.pct_change().dropna()
            volatility = float(returns.std() * 100) if len(returns) > 1 else 0
        else:
            current = 0
            month_change = 0
            sma_5 = 0
            sma_20 = 0
            volatility = 0

        return {
            "symbol": symbol,
            "name": info.get("shortName", info.get("longName", symbol)),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "dividend_yield": info.get("dividendYield", None),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", None),
            "price": round(current, 2),
            "month_change_pct": round(month_change, 2),
            "sma_5": round(sma_5, 2),
            "sma_20": round(sma_20, 2),
            "volatility": round(volatility, 2),
            "recommendation": info.get("recommendationKey", "none"),
        }
    except Exception as e:
        print(f"Error fetching detail for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


async def fetch_stock_detail(symbol: str) -> dict:
    """Async wrapper for stock detail."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_stock_detail_sync, symbol)


def _fetch_news_sync(symbol: str) -> list[dict]:
    """Fetch recent news for a stock."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return []

        # yfinance 1.x may return news differently
        results = []
        items = news if isinstance(news, list) else []

        for item in items[:5]:
            # Handle both old and new yfinance formats
            if isinstance(item, dict):
                # New format (1.x)
                content = item.get("content", item)
                if isinstance(content, dict):
                    title = content.get("title", item.get("title", "No title"))
                    provider = content.get("provider", {})
                    publisher = provider.get("displayName", item.get("publisher", "Unknown")) if isinstance(provider, dict) else str(provider)
                    canonical = content.get("canonicalUrl", {})
                    link = canonical.get("url", item.get("link", "")) if isinstance(canonical, dict) else str(canonical)
                    summary = content.get("summary", item.get("summary", ""))
                    pub_date = content.get("pubDate", item.get("providerPublishTime", ""))
                else:
                    title = item.get("title", "No title")
                    publisher = item.get("publisher", "Unknown")
                    link = item.get("link", "")
                    summary = item.get("summary", "")
                    pub_date = item.get("providerPublishTime", "")

                results.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "published": str(pub_date),
                    "summary": summary,
                })
        return results
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []


async def fetch_news(symbol: str) -> list[dict]:
    """Async wrapper for news."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_news_sync, symbol)


def _fetch_market_overview_sync() -> dict:
    """Get a quick market overview (major indices)."""
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
    }
    result = {}
    for symbol, name in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                current = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
                change_pct = ((current - prev) / prev * 100) if prev else 0
                result[name] = {
                    "value": round(current, 2),
                    "change_pct": round(change_pct, 2),
                }
        except Exception as e:
            print(f"  Index {symbol} error: {e}")
            continue
    return result


async def fetch_market_overview() -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_market_overview_sync)


def is_market_hours() -> bool:
    """Check if US market is currently open (rough check)."""
    now = datetime.now()
    # Market hours: Mon-Fri, 9:30 AM - 4:00 PM ET
    if now.weekday() >= 5:  # Weekend
        return False
    hour = now.hour
    return 9 <= hour <= 16
