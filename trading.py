"""
Paper trading engine - executes virtual trades and manages the portfolio.
"""
from database import (
    get_cash_balance, set_cash_balance,
    get_portfolio, get_holding, update_holding,
    add_trade, add_thought
)
from market_data import fetch_quotes


COMMISSION_PER_TRADE = 0.0  # Free trades like modern brokerages
SLIPPAGE_PCT = 0.01  # 0.01% slippage simulation


async def execute_buy(symbol: str, shares: float, reasoning: str = "") -> dict:
    """Execute a buy order. Returns result dict."""
    # Get current price
    quotes = await fetch_quotes([symbol])
    if symbol not in quotes:
        return {"success": False, "error": f"Could not get price for {symbol}"}

    price = quotes[symbol]["price"]
    # Apply slippage (buy at slightly higher price)
    exec_price = round(price * (1 + SLIPPAGE_PCT / 100), 2)
    total_cost = round(exec_price * shares, 2)

    # Check cash
    cash = await get_cash_balance()
    if total_cost > cash:
        max_shares = int(cash / exec_price)
        if max_shares <= 0:
            return {"success": False, "error": f"Not enough cash. Need ${total_cost:.2f}, have ${cash:.2f}"}
        return {"success": False, "error": f"Not enough cash for {shares} shares. Max affordable: {max_shares} shares"}

    # Update cash
    new_cash = round(cash - total_cost, 2)
    await set_cash_balance(new_cash)

    # Update holding
    existing = await get_holding(symbol)
    if existing:
        old_shares = existing["shares"]
        old_cost = existing["avg_cost"]
        new_shares = old_shares + shares
        # Weighted average cost
        new_avg = round(((old_shares * old_cost) + (shares * exec_price)) / new_shares, 2)
    else:
        new_shares = shares
        new_avg = exec_price

    await update_holding(symbol, new_shares, new_avg)
    await add_trade(symbol, "buy", shares, exec_price, total_cost, reasoning)

    return {
        "success": True,
        "symbol": symbol,
        "action": "buy",
        "shares": shares,
        "price": exec_price,
        "total": total_cost,
        "new_cash": new_cash,
        "new_shares": new_shares,
        "avg_cost": new_avg,
    }


async def execute_sell(symbol: str, shares: float, reasoning: str = "") -> dict:
    """Execute a sell order. Returns result dict."""
    # Check holding
    existing = await get_holding(symbol)
    if not existing or existing["shares"] <= 0:
        return {"success": False, "error": f"No shares of {symbol} to sell"}

    if shares > existing["shares"]:
        return {"success": False, "error": f"Only have {existing['shares']} shares of {symbol}, tried to sell {shares}"}

    # Get current price
    quotes = await fetch_quotes([symbol])
    if symbol not in quotes:
        return {"success": False, "error": f"Could not get price for {symbol}"}

    price = quotes[symbol]["price"]
    # Apply slippage (sell at slightly lower price)
    exec_price = round(price * (1 - SLIPPAGE_PCT / 100), 2)
    total_proceeds = round(exec_price * shares, 2)

    # Calculate P&L on this sale
    cost_basis = existing["avg_cost"] * shares
    pnl = round(total_proceeds - cost_basis, 2)

    # Update cash
    cash = await get_cash_balance()
    new_cash = round(cash + total_proceeds, 2)
    await set_cash_balance(new_cash)

    # Update holding
    remaining_shares = round(existing["shares"] - shares, 4)
    await update_holding(symbol, remaining_shares, existing["avg_cost"])

    await add_trade(symbol, "sell", shares, exec_price, total_proceeds, reasoning)

    return {
        "success": True,
        "symbol": symbol,
        "action": "sell",
        "shares": shares,
        "price": exec_price,
        "total": total_proceeds,
        "pnl": pnl,
        "new_cash": new_cash,
        "remaining_shares": remaining_shares,
    }


async def get_portfolio_summary() -> dict:
    """Get full portfolio summary with live prices."""
    holdings = await get_portfolio()
    cash = await get_cash_balance()

    if not holdings:
        return {
            "cash": cash,
            "holdings": [],
            "total_value": cash,
            "total_invested": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
        }

    # Fetch current prices for all holdings
    symbols = [h["symbol"] for h in holdings]
    quotes = await fetch_quotes(symbols)

    enriched = []
    total_market_value = 0
    total_cost_basis = 0

    for h in holdings:
        symbol = h["symbol"]
        shares = h["shares"]
        avg_cost = h["avg_cost"]
        cost_basis = round(shares * avg_cost, 2)

        if symbol in quotes:
            current_price = quotes[symbol]["price"]
            market_value = round(shares * current_price, 2)
            pnl = round(market_value - cost_basis, 2)
            pnl_pct = round((pnl / cost_basis * 100), 2) if cost_basis else 0
            day_change = quotes[symbol]["change_pct"]
        else:
            current_price = avg_cost
            market_value = cost_basis
            pnl = 0
            pnl_pct = 0
            day_change = 0

        total_market_value += market_value
        total_cost_basis += cost_basis

        enriched.append({
            "symbol": symbol,
            "shares": shares,
            "avg_cost": avg_cost,
            "current_price": current_price,
            "cost_basis": cost_basis,
            "market_value": market_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "day_change_pct": day_change,
        })

    total_value = round(cash + total_market_value, 2)
    total_pnl = round(total_market_value - total_cost_basis, 2)
    total_pnl_pct = round((total_pnl / total_cost_basis * 100), 2) if total_cost_basis else 0

    return {
        "cash": cash,
        "holdings": sorted(enriched, key=lambda x: x["market_value"], reverse=True),
        "total_value": total_value,
        "total_invested": round(total_cost_basis, 2),
        "total_market_value": round(total_market_value, 2),
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
    }

