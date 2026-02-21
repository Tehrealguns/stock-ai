"""
Paper trading engine - executes virtual trades and manages the portfolio.
"""
from database import (
    get_cash_balance, set_cash_balance,
    get_portfolio, get_holding, update_holding,
    add_trade, add_thought, get_risk_profile
)
from market_data import fetch_quotes


COMMISSION_PER_TRADE = 0.0  # Free trades like modern brokerages
SLIPPAGE_PCT = 0.01  # 0.01% slippage simulation

RISK_PROFILES = {
    "safe":       {"max_position_pct": 10, "min_cash_pct": 30, "max_holdings": 5},
    "moderate":   {"max_position_pct": 20, "min_cash_pct": 15, "max_holdings": 10},
    "aggressive": {"max_position_pct": 35, "min_cash_pct": 5,  "max_holdings": 15},
}


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

    # Enforce risk profile limits
    risk_profile = await get_risk_profile()
    limits = RISK_PROFILES.get(risk_profile, RISK_PROFILES["moderate"])

    portfolio = await get_portfolio()
    all_quotes = await fetch_quotes([h["symbol"] for h in portfolio] + [symbol])
    total_market_value = sum(
        h["shares"] * all_quotes.get(h["symbol"], {}).get("price", h["avg_cost"])
        for h in portfolio
    )
    total_portfolio = cash + total_market_value

    # Check max position size
    existing_holding = await get_holding(symbol)
    current_position_value = 0
    if existing_holding:
        current_position_value = existing_holding["shares"] * exec_price
    new_position_value = current_position_value + total_cost
    position_pct = (new_position_value / total_portfolio * 100) if total_portfolio > 0 else 100

    if position_pct > limits["max_position_pct"]:
        max_allowed = total_portfolio * limits["max_position_pct"] / 100 - current_position_value
        max_shares = int(max_allowed / exec_price) if max_allowed > 0 else 0
        return {
            "success": False,
            "error": f"Risk limit: {symbol} would be {position_pct:.1f}% of portfolio (max {limits['max_position_pct']}% in {risk_profile} mode). Max additional shares: {max_shares}"
        }

    # Check cash minimum
    new_cash_after = cash - total_cost
    new_cash_pct = (new_cash_after / total_portfolio * 100) if total_portfolio > 0 else 0
    if new_cash_pct < limits["min_cash_pct"]:
        return {
            "success": False,
            "error": f"Risk limit: cash would drop to {new_cash_pct:.1f}% (min {limits['min_cash_pct']}% in {risk_profile} mode)"
        }

    # Check max holdings count
    distinct_symbols = set(h["symbol"] for h in portfolio if h["shares"] > 0)
    if symbol not in distinct_symbols and len(distinct_symbols) >= limits["max_holdings"]:
        return {
            "success": False,
            "error": f"Risk limit: already at {len(distinct_symbols)} positions (max {limits['max_holdings']} in {risk_profile} mode)"
        }

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

