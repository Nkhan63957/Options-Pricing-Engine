# ── Cell: strategies.py ───────────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

def bs_price_simple(S, K, T, r, sigma, opt='call'):
    """Black-Scholes price. Returns 0 for expired options."""
    if T <= 0:
        return max(0, S - K) if opt == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, opt='call'):
    if T <= 0: return 0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) if opt == 'call' else norm.cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    if T <= 0: return 0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    if T <= 0: return 0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

# ── Strategy definitions ───────────────────────────────────────────────────────
STRATEGIES = {
    "Long Call": {
        "description": "Buy the right to purchase stock at strike. Unlimited upside, limited downside.",
        "legs": lambda S, K, T, r, sig: [
            {"type":"call","direction":1,"K":K,"premium":bs_price_simple(S,K,T,r,sig,'call')}
        ]
    },
    "Long Put": {
        "description": "Buy the right to sell stock at strike. Profits when stock falls.",
        "legs": lambda S, K, T, r, sig: [
            {"type":"put","direction":1,"K":K,"premium":bs_price_simple(S,K,T,r,sig,'put')}
        ]
    },
    "Covered Call": {
        "description": "Own the stock, sell a call. Generates income, caps upside.",
        "legs": lambda S, K, T, r, sig: [
            {"type":"stock","direction":1,"K":K,"premium":S},
            {"type":"call","direction":-1,"K":K,"premium":bs_price_simple(S,K,T,r,sig,'call')}
        ]
    },
    "Protective Put": {
        "description": "Own the stock, buy a put. Insurance against downside.",
        "legs": lambda S, K, T, r, sig: [
            {"type":"stock","direction":1,"K":K,"premium":S},
            {"type":"put","direction":1,"K":K,"premium":bs_price_simple(S,K,T,r,sig,'put')}
        ]
    },
    "Bull Call Spread": {
        "description": "Buy low strike call, sell high strike call. Reduced cost, capped profit.",
        "legs": lambda S, K, T, r, sig: [
            {"type":"call","direction":1, "K":K*0.95,"premium":bs_price_simple(S,K*0.95,T,r,sig,'call')},
            {"type":"call","direction":-1,"K":K*1.05,"premium":bs_price_simple(S,K*1.05,T,r,sig,'call')}
        ]
    },
    "Bear Put Spread": {
        "description": "Buy high strike put, sell low strike put. Profits on moderate decline.",
        "legs": lambda S, K, T, r, sig: [
            {"type":"put","direction":1, "K":K*1.05,"premium":bs_price_simple(S,K*1.05,T,r,sig,'put')},
            {"type":"put","direction":-1,"K":K*0.95,"premium":bs_price_simple(S,K*0.95,T,r,sig,'put')}
        ]
    },
    "Long Straddle": {
        "description": "Buy call and put at same strike. Profits from large moves in either direction.",
        "legs": lambda S, K, T, r, sig: [
            {"type":"call","direction":1,"K":K,"premium":bs_price_simple(S,K,T,r,sig,'call')},
            {"type":"put", "direction":1,"K":K,"premium":bs_price_simple(S,K,T,r,sig,'put')}
        ]
    },
    "Iron Condor": {
        "description": "Sell strangle, buy wider strangle. Profits when stock stays in a range.",
        "legs": lambda S, K, T, r, sig: [
            {"type":"put", "direction":1, "K":K*0.85,"premium":bs_price_simple(S,K*0.85,T,r,sig,'put')},
            {"type":"put", "direction":-1,"K":K*0.95,"premium":bs_price_simple(S,K*0.95,T,r,sig,'put')},
            {"type":"call","direction":-1,"K":K*1.05,"premium":bs_price_simple(S,K*1.05,T,r,sig,'call')},
            {"type":"call","direction":1, "K":K*1.15,"premium":bs_price_simple(S,K*1.15,T,r,sig,'call')}
        ]
    }
}

def payoff_at_expiry(legs, stock_prices):
    """
    Calculate total P&L of a multi-leg strategy at expiration
    across a range of stock prices.
    """
    total_pnl = np.zeros(len(stock_prices))
    total_premium = 0

    for leg in legs:
        direction = leg["direction"]
        K_leg     = leg["K"]
        premium   = leg["premium"]
        opt_type  = leg["type"]

        # Net premium paid/received
        total_premium += direction * premium

        for i, S_exp in enumerate(stock_prices):
            if opt_type == 'call':
                intrinsic = max(0, S_exp - K_leg)
            elif opt_type == 'put':
                intrinsic = max(0, K_leg - S_exp)
            else:  # stock
                intrinsic = S_exp - K_leg  # P&L vs purchase price

            total_pnl[i] += direction * intrinsic

    # Subtract net premium paid
    total_pnl -= total_premium
    return total_pnl, total_premium

def strategy_metrics(pnl, stock_prices, S):
    """Calculate max profit, max loss, breakeven points."""
    max_profit = np.max(pnl)
    max_loss   = np.min(pnl)

    # Breakeven: where pnl crosses zero
    breakevens = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i+1] < 0:  # sign change
            # Linear interpolation
            be = stock_prices[i] - pnl[i] * (stock_prices[i+1] - stock_prices[i]) / (pnl[i+1] - pnl[i])
            breakevens.append(round(be, 2))

    return {
        "max_profit": max_profit if max_profit < 1e6 else float('inf'),
        "max_loss":   max_loss   if max_loss   > -1e6 else float('-inf'),
        "breakevens": breakevens
    }

def plot_strategy(pnl, stock_prices, S, K, strategy_name, legs):
    """Create P&L diagram for the strategy."""
    fig = go.Figure()

    # Zero line
    fig.add_hline(y=0, line_color="#6b7280", line_width=1.5)

    # Current stock price
    fig.add_vline(x=S, line_dash="dash", line_color="#34d399",
                  line_width=1.5, annotation_text=f"Current ${S:.0f}",
                  annotation_position="top left")

    # Strike lines for each leg
    plotted_strikes = set()
    for leg in legs:
        if leg["type"] in ["call","put"] and leg["K"] not in plotted_strikes:
            fig.add_vline(x=leg["K"], line_dash="dot",
                         line_color="#f59e0b", line_width=1,
                         annotation_text=f"K=${leg['K']:.0f}")
            plotted_strikes.add(leg["K"])

    # P&L curve — color by profit/loss
    profit_mask = pnl >= 0
    loss_mask   = pnl < 0

    fig.add_trace(go.Scatter(
        x=stock_prices[profit_mask], y=pnl[profit_mask],
        mode='lines', name='Profit',
        line=dict(color='#34d399', width=3),
        fill='tozeroy', fillcolor='rgba(52, 211, 153, 0.15)'
    ))
    fig.add_trace(go.Scatter(
        x=stock_prices[loss_mask], y=pnl[loss_mask],
        mode='lines', name='Loss',
        line=dict(color='#f87171', width=3),
        fill='tozeroy', fillcolor='rgba(248, 113, 113, 0.15)'
    ))

    fig.update_layout(
        title=f"{strategy_name} — P&L at Expiration",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        template="plotly_dark",
        height=460,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    S, K, T, r, sig = 100, 100, 0.25, 0.05, 0.20
    stock_range = np.linspace(60, 140, 500)

    print("=" * 55)
    print("OPTIONS STRATEGY BUILDER — Test")
    print("=" * 55)

    for name, strat in STRATEGIES.items():
        legs = strat["legs"](S, K, T, r, sig)
        pnl, net_prem = payoff_at_expiry(legs, stock_range)
        metrics = strategy_metrics(pnl, stock_range, S)
        print(f"\n{name}")
        print(f"  Net premium : ${net_prem:+.4f}")
        max_p_str = 'Unlimited' if metrics['max_profit'] > 999 else f"${metrics['max_profit']:+.2f}"
        max_l_str = 'Unlimited' if metrics['max_loss'] < -999 else f"${metrics['max_loss']:+.2f}"
        print(f"  Max profit  : {max_p_str}")
        print(f"  Max loss    : {max_l_str}")
        print(f"  Breakevens  : {metrics['breakevens']}")

    print("\n" + "=" * 55)
