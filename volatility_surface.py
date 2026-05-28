import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq


def get_options_chain(ticker: str):
    """
    Pull live options chain data from Yahoo Finance.
    Returns calls and puts for all available expiration dates.
    """
    stock = yf.Ticker(ticker)
    
    try:
        spot_price = stock.history(period="1d")['Close'].iloc[-1]
    except Exception:
        raise ValueError(f"Could not fetch price for {ticker}.")
    
    expirations = stock.options
    if not expirations:
        raise ValueError(f"No options data available for {ticker}.")
    
    all_calls = []
    
    for exp_date in expirations[:8]:  # limit to 8 nearest expirations
        try:
            chain = stock.option_chain(exp_date)
            calls = chain.calls.copy()
            calls['expiration'] = exp_date
            all_calls.append(calls)
        except Exception:
            continue
    
    if not all_calls:
        raise ValueError(f"Could not retrieve options chain for {ticker}.")
    
    df = pd.concat(all_calls, ignore_index=True)
    return df, spot_price


def implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Calculate implied volatility by inverting Black-Scholes.
    Uses Brent's method to find the sigma that produces
    the observed market price.
    """
    if T <= 0 or market_price <= 0:
        return np.nan
    
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if market_price < intrinsic * 0.999:
        return np.nan

    def objective(sigma):
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) - market_price
        else:
            return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)) - market_price

    try:
        iv = brentq(objective, 1e-6, 20.0, xtol=1e-6, maxiter=500)
        return iv if 0.001 < iv < 20.0 else np.nan
    except Exception:
        return np.nan


def build_vol_surface(ticker='AAPL', r=0.05):
    """
    Build implied volatility surface for a given ticker.
    Returns strikes, expirations, and IV grid for 3D plotting.
    """
    print(f"Fetching options chain for {ticker}...")
    df, spot = get_options_chain(ticker)
    print(f"  Spot price: ${spot:.2f}")
    print(f"  Options contracts loaded: {len(df)}")

    from datetime import datetime
    today = datetime.today()

    records = []
    for _, row in df.iterrows():
        try:
            exp   = datetime.strptime(row['expiration'], '%Y-%m-%d')
            T     = max((exp - today).days / 365.0, 1/365)
            K     = float(row['strike'])
            mid   = (float(row['bid']) + float(row['ask'])) / 2

            if mid <= 0 or row['bid'] == 0:
                continue

            # Only use near-the-money options (80%-120% of spot)
            moneyness = K / spot
            if not (0.80 <= moneyness <= 1.20):
                continue

            iv = implied_volatility(mid, spot, K, T, r, 'call')
            if iv and not np.isnan(iv) and 0.01 < iv < 5.0:
                records.append({
                    'strike':     K,
                    'expiration': row['expiration'],
                    'T':          round(T, 4),
                    'iv':         round(iv, 4),
                    'moneyness':  round(moneyness, 4)
                })
        except Exception:
            continue

    if len(records) < 10:
        raise ValueError(
            f"Not enough valid options data for {ticker}. "
            f"Only {len(records)} valid IVs computed."
        )

    surface_df = pd.DataFrame(records)
    print(f"  Valid IV points computed: {len(surface_df)}")
    return surface_df, spot


def plot_vol_surface(surface_df, spot, ticker):
    """
    Render the implied volatility surface as an interactive 3D plot.
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=surface_df['T'],
        y=surface_df['strike'],
        z=surface_df['iv'] * 100,
        mode='markers',
        marker=dict(
            size=4,
            color=surface_df['iv'] * 100,
            colorscale='Viridis',
            colorbar=dict(title='IV (%)'),
            opacity=0.85
        ),
        hovertemplate=(
            'Expiry: %{x:.2f}yr<br>'
            'Strike: $%{y:.0f}<br>'
            'IV: %{z:.1f}%<br>'
            '<extra></extra>'
        )
    )])

    fig.update_layout(
        title=dict(
            text=f'{ticker} Implied Volatility Surface  |  Spot: ${spot:.2f}',
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='Time to Expiry (years)',
            yaxis_title='Strike Price ($)',
            zaxis_title='Implied Volatility (%)',
            bgcolor='#0e1117',
            xaxis=dict(backgroundcolor='#0e1117',
                       gridcolor='#374151', color='white'),
            yaxis=dict(backgroundcolor='#0e1117',
                       gridcolor='#374151', color='white'),
            zaxis=dict(backgroundcolor='#0e1117',
                       gridcolor='#374151', color='white'),
        ),
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=650,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    return fig


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    ticker = 'AAPL'
    
    print("=" * 55)
    print("IMPLIED VOLATILITY SURFACE")
    print("=" * 55)
    
    surface_df, spot = build_vol_surface(ticker)
    
    print(f"\nSample IV data:")
    print(surface_df.sort_values('T').head(10).to_string(index=False))
    
    fig = plot_vol_surface(surface_df, spot, ticker)
    fig.show()
    
    print(f"\nVolatility surface rendered for {ticker}")
    print("=" * 55)
