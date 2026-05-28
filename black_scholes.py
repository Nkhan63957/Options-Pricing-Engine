import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes option pricing formula.
    
    Parameters:
        S     : Current stock price
        K     : Strike price
        T     : Time to expiration (in years)
        r     : Risk-free interest rate (decimal, e.g. 0.05 = 5%)
        sigma : Volatility (decimal, e.g. 0.2 = 20%)
        option_type : 'call' or 'put'
    
    Returns:
        price : Option price
    """
    if T <= 0:
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate all five Greeks for an option.
    
    Returns:
        dict with delta, gamma, theta, vega, rho
    """
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta — sensitivity to stock price
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma — rate of change of delta (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta — time decay (per calendar day)
    if option_type == 'call':
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    # Vega — sensitivity to volatility (per 1% move in vol)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho — sensitivity to interest rates (per 1% move in rates)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': round(delta, 6),
        'gamma': round(gamma, 6),
        'theta': round(theta, 6),
        'vega':  round(vega, 6),
        'rho':   round(rho, 6)
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    call_price = black_scholes(S, K, T, r, sigma, 'call')
    put_price  = black_scholes(S, K, T, r, sigma, 'put')
    call_greeks = greeks(S, K, T, r, sigma, 'call')

    print("=" * 50)
    print("BLACK-SCHOLES PRICING ENGINE")
    print("=" * 50)
    print(f"Stock: ${S}  |  Strike: ${K}  |  T: {T}yr  |  r: {r*100}%  |  σ: {sigma*100}%")
    print(f"\nCall Price : ${call_price:.4f}")
    print(f"Put Price  : ${put_price:.4f}")
    print(f"\nGreeks (Call):")
    for name, val in call_greeks.items():
        print(f"  {name.capitalize():<8}: {val}")
    
    # Put-Call Parity check
    parity = call_price - put_price - S + K * np.exp(-r * T)
    print(f"\nPut-Call Parity check (should be ~0): {parity:.8f}")
    print("=" * 50)
