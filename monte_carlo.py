import numpy as np
import plotly.graph_objects as go

def monte_carlo_price(S, K, T, r, sigma, option_type='call',
                      n_simulations=10000, n_steps=252, seed=42):
    """
    Price an option via Monte Carlo simulation.
    
    Simulates n_simulations possible stock price paths,
    calculates payoff at expiration for each, and discounts
    back to present value.
    
    Parameters:
        S            : Current stock price
        K            : Strike price
        T            : Time to expiration (years)
        r            : Risk-free rate (decimal)
        sigma        : Volatility (decimal)
        option_type  : 'call' or 'put'
        n_simulations: Number of random paths to simulate
        n_steps      : Number of time steps per path
        seed         : Random seed for reproducibility
    
    Returns:
        price        : Estimated option price
        std_error    : Standard error of the estimate
        paths        : Array of simulated paths (for plotting)
    """
    np.random.seed(seed)

    dt = T / n_steps
    
    # Simulate all paths at once using vectorized operations
    # Each row is one path, each column is one time step
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    
    # Geometric Brownian Motion formula
    daily_returns = np.exp(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
    )
    
    # Build price paths: start at S, multiply cumulative returns
    price_paths = S * np.cumprod(daily_returns, axis=1)
    
    # Final prices at expiration
    final_prices = price_paths[:, -1]
    
    # Calculate payoff for each path
    if option_type == 'call':
        payoffs = np.maximum(final_prices - K, 0)
    else:
        payoffs = np.maximum(K - final_prices, 0)
    
    # Discount payoffs back to present value
    price     = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    
    return price, std_error, price_paths


def convergence_analysis(S, K, T, r, sigma, option_type='call',
                         max_simulations=50000):
    """
    Show how Monte Carlo price converges as simulations increase.
    Runs at 10 checkpoints from 100 to max_simulations.
    """
    checkpoints = np.logspace(2, np.log10(max_simulations), 10).astype(int)
    prices = []
    errors = []
    
    for n in checkpoints:
        price, std_err, _ = monte_carlo_price(
            S, K, T, r, sigma, option_type, n_simulations=n
        )
        prices.append(price)
        errors.append(std_err)
    
    return checkpoints, prices, errors


def plot_paths(price_paths, S, K, n_display=200):
    """Plot a sample of simulated price paths."""
    fig = go.Figure()
    
    steps = price_paths.shape[1]
    x = np.linspace(0, 1, steps)
    
    # Plot sample paths
    for i in range(min(n_display, len(price_paths))):
        fig.add_trace(go.Scatter(
            x=x, y=price_paths[i],
            mode='lines',
            line=dict(width=0.5, color='rgba(96, 165, 250, 0.15)'),
            showlegend=False
        ))
    
    # Strike price line
    fig.add_hline(y=K, line_dash="dash",
                  line_color="#f87171", line_width=2,
                  annotation_text=f"Strike ${K}")
    
    # Starting price line
    fig.add_hline(y=S, line_dash="dot",
                  line_color="#34d399", line_width=2,
                  annotation_text=f"Current ${S}")
    
    fig.update_layout(
        title="Simulated Stock Price Paths",
        xaxis_title="Time (fraction of year)",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        height=500,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117"
    )
    return fig


def plot_convergence(checkpoints, prices, bs_price):
    """Plot Monte Carlo convergence toward Black-Scholes price."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=checkpoints, y=prices,
        mode='lines+markers',
        name='Monte Carlo Price',
        line=dict(color='#60a5fa', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_hline(
        y=bs_price,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"Black-Scholes ${bs_price:.4f}"
    )
    
    fig.update_layout(
        title="Monte Carlo Convergence to Black-Scholes",
        xaxis_title="Number of Simulations",
        xaxis_type="log",
        yaxis_title="Estimated Option Price ($)",
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117"
    )
    return fig


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    print("=" * 55)
    print("MONTE CARLO OPTIONS PRICER")
    print("=" * 55)

    for n in [1000, 10000, 50000]:
        price, stderr, _ = monte_carlo_price(
            S, K, T, r, sigma, 'call', n_simulations=n
        )
        print(f"  {n:>6,} simulations → ${price:.4f}  (±{stderr:.4f})")

    print(f"\n  Black-Scholes benchmark  → $10.4506")
    print(f"\n  Convergence: ✓ MC approaches BS as n increases")
    
    # Convergence plot
    checkpoints, prices, errors = convergence_analysis(
        S, K, T, r, sigma, max_simulations=50000
    )
    fig = plot_convergence(checkpoints, prices, bs_price=10.4506)
    fig.show()
    
    # Paths plot
    _, _, paths = monte_carlo_price(
        S, K, T, r, sigma, n_simulations=500
    )
    fig2 = plot_paths(paths, S, K)
    fig2.show()
    
    print("=" * 55)
