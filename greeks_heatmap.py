# ── Cell: greeks_heatmap.py ───────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

def compute_greeks_grid(S_range, T_range, K, r, sigma, greek='delta', opt='call'):
    """
    Compute a Greek value across a 2D grid of stock prices × time to expiry.
    Returns a matrix ready for heatmap plotting.
    """
    grid = np.zeros((len(T_range), len(S_range)))

    for i, T in enumerate(T_range):
        for j, S in enumerate(S_range):
            if T <= 0:
                grid[i][j] = 0
                continue

            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)

            if greek == 'delta':
                grid[i][j] = norm.cdf(d1) if opt=='call' else norm.cdf(d1)-1
            elif greek == 'gamma':
                grid[i][j] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            elif greek == 'theta':
                term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                if opt == 'call':
                    grid[i][j] = (term1 - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
                else:
                    grid[i][j] = (term1 + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
            elif greek == 'vega':
                grid[i][j] = S * norm.pdf(d1) * np.sqrt(T) / 100
            elif greek == 'rho':
                if opt == 'call':
                    grid[i][j] = K*T*np.exp(-r*T)*norm.cdf(d2)/100
                else:
                    grid[i][j] = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100

    return grid


def plot_greeks_heatmap(S_range, T_range, grid, greek, opt, K):
    """
    Render Greeks heatmap: stock price on X, time to expiry on Y,
    Greek value as color intensity.
    """
    colorscales = {
        'delta': 'RdBu',
        'gamma': 'Viridis',
        'theta': 'Reds',
        'vega':  'Plasma',
        'rho':   'Cividis'
    }

    fig = go.Figure(data=go.Heatmap(
        x=S_range,
        y=T_range,
        z=grid,
        colorscale=colorscales.get(greek, 'Viridis'),
        colorbar=dict(title=greek.capitalize()),
        hovertemplate=(
            'Stock: $%{x:.1f}<br>'
            'Time: %{y:.2f}yr<br>'
            f'{greek.capitalize()}: %{{z:.4f}}<br>'
            '<extra></extra>'
        )
    ))

    # Strike line
    fig.add_vline(x=K, line_dash="dash",
                  line_color="white", line_width=2,
                  annotation_text=f"Strike K=${K}",
                  annotation_font_color="white")

    greek_descriptions = {
        'delta': 'Delta — How much the option moves per $1 of stock movement',
        'gamma': 'Gamma — Rate of change of Delta (acceleration)',
        'theta': 'Theta — Daily time decay ($/day)',
        'vega':  'Vega — Sensitivity to 1% change in volatility',
        'rho':   'Rho — Sensitivity to 1% change in interest rates'
    }

    fig.update_layout(
        title=f"{greek.capitalize()} Surface — {opt.upper()} | σ={25}% | r=5%<br>"
              f"<sub>{greek_descriptions.get(greek,'')}</sub>",
        xaxis_title="Stock Price ($)",
        yaxis_title="Time to Expiry (years)",
        template="plotly_dark",
        height=500,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color='white'),
        margin=dict(l=40, r=20, t=80, b=40)
    )
    return fig


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    K, r, sigma = 100, 0.05, 0.25
    S_range = np.linspace(60, 140, 80)
    T_range = np.linspace(0.02, 2.0, 60)

    print("=" * 55)
    print("GREEKS HEATMAP GENERATOR")
    print("=" * 55)

    for greek in ['delta', 'gamma', 'theta', 'vega']:
        grid = compute_greeks_grid(S_range, T_range, K, r, sigma,
                                    greek=greek, opt='call')
        fig = plot_greeks_heatmap(S_range, T_range, grid, greek, 'call', K)
        fig.show()
        print(f"  {greek.capitalize()} heatmap rendered ✓")

    print("=" * 55)
