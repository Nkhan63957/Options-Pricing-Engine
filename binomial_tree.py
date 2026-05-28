# ── Cell: binomial_tree.py ────────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go

def binomial_tree_price(S, K, T, r, sigma, N=100,
                         option_type='call', american=True):
    """
    Cox-Ross-Rubinstein binomial tree option pricer.

    Works for both European and American options.
    American options allow early exercise at every node —
    this is what Black-Scholes fundamentally cannot do.

    Parameters:
        N        : Number of time steps (more = more accurate)
        american : If True, check early exercise at every node
    """
    dt   = T / N
    u    = np.exp(sigma * np.sqrt(dt))         # up factor
    d    = 1 / u                                # down factor
    p    = (np.exp(r * dt) - d) / (u - d)      # risk-neutral probability
    disc = np.exp(-r * dt)                      # discount factor per step

    # Build stock price tree at expiration (terminal nodes only)
    # ST[j] = S * u^j * d^(N-j)
    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))

    # Option payoffs at expiration
    if option_type == 'call':
        values = np.maximum(ST - K, 0)
    else:
        values = np.maximum(K - ST, 0)

    # Backward induction through the tree
    for i in range(N - 1, -1, -1):
        # Roll back one step
        values = disc * (p * values[1:i+2] + (1 - p) * values[0:i+1])

        if american:
            # At each node, stock price is S * u^j * d^(i-j)
            j_nodes = np.arange(i + 1)
            S_nodes = S * (u ** j_nodes) * (d ** (i - j_nodes))

            if option_type == 'call':
                exercise = np.maximum(S_nodes - K, 0)
            else:
                exercise = np.maximum(K - S_nodes, 0)

            # Early exercise if intrinsic > hold value
            values = np.maximum(values, exercise)

    return float(values[0])


def build_tree_for_viz(S, K, T, r, sigma, N=8,
                        option_type='call', american=True):
    """
    Build full stock and option value trees for visualization.
    Limited to small N (<=12) for readability.
    """
    dt   = T / N
    u    = np.exp(sigma * np.sqrt(dt))
    d    = 1 / u
    p    = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Stock price tree: stock_tree[i][j] = price at step i, j up-moves
    stock_tree = [[0.0] * (i + 1) for i in range(N + 1)]
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[i][j] = S * (u ** j) * (d ** (i - j))

    # Option value tree — start from terminal nodes
    opt_tree = [[0.0] * (i + 1) for i in range(N + 1)]

    # Terminal payoffs
    for j in range(N + 1):
        ST = stock_tree[N][j]
        if option_type == 'call':
            opt_tree[N][j] = max(ST - K, 0)
        else:
            opt_tree[N][j] = max(K - ST, 0)

    # Backward induction
    early_exercise_nodes = []
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = disc * (p * opt_tree[i+1][j+1] + (1-p) * opt_tree[i+1][j])
            if american:
                S_node = stock_tree[i][j]
                exercise = max(S_node - K, 0) if option_type=='call' \
                           else max(K - S_node, 0)
                if exercise > hold and exercise > 0:
                    opt_tree[i][j] = exercise
                    early_exercise_nodes.append((i, j))
                else:
                    opt_tree[i][j] = hold
            else:
                opt_tree[i][j] = hold

    return stock_tree, opt_tree, early_exercise_nodes, p


def plot_binomial_tree(stock_tree, opt_tree, early_exercise_nodes,
                        option_type, K, N):
    """
    Visualize the binomial tree as a network diagram.
    Nodes colored by option value, early exercise nodes highlighted.
    """
    node_x, node_y = [], []
    node_text, node_color, node_size = [], [], []
    edge_x, edge_y = [], []

    early_set = set(early_exercise_nodes)
    max_opt = max(v for row in opt_tree for v in row if v > 0) or 1

    for i in range(N + 1):
        for j in range(i + 1):
            x = i
            y = j - i / 2  # center vertically
            node_x.append(x)
            node_y.append(y)

            S_val   = stock_tree[i][j]
            opt_val = opt_tree[i][j]

            node_text.append(
                f"Step {i}, {j} up<br>"
                f"S = ${S_val:.2f}<br>"
                f"V = ${opt_val:.3f}"
                + (" ⚡Early exercise" if (i,j) in early_set else "")
            )

            # Color: early exercise = red, else gradient by value
            if (i, j) in early_set:
                node_color.append('#f87171')
            else:
                node_color.append(opt_val / max_opt)

            node_size.append(18 if i < N else 14)

            # Draw edges to children
            if i < N:
                x_child, y_up = i+1, (j+1) - (i+1)/2
                x_child, y_dn = i+1, j     - (i+1)/2
                edge_x += [x, x_child, None, x, x_child, None]
                edge_y += [y, y_up,    None, y, y_dn,    None]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='rgba(148,163,184,0.25)', width=1),
        hoverinfo='none', showlegend=False
    ))

    # Nodes (non-early-exercise)
    non_early_mask = [(i,j) not in early_set
                      for i in range(N+1) for j in range(i+1)]
    ne_x = [node_x[k] for k in range(len(node_x)) if non_early_mask[k]]
    ne_y = [node_y[k] for k in range(len(node_y)) if non_early_mask[k]]
    ne_c = [node_color[k] for k in range(len(node_color))
            if non_early_mask[k] and isinstance(node_color[k], float)]
    ne_t = [node_text[k] for k in range(len(node_text)) if non_early_mask[k]]

    fig.add_trace(go.Scatter(
        x=ne_x, y=ne_y, mode='markers',
        marker=dict(
            size=16, color=ne_c,
            colorscale='Blues',
            colorbar=dict(title='Option Value (normalized)'),
            line=dict(color='#60a5fa', width=1)
        ),
        text=ne_t, hoverinfo='text', showlegend=False
    ))

    # Early exercise nodes highlighted separately
    ee_x = [node_x[k] for k in range(len(node_x))
            if not non_early_mask[k]]
    ee_y = [node_y[k] for k in range(len(node_y))
            if not non_early_mask[k]]
    ee_t = [node_text[k] for k in range(len(node_text))
            if not non_early_mask[k]]

    if ee_x:
        fig.add_trace(go.Scatter(
            x=ee_x, y=ee_y, mode='markers',
            marker=dict(size=18, color='#f87171',
                       symbol='star',
                       line=dict(color='white', width=1.5)),
            text=ee_t, hoverinfo='text',
            name='Early Exercise ⚡',
            showlegend=True
        ))

    fig.update_layout(
        title=f"Binomial Tree ({N} steps) — Hover nodes for prices",
        template="plotly_dark",
        height=520,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title="Time Steps →"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(color='white'),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.30

    print("=" * 60)
    print("BINOMIAL TREE OPTION PRICER — CRR Model")
    print("=" * 60)

    for opt in ['call', 'put']:
        bs_val = None
        from scipy.stats import norm
        d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if opt == 'call':
            bs_val = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            bs_val = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

        eu_price = binomial_tree_price(S, K, T, r, sigma, N=200,
                                        option_type=opt, american=False)
        am_price = binomial_tree_price(S, K, T, r, sigma, N=200,
                                        option_type=opt, american=True)
        premium  = am_price - eu_price

        print(f"\n  {opt.upper()} option:")
        print(f"    Black-Scholes (European) : ${bs_val:.4f}")
        print(f"    Binomial     (European)  : ${eu_price:.4f}")
        print(f"    Binomial     (American)  : ${am_price:.4f}")
        print(f"    Early exercise premium   : ${premium:.4f}")

    print("\n  Tree visualization (N=8):")
    st_tree, ov_tree, ee_nodes, p = build_tree_for_viz(
        S, K, T, r, sigma, N=8, option_type='put', american=True
    )
    print(f"    Risk-neutral probability p = {p:.4f}")
    print(f"    Early exercise nodes: {ee_nodes}")
    fig = plot_binomial_tree(st_tree, ov_tree, ee_nodes, 'put', K, N=8)
    fig.show()
    print("=" * 60)
