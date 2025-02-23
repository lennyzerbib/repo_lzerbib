import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import time


def generate_greek_data(params, greek_name, option_type):
    """Generate data for plotting Greeks"""
    points = 100

    # Define range based on selected Greek
    if greek_name in ['delta', 'gamma']:
        var_param = 'S'
        x_range = np.linspace(params['S'] * 0.5, params['S'] * 1.5, points)
    elif greek_name == 'theta':
        var_param = 'T'
        x_range = np.linspace(0.1, 2, points)
    elif greek_name == 'vega':
        var_param = 'sigma'
        x_range = np.linspace(0.1, 0.5, points)
    else:  # rho
        var_param = 'r'
        x_range = np.linspace(0.01, 0.1, points)

    y_values = []
    for x in x_range:
        temp_params = params.copy()
        temp_params[var_param] = x
        greeks = calculate_greeks(
            temp_params['S'], temp_params['K'], temp_params['T'],
            temp_params['r'], temp_params['q'], temp_params['sigma'],
            option_type
        )
        y_values.append(greeks[greek_name.lower()])

    return x_range, np.array(y_values)


def calculate_greeks(S, K, T, r, q, sigma, option_type='call'):
    """Calculate all Greeks for given parameters"""
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    sign = 1 if option_type == 'call' else -1

    price = sign * (S * np.exp(-q * T) * norm.cdf(sign * d1) - K * np.exp(-r * T) * norm.cdf(sign * d2))
    delta = sign * np.exp(-q * T) * norm.cdf(sign * d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
    theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) - \
            sign * r * K * np.exp(-r * T) * norm.cdf(sign * d2) + \
            sign * q * S * np.exp(-q * T) * norm.cdf(sign * d1)
    rho = sign * K * T * np.exp(-r * T) * norm.cdf(sign * d2)

    return {'price': price, 'delta': delta, 'gamma': gamma,
            'vega': vega, 'theta': theta, 'rho': rho}


def generate_multi_param_data(base_params, vary_params, steps=20):
    """Generate data for multiple varying parameters"""
    param_ranges = {
        'S': (base_params['S'] * 0.5, base_params['S'] * 1.5),
        'K': (base_params['K'] * 0.5, base_params['K'] * 1.5),
        'T': (0.1, 2.0),
        'r': (0.01, 0.10),
        'sigma': (0.1, 0.5),
        'q': (0.0, 0.10)
    }

    # Create parameter combinations
    param_values = {}
    for param in vary_params:
        start, end = param_ranges[param]
        param_values[param] = np.linspace(start, end, steps)

    # Generate all combinations of parameter values
    from itertools import product
    combinations = list(product(*[param_values[param] for param in vary_params]))

    # Calculate Greeks for each combination
    results = []
    for combo in combinations:
        temp_params = base_params.copy()
        for param, value in zip(vary_params, combo):
            temp_params[param] = value
        greeks = calculate_greeks(**temp_params)
        results.append({**dict(zip(vary_params, combo)), **greeks})

    return results


def create_animated_plot(data, vary_params, greeks=['delta', 'gamma', 'vega', 'theta', 'rho']):
    """Create an animated plot showing how Greeks change with multiple parameters"""
    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=tuple(greek.capitalize() for greek in greeks),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1)

    # Define positions and colors
    positions = {
        'delta': (1, 1),
        'gamma': (1, 2),
        'vega': (2, 1),
        'theta': (2, 2),
        'rho': (3, 1)
    }

    colors = {
        'delta': '#1f77b4',
        'gamma': '#ff7f0e',
        'vega': '#2ca02c',
        'theta': '#d62728',
        'rho': '#9467bd'
    }

    # Create frames for animation
    frames = []
    for i in range(len(data)):
        frame_data = []
        for greek in greeks:
            row, col = positions[greek]
            trace = go.Scatter(
                x=[data[j][vary_params[0]] for j in range(i + 1)],
                y=[data[j][greek] for j in range(i + 1)],
                mode='lines',
                line=dict(color=colors[greek], width=2),
                name=greek.capitalize()
            )
            frame_data.append(trace)
        frames.append(go.Frame(data=frame_data))

    # Add initial empty traces
    for greek in greeks:
        row, col = positions[greek]
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode='lines',
                line=dict(color=colors[greek], width=2),
                name=greek.capitalize()
            ),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=False,
        title_text=f"Greeks Sensitivity to {', '.join(vary_params)}",
        plot_bgcolor='white',
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 50, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Time step: '},
            'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                           'mode': 'immediate'}],
                       'label': str(k),
                       'method': 'animate'} for k, f in enumerate(frames)]
        }]
    )

    fig.frames = frames

    return fig


# Page setup
st.set_page_config(page_title="Greeks Sensitivity Analysis", layout="wide")
st.title("Options Greeks Sensitivity Analysis")

# Add tab selection
tab1, tab2 = st.tabs(["Static Analysis", "Animated Analysis"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        option_type = st.selectbox("Option Type", ["Call", "Put"]).lower()

    with col2:
        exercise_style = st.selectbox("Exercise Style", ["European", "American"]).lower()

    with col3:
        selected_greek = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"])

    # Base parameters in collapsible section
    with st.expander("Base Parameters", expanded=False):
        col_params1, col_params2 = st.columns(2)

        with col_params1:
            S = st.slider("Stock Price (S)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
            K = st.slider("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
            T = st.slider("Time to Expiration (T) in years", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

        with col_params2:
            r = st.slider("Risk-free Rate (r)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
            q = st.slider("Dividend Yield (q)", min_value=0.0, max_value=0.10, value=0.02, step=0.01)
            sigma = st.slider("Volatility (σ)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    params = {
        'S': S, 'K': K, 'T': T, 'r': r, 'q': q, 'sigma': sigma
    }

    x_values, y_values = generate_greek_data(params, selected_greek.lower(), option_type)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name=selected_greek
    ))

    # Update layout based on selected Greek
    x_label = {
        'Delta': 'Stock Price',
        'Gamma': 'Stock Price',
        'Vega': 'Volatility',
        'Theta': 'Time to Expiration',
        'Rho': 'Risk-free Rate'
    }[selected_greek]

    fig.update_layout(
        title=f"{selected_greek} vs {x_label}",
        xaxis_title=x_label,
        yaxis_title=selected_greek,
        height=600,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )

    st.plotly_chart(fig, use_container_width=True)

    current_values = calculate_greeks(S, K, T, r, q, sigma, option_type)

    st.subheader("Current Values")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

    with col_metrics1:
        st.metric("Price", f"{current_values['price']:.4f}")
        st.metric("Delta", f"{current_values['delta']:.4f}")

    with col_metrics2:
        st.metric("Gamma", f"{current_values['gamma']:.4f}")
        st.metric("Vega", f"{current_values['vega']:.4f}")

    with col_metrics3:
        st.metric("Theta", f"{current_values['theta']:.4f}")
        st.metric("Rho", f"{current_values['rho']:.4f}")

    if exercise_style == "american":
        st.info(
            "Note: For American options, this calculator uses the Black-Scholes model as an approximation. "
            "Actual values may differ due to early exercise premium.")

with tab2:
    st.subheader("Animated Greeks Sensitivity Analysis")

    # Controls for animated analysis
    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option Type (Animated)", ["Call", "Put"]).lower()
    with col2:
        vary_params = st.multiselect(
            "Select Parameters to Vary",
            ["Stock Price (S)", "Strike Price (K)", "Time (T)",
             "Risk-free Rate (r)", "Volatility (σ)", "Dividend Yield (q)"],
            default=["Stock Price (S)"]
        )

    # Parameter mapping
    param_map = {
        "Stock Price (S)": "S",
        "Strike Price (K)": "K",
        "Time (T)": "T",
        "Risk-free Rate (r)": "r",
        "Volatility (σ)": "sigma",
        "Dividend Yield (q)": "q"
    }

    # Convert selected parameters to internal names
    selected_params = [param_map[p] for p in vary_params]

    if len(selected_params) > 0:
        # Prepare base parameters
        base_params = {
            'S': S, 'K': K, 'T': T, 'r': r, 'q': q, 'sigma': sigma,
            'option_type': option_type
        }

        # Generate data for animation
        animation_data = generate_multi_param_data(base_params, selected_params)

        # Create and display animated plot
        fig = create_animated_plot(animation_data, selected_params)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### How to Use the Animation:
        1. Select the parameters you want to vary using the multi-select dropdown
        2. Click the 'Play' button to start the animation
        3. Use the slider to manually control the animation progress
        4. Watch how the Greeks evolve as the selected parameters change

        The animation shows the continuous evolution of all Greeks as the selected parameters
        vary across their ranges, helping visualize the relationships between multiple parameters
        and their effects on option sensitivity.
        """)
    else:
        st.warning("Please select at least one parameter to vary.")