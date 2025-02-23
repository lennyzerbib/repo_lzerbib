import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


def calculate_d1d2(S, K, T, r, q, sigma):
    """Calculate d1 and d2 parameters for Black-Scholes"""
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def calculate_greeks(S, K, T, r, q, sigma, option_type='call'):
    """Calculate option price and all Greeks"""
    d1, d2 = calculate_d1d2(S, K, T, r, q, sigma)

    sign = 1 if option_type == 'call' else -1

    # Calculate option price and Greeks
    price = sign * (S * np.exp(-q * T) * norm.cdf(sign * d1) - K * np.exp(-r * T) * norm.cdf(sign * d2))
    delta = sign * np.exp(-q * T) * norm.cdf(sign * d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
    theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) - \
            sign * r * K * np.exp(-r * T) * norm.cdf(sign * d2) + \
            sign * q * S * np.exp(-q * T) * norm.cdf(sign * d1)
    rho = sign * K * T * np.exp(-r * T) * norm.cdf(sign * d2)

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


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


# Page configuration
st.set_page_config(page_title="Options Greeks Calculator", layout="wide")
st.title("Options Greeks Calculator")

# Main selection controls in a horizontal layout
col1, col2, col3 = st.columns(3)

with col1:
    option_type = st.selectbox("Option Type", ["Call", "Put"]).lower()

with col2:
    exercise_style = st.selectbox("Exercise Style", ["European", "American"]).lower()

with col3:
    selected_greek = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"])

# Parameters in collapsible section
with st.expander("Parameters", expanded=False):
    col_params1, col_params2 = st.columns(2)

    with col_params1:
        S = st.slider("Stock Price (S)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
        K = st.slider("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
        T = st.slider("Time to Expiration (T) in years", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    with col_params2:
        r = st.slider("Risk-free Rate (r)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
        q = st.slider("Dividend Yield (q)", min_value=0.0, max_value=0.10, value=0.02, step=0.01)
        sigma = st.slider("Volatility (σ)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

# Store parameters
params = {
    'S': S, 'K': K, 'T': T, 'r': r, 'q': q, 'sigma': sigma
}

# Calculate current values
current_values = calculate_greeks(S, K, T, r, q, sigma, option_type)

# Generate and display plot
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


# Display current values in a clean grid
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
        "Note: For American options, this calculator uses the Black-Scholes model as an approximation. Actual values may differ due to early exercise premium.")