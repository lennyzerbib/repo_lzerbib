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

    # Adjust calculations based on option type
    sign = 1 if option_type == 'call' else -1

    # Calculate option price
    price = sign * (S * np.exp(-q * T) * norm.cdf(sign * d1) - K * np.exp(-r * T) * norm.cdf(sign * d2))

    # Calculate Greeks
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


# Streamlit UI
st.set_page_config(page_title="Options Greeks Calculator", layout="wide")
st.title("Options Greeks Calculator")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Option Parameters")

    # Input parameters
    S = st.number_input("Stock Price (S)", value=100.0, min_value=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, min_value=1.0)
    T = st.number_input("Time to Expiration (T) in years", value=1.0, min_value=0.1)
    r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0)
    q = st.number_input("Dividend Yield (q)", value=0.02, min_value=0.0, max_value=1.0)
    sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=1.0)

    # Option type selection
    option_type = st.selectbox("Option Type", ["Call", "Put"]).lower()
    exercise_style = st.selectbox("Exercise Style", ["European", "American"]).lower()

    # Store parameters in a dictionary
    params = {
        'S': S, 'K': K, 'T': T, 'r': r, 'q': q, 'sigma': sigma
    }

with col2:
    # Calculate current values
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

# Greek selection and plotting
selected_greek = st.selectbox("Select Greek to Plot", ["Delta", "Gamma", "Vega", "Theta", "Rho"])

# Generate plot data
x_values, y_values = generate_greek_data(params, selected_greek.lower(), option_type)

# Create plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_values,
    y=y_values,
    mode='lines',
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
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Add notes about calculations
st.info("""
Note: This calculator uses the Black-Scholes model for European options. 
For American options, it uses the same calculations as an approximation. 
The actual values for American options may differ due to early exercise premium.
""")