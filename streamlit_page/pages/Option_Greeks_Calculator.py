import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import time
import imageio
import os
from io import BytesIO
from itertools import product


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


# Désactiver le cache pour cette fonction
@st.cache_data(persist=True)
def generate_gif_frames(base_params, greek_name, option_type, vary_param, output_path, _timestamp):
    frames = []
    points = 50  # Nombre de frames pour l'animation

    # Définir la plage pour l'axe x selon la grecque
    if greek_name.lower() in ['delta', 'gamma']:
        plot_param = 'S'
        x_range = np.linspace(base_params['S'] * 0.5, base_params['S'] * 1.5, points)
    elif greek_name.lower() == 'theta':
        plot_param = 'T'
        x_range = np.linspace(0.1, 2, points)
    elif greek_name.lower() == 'vega':
        plot_param = 'sigma'
        x_range = np.linspace(0.1, 0.5, points)
    else:  # rho
        plot_param = 'r'
        x_range = np.linspace(0.01, 0.1, points)

    # Générer la courbe de base (pointillés)
    base_x, base_y = generate_greek_data(base_params, greek_name.lower(), option_type)

    # Définir une plage de valeurs pour le paramètre choisi (vary_param)
    param_ranges = {
        'S': (base_params['S'] * 0.5, base_params['S'] * 1.5),
        'K': (base_params['K'] * 0.5, base_params['K'] * 1.5),
        'T': (0.1, 2.0),
        'r': (0.01, 0.10),
        'sigma': (0.1, 0.5),
        'q': (0.0, 0.10)
    }
    vary_values = np.linspace(param_ranges[vary_param][0], param_ranges[vary_param][1], points)

    # Générer les données pour chaque valeur du paramètre variable
    new_curves = []
    for value in vary_values:
        temp_params = base_params.copy()
        temp_params[vary_param] = value
        x, y = generate_greek_data(temp_params, greek_name.lower(), option_type)
        new_curves.append((x, y))

    # Création des frames pour le GIF
    for i in range(points):
        fig = go.Figure()

        # Courbe de base en pointillés
        fig.add_trace(go.Scatter(
            x=base_x,
            y=base_y,
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name='Base Parameters'
        ))

        # Courbe évolutive en noir pour la valeur actuelle de vary_param
        new_x, new_y = new_curves[i]
        fig.add_trace(go.Scatter(
            x=new_x,
            y=new_y,
            mode='lines',
            line=dict(color='black', width=2),
            name=f'{vary_param} = {vary_values[i]:.2f}'
        ))

        # Mise à jour du layout
        fig.update_layout(
            title=f"{greek_name.capitalize()} Evolution with Varying {vary_param}",
            xaxis_title=plot_param,
            yaxis_title=greek_name.capitalize(),
            height=500,
            width=700,
            showlegend=True,
            plot_bgcolor='white'
        )

        # Conversion en image pour le GIF
        img_bytes = fig.to_image(format="png")
        frames.append(imageio.imread(img_bytes))

    # Sauvegarde du GIF
    imageio.mimsave(output_path, frames, duration=0.1)
    return output_path


# Page setup
st.set_page_config(page_title="Greeks Sensitivity Analysis", layout="wide")
st.title("Options Greeks Sensitivity Analysis")

# Add tab selection
tab1, tab2 = st.tabs(["Static Analysis", "Animated Analysis"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        option_type = st.selectbox("Option Type", ["Call", "Put"], key="option_type_tab1").lower()

    with col2:
        exercise_style = st.selectbox("Exercise Style", ["European", "American"], key="exercise_style_tab1").lower()

    with col3:
        selected_greek = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"], key="greek_tab1")

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
    st.subheader("Animated Greek Evolution (GIF)")

    col1, col2, col3 = st.columns(3)
    with col1:
        option_type = st.selectbox("Option Type", ["Call", "Put"], key="option_type_tab2").lower()
    with col2:
        greek_name = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"], key="greek_tab2")
    with col3:
        vary_param = st.selectbox("Parameter to Vary", ["S", "K", "T", "r", "q", "sigma"], key="param_tab2")

    # Paramètres de base repris de tab1
    base_params = {
        'S': S, 'K': K, 'T': T, 'r': r, 'q': q, 'sigma': sigma
    }

    if st.button("Generate GIF", key="generate_gif_button"):
        # Utiliser un timestamp pour garantir une exécution fraîche
        timestamp = int(time.time())
        output_path = f"greek_evolution_{timestamp}.gif"
        gif_path = generate_gif_frames(base_params, greek_name, option_type, vary_param, output_path, timestamp)

        # Affichage du GIF dans Streamlit
        with open(gif_path, "rb") as file:
            gif_bytes = file.read()
        st.image(gif_bytes, caption=f"{greek_name} Evolution with Varying {vary_param}", use_column_width=True)

        # Nettoyage du fichier temporaire
        if os.path.exists(gif_path):
            os.remove(gif_path)