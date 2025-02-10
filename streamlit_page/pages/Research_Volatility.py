import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

import sys


#ENTRER LE PATH VERS L'ENDROIT OU TU AS TELECHARGER LES 2 DOCUMENTS
sys.path.append('/Users/lennyzerbib/Desktop/Dauphine/203/Mémoire 203')
import get_volatility as vol

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 200px;
        max-width: 250px;
        }
    """,
    unsafe_allow_html=True,
    )


try:
    _,col_1 = st.columns([9,1])
    empty_button = col_1.button(label='Clear Cache', key='Clear Cache')
    if st.session_state['Clear Cache']  != False : st.cache_data.clear()
except:
    pass


###################################################################################################################################
#                                                           FIGURE VOLATILITY                                                     #
###################################################################################################################################


def figure_implied_volatility(df):
    fig_vol = make_subplots(rows=1, cols=2, subplot_titles = ('Implied Volatility', 'Market Skew'))
    fig_vol.add_trace(go.Scatter(x=df['Date'], y=df['ATMF_Implied_Volatility'], name='ATMF Implied Vol', line_color='darkorange'), row=1, col=1)
    fig_vol.add_trace(go.Scatter(x=df['Date'], y=df['Market_Skew'], name='Market Skew', line_color='darkorange'), row=1, col=2)

    fig_vol.add_hline(y=0, line_width=0.75, line_dash='dash', line_color='black')
    fig_vol.update_xaxes(title_text='<b>Maturity<b>')
    fig_vol.update_yaxes(title_text='<b>ATMF Implied Volatility<b>', row=1, col=1)
    fig_vol.update_yaxes(title_text='<b>Market Skew<b>', row=1, col=2)

    fig_vol.update_layout(height=500, width=400, title_text='ATMF Implied Volatility and Market Skew over different Maturities for '+ticker)
    st.plotly_chart(fig_vol, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig_vol


def figure_proxy(df):
    fig_proxy = make_subplots(rows=1, cols=2, subplot_titles = ('Volatility-of-Volatility for '+ ticker, 'Correlation Spot-Volatility for '+ticker))
    fig_proxy.add_trace(go.Scatter(x=df['Date'], y=df['Volatility'], name='Vol-of-Vol', line_color='darkorange'), row=1, col=1)
    fig_proxy.add_trace(go.Scatter(x=df['Date'], y=df['Correlation'], name='Correlation Spot-Vol', line_color='darkorange'), row=1, col=2)

    fig_proxy.add_hline(y=0, line_width=0.75, line_dash='dash', line_color='black')
    fig_proxy.update_xaxes(title_text='<b>Maturity<b>')
    fig_proxy.update_yaxes(title_text='<b>Vvol<b>', row=1, col=1)
    fig_proxy.update_yaxes(title_text='<b>Correl<b>', row=1, col=2)

    fig_proxy.update_layout(height=500, width=400, title_text='Breakeven of '+ticker+' with parameters: (k=1.5, p=-0.8, v=2)')
    st.plotly_chart(fig_proxy, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig_proxy


def fig_nelder_mead(df, df_algo):
    fig_proxy = make_subplots(rows=1, cols=2, subplot_titles = ('Volatility-of-Volatility for '+ ticker, 'Correlation Spot-Volatility for '+ticker))
    fig_proxy.add_trace(go.Scatter(x=df['Date'], y=df['Volatility'], name='Vol-of-Vol', line_color='darkorange', legendgroup='1'), row=1, col=1)
    fig_proxy.add_trace(go.Scatter(x=df['Date'], y=df['Correlation'], name='Correlation Spot-Vol', line_color='darkorange', legendgroup='1'), row=1, col=2)

    fig_proxy.add_trace(go.Scatter(x=df['Date'], y=df['Volatility'].apply(lambda x: x+0.1), name='Bump Vol-of-Vol', line_color='lightgrey', legendgroup='2'), row=1, col=1)
    fig_proxy.add_trace(go.Scatter(x=df['Date'], y=df['Correlation'].apply(lambda x : x+0.1), name='Bump Correlation Spot-Vol', line_color='lightgrey', legendgroup='2'), row=1, col=2)

    fig_proxy.add_trace(go.Scatter(x=df_algo['Date'], y=df_algo['Volatility'], name='Nelder-Mead Vol-of-Vol', line_color='steelblue', legendgroup='3'), row=1, col=1)
    fig_proxy.add_trace(go.Scatter(x=df_algo['Date'], y=df_algo['Correlation'], name='Nelder-Mead Correlation Spot-Vol', line_color='steelblue', legendgroup='3'), row=1, col=2)

    fig_proxy.add_hline(y=0, line_width=0.75, line_dash='dash', line_color='black')
    fig_proxy.update_xaxes(title_text='<b>Maturity<b>')
    fig_proxy.update_yaxes(title_text='<b>Vvol<b>', row=1, col=1)
    fig_proxy.update_yaxes(title_text='<b>Correl<b>', row=1, col=2)

    fig_proxy.update_layout(height=500, width=400, title_text='Nelder Mead Algorithm on '+ticker)
    st.plotly_chart(fig_proxy, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig_proxy


def fig_cost_function(df, fig, row, col):
    df = df.reset_index()
    trace = go.Scatter(x=df['index'], y=df['Loss'], name='Cost Function', line_color='purple', showlegend=False)
    fig.add_trace(trace, row=row, col=col)
    
    fig.update_xaxes(title_text='<b>Iteration</b>', row=row, col=col)
    fig.update_yaxes(title_text='<b>Cost Function</b>', row=row, col=col)


def fig_3d_scatter(df, fig, row, col):
    scatter = px.scatter_3d(df, x='kappa', y='rho', z='nu', color='Loss', color_continuous_scale='Plotly3_r', range_color=(0, 1))
    for trace in scatter.data:
        fig.add_trace(trace, row=row, col=col)
    fig.update_scenes(xaxis_title='Kappa', yaxis_title='Rho', zaxis_title='Nu', row=row, col=col)
    fig.update_layout(coloraxis_colorbar=dict(title='Cost Function'))


def fig_convergence(df_cost):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Conververgence on Cost Function", "Convergence on Parameters"), specs=[[{'type': 'xy'}, {'type': 'scatter3d'}]])
    fig_cost_function(df_cost, fig, 1, 1)
    fig_3d_scatter(df_cost, fig, 1, 2)
    fig.update_layout(height=400, width=800, title_text="Convergence of the Nelder Mead Algorithm on "+ticker)
    st.plotly_chart(fig, use_container_width=True)
    return fig


def fig_spot_path(S_bs, S_lsv):
    fig_spot_path = make_subplots(rows=1, cols=2, subplot_titles = ('Stock Paths with LSV Model for '+ticker, 'Stock Paths with BS Model for '+ticker))
    for i in range(100):
        fig_spot_path = fig_spot_path.add_trace(go.Scatter(x=S_bs['index'].apply(lambda x : x/len(S_bs)), y=S_bs[i], showlegend=False), row=1, col=1)
    for i in range(100):
        fig_spot_path = fig_spot_path.add_trace(go.Scatter(x=S_lsv['index'].apply(lambda x : x/len(S_lsv)), y=S_lsv[i], showlegend=False), row=1, col=2)
    fig_spot_path.update_xaxes(title_text='<b>Time Step<b>')
    fig_spot_path.update_yaxes(title_text='<b>Stock Price<b>')
    
    fig_spot_path.update_layout(height=500, width=400, title_text='Simulation of '+ticker+' spot path with LSV1F and BS Model Over 1y')
    st.plotly_chart(fig_spot_path, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig_spot_path


def fig_distrib(ticker):
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    maturities = ['3m', '6m', '9m', '1y']

    for ax, maturity in zip(axes.flatten(), maturities):
        X = vol.get_distribution(ticker, maturity)
        
        sns.histplot(X, x=f'Spot with {maturity} maturity', hue='Model', element='poly', ax=ax)
        
        ax.set_xlim(left=0, right=X[f'Spot with {maturity} maturity'].quantile(0.95))

        ax.set_title(f'{ticker} Spot Path with a {maturity} Maturity')
        ax.set_xlabel('Spot')
        ax.set_ylabel('Frequency')

    fig.tight_layout()
    st.pyplot(fig)
    return fig

def fig_cdf(ticker):
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    maturities = ['3m', '6m', '9m', '1y']

    for ax, maturity in zip(axes.flatten(), maturities):
        X = vol.get_distribution(ticker, maturity)
        
        sns.ecdfplot(X, x=f'Spot with {maturity} maturity', hue='Model', ax=ax)
        ax.set_xlim(left=0, right=X[f'Spot with {maturity} maturity'].quantile(0.95))

        ax.set_title(f'{ticker} Spot Path with a {maturity} Maturity')
        ax.set_xlabel('Spot')
        ax.set_ylabel('Cummulative Probability')

    fig.tight_layout()
    st.pyplot(fig)
    return fig

def fig_distribution(ticker):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    X = vol.get_distribution(ticker, '1y')
    i = 0
    for ax in axes.flatten():
        if i==0:
            sns.histplot(X, x='Spot with 1y maturity', hue='Model', element='poly', ax=ax)
            
            ax.set_xlim(left=0, right=X['Spot with 1y maturity'].quantile(0.95))

            ax.set_xlabel('Spot')
            ax.set_ylabel('Frequency')
        else:
            sns.ecdfplot(X, x='Spot with 1y maturity', hue='Model', ax=ax)
            ax.set_xlim(left=0, right=X['Spot with 1y maturity'].quantile(0.95))

            ax.set_xlabel('Spot')
            ax.set_ylabel('Cum. Probability')
        i+=1
    
    fig.suptitle(f'Distribution of {ticker} Spot Path with a 1y Maturity', fontsize=16)
    fig.tight_layout()
    st.pyplot(fig)
    return fig

###################################################################################################################################
#                                                          STARTING POINT                                                         #
###################################################################################################################################

st.title('LSV1F: Bumping Parameters')

_,col_1,_ = st.columns([2,12,2])
with col_1:
    ticker = st.text_input('Enter a Ticker:').upper()
col_1.caption('\*Exemple:  _AAPL_')

ticker_exist = False
agree_nelder_mead=False

if len(ticker)!=0:
    try:
        Nappe_Vol = vol.get_implied_volatility(ticker)
        ticker_exist=True
    except:
        col_1.warning(ticker+' doesn\'t work: _try another US stock_')

if ticker_exist:
    Nappe_Vol = vol.get_implied_volatility(ticker)
    Nappe_Vol.reset_index(inplace=True)
    figure_implied_volatility(Nappe_Vol)
    _,col_1,_ = st.columns([10,12,2])
    with col_1:
        agree_proxy = st.toggle('Get Proxy')

if ticker_exist and agree_proxy:
    st.write('---')
    with st.spinner('Running Proxy ...'):
        proxy_ticker = vol.get_proxy(ticker, 1.5, -0.8, 2)
    figure_proxy(proxy_ticker)
    
    _,col_1,_ = st.columns([10,12,2])
    with col_1:
        agree_nelder_mead = st.toggle('Nelder Mead Algorithm')

if ticker_exist and agree_nelder_mead:
    st.write('---')
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        parallel_user = st.checkbox('Parallel Bump')
    with col_2:
        vvol_user = st.checkbox('Volatility-of-Volatility Bump')
    with col_3:
        correl_user = st.checkbox('Correlation Spot-Volatility Bump')
    
    if parallel_user or vvol_user or correl_user:
        bump_vvol = 0 if correl_user else 0.1
        bump_correl = 0 if vvol_user else 0.1
        with st.spinner('Running Nelder Mead Algorithm ...'):
            nm_params, dict_cost_function = vol.nelder_mead(ticker, bump_vvol, bump_correl)
            algo_ticker = vol.get_proxy(ticker, nm_params[0], nm_params[1], nm_params[2])
        
        params = list(map(lambda x: round(x,3), nm_params))
        df = pd.DataFrame({'':['Old', 'New'], 'Kappa':[1.5, params[0]], 'Rho':[-0.8, params[1]], 'Nu':[2, params[2]]}).set_index('')
        with col_2:
            st.dataframe(df)
        fig_nelder_mead(proxy_ticker, algo_ticker)
        dist_vvol = (algo_ticker['Volatility']-proxy_ticker['Volatility']).mean()*100
        dist_correl = (algo_ticker['Correlation'] - proxy_ticker['Correlation']).mean()*100
        col_1, col_2 = st.columns(2)
        col_1.warning('Distance L1 in Volatility: '+ str(round(dist_vvol,2))+'%', icon="⚠️")
        col_2.warning('Distance L1 in Correlation: '+ str(round(dist_correl,2))+'%', icon="⚠️")
        
        df_cost_function = pd.DataFrame(dict_cost_function.items())
        df_cost_function[['kappa','rho','nu']] = df_cost_function[0].astype(str).str.replace('(','').str.replace(')','').str.split(',',expand=True).astype(float)
        df_cost_function['Loss'] = df_cost_function[1]
        # fig_3d_scatter(df_cost_function)
        # fig_cost_function(df_cost_function)
        fig_convergence(df_cost_function)


if ticker_exist:
    st.write('---')
    st.title('Playground')
    _,col_1,col_2,col_3,col_4,col_5,_ = st.columns([2,12,2,12,2,12,2])
    kappa_user = col_1.text_input('Enter Kappa:')
    col_1.caption('\* Kappa = 1.5')
    rho_user = col_3.text_input('Enter Rho:')
    col_3.caption('\*Rho = -0.8')
    nu_user = col_5.text_input('Enter Nu:')
    col_5.caption('\*Nu = 2')
    if len(kappa_user)!=0 and len(rho_user)!=0 and len(nu_user)!=0:
        with st.spinner('Running Proxy ...'):
            proxy_user = vol.get_proxy(ticker, float(kappa_user), float(rho_user), float(nu_user))
        figure_proxy(proxy_user)
    

if ticker_exist:
    st.write('---')
    st.title('Pricer of Exotic Products w. LSV1F and BS Model')
    
    S_lsv = pd.DataFrame(vol.simulate_lsv_paths(ticker)).reset_index()
    S_bs = pd.DataFrame(vol.simulate_bs_paths(ticker)).reset_index()
    
    fig_spot_path(S_lsv, S_bs)
    
    _,col_1,_ = st.columns(3)
    with col_1:
        toggle_distribution = st.toggle('Get The Distribution of the Stock Paths')
        st.write('')
    if toggle_distribution:
        fig_distribution(ticker)
        pvalue = vol.kolmogorov_test(ticker, '1y')
        if pvalue <=0.05:
            st.write(f'**The pvalue on the Kolmogorov Smirnov Test is : {pvalue} < 5%, so we reject H0 : the two distributions are not equals**')
        else:
            st.write(f'**The pvalue on the Kolmogorov Smirnov Test is : {pvalue} > 5%, so we accept H0 : the two distributions are equals*')
        
        _,col_1,_ = st.columns(3)
        with col_1:
            toggle_details = st.toggle('See The Evolution over [0,1y] period')
        if toggle_details:
            st.write('---')
            fig_distrib(ticker)
            st.write('---')
            fig_cdf(ticker)
        
        st.write('---')
    
    col_1, col_2, col_3 = st.columns(3)
    
    with col_1:
        product = st.selectbox("Choose the product you want to price:", ("", "Call", "Put", "Call Up and Out", "Put Down and In"))
    with col_2:
        maturity = st.selectbox('Select the maturity for the product:', ('','6m', '1y', '2y', '5y', '10y'))
    with col_3:
        barrier = st.text_input('Enter the Barrier for the product _(in %)_:', disabled=True if product in ['Call', 'Put'] else False)
        complete_analysis = st.toggle('Complete Nelder Mead Analysis _(~5min)_')
    
    if product in ['Call', 'Put']:
        barrier=100
    
    if len(product)!=0 and len(maturity)!=0 and barrier!='':
        T = 0.5 if maturity=='6m' else int(maturity[:-1])
        table_analysis = vol.get_prices(ticker, product, T, barrier, complete_analysis)
        if not complete_analysis:
            with col_2:
                st.dataframe(table_analysis)
        else:
            _,col_1,_ = st.columns([2,25,2])
            with col_1:
                st.dataframe(table_analysis)
            

