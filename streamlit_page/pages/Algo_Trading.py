import numpy as np
import pandas as pd
import datetime as dt
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import pytz
import requests
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

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
#                                                           GLOBAL VARIABLES                                                      #
###################################################################################################################################


API_KEY_ID     = "PKL9BT6QPZNC5690ELR5"
API_SECRET_KEY = "8N12VHY2aZ4qbMHar3gvWdDJpaws5fRZMyQZO3Bf"

mapping_pair = {'':'', 'COPPER' : ('COPX', 'CPER'), 'OIL' : ('XLE', 'USO'), 'GOLD' : ('GDX', 'GLD'), 'SILVER' : ('SIL', 'SLV')}

TD = dt.datetime.today()

backtest_start = (TD - relativedelta(days=180)).strftime("%Y-%m-%d")
rolling_beta_horizon = 60
NNN = 1e6

n_std_entry = 1.25
first_limit = (1.75 - n_std_entry)
first_limit_qty = 1

second_limit = (2.5 - n_std_entry)
second_limit_qty = 2

limit_take_profit = -(0.25 - n_std_entry)
limit_stop_loss = (3.25 - n_std_entry)

list_period = [5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180]
mapping_position = {'entries' : 'yellow', 'adds' : 'orange', 'stop_loss' : 'red', 'take_profit' : 'lightgreen',
                    'take_profit_and_invert' : 'darkgreen'}


###################################################################################################################################
#                                                           PLOT FUNCTIONS                                                        #
###################################################################################################################################


def plot_position(df):
    fig = px.line(df[f'rolling_beta_{rolling_beta_horizon}'], title=f'ROLLING_BETA_{rolling_beta_horizon}')
    fig.update_yaxes({'zerolinecolor' : 'grey'}, range = [0, round(df[f'rolling_beta_{rolling_beta_horizon}'].max() * 1.2, 1)])
    st.plotly_chart(fig, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig


def plot_spread(df, period):
    fig = px.line(df[f'spread_rolling_spread_{period}'], title = f'ROLLING_STD_{period}D', labels={'variable' : 'spread'})
    st.plotly_chart(fig, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig


def plot_bollinger_position(df, dict_position, period):
    df_plot = df[['spread', f'spread_rolling_mean_{period}', 'entry_level_long', 'entry_level_short', 'target_level', 'target_level_adjusted',
                  f'{n_std_entry}std', f'-{n_std_entry}std']]
    fig = go.Figure()
    fig.update_layout(title='Bollinger Bands and Entry Points')

    # Change sets to tuples
    mapping = {
        'spread': ('solid', 'grey'),  # Tuple instead of set
        f'spread_rolling_mean_{period}': ('dash', 'darkgrey'),
        'entry_level_long': ('dot', 'green'),
        'entry_level_short': ('dot', 'red'),
        'target_level': ('dot', 'purple'),
        'target_level_adjusted': ('dot', 'rebeccapurple'),
        f'{n_std_entry}std': ('dash', 'darkgrey'),
        f'-{n_std_entry}std': ('dash', 'darkgrey')
    }
    
    # Add traces to the plot
    for y in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot[y],
            name=y,
            line=dict(color=mapping[y][1], dash=mapping[y][0])
        ))
    
    # Add vertical lines for positions
    for position in dict_position.keys():
        for date in dict_position[position]:
            fig.add_vline(x=date, line_width=2, line_dash='dash', line_color=mapping_position[position])
        
    st.plotly_chart(fig, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig



def plot_pnl(df, dict_position):
    fig = px.line(df[['return']], title='PNL')
    
    for position in dict_position.keys():
        for date in dict_position[position]:
            fig.add_vline(x=date, line_width=2, line_dash='dash', line_color=mapping_position[position])

    st.plotly_chart(fig, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig


def plot_inventory(df, ticker1, ticker2):
    fig = px.line(df[[f'inventory_{ticker1}', f'inventory_{ticker2}']], title='INVENTORY')
    st.plotly_chart(fig, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig


def plot_valorisation(df, ticker1, ticker2):
    fig = px.line(df[[f'valo_{ticker1}', f'valo_{ticker2}', 'valo']], title='ASSETS')
    st.plotly_chart(fig, use_container_width=True, theme='streamlit', width=200, height=700)
    return fig

def plot_all(ticker1, ticker2, period, df, dict_position):
    plot_position(df)
    plot_spread(df, period)
    plot_bollinger_position(df, dict_position, period)
    plot_pnl(df, dict_position)
    plot_inventory(df, ticker1, ticker2)
    plot_valorisation(df, ticker1, ticker2)


###################################################################################################################################
#                                                           RECUPERATE DATA                                                       #
###################################################################################################################################


def fetch_alpaca_data(symbol, timeframe, start_date, end_date):
    """
    Fetch stock bars data from Alpaca Markets based on the given timeframe.
    """
    url = 'https://data.alpaca.markets/v2/stocks/bars'
    headers = {
        'APCA-API-KEY-ID': API_KEY_ID,
        'APCA-API-SECRET-KEY': API_SECRET_KEY
    }

    params = {
        'symbols': symbol,
        'timeframe': timeframe,
        'start': dt.datetime.strptime(start_date, "%Y-%m-%d").isoformat() + 'Z',
        'end': dt.datetime.strptime(end_date, "%Y-%m-%d").isoformat() + 'Z',
        'limit': 10000,
        'adjustment': 'raw',
        'feed': 'sip'
    }

    data_list = []
    eastern = pytz.timezone('America/New_York')
    utc = pytz.utc

    market_open  = dt.time(9, 30)  # Market opens at 9:30 AM
    market_close = dt.time(15, 59)  # Market closes just before 4:00 PM

    print("Starting data fetch...")
    while True:
        # print(f"Fetching data for symbols: {symbol} from {start_date} to {end_date}")
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching data with status code {response.status_code}: {response.text}")
            break

        data = response.json()

        bars = data.get('bars')

        for symbol, entries in bars.items():
            # print(f"Processing {len(entries)} entries for symbol: {symbol}")
            for entry in entries:
                try:
                    utc_time = dt.datetime.fromisoformat(entry['t'].rstrip('Z')).replace(tzinfo=utc)
                    eastern_time = utc_time.astimezone(eastern)

                    # Apply market hours filter for '1Min' timeframe
                    if timeframe == '1Min' and not (market_open <= eastern_time.time() <= market_close):
                        continue  # Skip entries outside market hours

                    data_entry = {
                        'DATE': eastern_time.strftime('%Y-%m-%d'),
                        f'{symbol}': entry['c']
                    }
                    data_list.append(data_entry)
                    # print(f"Appended data for {symbol} at {eastern_time}")
                except Exception as e:
                    # print(f"Error processing entry: {entry}, {e}")
                    continue

        if 'next_page_token' in data and data['next_page_token']:
            params['page_token'] = data['next_page_token']
            print("Fetching next page...")
        else:
            # print("No more pages to fetch.")
            break

    df = pd.DataFrame(data_list)
    print("Data fetching complete.")
    return df


def get_closes_betas(ticker1, ticker2):
    df1 = fetch_alpaca_data(ticker1, '1D', (TD - relativedelta(years=2)).strftime('%Y-%m-%d'), TD.strftime('%Y-%m-%d'))
    df2 = fetch_alpaca_data(ticker2, '1D', (TD - relativedelta(years=2)).strftime('%Y-%m-%d'), TD.strftime('%Y-%m-%d'))
    df = pd.merge(df1, df2, on='DATE')
    df[f'Return_{ticker1}'], df[f'Return_{ticker2}'] = df[f'{ticker1}'].pct_change(), df[f'{ticker2}'].pct_change()

    model, betas = LinearRegression(), []
    for i in range(rolling_beta_horizon, len(df)):
        window = df.iloc[i-rolling_beta_horizon:i].dropna()
        if len(window) == rolling_beta_horizon:
            X, y = window[f'Return_{ticker2}'].values.reshape(-1, 1), window[f'Return_{ticker1}'].values
            betas.append(model.fit(X, y).coef_[0])
        else:
            betas.append(np.nan)
    df[f'rolling_beta_{rolling_beta_horizon}'] = [np.nan] * rolling_beta_horizon + betas
    df[f'rolling_beta_{rolling_beta_horizon}'] = df[f'rolling_beta_{rolling_beta_horizon}'].bfill()
    df.drop(columns=[f'Return_{ticker1}', f'Return_{ticker2}'], axis=1, inplace=True)
    df.set_index('DATE', inplace=True)
    return df


###################################################################################################################################
#                                                           BACKTEST STRATEGY                                                     #
###################################################################################################################################


def backtest(pair, period, beta_adjust=True, std_factor=1):
    ticker1, ticker2 = pair
    df = get_closes_betas(ticker1, ticker2)
    df[[f'{ticker1}_diff', f'{ticker2}_diff']] = ( df[[ticker1, ticker2]] / df[df.index > backtest_start][[ticker1, ticker2]].iloc[0] ) - 1
    df['spread'] = df[f'{ticker1}_diff'] - df[f'{ticker1}_diff'] * (df[f'rolling_beta_{rolling_beta_horizon}'] if beta_adjust else 1)
    df[f'spread_rolling_mean_{period}'] = df['spread'].bfill().rolling(window=period).mean()
    df[f'spread_rolling_spread_{period}'] = df['spread'].bfill().rolling(window=period).std() * std_factor

    df[f'-{n_std_entry}std'] = df[f'spread_rolling_mean_{period}'] - n_std_entry * df[f'spread_rolling_spread_{period}']
    df[f'{n_std_entry}std'] = df[f'spread_rolling_mean_{period}'] + n_std_entry * df[f'spread_rolling_spread_{period}']
    
    df = df[df.index > backtest_start]

    inventory_ticker1, inventory_ticker2 = 0, 0
    cash, mode, mode_level = 0, 0, 0
    entry_std, entry_level, target_level = np.nan, np.nan, np.nan
    entry_levels, target_levels, target_levels_adjusted = {}, {}, {}
    dict_positions = {'entries' : [], 'adds' : [], 'stop_loss' : [], 'take_profit' : [], 'take_profit_and_invert' : []}

    for date, row in df.iterrows():
        live, std, mid = row['spread'], row[f'spread_rolling_spread_{period}'], row[f'spread_rolling_mean_{period}']
        lower, upper = row[f'-{n_std_entry}std'], row[f'{n_std_entry}std']
        ticker1_spot, ticker2_spot = row[ticker1], row[ticker2]
        beta = row[f'rolling_beta_{rolling_beta_horizon}']

        current_inventory_ticker1, current_inventory_ticker2 = inventory_ticker1, inventory_ticker2
        target_level_adjusted = np.nan

        if mode == 0:

            if live > upper:
                inventory_ticker1 = - NNN / ticker1_spot
                inventory_ticker2 = NNN / ticker2_spot * beta
                mode, mode_level = -1, 1
                entry_std, entry_level = std, live
                entry_levels[date] = live
                target_level = live - limit_take_profit * std
                dict_positions['entries'].append(date)
            
            elif live < lower:
                inventory_ticker1 = NNN / ticker1_spot
                inventory_ticker2 = - NNN / ticker2_spot * beta
                mode, mode_level = 1, 1
                entry_std, entry_level = std, live
                entry_levels[date] = live
                target_level = live + limit_take_profit * std
                dict_positions['entries'].append(date)
        
        elif mode == -1:

            target_level_adjusted = max(target_level, 0.5 * (entry_level - limit_take_profit * entry_std) + 0.5 * mid)
            tgt_level = target_level_adjusted if mode_level == 2 else target_level

            if live > entry_level + limit_stop_loss * std:
                inventory_ticker1, inventory_ticker2 = 0, 0
                mode, mode_level = 0, 0
                entry_level, target_level = np.nan, np.nan
                dict_positions['stop_loss'].append(date)
            
            elif live > entry_level + second_limit * entry_std:
                if mode_level==1:
                    inventory_ticker1+= - second_limit_qty * NNN / ticker1_spot
                    inventory_ticker2+= second_limit_qty * NNN / ticker2_spot * beta
                    mode_level = 2
                    dict_positions['adds'].append(date)
                elif mode_level==2:
                    inventory_ticker1+= - first_limit_qty * NNN / ticker1_spot
                    inventory_ticker2+= first_limit_qty * NNN / ticker2_spot * beta
                    mode_level = 3
                    dict_positions['adds'].append(date)
        
            elif live < min(tgt_level, lower):
                inventory_ticker1 = NNN / ticker1_spot
                inventory_ticker2 = - NNN / ticker2_spot * beta
                mode, mode_level = 1, 1
                entry_std, entry_level = std, live
                entry_levels[date] = live
                target_level = live + limit_take_profit * std
                dict_positions['take_profit_and_invert'].append(date)
            
            elif live < entry_level - limit_take_profit * entry_std:
                inventory_ticker1, inventory_ticker2 = 0, 0
                mode, mode_level = 0, 0
                entry_level, target_level = np.nan, np.nan
                dict_positions['take_profit'].append(date)
        
        elif mode == 1:

            target_level_adjusted = min(target_level, 0.5 * (entry_level + limit_take_profit * entry_std) + 0.5 * mid)
            tgt_level = target_level_adjusted if mode_level == 2 else target_level

            if live < entry_level - limit_stop_loss * entry_std:
                inventory_ticker1, inventory_ticker2 = 0, 0
                mode, mode_level = 0, 0
                entry_level, target_level = np.nan, np.nan
                dict_positions['stop_loss'].append(date)
            
            elif live < entry_level - second_limit * entry_std:
                if mode_level == 1:
                    inventory_ticker1 += second_limit_qty * NNN / ticker1_spot
                    inventory_ticker2 += - second_limit_qty * NNN / ticker2_spot * beta
                    mode_level = 2
                    dict_positions['adds'].append(date)
                elif mode_level == 2:
                    inventory_ticker1 += first_limit_qty * NNN / ticker1_spot
                    inventory_ticker2 += - first_limit_qty * NNN / ticker2_spot * beta
                    mode_level = 3
                    dict_positions['adds'].append(date)
            
            elif live < entry_level - first_limit * entry_std:
                if mode_level == 1:
                    inventory_ticker1 += NNN / ticker1_spot
                    inventory_ticker2 += - NNN / ticker2_spot * beta
                    mode_level = 2
                    dict_positions['adds'].append(date)
            
            elif live > max(tgt_level, upper):
                inventory_ticker1 = - NNN / ticker1_spot
                inventory_ticker2 = NNN / ticker2_spot * beta
                mode, mode_level = -1, 1
                entry_std, entry_level = std, live
                entry_levels[date] = entry_level
                target_level = live - limit_take_profit * std
                dict_positions['take_profit_and_invert'].append(date)
            
            elif live > tgt_level:
                inventory_ticker1, inventory_ticker2 = 0, 0
                mode, mode_level = 0, 0
                entry_level, target_level = np.nan, np.nan
                dict_positions['take_profit'].append(date)
        
        target_levels[date] = target_level
        target_levels_adjusted[date] = target_level_adjusted

        exec_ticker1 = inventory_ticker1 - current_inventory_ticker1
        exec_ticker2 = inventory_ticker2 - current_inventory_ticker2
        cash -= exec_ticker1 * ticker1_spot + exec_ticker2 * ticker2_spot
        valo_ticker1 = inventory_ticker1 * ticker1_spot
        valo_ticker2 = inventory_ticker2 * ticker2_spot
        valo = cash + valo_ticker1 + valo_ticker2

        df.loc[date, 'mode'] = mode
        df.loc[date, 'mode_level'] = mode_level
        df.loc[date, 'entry_level'] = entry_level
        df.loc[date, 'target_level'] = target_level
        df.loc[date, 'target_level_adjusted'] = target_level_adjusted
        df.loc[date, f'inventory_{ticker1}'] = inventory_ticker1
        df.loc[date, f'inventory_{ticker2}'] = inventory_ticker2
        df.loc[date, f'valo_{ticker1}'] = valo_ticker1
        df.loc[date, f'valo_{ticker2}'] = valo_ticker2
        df.loc[date, 'cash'] = cash
        df.loc[date, 'valo'] = valo
    
    df['entry_level_long'] = np.where(df['mode'] > 0, df['entry_level'], np.nan)
    df['entry_level_short'] = np.where(df['mode'] < 0, df['entry_level'], np.nan)

    df['valo'] = df['valo'].round(2)
    df['return'] = df['valo'] / NNN
    df['daily_return'] = df['return'] - df['return'].shift(1)
    
    return df, dict_positions


###################################################################################################################################
#                                                          STARTING POINT                                                         #
###################################################################################################################################

st.title('Algorithm Pair Trading')

_,col_1,_,col_2,_ = st.columns([2,12,2,12,2])
with col_1:
    pair_ticker = st.selectbox("Choose the product you want to price:", list(mapping_pair.keys()))
    pair_ticker = mapping_pair[pair_ticker]
with col_2:
    period_user = st.selectbox("Choose the period of the backtest:", list_period)

if pair_ticker!="":
    full_pos, dict_pos = backtest(pair_ticker, period=period_user)
    st.subheader(f"PNL : {round(full_pos['valo'].iloc[-1] / NNN * 100, 2)}%", divider=True)
    plot_all(pair_ticker[0], pair_ticker[1], period_user, full_pos, dict_pos)