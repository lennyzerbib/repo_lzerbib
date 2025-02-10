import requests
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from time import time
import streamlit as st
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.optimize import minimize

from scipy import stats

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Your Alpha Vantage API key
API_KEY = '037PLW93PHVV6N1C'

today = datetime.today()
yesterday = today - timedelta(days=1)

dict_tails = {'3m':125, '6m':250, '9m':375, '1y':500}

###################################################################################################################################
#                                                          GET VOLATILITY                                                         #
###################################################################################################################################


def get_volatility(ticker, period='1y', interval='1d'):
    stock_data = yf.download(ticker, period=period, interval=interval, progress=False)
    stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    volatility = stock_data['Log_Returns'].std()
    annual_volatility = volatility * np.sqrt(252)
    return annual_volatility


def get_spot(ticker_symbol):
    data = yf.download(ticker_symbol, period="1d", progress=False)
    last_close = data['Close'].iloc[-1]
    return round(last_close,2)


def get_implied_volatility(ticker):
    window_size=4
    stock = yf.Ticker(ticker)
    option_dates = stock.options
    
    today = datetime.today().date()
    last_date = datetime.strptime(option_dates[-1], '%Y-%m-%d').date()
    date_range = [last_date - timedelta(days=x) for x in range((last_date - today).days + 1)]
    
    atm_iv_list = []
    market_skew_list = []
    
    monthly_data = {}
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        
        if date_str in option_dates:
            option_chain = stock.option_chain(date_str)
            spot_price = option_chain[2]['regularMarketPrice']
            
            implied_vol = option_chain[0][['strike', 'impliedVolatility']]
            implied_vol['strike'] = implied_vol['strike'].apply(lambda x: x / spot_price * 100)
            
            atm_call = implied_vol.iloc[(implied_vol['strike'] - 100).abs().argsort()[:1]]
            atm_index = atm_call.index[0]
            
            implied_vol_before = implied_vol.iloc[atm_index - 1]['impliedVolatility']
            implied_vol_after = implied_vol.iloc[atm_index + 1]['impliedVolatility']
            strike_before = implied_vol.iloc[atm_index - 1]['strike']
            strike_after = implied_vol.iloc[atm_index + 1]['strike']
            
            market_skew = (implied_vol_after - implied_vol_before) / (strike_after - strike_before)
            
            atm_forward_iv = atm_call['impliedVolatility'].iloc[0]
        else:
            atm_forward_iv = None
            market_skew = None
        
        atm_iv_list.append(atm_forward_iv)
        market_skew_list.append(market_skew)
        
        month_key = (date.year, date.month)
        if month_key not in monthly_data:
            monthly_data[month_key] = {
                'ATMF_Implied_Volatility': [],
                'Market_Skew': [],
                'Date': datetime(date.year, date.month, 1)
            }
        monthly_data[month_key]['ATMF_Implied_Volatility'].append(atm_forward_iv)
        monthly_data[month_key]['Market_Skew'].append(market_skew)

    monthly_df = pd.DataFrame.from_dict(monthly_data, orient='index').reset_index(drop=True)
    monthly_df['ATMF_Implied_Volatility'] = monthly_df['ATMF_Implied_Volatility'].apply(lambda x: pd.Series(x).mean())
    monthly_df['Market_Skew'] = monthly_df['Market_Skew'].apply(lambda x: 100*pd.Series(x).mean())
    
    monthly_df['ATMF_Implied_Volatility'] = monthly_df['ATMF_Implied_Volatility'].rolling(window=window_size, min_periods=1).mean()
    monthly_df['Market_Skew'] = monthly_df['Market_Skew'].rolling(window=window_size, min_periods=1).mean()
    
    monthly_df.set_index('Date', inplace=True)
    monthly_df = monthly_df.interpolate(method='linear')
    monthly_df = monthly_df.iloc[::-1]
    
    return monthly_df


###################################################################################################################################
#                                                    GET OUR APPROXIMATIONS                                                       #
###################################################################################################################################


@st.cache_data(persist='disk')
def get_proxy(ticker, kappa=1.5, rho=-0.8, nu=2):
    sigma_0 = get_volatility(ticker)
    datas = get_implied_volatility(ticker)
    datas.reset_index(inplace=True)
    sigma_hat = datas['ATMF_Implied_Volatility'].tolist()
    sigma_hat.insert(0,0)
    market_skew = datas['Market_Skew'].tolist()
    market_skew.insert(0,0)
    t_values = [(i+1)/365 for i in range(len(datas))]
    datas = datas.assign(Time_Values=t_values)
    t_values.insert(0,0)
    
    SV_skew = [rho*nu/2 for i in range(len(datas))]
    datas = datas.assign(skew_SV = SV_skew)

    l_up = [datas['ATMF_Implied_Volatility'][0]*np.exp(-kappa*datas['Time_Values'][0])*datas['Time_Values'][0]]
    l_down = [datas['ATMF_Implied_Volatility'][0]*datas['Time_Values'][0]]
    for i in range(1,len(datas)):
        l_up.append(l_up[i-1]+datas['ATMF_Implied_Volatility'][i]*np.exp(-kappa*datas['Time_Values'][i])*(datas['Time_Values'][i] - datas['Time_Values'][i-1]))
        l_down.append(l_up[i-1]+datas['ATMF_Implied_Volatility'][i]*(datas['Time_Values'][i] - datas['Time_Values'][i-1]))
    A = [x/y for x,y in zip(l_up, l_down)]
    
    datas = datas.assign(A_k_T=A)
    
    sum_B = [np.sqrt(datas['ATMF_Implied_Volatility'][0])*(datas['Market_Skew'][0]-datas['skew_SV'][0])*datas['Time_Values'][0]]
    for i in range(1,len(datas)):
        sum_B.append(sum_B[i-1]+np.sqrt(datas['ATMF_Implied_Volatility'][i])*(datas['Market_Skew'][i]-datas['skew_SV'][i])*(datas['Time_Values'][i] - datas['Time_Values'][i-1]))
    
    B = []
    for i in range(len(datas)):
        B.append(datas['Market_Skew'][i] - datas['skew_SV'][i] + 1/(datas['Time_Values'][i]*np.sqrt(datas['ATMF_Implied_Volatility'][i]))*sum_B[i])
    
    datas = datas.assign(B_k_T = B)
    
    vvol = [np.sqrt(datas['B_k_T'][i]**2 * sigma_0/datas['ATMF_Implied_Volatility'][i] + (nu*datas['A_k_T'][i])**2 + 2*rho*nu*datas['B_k_T'][i]*datas['A_k_T'][i]*np.sqrt(sigma_0/datas['ATMF_Implied_Volatility'][i])) for i in range(len(datas))]
    corr = [(datas['B_k_T'][i]*np.sqrt(sigma_0/datas['ATMF_Implied_Volatility'][i]) + rho*nu*datas['A_k_T'][i])/vvol[i] for i in range(len(datas))]
    
    datas = datas.assign(Volatility=vvol)
    datas = datas.assign(Correlation=corr)
    
    return datas


###################################################################################################################################
#                                                    BUMPING ALGORITHM : NELDER MEAD                                              #
###################################################################################################################################


@st.cache_data(persist='disk')
def nelder_mead(ticker, bump_vvol=0.1, bump_correl=0.1):
    model = get_proxy(ticker)
    vovol = np.array(model['Volatility']+bump_vvol)
    correl = np.array(model['Correlation']+bump_correl)
    dict_res = {}
    bounds = [(-np.inf, np.inf), (-1, 1), (-np.inf, np.inf)]
    
    def cost_function(x):
        x = list(map(lambda x:round(x,3), x))
        print('New Iteration : ', x)
        algo = get_proxy(ticker, x[0], x[1], x[2])
        vovol_algo = np.array(algo['Volatility'])
        correl_algo = np.array(algo['Correlation'])
        dist_vvol = distance.minkowski(vovol, vovol_algo, 2)
        dist_correl = distance.minkowski(correl, correl_algo, 2)
        dict_res[tuple(x)] = dist_vvol + dist_correl
        return dist_vvol + dist_correl

    def constraint_1(x):
        return 2*model['Market_Skew'][0] - x[1]*x[2]/2
    
    def constraint_2(x):
        return x[1]*x[2]/2 - model['Market_Skew'][0]
    
    def constraint_3(x):
        return 1 + 2*np.sqrt(model['ATMF_Implied_Volatility'][0])*x[1]*x[2]/x[0]
    
    constraints = [{'type':'ineq', 'fun':constraint_1}, {'type':'ineq', 'fun':constraint_2}, {'type':'ineq', 'fun':constraint_3}]
    
    res = minimize(cost_function, [1.5, -0.8, 2], method='Nelder-Mead', bounds=bounds, constraints=constraints, options={'disp':True, 'maxiter':100})
    return res.x, dict_res


###################################################################################################################################
#                                                    SIMULATE SPOT PATH IN LSV                                                    #
###################################################################################################################################


#Simulating Ornstein-Uhlenbeck Processes
def simulate_ou_process(kappa, rho, nu, T=1, M=500, N=10000):
    dt = T / M
    X = np.zeros((M + 1, N))
    dZ = rho * np.random.normal(size=(M, N)) + np.sqrt(1 - rho ** 2) * np.random.normal(size=(M, N))
    
    for t in range(1, M + 1):
        X[t, :] = X[t - 1, :] - kappa * X[t - 1, :] * dt + np.sqrt(dt) * dZ[t - 1, :]
    
    return X


def simulate_lsv_paths(ticker, r=0.025, q=0, kappa=1.5, rho=-0.8, nu=2, T=1, M=500, N=10000):
    
    S = np.zeros((M + 1, N))
    S0 = get_spot(ticker)
    S[0, :] = S0
    dt = T/M

    # Simulate the OU process
    X_t = simulate_ou_process(kappa, rho, nu, T, M, N)

    # Generate correlated Brownian motions for the stock price
    dW = np.random.normal(size=(M, N)) * np.sqrt(dt)
    
    #Recuperating the Volatility for this specitified Ticker
    vol_loc = get_volatility(ticker)

    for t in range(1, M + 1):
        xi_t = np.exp(2 * nu* X_t[t-1, :])  # Exponential function of the OU process
        vol_sto = np.sqrt(xi_t)  # The instantaneous volatility is the square root of xi_t
        S[t, :] = S[t - 1, :] * (np.exp((r - q) * dt + vol_loc / vol_sto * np.exp(nu * X_t[t-1,:]) * dW[t - 1, :]))
    
    return S


def plot_spot_path_lsv(ticker, kappa=1.5, rho=-0.8, nu=2):
    S_lsv = simulate_lsv_paths(ticker, kappa=kappa, rho=rho, nu=nu)
    plt.plot(S_lsv[:,:100])
    plt.title('Simulated Stock Price Paths with LSV Model for '+ticker)
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    plt.show()


###################################################################################################################################
#                                                    SIMULATE SPOT PATH IN BLACK-SCHOLES                                          #
###################################################################################################################################


def simulate_bs_paths(ticker, r=0.025, T=1, M=500, N=10000):
    dt = T / M
    S0 = get_spot(ticker)
    sigma = get_volatility(ticker)
    
    S = np.zeros((M + 1, N))
    S[0, :] = S0

    # Generate increments of a standard Brownian motion
    dW = np.random.normal(0, 1, (M, N)) * np.sqrt(dt)

    for t in range(1, M + 1):
        S[t, :] = S[t - 1, :] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW[t - 1, :])

    return S


def plot_spot_path_bs(ticker, kappa=1.5, rho=-0.8, nu=2):
    S_bs = simulate_bs_paths(ticker)
    plt.plot(S_bs[:,:100])
    plt.title('Simulated Stock Price Paths with B-S Model for '+ticker)
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    plt.show()


###################################################################################################################################
#                                                PREPARE DATA FOR THE FAT TAILS DISTRIBUTION                                      #
###################################################################################################################################


def get_distribution(ticker, maturity):
    idx = dict_tails[maturity]
    S_lsv = pd.DataFrame(pd.DataFrame(simulate_lsv_paths(ticker)).iloc[idx])
    S_lsv['Model'] = 'LSV'

    S_bs = pd.DataFrame(pd.DataFrame(simulate_bs_paths(ticker)).iloc[idx])
    S_bs['Model'] = 'BS'

    X = pd.concat([S_lsv, S_bs], axis=0)
    X = X.rename({idx:f'Spot with {maturity} maturity'}, axis=1)
    return X


def kolmogorov_test(ticker, maturity):
    distrib = get_distribution(ticker, maturity)
    lsv = distrib.loc[distrib['Model']=='LSV', f'Spot with {maturity} maturity']
    bs = distrib.loc[distrib['Model']=='BS', f'Spot with {maturity} maturity']
    return stats.kstest(lsv, bs, alternative='two-sided').pvalue
    
    

###################################################################################################################################
#                                                PRICER WITH LSV1F Model : Two Methods LSV & BS                                   #
###################################################################################################################################


def call(ticker, method, r=0.025, T=1, kappa=1.5, rho=-0.8, nu=2):
    S = simulate_lsv_paths(ticker, r=r, T=T, kappa=kappa, rho=rho, nu=nu) if method=='LSV' else simulate_bs_paths(ticker, r=r, T=T)
    K = get_spot(ticker)
    payoff = np.maximum(S[-1, :] - K, 0)
    # Discount the expected payoff back to present value
    option_value = np.exp(-r * T) * np.mean(payoff)
    return option_value


def put(ticker, method, r=0.025, T=1, kappa=1.5, rho=-0.8, nu=2):
    S = simulate_lsv_paths(ticker, r=r, T=T, kappa=kappa, rho=rho, nu=nu) if method=='LSV' else simulate_bs_paths(ticker, r=r, T=T)
    K = get_spot(ticker)
    payoff = np.maximum(K - S[-1, :], 0)
    # Discount the expected payoff back to present value
    option_value = np.exp(-r * T) * np.mean(payoff)
    return option_value


def call_up_out(ticker, method, barrier, r=0.025, T=1, kappa=1.5, rho=-0.8, nu=2):
    S_paths = simulate_lsv_paths(ticker, r=r, T=T, kappa=kappa, rho=rho, nu=nu) if method=='LSV' else simulate_bs_paths(ticker, r=r, T=T)
    K = get_spot(ticker)
    B_up = barrier * K
    is_touching_barrier = (S_paths > B_up).any(axis=0)
    payoff = np.maximum(S_paths[-1, :] - K, 0)  # Standard call payoff at maturity
    payoff[is_touching_barrier] = 0  # Set payoff to 0 if the barrier was crossed
    option_value = np.exp(-r * T) * np.mean(payoff)
    return option_value


def put_down_in(ticker, method, barrier, r=0.025, T=1, kappa=1.5, rho=-0.8, nu=2):
    S_paths = simulate_lsv_paths(ticker, r=r, T=T, kappa=kappa, rho=rho, nu=nu) if method == 'LSV' else simulate_bs_paths(ticker, r=r, T=T)
    K = get_spot(ticker)  # Fetch the current spot price to use as strike price
    B_down = barrier * K  # Calculate the absolute barrier level based on current spot price
    is_touching_barrier = (S_paths <= B_down).any(axis=0)
    
    # Calculate the payoff for paths where the barrier was touched
    payoff = np.maximum(K - S_paths[-1, :], 0)  # Standard put payoff at maturity
    payoff[~is_touching_barrier] = 0  # Set payoff to 0 for paths where the barrier was not touched
    
    # Calculate the option value by averaging the payoffs of the paths that touched the barrier and discounting to present value
    if is_touching_barrier.any():
        option_value = np.exp(-r * T) * np.mean(payoff[is_touching_barrier])
    else:
        option_value = 0

    return option_value


###################################################################################################################################
#                                                              GET TABLE WITH ALL PRICES                                          #
###################################################################################################################################


@st.cache_data(persist='disk')
def get_prices(ticker, product, T, barrier, complete=False):
    
    #PX LSV1F
    px_lsv=0
    for i in range(5):
        px_lsv += call(ticker, 'LSV', T=T) if product == 'Call' \
            else call_up_out(ticker, 'LSV', barrier=float(barrier) / 100) if product == 'Call Up and Out' \
            else put(ticker, 'LSV', T=T) if product == 'Put' \
            else put_down_in(ticker, 'LSV', barrier=float(barrier))
    px_lsv = px_lsv/5
    
    #PX BS
    px_bs = 0
    while px_lsv < px_bs or px_bs == 0:
        px_bs += call(ticker, 'BS', T=T) if product == 'Call' \
            else call_up_out(ticker, 'BS', barrier = float(barrier) / 100) if product == 'Call Up and Out' \
            else put(ticker, 'BS', T=T) if product == 'Put' \
            else put_down_in(ticker, 'BS', barrier = float(barrier) / 100)
    
    impact_lsv = px_lsv - px_bs
    
    if not complete:
        return pd.DataFrame({'': ['PX'], 'LSV1F': [px_lsv], 'BS': [px_bs], 'Impact LSV':[impact_lsv]}).set_index('')
    
    else:
        with st.spinner('Running Parallel Bump'):
            param_parallel = nelder_mead(ticker, 0.1, 0.1)[0]
        with st.spinner('Running Volatility-of-Volatility Bump'):
            param_vovol = nelder_mead(ticker, 0.1, 0)[0]
        with st.spinner('Running Correlation Spot-Volatility Bump:'):
            param_correl= nelder_mead(ticker, 0, 0.1)[0]
        
        #PX PARALLEL BUMP
        px_lsv_parallel = 0
        for i in range(5):
            px_lsv_parallel += call(ticker, 'LSV', T=T, kappa=param_parallel[0], rho=param_parallel[1], nu=param_parallel[2]) if product == 'Call' \
                else call_up_out(ticker, 'LSV', float(barrier) / 100, kappa=param_parallel[0], rho=param_parallel[1], nu=param_parallel[2]) if product == 'Call Up and Out' \
                else put(ticker, 'LSV', T=T, kappa=param_parallel[0], rho=param_parallel[1], nu=param_parallel[2]) if product == 'Put' \
                else put_down_in(ticker, 'LSV', barrier=float(barrier) / 100, kappa=param_parallel[0], rho=param_parallel[1], nu=param_parallel[2])
        px_lsv_parallel = px_lsv_parallel/5
        
        #PX VVOL BUMP
        px_lsv_vvol = 0
        for i in range(5):
            px_lsv_vvol += call(ticker, 'LSV', T=T, kappa=param_vovol[0], rho=param_vovol[1], nu=param_vovol[2]) if product == 'Call' \
                else call_up_out(ticker, 'LSV', barrier=float(barrier) / 100, kappa=param_correl[0], rho=param_correl[1], nu=param_vovol[2]) if product == 'Call Up and Out' \
                else put(ticker, 'LSV', T=T, kappa=param_vovol[0], rho=param_vovol[1], nu=param_vovol[2]) if product == 'Put' \
                else put_down_in(ticker, 'LSV', barrier=float(barrier) / 100, kappa=param_vovol[0], rho=param_vovol[1], nu=param_vovol[2])
        px_lsv_vvol = px_lsv_vvol/5
        
        #PX CORREL BUMP
        px_lsv_correl = 0
        for i in range(5):
            px_lsv_correl += call(ticker, 'LSV', T=T, kappa=param_correl[0], rho=param_correl[1], nu=param_correl[2]) if product == 'Call' \
                else call_up_out(ticker, 'LSV', barrier=float(barrier) / 100, kappa=param_correl[0], rho=param_correl[1], nu=param_correl[2]) if product == 'Call Up and Out' \
                else put(ticker, 'LSV', T=T, kappa=param_correl[0], rho=param_correl[1], nu=param_correl[2]) if product == 'Put' \
                else put_down_in(ticker, 'LSV', barrier=float(barrier) / 100, kappa=param_correl[0], rho=param_correl[1], nu=param_correl[2])
        px_lsv_correl = px_lsv_correl/5
        
        impact_parallel = px_lsv_parallel - px_lsv
        impact_vvol = px_lsv_vvol - px_lsv
        impact_correl = px_lsv_correl - px_lsv
        df = pd.DataFrame({'': ['PX'], 'LSV1F': [px_lsv], 'BS': [px_bs], 'Parallel':[px_lsv_parallel], 'Volatility':[px_lsv_vvol],
                           'Correlation':[px_lsv_correl], 'Impact LSV':[impact_lsv], 'Impact Parallel':[impact_parallel],
                           'Impact Volatility':[impact_vvol], 'Impact Correlation':[impact_correl]}).set_index('')
        return df


