import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
import base64
from io import BytesIO
from datetime import datetime as dt, timedelta as td


st.set_page_config(page_title="Black-Scholes Pricing Model", layout="wide")

st.title("Black Scholes Pricing Model")
st.markdown("### By Kafui Avevor")

with st.sidebar:
    # Initialise session state for parameters if not already set
    if 'spot_price' not in st.session_state:
        st.session_state.spot_price = 50.00
    if 'strike_price' not in st.session_state:
        st.session_state.strike_price = 55.00
    if 'selected_call_strike' not in st.session_state:
        st.session_state.selected_call_strike = st.session_state.strike_price
    if 'selected_put_strike' not in st.session_state:
        st.session_state.selected_put_strike = st.session_state.strike_price
    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = 5.0
    def next_friday():
        today = dt.today()
        days_until_friday = (4 - today.weekday()) % 7  # 4 represents Friday (0 = Monday, 1 = Tuesday, ..., 4 = Friday)
        return today + td(days=days_until_friday)
    if 'maturity_date' not in st.session_state:
        st.session_state.maturity_date = next_friday()
    if 'time_to_expiry' not in st.session_state: 
        today_str = str(dt.today().date())
        maturity_str = str(st.session_state.maturity_date)
        days_to_expiry = pd.date_range(today_str, maturity_str).size
        st.session_state.time_to_expiry = days_to_expiry / 365
    
    if 'volatility' not in st.session_state:
        st.session_state.volatility = 20.0

    if 'implied_volatility_put' not in st.session_state:
        st.session_state.implied_volatility_put = 20.0
    if 'implied_volatility_call' not in st.session_state:
        st.session_state.implied_volatility_call = 20.0

    if 'currency' not in st.session_state:
        st.session_state.currency = 'USD'

    st.write("#### Fetch Live Data")
    ticker = st.text_input("Enter Stock Ticker", value="AAPL", help="Enter the ticker symbol (e.g., AAPL, MSFT, GOOG)")
    fetch_expirations = st.button("Fetch Available Maturities")

    if fetch_expirations:
        try:
            # Fetch available maturities for the ticker
            stock = yf.Ticker(ticker)
            available_expirations = stock.options  

            if not available_expirations:
                st.error(f"No expiration dates available for {ticker}.")
            else:
                st.session_state.available_expirations = available_expirations
                st.success(f"Available expirations for {ticker} fetched successfully!")
        except Exception as e:
            st.error(f"Error fetching expiration dates: {e}")

    if "available_expirations" in st.session_state:
        st.session_state.maturity_date = st.selectbox(
        "Pick a Maturity Date",
            st.session_state.available_expirations,
            help="Select an expiration date from the available options",
    )

            
        if st.session_state.maturity_date:
            # Fetch options data for the selected expiration date
            try:
                stock = yf.Ticker(ticker)
                option_chain = stock.option_chain(st.session_state.maturity_date)
                call_strike_prices = option_chain.calls['strike'].tolist()
                put_strike_prices = option_chain.puts['strike'].tolist()

                st.session_state.call_strike_prices = call_strike_prices
                st.session_state.put_strike_prices = put_strike_prices

                st.success(f"Strike prices fetched for expiration {st.session_state.maturity_date}.")
            except Exception as e:
                st.error(f"Error fetching strike prices: {e}")


        if "call_strike_prices" in st.session_state:
            st.session_state.selected_call_strike = st.selectbox(
                "Select a Call Strike Price",
                st.session_state.call_strike_prices,
                help="Choose a strike price for call options.",
            )
            st.write(f"You selected call strike price: {st.session_state.selected_call_strike}")
            
        if "put_strike_prices" in st.session_state:
            st.session_state.selected_put_strike = st.selectbox(
            "Select a Put Strike Price",
                st.session_state.put_strike_prices,
                help="Choose a strike price for put options.",
            )
            st.write(f"You selected put strike price: {st.session_state.selected_put_strike}")
    today_str = str(dt.today().date())
    call_strike = st.session_state.selected_call_strike
    put_strike = st.session_state.selected_put_strike
    maturity_date = st.session_state.maturity_date
    
    maturity_str = str(st.session_state.maturity_date)
    days_to_expiry = pd.date_range(today_str, maturity_str).size
    st.session_state.time_to_expiry = days_to_expiry / 365
    


    fetch_live = st.button("Fetch Live Data")
        
    if fetch_live:
        # Function to fetch live data
        @st.cache_data
        def get_live_data(ticker, maturity_date, call_strike, put_strike):
            try:
                stock = yf.Ticker(ticker)
                currency = stock.info['currency']
                hist = stock.history(period="1y")  # 1 year of historical data
                current_price = stock.fast_info["last_price"]
                close_price = hist['Close'][-1]
                options = stock.option_chain(maturity_date)
                puts_volume = options.puts['volume'].sum()
                call_volume = options.calls['volume'].sum()
                put_call_ratio = puts_volume/call_volume if call_volume > 0 else None
                specific_call = options.calls[options.calls['strike'] == call_strike]
                specific_put = options.puts[options.puts['strike'] == put_strike]
                iv_call = specific_call['impliedVolatility'].iloc[0]
                iv_put = specific_put['impliedVolatility'].iloc[0]
                

                

                    
                return {
                    'current_price': current_price,
                    'historical_prices': hist['Close'],
                    'currency': currency,
                    'put_call_ratio': put_call_ratio,
                    'iv_put': iv_put,
                    'iv_call': iv_call,
                   
                }
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {e}")
                return None
            
        def calculate_historical_volatility(historical_prices):
            log_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
            volatility = log_returns.std() * np.sqrt(365)  
            return volatility            
            
        live_data = get_live_data(ticker, maturity_date, call_strike, put_strike)
        if live_data:
            st.session_state.spot_price = live_data['current_price']
            st.session_state.currency = live_data['currency']
            st.session_state.put_call_ratio = live_data['put_call_ratio']
            st.session_state.implied_volatility_call = live_data['iv_call']
            st.session_state.implied_volatility_put = live_data['iv_put']
            st.session_state.volatility = calculate_historical_volatility(live_data['historical_prices']) * 100  # Convert to percentage


            
                
        

        if st.session_state.time_to_expiry <= 1:
            st.session_state.risk_free_rate = st.session_state.risk_free_rate = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] 
        else:
            st.session_state.risk_free_rate = st.session_state.risk_free_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] 
                    
        st.success(f"Live data for {ticker.upper()} fetched successfully!")
        st.write(f"**Current Spot Price:** {st.session_state.currency.upper()} {st.session_state.spot_price:,.2f}")
        st.write(f"**Historical Volatility:** {st.session_state.volatility:,.2f}%")
        st.write(f"**Risk-Free Rate:** {st.session_state.risk_free_rate:,.2f}")
        st.write(f"**Implied Volatility Call:** {st.session_state.implied_volatility_call*100:,.2f}")
        st.write(f"**Implied Volatility Put:** {st.session_state.implied_volatility_put*100:,.2f}")
            
        st.write(f"**Open-Interest Put-Call Ratio:** {st.session_state.put_call_ratio:,.2f}")

        st.write(f"**Maturity Date:** {st.session_state.maturity_date}")
            
        st.write("#### Last 5 Days of Closing Prices")
        st.dataframe(live_data['historical_prices'].tail())
                

                

    st.markdown("---")
    st.header("Heatmap Parameters")
    col1, col2 = st.columns(2)
    min_vol = col1.slider("Min Volatility (%)", 0.00, 100.00, float(st.session_state.volatility) * 0.67, step=0.1)
    max_vol = col2.slider("Max Volatility (%)", 0.00, 100.00, float(st.session_state.volatility) * 1.5, step=0.1)
    min_spot = col1.number_input("Min Spot Price ($)", 0.00, 1000000.00, float(st.session_state.spot_price) * 0.9, step=0.1)
    max_spot = col2.number_input("Max Spot Price ($)", 0.00, 1000000.00, float(st.session_state.spot_price) * 1.1, step=0.1)
    

def black_scholes(spot_price, strike_price, risk_free_rate, time_to_expiry, volatility, option_type="call"):
    # Convert percentages to decimals
    risk_free_rate_decimal = risk_free_rate / 100
    volatility_decimal = volatility / 100

    d1 = (np.log(spot_price / strike_price) + (risk_free_rate_decimal + volatility_decimal**2 / 2) * time_to_expiry) / (volatility_decimal * np.sqrt(time_to_expiry))
    d2 = d1 - volatility_decimal * np.sqrt(time_to_expiry)

    if option_type == "call":
        price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(d2)
    elif option_type == "put":
        price = strike_price * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Please use 'call' or 'put'.")
    return price

call_price = black_scholes(st.session_state.spot_price, st.session_state.selected_call_strike, st.session_state.risk_free_rate, st.session_state.time_to_expiry, st.session_state.implied_volatility_call, option_type="call")
put_price = black_scholes(st.session_state.spot_price, st.session_state.selected_put_strike, st.session_state.risk_free_rate, st.session_state.time_to_expiry, st.session_state.implied_volatility_put, option_type="put")


st.write("### Option Price (European)")
col1, col2 = st.columns(2)
col1.metric(label="European Call Price", value=f"{st.session_state.currency.upper()} {call_price:,.3f}")
col2.metric(label="European Put Price", value=f"{st.session_state.currency.upper()} {put_price:,.3f}")

st.write("### Heatmaps of European Call and Put Prices with Spot Price and Volatility")

spot_range = np.linspace(min_spot, max_spot, 10)  
volatility_range = np.linspace(min_vol, max_vol, 10)

call_prices = np.zeros((len(volatility_range), len(spot_range)))
put_prices = np.zeros((len(volatility_range), len(spot_range)))

for i, vol in enumerate(volatility_range):
    for j, spot in enumerate(spot_range):
        call_prices[i, j] = black_scholes(spot, st.session_state.selected_call_strike, st.session_state.risk_free_rate, st.session_state.time_to_expiry, vol, option_type="call")
        put_prices[i, j] = black_scholes(spot, st.session_state.selected_put_strike, st.session_state.risk_free_rate, st.session_state.time_to_expiry, vol, option_type="put")

fig, (ax_call, ax_put) = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(call_prices, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2),
            yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_call)
ax_call.set_title('Call Option Prices Heatmap')
ax_call.set_xlabel('Spot Price ($)')
ax_call.set_ylabel('Volatility (%)')

sns.heatmap(put_prices, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2),
            yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_put)
ax_put.set_title('Put Option Prices Heatmap')
ax_put.set_xlabel('Spot Price ($)')
ax_put.set_ylabel('Volatility (%)')

plt.tight_layout()
st.pyplot(fig)

def download_heatmap(heatmap_data, title, spot_range, volatility_range):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(volatility_range, 2))
    plt.title(title)
    plt.xlabel('Spot Price ($)')
    plt.ylabel('Volatility (%)')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)  # Close the figure to free memory
    return f'data:image/png;base64,{b64}'

if st.button("Download Call Price Heatmap"):
    call_heatmap_download = download_heatmap(call_prices, "Call Prices Heatmap", spot_range, volatility_range)
    st.markdown(f'<a href="{call_heatmap_download}" download="call_prices_heatmap.png">Download Call Prices Heatmap</a>', unsafe_allow_html=True)

if st.button("Download Put Price Heatmap"):
    put_heatmap_download = download_heatmap(put_prices, "Put Prices Heatmap", spot_range, volatility_range)
    st.markdown(f'<a href="{put_heatmap_download}" download="put_prices_heatmap.png">Download Put Prices Heatmap</a>', unsafe_allow_html=True)

def calculate_greeks(spot_price, strike_price, risk_free_rate, time_to_expiry, volatility):
    # Convert percentages to decimals
    risk_free_rate_decimal = risk_free_rate / 100
    volatility_decimal = volatility / 100

    d1 = (np.log(spot_price / strike_price) + (risk_free_rate_decimal + (volatility_decimal**2) / 2) * time_to_expiry) / (volatility_decimal * np.sqrt(time_to_expiry))
    d2 = d1 - volatility_decimal * np.sqrt(time_to_expiry)

    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (spot_price * volatility_decimal * np.sqrt(time_to_expiry))
    theta_call = (- (spot_price * norm.pdf(d1) * volatility_decimal) / (2 * np.sqrt(time_to_expiry))
                  - risk_free_rate_decimal * strike_price * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(d2))/100
    theta_put = (- (spot_price * norm.pdf(d1) * volatility_decimal) / (2 * np.sqrt(time_to_expiry))
                 + risk_free_rate_decimal * strike_price * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(-d2))/100
    rho_call = strike_price * time_to_expiry * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(d2) / 100  
    rho_put = -strike_price * time_to_expiry * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(-d2) / 100
    vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)   

    return {
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'theta_call': theta_call,
        'theta_put': theta_put,
        'rho_call': rho_call,
        'rho_put': rho_put,
        'vega': vega
    }

greeks_call = calculate_greeks(st.session_state.spot_price, st.session_state.selected_call_strike, st.session_state.risk_free_rate, st.session_state.time_to_expiry, st.session_state.volatility)
greeks_put = calculate_greeks(st.session_state.spot_price, st.session_state.selected_put_strike, st.session_state.risk_free_rate, st.session_state.time_to_expiry, st.session_state.volatility)

# Display the Greeks
st.write("### Greeks")
col1, col2 = st.columns(2)
col1.metric(label="Call Delta", value=f"{greeks_call['delta_call']:,.3f}")
col2.metric(label="Put Delta", value=f"{greeks_put['delta_put']:,.3f}")

col1.metric(label="Call Gamma", value=f"{greeks_call['gamma']:,.3f}")
col2.metric(label="Put Gamma", value=f"{greeks_put['gamma']:,.3f}")

col1.metric(label="Call Vega", value=f"{greeks_call['vega']:,.3f}")
col2.metric(label="Put Vega", value=f"{greeks_put['vega']:,.3f}")

col1.metric(label="Call Theta", value=f"{greeks_call['theta_call']:,.3f}")
col2.metric(label="Put Theta", value=f"{greeks_put['theta_put']:,.3f}")

col1.metric(label="Call Rho", value=f"{greeks_call['rho_call']:,.3f}")
col2.metric(label="Put Rho", value=f"{greeks_put['rho_put']:,.3f}")

st.markdown("---")
st.markdown("### Developed by Kafui Avevor")
st.markdown("### [LinkedIn](https://www.linkedin.com/in/kafui-avevor/) | [GitHub](https://github.com/kafuiavevor)")
