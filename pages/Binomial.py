import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
import base64
from io import BytesIO
from datetime import datetime as dt
from datetime import timedelta as td 

st.set_page_config(page_title="Binomial Pricing Model", layout="wide")

st.title("Binomial Pricing Model")
st.markdown("### By Kafui Avevor")


with st.sidebar:

    if 'spot_price' not in st.session_state:
        st.session_state.spot_price = 50.00
    if 'strike_price' not in st.session_state:
        st.session_state.strike_price = st.session_state.spot_price + 5
    if 'selected_call_strike' not in st.session_state:
        st.session_state.selected_call_strike = st.session_state.strike_price
    if 'selected_put_strike' not in st.session_state:
        st.session_state.selected_put_strike = st.session_state.strike_price
    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = 5.0
    def next_friday():
        today = dt.today()
        days_until_friday = (4 - today.weekday()) % 7 
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
    if 'number_of_steps' not in st.session_state:
        st.session_state.number_of_steps = 100
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

    # Show dropdown only if expirations are fetched
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

        # Dropdown for put strike prices
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
        def get_live_data(ticker, maturity_date):
            try:
                stock = yf.Ticker(ticker)
                currency = stock.info['currency']
                hist = stock.history(period="1y")  # 1 year of historical data
                close_price = hist['Close'][-1]
                current_price = stock.fast_info["last_price"]
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
            volatility = log_returns.std() * np.sqrt(252)  # Annualise
            return volatility
                    
            
        live_data = get_live_data(ticker, maturity_date)
        if live_data:
            st.session_state.spot_price = live_data['current_price']
            st.session_state.currency = live_data['currency']
            st.session_state.put_call_ratio = live_data['put_call_ratio']
            st.session_state.implied_volatility_call = live_data['iv_call']
            st.session_state.implied_volatility_put = live_data['iv_put']
            st.session_state.volatility = calculate_historical_volatility(live_data['historical_prices']) * 100 
            

            
                

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
    min_vol = col1.slider("Min Volatility (%)", 0.00, 100.00, float(st.session_state.volatility) * 0.5, step=0.1)
    max_vol = col2.slider("Max Volatility (%)", 0.00, 100.00, float(st.session_state.volatility) * 1.5, step=0.1)
    min_spot = col1.number_input("Min Spot Price ($)", 0.00, 1000000.00, float(st.session_state.spot_price) * 0.9, step=0.1)
    max_spot = col2.number_input("Max Spot Price ($)", 0.00, 1000000.00, float(st.session_state.spot_price) * 1.1, step=0.1)

# Binomial Model for American Options
def binomial_american_option(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, number_of_steps, option_type='call'):
    """
    Binomial model for pricing American options.
    """
    # Calculate the time step
    dt = time_to_expiry / number_of_steps
    # Up and down factors
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    # Risk-neutral probability
    p = (np.exp(risk_free_rate * dt) - d) / (u - d)
    
    # Initialise asset prices at maturity
    asset_prices = np.array([spot_price * (u ** i) * (d ** (number_of_steps - i)) for i in range(number_of_steps + 1)])
    
    # Initialise option values at maturity
    if option_type == 'call':
        option_values = np.maximum(asset_prices - strike_price, 0)
    elif option_type == 'put':
        option_values = np.maximum(strike_price - asset_prices, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Step back through the tree
    for j in range(number_of_steps - 1, -1, -1):
        for i in range(j + 1):
            # Calculate asset price at node (j, i)
            S = spot_price * (u ** i) * (d ** (j - i))
            # Continuation value
            hold_value = np.exp(-risk_free_rate * dt) * (p * option_values[i + 1] + (1 - p) * option_values[i])
            # Early exercise value
            if option_type == 'call':
                exercise_value = max(S - strike_price, 0)
            else:  # put
                exercise_value = max(strike_price - S, 0)
            # Option value at node (j, i)
            option_values[i] = max(hold_value, exercise_value)
    
    return option_values[0]

american_call_price = binomial_american_option(st.session_state.spot_price, st.session_state.selected_call_strike,  st.session_state.time_to_expiry,st.session_state.risk_free_rate/100, st.session_state.implied_volatility_call/100, st.session_state.number_of_steps, option_type="call")
american_put_price =  binomial_american_option(st.session_state.spot_price, st.session_state.selected_put_strike, st.session_state.time_to_expiry, st.session_state.risk_free_rate/100,  st.session_state.implied_volatility_put/100, st.session_state.number_of_steps, option_type="put")



st.write("### Option Price (American)")
col1, col2 = st.columns(2)
col1.metric(label="American Call Price", value=f"{st.session_state.currency.upper()} {american_call_price:,.3f}")
col2.metric(label="American Put Price", value=f"{st.session_state.currency.upper()} {american_put_price:,.3f}")

st.write("### Heatmaps of American Call and Put Prices with Spot Price and Volatility")
spot_range = np.linspace(min_spot, max_spot, 10)  
volatility_range = np.linspace(min_vol, max_vol, 10)
call_prices = np.zeros((len(volatility_range), len(spot_range)))
put_prices = np.zeros((len(volatility_range), len(spot_range)))
for i, vol in enumerate(volatility_range):
    for j, spot in enumerate(spot_range):
        call_prices[i, j] = binomial_american_option(spot, st.session_state.selected_call_strike,  st.session_state.time_to_expiry,st.session_state.risk_free_rate/100, vol/100, st.session_state.number_of_steps, option_type="call")
        put_prices[i, j] = binomial_american_option(spot, st.session_state.selected_put_strike,  st.session_state.time_to_expiry,st.session_state.risk_free_rate/100, vol/100, st.session_state.number_of_steps, option_type="put")


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
