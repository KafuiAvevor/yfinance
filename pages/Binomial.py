import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Binomial Pricing Model", layout="wide")

st.title("Binomial Pricing Model")
st.markdown("### By Kafui Avevor")


# Sidebar Inputs
import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm

# Sidebar Inputs
with st.sidebar:
    st.write("### Data Input Method")
    data_input_method = st.radio("Choose Data Input Method", ("Manual Input", "Live Data from Yahoo Finance"))

    # Initialize session state for parameters if not already set
    if 'spot_price' not in st.session_state:
        st.session_state.spot_price = 50.00
    if 'strike_price' not in st.session_state:
        st.session_state.strike_price = 55.00
    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = 5.0
    if 'time_to_expiry' not in st.session_state:
        st.session_state.time_to_expiry = 1.00
    if 'volatility' not in st.session_state:
        st.session_state.volatility = 20.0
    if 'number_of_steps' not in st.session_state:
        st.session_state.number_of_steps = 100

    if data_input_method == "Manual Input":
        st.write("#### Manual Input Parameters")
        col1, col2 = st.columns(2)
        st.session_state.spot_price = col1.number_input("Spot Price ($)", min_value=0.00, value=st.session_state.spot_price, step=0.1, help="Current price of the underlying asset")
        st.session_state.strike_price = col1.number_input("Strike Price ($)", min_value=0.00, value=st.session_state.strike_price, step=0.1, help="Strike price of the option")
        st.session_state.risk_free_rate = col1.number_input("Risk Free Rate (%)", min_value=0.00, value=st.session_state.risk_free_rate, step=0.1, help="Annual risk-free interest rate in percentage (e.g., 5 for 5%)")
        st.session_state.time_to_expiry = col2.number_input("Time to Expiry (Years)", min_value=0.00, value=st.session_state.time_to_expiry, step=0.1, help="Time until option expires in years")
        st.session_state.volatility = col2.number_input("Volatility (%)", min_value=0.00, value=st.session_state.volatility, step=0.1, help="Annualized volatility in percentage (e.g., 20 for 20%)")
        st.session_state.number_of_steps = col2.number_input("Number of Steps for Heatmap", min_value=1, value=st.session_state.number_of_steps, step=1, help="Number of steps")
   
    else:
        st.write("#### Fetch Live Data")
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", help="Enter the ticker symbol (e.g., AAPL, MSFT, GOOG)")
        fetch_live = st.button("Fetch Live Data")
        if fetch_live:
            # Function to fetch live data
            @st.cache_data
            def get_live_data(ticker):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1y")  # 1 year of historical data
                    current_price = hist['Close'][-1]
                    return {
                        'current_price': current_price,
                        'historical_prices': hist['Close']
                    }
                except Exception as e:
                    st.error(f"Error fetching data for {ticker}: {e}")
                    return None

            live_data = get_live_data(ticker)
            if live_data:
                st.session_state.spot_price = live_data['current_price']
                
                # Function to calculate historical volatility
                def calculate_historical_volatility(historical_prices):
                    log_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
                    volatility = log_returns.std() * np.sqrt(252)  # Annualize
                    return volatility

                st.session_state.volatility = calculate_historical_volatility(live_data['historical_prices']) * 100  # Convert to percentage

                st.success(f"Live data for {ticker.upper()} fetched successfully!")
                st.write(f"**Current Spot Price:** ${st.session_state.spot_price:,.2f}")
                st.write(f"**Historical Volatility:** {st.session_state.volatility:,.2f}%")
                st.write("#### Last 5 Days of Closing Prices")
                st.dataframe(live_data['historical_prices'].tail())

                # Allow user to input or adjust other parameters
                col1, col2 = st.columns(2)
                st.session_state.strike_price = col1.number_input("Strike Price ($)", min_value=0.00, value=st.session_state.spot_price, step=0.1, help="Strike price of the option")
                st.session_state.risk_free_rate = col1.number_input("Risk Free Rate (%)", min_value=0.00, value=st.session_state.risk_free_rate, step=0.1, help="Annual risk-free interest rate in percentage (e.g., 5 for 5%)")
                st.session_state.time_to_expiry = col2.number_input("Time to Expiry (Years)", min_value=0.00, value=st.session_state.time_to_expiry, step=0.1, help="Time until option expires in years")

    st.markdown("---")
    st.header("Heatmap Parameters")
    col1, col2 = st.columns(2)
    min_vol = col1.slider("Min Volatility (%)", 0.00, 100.00, float(st.session_state.volatility) * 0.5, step=0.1)
    max_vol = col2.slider("Max Volatility (%)", 0.00, 100.00, float(st.session_state.volatility) * 1.5, step=0.1)
    min_spot = col1.number_input("Min Spot Price ($)", 0.00, 1000000.00, float(st.session_state.spot_price) * 0.5, step=0.1)
    max_spot = col2.number_input("Max Spot Price ($)", 0.00, 1000000.00, float(st.session_state.spot_price) * 1.5, step=0.1)

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
    
    # Initialize asset prices at maturity
    asset_prices = np.array([spot_price * (u ** i) * (d ** (number_of_steps - i)) for i in range(number_of_steps + 1)])
    
    # Initialize option values at maturity
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

# Calculate Prices
american_call_price = binomial_american_option(st.session_state.spot_price, st.session_state.strike_price,  st.session_state.time_to_expiry,st.session_state.risk_free_rate/100, st.session_state.volatility/100, st.session_state.number_of_steps, option_type="call")
american_put_price =  binomial_american_option(st.session_state.spot_price, st.session_state.strike_price, st.session_state.time_to_expiry, st.session_state.risk_free_rate/100,  st.session_state.volatility/100, st.session_state.number_of_steps, option_type="put")


# Display the option price
st.write("### Option Price (American)")
col1, col2 = st.columns(2)
col1.metric(label="American Call Price", value=f"${american_call_price:,.3f}")
col2.metric(label="American Put Price", value=f"${american_put_price:,.3f}")

# Generate the heatmap data (for Call and Put Prices with different Spot Prices and Volatilities)
st.write("### Heatmaps of American Call and Put Prices with Spot Price and Volatility")

# Define heatmap resolution based on user input
spot_range = np.linspace(min_spot, max_spot, 10)  
volatility_range = np.linspace(min_vol, max_vol, 10)

# Create 2D arrays for call and put prices based on spot prices and volatilities
call_prices = np.zeros((len(volatility_range), len(spot_range)))
put_prices = np.zeros((len(volatility_range), len(spot_range)))

# Calculate call and put prices for each combination of volatility and spot price
for i, vol in enumerate(volatility_range):
    for j, spot in enumerate(spot_range):
        call_prices[i, j] = binomial_american_option(spot, st.session_state.strike_price,  st.session_state.time_to_expiry,st.session_state.risk_free_rate/100, vol/100, st.session_state.number_of_steps, option_type="call")
        put_prices[i, j] = binomial_american_option(spot, st.session_state.strike_price,  st.session_state.time_to_expiry,st.session_state.risk_free_rate/100, vol/100, st.session_state.number_of_steps, option_type="put")

# Plotting heatmaps
fig, (ax_call, ax_put) = plt.subplots(1, 2, figsize=(20, 8))

# Plot the heatmap for Call Prices on the first subplot
sns.heatmap(call_prices, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2),
            yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_call)
ax_call.set_title('Call Option Prices Heatmap')
ax_call.set_xlabel('Spot Price ($)')
ax_call.set_ylabel('Volatility (%)')

# Plot the heatmap for Put Prices on the second subplot
sns.heatmap(put_prices, annot=True, fmt=".2f", xticklabels=np.round(spot_range, 2),
            yticklabels=np.round(volatility_range, 2), cmap="RdYlGn", ax=ax_put)
ax_put.set_title('Put Option Prices Heatmap')
ax_put.set_xlabel('Spot Price ($)')
ax_put.set_ylabel('Volatility (%)')

# Adjust layout and display heatmaps
plt.tight_layout()
st.pyplot(fig)

# Function to download heatmaps
def download_heatmap(heatmap_data, title, spot_range, volatility_range):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(volatility_range, 2))
    plt.title(title)
    plt.xlabel('Spot Price ($)')
    plt.ylabel('Volatility (%)')
    plt.tight_layout()

    # Save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode to base64
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)  # Close the figure to free memory
    return f'data:image/png;base64,{b64}'

# Add download buttons
if st.button("Download Call Price Heatmap"):
    call_heatmap_download = download_heatmap(call_prices, "Call Prices Heatmap", spot_range, volatility_range)
    st.markdown(f'<a href="{call_heatmap_download}" download="call_prices_heatmap.png">Download Call Prices Heatmap</a>', unsafe_allow_html=True)

if st.button("Download Put Price Heatmap"):
    put_heatmap_download = download_heatmap(put_prices, "Put Prices Heatmap", spot_range, volatility_range)
    st.markdown(f'<a href="{put_heatmap_download}" download="put_prices_heatmap.png">Download Put Prices Heatmap</a>', unsafe_allow_html=True)

# Calculate the Greeks
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
                  - risk_free_rate_decimal * strike_price * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(d2))
    theta_put = (- (spot_price * norm.pdf(d1) * volatility_decimal) / (2 * np.sqrt(time_to_expiry))
                 + risk_free_rate_decimal * strike_price * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(-d2))
    rho_call = strike_price * time_to_expiry * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(d2) / 100  
    rho_put = -strike_price * time_to_expiry * np.exp(-risk_free_rate_decimal * time_to_expiry) * norm.cdf(-d2) / 100
    vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100  

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

greeks = calculate_greeks(st.session_state.spot_price, st.session_state.strike_price, st.session_state.risk_free_rate, st.session_state.time_to_expiry, st.session_state.volatility)

# Display the Greeks
st.write("### Greeks")
col1, col2 = st.columns(2)
col1.metric(label="Call Delta", value=f"{greeks['delta_call']:,.3f}")
col2.metric(label="Put Delta", value=f"{greeks['delta_put']:,.3f}")

col1, col2 = st.columns(2)
col1.metric(label="Gamma", value=f"{greeks['gamma']:,.3f}")
col2.metric(label="Vega", value=f"{greeks['vega']:,.3f}")

col1, col2 = st.columns(2)
col1.metric(label="Call Theta", value=f"{greeks['theta_call']:,.3f}")
col2.metric(label="Put Theta", value=f"{greeks['theta_put']:,.3f}")

col1, col2 = st.columns(2)
col1.metric(label="Call Rho", value=f"{greeks['rho_call']:,.3f}")
col2.metric(label="Put Rho", value=f"{greeks['rho_put']:,.3f}")

st.markdown("---")
st.markdown("### Developed by Kafui Avevor")
st.markdown("### [LinkedIn](https://www.linkedin.com/in/kafui-avevor/) | [GitHub](https://github.com/kafuiavevor)")