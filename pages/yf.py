import streamlit as st
import yfinance as yf

# Text input for ticker
ticker = st.text_input("Enter Stock Ticker", value="AAPL", help="Enter the ticker symbol (e.g., AAPL, MSFT, GOOG)")

# Button to fetch live data
fetch_live = st.button("Fetch Live Data")

# Caching the live data fetching function
@st.cache_data
def get_live_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")  # 1 year of historical data
        current_price = hist['Close'][-1]
        return {
            'current_price': current_price,
            'historical_prices': hist['Close'],
            'dates': hist.index
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Fetch and display data when the button is clicked
if fetch_live:
    data = get_live_data(ticker)
    if data:
        st.write(f"Current Price of {ticker}: ${data['current_price']:.2f}")
        st.line_chart(data['historical_prices'])  # Visualize historical prices
