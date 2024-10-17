import pandas as pd
import streamlit as st
import yfinance as yf
import financedatabase as fd
import datetime as dt


# Set the title of the app
st.title("Stock Data Viewer")

# Fetch data for a specific stock, e.g., Apple (AAPL)
ticker_symbol = "AAPL"
ticker = yf.Ticker(ticker_symbol)
data = ticker.history(period="5d")  # Fetch last 5 days of data

# Display the fetched data in the Streamlit app
st.header(f"Last 5 Days of {ticker_symbol} Stock Data")
st.write(data)

# Optionally, you can plot the closing price using Streamlit
st.line_chart(data['Close'])

# Load ticker list from database
ticker_list = load_data()

with st.sidebar:
    # Multiselect for tickers
    sel_tickers = st.multiselect('Portfolio Builder', placeholder="Search tickers", options=ticker_list.symbol_name)
    sel_tickers_list = ticker_list[ticker_list.symbol_name.isin(sel_tickers)].symbol

    # Display ticker logos or names in sidebar columns
    cols = st.columns(4)
    for i, ticker in enumerate(sel_tickers_list):
        try:
            website = yf.Ticker(ticker).info.get('website', '')
            logo_url = f'https://logo.clearbit.com/{website.replace("https://www.", "")}'
            cols[i % 4].image(logo_url, width=65)
        except:
            cols[i % 4].subheader(ticker)  # Fallback if no logo available
            
# Create two date input columns for start and end date
cols = st.columns(2)
sel_dt1 = cols[0].date_input('Start Date', value=dt.datetime(2024, 1, 1), format='YYYY-MM-DD')
sel_dt2 = cols[1].date_input('End Date', format='YYYY-MM-DD')

# Ensure valid date range selection
if sel_dt1 >= sel_dt2:
    st.error("End date must be after the start date.")
else:
    # Only fetch data if tickers are selected
    if len(sel_tickers) != 0:
        try:
            # Create an empty DataFrame to store data for all tickers
            all_data = pd.DataFrame()

            # Fetch historical data for each ticker
            for ticker in sel_tickers_list:
                ticker_data = yf.Ticker(ticker).history(start=sel_dt1, end=sel_dt2)[['Close']]
                ticker_data['ticker'] = ticker  # Add ticker column
                ticker_data = ticker_data.reset_index()  # Reset index to have date as a column
                all_data = pd.concat([all_data, ticker_data], ignore_index=True)

            # Reshape and calculate percentage changes
            all_data['price_start'] = all_data.groupby('ticker')['Close'].transform('first')
            all_data['price_pct_daily'] = all_data.groupby('ticker')['Close'].pct_change()
            all_data['price_pct'] = (all_data['Close'] - all_data['price_start']) / all_data['price_start']

            # Display the percentage change chart
            st.line_chart(all_data.pivot(index='Date', columns='ticker', values='price_pct'))

        except Exception as e:
            st.error(f"An error occurred: {e}")
