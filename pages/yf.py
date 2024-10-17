import pandas as pd
import streamlit as st
import yfinance as yf
import financedatabase as fd
import datetime as dt

# Caching the ticker data with a time-to-live (ttl) to refresh periodically
@st.cache_data(ttl=60*60*24)  # Cache for 1 day
def load_data():
    ticker_list = pd.concat([fd.ETFs().select().reset_index()[['symbol', 'name']], 
                             fd.Equities().select().reset_index()[['symbol', 'name']]])
    ticker_list = ticker_list[ticker_list.symbol.notna()]
    ticker_list['symbol_name'] = ticker_list.symbol + '-' + ticker_list.name
    return ticker_list

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
            # Fetch historical data from Yahoo Finance
            yfdata = yf.download(list(sel_tickers_list), start=sel_dt1, end=sel_dt2)['Close'].reset_index().melt(
                id_vars=['Date'], var_name='ticker', value_name='price')

            # Calculate price percentage changes
            yfdata['price_start'] = yfdata.groupby('ticker').price.transform('first')
            yfdata['price_pct_daily'] = yfdata.groupby('ticker').price.pct_change()
            yfdata['price_pct'] = (yfdata.price - yfdata.price_start) / yfdata.price_start

            # Display the percentage change chart
            st.line_chart(yfdata.pivot(index='Date', columns='ticker', values='price_pct'))

        except Exception as e:
            st.error(f"An error occurred: {e}")
