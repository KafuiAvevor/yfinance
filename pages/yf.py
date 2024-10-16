import pandas as pd
import streamlit as st
import yfinance as yf
import financedatabase as fd

@st.cache_data
def load_data():
    ticker_list = pd.concat([fd.ETFs().select().reset_index()[['symbol', 'name']], fd.Equities().select().reset_index()[['symbol', 'name']]])
    ticker_list = ticker_list[ticker_list.symbol.notna()]
    ticker_list['symbol_name'] = ticker_list.symbol +'-' + ticker_list.name
    return ticker_list
ticker_list = load_data()

with st.sidebar:
    sel_tickers = st.multiselect('Portfolio Builder', placeholder="Search tickers", options=ticker_list.symbol_name)
    sel_tickers_list = ticker_list[ticker_list.symbol_name.isin(sel_tickers)].symbol

    cols = st.columns(4)
    for i, ticker in enumerate(sel_tickers_list):
        try:
            cols[i % 4].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.',''), width=65)
        except:
            cols[i % 4].subheader(ticker)
        
           

