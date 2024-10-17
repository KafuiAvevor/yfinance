import pandas as pd
import streamlit as st
import yfinance as yf
import financedatabase as fd
import datetime as dt


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
        
           
cols = st.columns(2)
sel_dt1 = cols[0].date_input('Start Date', value=dt.datetime(2024,1,1), format='YYYY-MM-DD')
sel_dt2 = cols[1].date_input('End Date', format='YYYY-MM-DD')

if len(sel_tickers) != 0:
    yfdata = yf.sownload(list(sel_tickers_list), start=sel_dt1, end=sel_st2)['Close'].reset_index().melt(id_vars = ['Date'], var_name = 'ticker', value_name='price')
    yfdata['price_start'] = yfdata.groupby('ticker').price.transform('first')
    yfdata['price_pct_daily'] = yfdata.groupby('ticker').price.pct_change()
    yfdata['price_pct'] = (yfdata.price - yfdata.price_start)/ yfdata.price_start

    st.line_chart(yfdata.pivot(index='Date', columns='ticker', values='price_pct'))


