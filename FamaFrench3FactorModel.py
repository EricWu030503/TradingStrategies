import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Gather Data

# Get S&P 500 tickers
sp500 = yf.Tickers("^GSPC")
sp500_tickers = [ticker for ticker in sp500.tickers]

# Get historical data for these tickers
data = {ticker: yf.download(ticker, period="5y")["Close"] for ticker in sp500_tickers[:50]}  # Limiting to 50 due to time constraints. You might want to remove this limit.

# Get Fama-French three-factor data
ff_data = web.DataReader("F-F_Research_Data_Factors", "famafrench")[0]
ff_data = ff_data.resample('M').last()  # Convert to end-of-month data to align with stock data

# Calculate excess returns

stock_returns = pd.DataFrame(data).resample('M').last().pct_change().dropna()
excess_returns = stock_returns.sub(ff_data["RF"].values, axis=0)

# Regression for each stock

betas = {}

for ticker in sp500_tickers[:50]:  # Again, limited for simplification
    X = sm.add_constant(ff_data[["Mkt-RF", "SMB", "HML"]])
    y = excess_returns[ticker]
    model = sm.OLS(y, X).fit()
    betas[ticker] = model.params

# Stock selection
selected_stocks = sorted(betas.keys(), key=lambda x: betas[x]['HML'], reverse=True)[:50]

# Construct Portfolio
portfolio_returns = stock_returns[selected_stocks].mean(axis=1)

# Backtest

sp500_returns = yf.download("^GSPC", period="5y")["Close"].resample('M').last().pct_change().dropna()

cumulative_portfolio = (1 + portfolio_returns).cumprod()
cumulative_sp500 = (1 + sp500_returns).cumprod()

plt.plot(cumulative_portfolio, label="Portfolio")
plt.plot(cumulative_sp500, label="S&P 500")
plt.legend()
plt.show()
