import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# Fetch data
def fetch_data(ticker, period):
    data = yf.Ticker(ticker)
    df = data.history(period=period)
    return df


# Calculate Bollinger Bands
def calculate_bollinger_bands(df, window, num_std_dev):
    df['Rolling Mean'] = df['Close'].rolling(window).mean()
    df['Bollinger High'] = df['Rolling Mean'] + (df['Close'].rolling(window).std() * num_std_dev)
    df['Bollinger Low'] = df['Rolling Mean'] - (df['Close'].rolling(window).std() * num_std_dev)
    return df


# Calculate ATR
def calculate_atr(df, window):
    tr = np.maximum((df['High'] - df['Low']),
                    np.maximum(abs(df['High'] - df['Close'].shift()),
                               abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = tr.rolling(window).mean()
    return df

'''
# Generate signals
def generate_signals(df):
    signals = []
    position = 0  # 1 indicates long, -1 indicates short, 0 indicates no position
    trailing_stop = None  # Initialize trailing stop

    for i in range(len(df)):
        close = df['Close'].iloc[i]
        lower_band = df['Bollinger Low'].iloc[i]
        upper_band = df['Bollinger High'].iloc[i]
        mean = df['Rolling Mean'].iloc[i]
        atr = df['ATR'].iloc[i]

        # Entry logic
        if position == 0:
            if close > upper_band:
                signals.append('Buy')
                position = 1
                trailing_stop = close - 2*atr  # Set trailing stop
            elif close < lower_band:
                signals.append('Short')
                position = -1
                trailing_stop = close + 2*atr  # Set trailing stop
            else:
                signals.append("Hold")

        # Exit logic
        elif position == 1 and close < trailing_stop:
            signals.append('Sell')
            position = 0
        elif position == -1 and close > trailing_stop:
            signals.append('Cover')
            position = 0

        else:
            signals.append('Hold')

        # Update trailing stop
        if position == 1:
            trailing_stop = max(trailing_stop, close - 2.5*atr)
        elif position == -1:
            trailing_stop = min(trailing_stop, close + 2.5*atr)

    df['Signal'] = signals
    return df
'''

def generate_signals(df):
    signals = []
    position = 0  # 1 indicates long, -1 indicates short, 0 indicates no position
    close = df['Close'].iloc[0]
    mean = df['Rolling Mean'].iloc[0]
    if close >= mean:
        price_greater_than_ma = True
    else:
        price_greater_than_ma = False
    signals.append("Hold")

    for i in range(1, len(df)):
        close = df['Close'].iloc[i]
        mean = df['Rolling Mean'].iloc[i]
        temp = price_greater_than_ma
        if close >= mean:
            price_greater_than_ma = True
        else:
            price_greater_than_ma = False

        # Entry logic
        if position == 0:
            if temp == False and price_greater_than_ma == True:
                signals.append('Buy')
                position = 1
            elif temp == True and price_greater_than_ma == False:
                signals.append('Short')
                position = -1
            else:
                signals.append("Hold")

        # Exit logic
        elif position == 1 and (temp == True and price_greater_than_ma == False):
            signals.append('Sell')
            position = -1
        elif position == -1 and (temp == False and price_greater_than_ma == True):
            signals.append('Cover')
            position = 1

        else:
            signals.append('Hold')

    df['Signal'] = signals
    return df

def plot_data(df):
    fig, axs = plt.subplots(2, figsize=(10, 10))

    # Plot Portfolio Value
    axs[0].plot(df['Portfolio Value'], label='Portfolio Value')
    axs[0].set_title('Portfolio Value Over Time')
    axs[0].legend(loc='best')

    # Plot the closing prices, Bollinger Bands and Buy/Sell/Short/Cover signals
    axs[1].plot(df['Close'], label='Price')
    axs[1].plot(df['Rolling Mean'], label='Rolling Mean')
    axs[1].plot(df['Bollinger High'], label='Upper Band')
    axs[1].plot(df['Bollinger Low'], label='Lower Band')

    buy_signals = df[df['Signal'] == 'Buy']
    sell_signals = df[df['Signal'] == 'Sell']
    short_signals = df[df['Signal'] == 'Short']
    cover_signals = df[df['Signal'] == 'Cover']

    axs[1].plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', label='Buy Signal')
    axs[1].plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', label='Sell Signal')
    axs[1].plot(short_signals.index, short_signals['Close'], '1', markersize=10, color='m', label='Short Signal')  # using '1' for downward pointing triangle
    axs[1].plot(cover_signals.index, cover_signals['Close'], '2', markersize=10, color='c', label='Cover Signal')  # using '2' for upward pointing triangle

    axs[1].set_title('Price, Bollinger Bands, and Buy/Sell/Short/Cover Signals Over Time')
    axs[1].legend(loc='best')

    plt.tight_layout()
    plt.show()


'''
def backtest(df, initial_capital):
    shares_held = 0  # Tracks our positions over time
    cash = initial_capital  # Tracks our cash over time


    for i in range(len(df)):
        signal = df['Signal'].iloc[i]
        price = df['Close'].iloc[i]

        if signal == 'Buy' and cash >= price * 100:
            shares_held = int(cash / price)  # Calculate the maximum number of shares that can be bought
            cash -= price * shares_held  # Update cash

        elif signal == 'Sell' and shares_held > 0:
            cash += price * shares_held  # Update cash
            shares_held = 0

        elif signal == 'Short' and cash >= price * 100:
            shares_held = int(cash / price) * (-1)  # Calculate the maximum number of shares that can be shorted
            cash += price * (-1 * shares_held)  # Update cash

        elif signal == 'Cover' and shares_held < 0:
            cash += price * shares_held  # Update cash
            shares_held = 0

        df.at[df.index[i], 'Positions'] = shares_held  # Update Positions column
        df.at[df.index[i], 'Cash'] = cash  # Update Cash column
        df.at[df.index[i], 'Portfolio Value'] = cash + (shares_held * price)  # Update Portfolio Value column


    df['Returns'] = df['Portfolio Value'].pct_change()

    # Performance metrics
    num_trading_days = len(df)
    num_years = num_trading_days / 252
    total_return = (df['Portfolio Value'].iloc[-1] / df['Portfolio Value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    annualized_volatility = df['Returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    print(f'Total Return: {total_return * 100:.2f}%')
    print(f'Annualized Return: {annualized_return * 100:.2f}%')
    print(f'Annualized Volatility: {annualized_volatility * 100:.2f}%')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    print(" ")
    return annualized_return
'''

def backtest(df, initial_capital):
    shares_held = 0  # Tracks our positions over time
    cash = initial_capital  # Tracks our cash over time


    for i in range(len(df)):
        signal = df['Signal'].iloc[i]
        price = df['Close'].iloc[i]

        if signal == 'Buy' and cash >= price * 100:
            shares_held = int(cash / price)  # Calculate the maximum number of shares that can be bought
            cash -= price * shares_held  # Update cash

        elif signal == 'Sell' and shares_held > 0:
            cash += price * shares_held  # Update cash
            shares_held = 0
            shares_held = int(cash / price) * (-1)  # Calculate the maximum number of shares that can be shorted
            cash += price * (-1 * shares_held)  # Update cash

        elif signal == 'Short' and cash >= price * 100:
            shares_held = int(cash / price) * (-1)  # Calculate the maximum number of shares that can be shorted
            cash += price * (-1 * shares_held)  # Update cash

        elif signal == 'Cover' and shares_held < 0:
            cash += price * shares_held  # Update cash
            shares_held = 0
            shares_held = int(cash / price)  # Calculate the maximum number of shares that can be bought
            cash -= price * shares_held  # Update cash

        df.at[df.index[i], 'Positions'] = shares_held  # Update Positions column
        df.at[df.index[i], 'Cash'] = cash  # Update Cash column
        df.at[df.index[i], 'Portfolio Value'] = cash + (shares_held * price)  # Update Portfolio Value column


    df['Returns'] = df['Portfolio Value'].pct_change()

    # Performance metrics
    num_trading_days = len(df)
    num_years = num_trading_days / 252
    total_return = (df['Portfolio Value'].iloc[-1] / df['Portfolio Value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    annualized_volatility = df['Returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    print(f'Total Return: {total_return * 100:.2f}%')
    print(f'Annualized Return: {annualized_return * 100:.2f}%')
    print(f'Annualized Volatility: {annualized_volatility * 100:.2f}%')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    print(" ")
    return annualized_return

def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return table['Symbol'].tolist()


def main():
    period = "3y"
    window = 30
    num_std_dev = 2
    initial_capital = 10000000
    annualized_returns = []

    # User input for mode and plotting
    mode = input("Choose the mode (single/all): ").lower().strip()
    plot_graph = input("Do you want to plot the graph? (yes/no): ").lower().strip() == "yes"

    if mode == 'single':
        ticker = input("Enter the ticker symbol (e.g., 'AAPL'): ").strip()
        tickers = [ticker]
    elif mode == 'all':
        tickers = get_sp500_tickers()
    else:
        print("Invalid mode selected.")
        return

    for ticker in tickers:
        print(f"Processing {ticker}...")
        try:
            df = fetch_data(ticker, period)
            df = calculate_bollinger_bands(df, window, num_std_dev)
            df = calculate_atr(df, window)
            df.dropna(inplace=True)
            df = generate_signals(df)
            annualized_return = backtest(df, initial_capital)
            annualized_returns.append(annualized_return)

            if plot_graph:
                plot_data(df)
        except Exception as e:
            print(f"Could not process {ticker} due to {str(e)}")
            print(" ")

    if annualized_returns:
        print(f"The Portfolio annualized return: {sum(annualized_returns)/len(annualized_returns)*100:.2f}%")
    else:
        print("No valid returns calculated.")

if __name__ == "__main__":
    main()


