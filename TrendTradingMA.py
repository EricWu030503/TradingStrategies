import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def calculate_adx(df, window):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift())
    df['L-PC'] = abs(df['Low'] - df['Close'].shift())

    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    df['+DM'] = np.where((df['High'] > df['High'].shift()) &
                         (df['High'] - df['High'].shift() > df['Low'].shift() - df['Low']),
                         df['High'] - df['High'].shift(), 0)

    df['-DM'] = np.where((df['Low'] < df['Low'].shift()) &
                         (df['High'].shift() - df['High'] < df['Low'].shift() - df['Low']),
                         df['Low'].shift() - df['Low'], 0)

    df['TR'] = df['TR'].rolling(window).sum()
    df['+DM'] = df['+DM'].rolling(window).sum()
    df['-DM'] = df['-DM'].rolling(window).sum()

    df['+DI'] = (df['+DM'] / df['TR']) * 100
    df['-DI'] = (df['-DM'] / df['TR']) * 100

    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window).mean()

    return df


def is_trending(df, threshold):
    return df['ADX'].iloc[-1] > threshold


# Fetch data
def fetch_data(ticker, period):
    data = yf.Ticker(ticker)
    df = data.history(period=period)
    return df


# Calculate Moving Average
def calculate_moving_average(df, window):
    df['Moving Average'] = df['Close'].rolling(window).mean()
    return df


# Generate signals
def generate_signals(df,threshold):
    signals = []
    position = 0  # 0 indicates no position, 1 indicates long, -1 indicates short
    # Determine if the price crosses the moving average
    for i in range(1, len(df)):
        current_data = df.iloc[:i + 1]
        trending = is_trending(current_data,threshold)
        if trending:
            previous_price = df['Close'].iloc[i - 1]
            current_price = df['Close'].iloc[i]
            previous_ma = df['Moving Average'].iloc[i - 1]
            current_ma = df['Moving Average'].iloc[i]

            if position != 1 and previous_price < previous_ma and current_price > current_ma:
                signals.append('Buy')
                position = 1  # Set position to long
            elif position != -1 and previous_price > previous_ma and current_price < current_ma:
                signals.append('Sell')
                position = -1  # Set position to short
            else:
                signals.append('Hold')
        else:
            signals.append('Hold')

    # Add a Hold at the beginning since the first row doesn't have a previous day to compare
    signals.insert(0, 'Hold')
    df['Signal'] = signals
    return df


def backtest(df, initial_capital):
    shares_held = 0
    cash = initial_capital

    for i in range(len(df)):
        signal = df['Signal'].iloc[i]
        price = df['Close'].iloc[i]

        if signal == 'Buy':
            shares_bought = int(cash / price)
            cash -= shares_bought * price
            shares_held += shares_bought
        elif signal == 'Sell':
            if shares_held > 0:
                cash += shares_held * price * 2
                shares_held *= -1
            else:
                shares_to_sell = int(cash/price)
                cash += shares_to_sell * price
                shares_held = -1 * shares_to_sell


        df.at[df.index[i], 'Positions'] = shares_held
        df.at[df.index[i], 'Cash'] = cash
        df.at[df.index[i], 'Portfolio Value'] = cash + (shares_held * price)

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


def plot_data(df):
    fig, axs = plt.subplots(2, figsize=(10, 10))

    # Plot Portfolio Value
    axs[0].plot(df['Portfolio Value'], label='Portfolio Value')
    axs[0].set_title('Portfolio Value Over Time')
    axs[0].legend(loc='best')

    # Plot the closing prices, moving average and Buy/Sell signals
    axs[1].plot(df['Close'], label='Price')
    axs[1].plot(df['Moving Average'], label='Moving Average')

    buy_signals = df[df['Signal'] == 'Buy']
    sell_signals = df[df['Signal'] == 'Sell']

    axs[1].plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', label='Buy Signal')
    axs[1].plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', label='Sell Signal')

    axs[1].set_title('Price, Moving Average, and Buy/Sell Signals Over Time')
    axs[1].legend(loc='best')

    plt.tight_layout()
    plt.show()


def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return table['Symbol'].tolist()


def main():
    period = '3y'
    window = 30
    initial_capital = 10000000
    annualized_returns = []

    adx_threshold = 25  # Define your threshold for trending market

    sp500_tickers = get_sp500_tickers()

    for ticker in sp500_tickers:
        print(f"Processing {ticker}...")
        try:
            df = fetch_data(ticker, period)
            df = calculate_adx(df, window)
            df = calculate_moving_average(df, window)
            df.dropna(inplace=True)  # Drop rows with missing values
            df = generate_signals(df, adx_threshold)
            annualized_return = backtest(df, initial_capital)
            if not np.isnan(annualized_return):
                annualized_returns.append(annualized_return)
            #plot_data(df)
        except Exception as e:
            print(f"Could not process {ticker} due to {str(e)}")
            print(" ")

    print(f"The Portfolio annualized return: {sum(annualized_returns) / len(annualized_returns) * 100}%")


if __name__ == "__main__":
    main()
