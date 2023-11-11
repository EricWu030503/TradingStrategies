import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import math



def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date,progress=False)


def calculate_bollinger_bands(data, window=20, num_of_sd=2):
    data['SMA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper_Band'] = data['SMA'] + (data['STD'] * num_of_sd)
    data['Lower_Band'] = data['SMA'] - (data['STD'] * num_of_sd)


def generate_signals(data):
    data['Position'] = None
    data['Buy_Signal'] = 0
    data['Sell_Signal'] = 0
    data['Short_Signal'] = 0
    data['Cover_Signal'] = 0

    for i in range(1, len(data)):
        close = data['Close'].iloc[i]
        lower_band = data['Lower_Band'].iloc[i]
        upper_band = data['Upper_Band'].iloc[i]
        sma = data['SMA'].iloc[i]
        prev_position = data['Position'].iloc[i - 1]
        idx = data.index[i]

        if close <= lower_band and prev_position is None:
            data.loc[idx, 'Position'] = 'long'
            data.loc[idx, 'Buy_Signal'] = 1
        elif close >= upper_band and prev_position is None:
            data.loc[idx, 'Position'] = 'short'
            data.loc[idx, 'Short_Signal'] = 1
        elif close >= sma and prev_position == 'long':
            data.loc[idx, 'Position'] = None
            data.loc[idx, 'Sell_Signal'] = 1
        elif close <= sma and prev_position == 'short':
            data.loc[idx, 'Position'] = None
            data.loc[idx, 'Cover_Signal'] = 1
        else:
            data.loc[idx, 'Position'] = prev_position

    return data


def backtest_strategy(data, initial_capital=1000000):
    cash = initial_capital
    position = 0
    portfolio_values = [initial_capital]

    for index, row in data.iterrows():
        if row['Buy_Signal'] == 1 and cash > row['Close'] * 100:
            position += int (cash / row['Close'])
            cash -= position * row['Close']
        elif row['Sell_Signal'] == 1:
            cash = position * row['Close']
            position = 0
        elif row['Short_Signal'] == 1 and cash > row['Close'] * 100:
            position -= int (cash / row['Close'])
            cash -= position * row['Close']
        elif row['Cover_Signal'] == 1:
            cash += position * row['Close']
            position = 0

        portfolio_values.append(cash + position * row['Close'])

    data['Portfolio_Value'] = portfolio_values[1:]
    total_return = (data['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital
    days = (data.index[-1] - data.index[0]).days
    annualized_return = (1 + total_return) ** (365.25 / days) - 1
    print(f"Total return: {total_return * 100:.2f}%")
    print(f"Annualized return: {annualized_return * 100:.2f}% \n")

    return total_return, annualized_return


def plot_strategy(data, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # First graph on ax1
    ax1.plot(data['Close'], label='Close Price', alpha=0.5)
    ax1.plot(data['SMA'], label='Simple Moving Average', alpha=0.5)
    ax1.plot(data['Upper_Band'], label='Upper Bollinger Band', alpha=0.5)
    ax1.plot(data['Lower_Band'], label='Lower Bollinger Band', alpha=0.5)

    # Extracting signal data
    buy_signals = data[data['Buy_Signal'] == 1]
    sell_signals = data[data['Sell_Signal'] == 1]
    short_signals = data[data['Short_Signal'] == 1]
    cover_signals = data[data['Cover_Signal'] == 1]

    # Plotting signals on the graph
    ax1.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', alpha=1, color='green')
    ax1.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', alpha=1, color='red')
    ax1.scatter(short_signals.index, short_signals['Close'], label='Short Signal', marker='^', alpha=1, color='purple')
    ax1.scatter(cover_signals.index, cover_signals['Close'], label='Cover Signal', marker='v', alpha=1, color='blue')

    ax1.set_title(f"{ticker} - Bollinger Band Strategy")
    ax1.legend()

    # Second graph on ax2
    ax2.plot(data['Portfolio_Value'], label='Portfolio Value', alpha=0.8, color='magenta', linestyle='--')
    ax2.set_title(f"{ticker} - Portfolio Value Over Time")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def backtest_ticker(ticker, start_date, end_date, single):
    data = fetch_data(ticker, start_date, end_date)
    calculate_bollinger_bands(data)
    data = generate_signals(data)
    total_return, annualized_return = backtest_strategy(data)
    if single:
        plot_strategy(data, ticker)
    return total_return, annualized_return


def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return table['Symbol'].tolist()



if __name__ == "__main__":
    start_date = input("Please input the start date for backtesting (format: yyyy-mm-dd): ")
    end_date = input("Please input the end date for backtesting (format: yyyy-mm-dd): ")

    while True:

        choice = input("Do you want to test a single stock or all S&P 500 stocks? (single/some/all): ")

        if choice == "single":
            ticker = input("Enter the stock ticker (e.g. TSLA): ")
            total_return, annualized_return = backtest_ticker(ticker, start_date, end_date, True)
            break
        elif choice == "some":
            number = int(input("Number of companies you want to test: "))
            tickers = get_sp500_tickers()
            annualized_returns = []

            for i in range(number):
                print(f"Backtesting for {tickers[i]}...")
                try:
                    _, annualized_return = backtest_ticker(tickers[i], start_date, end_date, False)
                    annualized_returns.append(annualized_return)
                except Exception as e:
                    print(f"Error with {tickers[i]}: {e}")
            if annualized_returns:
                mean_annualized_return = sum(annualized_returns) / len(annualized_returns)
                print(f"Mean annualized return for these {number} companies: {mean_annualized_return * 100:.2f}%")
            else:
                print("Failed to retrieve results for all tickers.")
            break

        elif choice == "all":
            tickers = get_sp500_tickers()
            annualized_returns = []

            for ticker in tickers:
                print(f"Backtesting for {ticker}...")
                try:
                    _, annualized_return = backtest_ticker(ticker, start_date, end_date, False)
                    if not math.isnan(annualized_return):
                        annualized_returns.append(annualized_return)
                except Exception as e:
                    print(f"Error with {ticker}: {e}")
            if annualized_returns:
                mean_annualized_return = sum(annualized_returns) / len(annualized_returns)
                print(f"Mean annualized return for S&P 500: {mean_annualized_return * 100:.2f}%")
            else:
                print("Failed to retrieve results for all tickers.")
            break
        else:
            print("Invalid choice.")


