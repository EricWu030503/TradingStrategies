import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def gap_and_go_strategy(data, gap_percentage_threshold, profit_target, stop_loss):
    data['Previous Close'] = data['Close'].shift(1)
    data['Gap Percentage'] = (data['Open'] - data['Previous Close']) / data['Previous Close'] * 100

    # Initialize columns for tracking trades
    data['Entry Signal'] = False
    data['Exit Signal'] = False
    data['Entry Price'] = None
    data['Exit Price'] = None
    data['Target Price'] = None
    data['Stop Loss Price'] = None

    in_trade = False
    entry_price = None
    target_price = None
    stop_price = None

    for index, row in data.iterrows():
        if not in_trade:
            # Check for gap and enter trade
            if abs(row['Gap Percentage']) > gap_percentage_threshold:
                in_trade = True
                entry_price = row['Open']
                target_price = entry_price * (1 + profit_target / 100)
                stop_price = entry_price * (1 - stop_loss / 100)
                data.at[index, 'Entry Signal'] = True
                data.at[index, 'Entry Price'] = entry_price
                data.at[index, 'Target Price'] = target_price
                data.at[index, 'Stop Loss Price'] = stop_price
        else:
            # Check for exit conditions
            if row['High'] >= target_price or row['Low'] <= stop_price:
                in_trade = False
                exit_price = target_price if row['High'] >= target_price else stop_price
                data.at[index, 'Exit Signal'] = True
                data.at[index, 'Exit Price'] = exit_price

    return data

def backtest_strategy(data, initial_portfolio_value=10000):
    in_trade = False
    entry_price = 0
    portfolio_value = initial_portfolio_value
    portfolio_values = [initial_portfolio_value]

    for index, row in data.iterrows():
        if row['Entry Signal'] and not in_trade:
            in_trade = True
            entry_price = row['Open']
            shares = portfolio_value / entry_price

        if in_trade:
            current_portfolio_value = shares * row['Close']
            portfolio_values.append(current_portfolio_value)
        else:
            portfolio_values.append(portfolio_value)

        if row['Exit Signal'] and in_trade:
            in_trade = False
            exit_price = row['Open']
            portfolio_value = shares * exit_price
            shares = 0

    data['Portfolio Value'] = portfolio_values[:len(data)]
    total_return = portfolio_value / initial_portfolio_value - 1
    return total_return



def plot_performance(data):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the stock price and signals
    axes[0].plot(data.index, data['Close'], label='Stock Price', color='blue')
    axes[0].scatter(data[data['Entry Signal']].index, data[data['Entry Signal']]['Close'], label='Entry Signal', color='green', marker='^')
    axes[0].scatter(data[data['Exit Signal']].index, data[data['Exit Signal']]['Close'], label='Exit Signal', color='red', marker='v')
    axes[0].set_title('Stock Price and Trade Signals')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)

    # Plot the portfolio value
    axes[1].plot(data.index, data['Portfolio Value'], label='Portfolio Value', color='purple')
    axes[1].set_title('Portfolio Value Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()




def fetch_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_tickers = table[0]['Symbol'].tolist()
    return sp500_tickers

def test_single_ticker(ticker, start_date, end_date, gap_percentage_threshold, profit_target, stop_loss):
    data = fetch_data(ticker, start_date, end_date)
    data_with_signals = gap_and_go_strategy(data, gap_percentage_threshold, profit_target, stop_loss)
    total_return = backtest_strategy(data_with_signals)
    plot_performance(data_with_signals)
    print(f"Total Return for {ticker}: {total_return * 100:.2f}%")

def test_sp500_tickers(start_date, end_date, gap_percentage_threshold, profit_target, stop_loss):
    tickers = fetch_sp500_tickers()
    annualized_returns = []
    for ticker in tickers:
        try:
            data = fetch_data(ticker, start_date, end_date)
            if data.empty:
                continue
            data_with_signals = gap_and_go_strategy(data, gap_percentage_threshold, profit_target, stop_loss)
            total_return = backtest_strategy(data_with_signals)
            num_days = (data.index[-1] - data.index[0]).days
            annualized_return = ((1 + total_return) ** (365 / num_days) - 1) * 100
            annualized_returns.append(annualized_return)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    average_return = sum(annualized_returns) / len(annualized_returns) if annualized_returns else 0
    print(f"Average Annualized Return for S&P 500: {average_return:.2f}%")

# User input to choose the mode
mode = input("Choose the mode (single/all): ").strip().lower()
ticker = ''
if mode == 'single':
    ticker = input("Enter the ticker symbol (e.g., 'AAPL'): ").strip()
elif mode != 'all':
    print("Invalid mode selected.")
    exit()

# Common parameters for the strategy
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")
gap_percentage_threshold = 2  # 2 percent gap
profit_target = 5  # 5 percent profit target
stop_loss = 2  # 2 percent stop loss

# Execute based on the chosen mode
if mode == 'single':
    test_single_ticker(ticker, start_date, end_date, gap_percentage_threshold, profit_target, stop_loss)
elif mode == 'all':
    test_sp500_tickers(start_date, end_date, gap_percentage_threshold, profit_target, stop_loss)

