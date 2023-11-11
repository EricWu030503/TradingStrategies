import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

currency_pairs = [
    'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X',
    'JPYUSD=X', 'CADUSD=X', 'CHFUSD=X', 'CNYUSD=X'
]

def fetch_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    df = df['Close']
    df = df.dropna()
    return df

def find_cointegrated_pairs(currency_pairs, start_date, end_date):
    cointegrated_pairs = []
    for i in range(len(currency_pairs)):
        for j in range(i + 1, len(currency_pairs)):
            pair1 = currency_pairs[i]
            pair2 = currency_pairs[j]
            df = fetch_data([pair1, pair2], start_date, end_date)
            if df.empty:
                continue
            try:
                _, pvalue, _ = coint(df[pair1], df[pair2])
                if pvalue < 0.05:
                    print(f'Cointegrated pairs found: {pair1.replace("=X", "")} and {pair2.replace("=X", "")} with p-value: {pvalue}')
                    cointegrated_pairs.append((df, pair1, pair2))
            except Exception as e:
                print(f"Error processing pair {pair1} and {pair2}: {e}")
                continue

    if cointegrated_pairs:
        return cointegrated_pairs
    else:
        return None

def generate_signals(df, pair1, pair2, window=20):
    df['Spread'] = df[pair1] - df[pair2]
    mean = df['Spread'].rolling(window=window).mean()
    std = df['Spread'].rolling(window=window).std()
    df['z_score'] = (df['Spread'] - mean) / std

    df['Signal'] = 0
    df['Position'] = 0

    entry_threshold = 2.0
    exit_threshold = 1.0

    for i in range(1, len(df)):
        if df.loc[df.index[i - 1], 'Position'] == 0:
            if df.loc[df.index[i], 'z_score'] > entry_threshold:
                df.loc[df.index[i], 'Signal'] = 2  # Signal to short pair1 and long pair2
                df.loc[df.index[i], 'Position'] = 2
            elif df.loc[df.index[i], 'z_score'] < -entry_threshold:
                df.loc[df.index[i], 'Signal'] = 1  # Signal to short pair2 and long pair1
                df.loc[df.index[i], 'Position'] = 1
        elif df.loc[df.index[i - 1], 'Position'] == 1 and abs(df.loc[df.index[i], 'z_score']) <= exit_threshold:
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'Position'] = 0
        elif df.loc[df.index[i - 1], 'Position'] == 2 and abs(df.loc[df.index[i], 'z_score']) <= exit_threshold:
            df.loc[df.index[i], 'Signal'] = -2
            df.loc[df.index[i], 'Position'] = 0
        else:
            df.loc[df.index[i], 'Position'] = df.loc[df.index[i - 1], 'Position']

    return df

def backtest(df, pair1, pair2):
    base_currency1 = pair1[0:3]
    base_currency2 = pair2[0:3]

    cash_balances = {'USD': 10000, base_currency1: 0, base_currency2: 0}

    df['Portfolio_Value'] = 0
    df['USD'] = 0
    df[base_currency1] = 0
    df[base_currency2] = 0

    for i in range(len(df)):
        signal = df.loc[df.index[i], 'Signal']
        price_pair1 = df.loc[df.index[i], pair1]
        price_pair2 = df.loc[df.index[i], pair2]

        if signal == 1:  # Long pair1 and short pair2
            # Long pair1
            cash_balances[base_currency1] += cash_balances['USD']/2/price_pair1
            cash_balances['USD'] /= 2

            # Short pair2
            cash_balances[base_currency2] -= cash_balances['USD']/price_pair2
            cash_balances['USD'] *= 2

        elif signal == -1:
            #sell pair1
            cash_balances['USD'] += cash_balances[base_currency1]*price_pair1
            cash_balances[base_currency1] = 0

            #buy back pair2
            cash_balances['USD'] += cash_balances[base_currency2]*price_pair2
            cash_balances[base_currency2] = 0

        elif signal == 2:  # Long pair2 and short pair1
            # Long pair2
            cash_balances[base_currency2] += cash_balances['USD'] / 2 / price_pair2
            cash_balances['USD'] /= 2

            # Short pair1
            cash_balances[base_currency1] -= cash_balances['USD'] / price_pair1
            cash_balances['USD'] *= 2

        elif signal == -2:
            # sell pair2
            cash_balances['USD'] += cash_balances[base_currency2] * price_pair2
            cash_balances[base_currency2] = 0

            # buy back pair1
            cash_balances['USD'] += cash_balances[base_currency1] * price_pair1
            cash_balances[base_currency1] = 0

        df.loc[df.index[i], 'USD'] = cash_balances['USD']
        df.loc[df.index[i], base_currency2] = cash_balances[base_currency2]
        df.loc[df.index[i], base_currency1] = cash_balances[base_currency1]

        # Update the portfolio value in USD
        df.loc[df.index[i], 'Portfolio_Value'] = cash_balances['USD'] + \
                                                 cash_balances[base_currency1] * price_pair1 + \
                                                 cash_balances[base_currency2] * price_pair2

    # Calculate returns
    print(f'{base_currency1}/USD and {base_currency2}/USD：')
    total_return = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0] - 1) * 100
    print(f"Total Return: {total_return:.2f}%")

    # Calculate annualized return
    num_days = (df.index[-1] - df.index[0]).days
    annualized_return = ((1 + total_return / 100) ** (365 / num_days) - 1) * 100
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(' ')

def plot_performance(df, pair1, pair2):
    # Create a figure with a specified size
    fig = plt.figure(figsize=(14, 18))

    # Plot the price trend for pair1
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(df.index, df[pair1], label=pair1 + ' Price', color='blue')
    ax1.scatter(df[df['Signal'] == 2].index, df[df['Signal'] == 2][pair1], color='red', marker='v', label='Short ' + pair1)
    ax1.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1][pair1], color='orange', marker='^', label='Long ' + pair1)
    ax1.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1][pair1], color='gray', marker='x', label='Exit Long ' + pair1)
    ax1.scatter(df[df['Signal'] == -2].index, df[df['Signal'] == -2][pair1], color='brown', marker='x', label='Exit Short ' + pair1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Price Trend of {pair1}')
    ax1.legend()
    ax1.grid(True)

    # Plot the price trend for pair2
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(df.index, df[pair2], label=pair2 + ' Price', color='green')
    ax2.scatter(df[df['Signal'] == 2].index, df[df['Signal'] == 2][pair2], color='black', marker='^', label='Long ' + pair2)
    ax2.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1][pair2], color='purple', marker='v', label='Short ' + pair2)
    ax2.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1][pair2], color='gray', marker='x', label='Exit Short ' + pair2)
    ax2.scatter(df[df['Signal'] == -2].index, df[df['Signal'] == -2][pair2], color='brown', marker='x', label='Exit Long ' + pair2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.set_title(f'Price Trend of {pair2}')
    ax2.legend()
    ax2.grid(True)

    # Plot the portfolio value
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(df.index, df['Portfolio_Value'], label='Portfolio Value', color='red')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Value')
    ax3.set_title('Portfolio Value Over Time')
    ax3.legend()
    ax3.grid(True)

    plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05)

    plt.show()

start_date = input("Please input the start date for backtesting (format: yyyy-mm-dd): ")
end_date = input("Please input the end date for backtesting (format: yyyy-mm-dd): ")

cointegrated_pairs = find_cointegrated_pairs(currency_pairs, start_date, end_date)

if cointegrated_pairs:
    for df, pair1, pair2 in cointegrated_pairs:
        generate_signals(df, pair1, pair2)
        backtest(df, pair1, pair2)
        plot_performance(df, pair1, pair2)
else:
    print("No cointegrated pairs suitable for pairs trading were found.")