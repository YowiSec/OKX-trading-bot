import pandas as pd
import ta

# Load your data into a DataFrame
# Assuming the DataFrame is named 'df' and contains 'open', 'high', 'low', 'close', and 'volume' columns
# Load the dataset
df = pd.read_csv('data/data.csv')

# Calculate Money Flow Index (MFI)
df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
#print(df['mfi'])
# Calculate Exponential Moving Averages (EMA)
short_window = 12
long_window = 24
df['ema_short'] = ta.trend.EMAIndicator(close=df['close'], window=short_window).ema_indicator()
df['ema_long'] = ta.trend.EMAIndicator(close=df['close'], window=long_window).ema_indicator()

# Calculate Average True Range (ATR)
df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
# Drop the NaN values from the DataFrame after calculating the indicators
df = df.dropna()
# Define the trading signals
signals = []

for i in range(len(df)):
    if df['mfi'].iloc[i] < 10 and df['ema_short'].iloc[i] > df['ema_long'].iloc[i]:
        signals.append('BUY')
    elif df['mfi'].iloc[i] > 90 and df['ema_short'].iloc[i] < df['ema_long'].iloc[i]:
        signals.append('SELL')
    else:
        signals.append('HOLD')

# Add signals to the DataFrame
df['signals'] = signals

# Use raw prices for transactions

""" prices = df['close'].values[-len(signals):]
initial_balance = 10000  # starting with $10,000 for example
balance = initial_balance
stock_quantity = 0

# Use log returns for buy/sell decisions
# returns = new_data['log_return'].values[-len(signals):]

allocation = 0.1  # Or whatever percentage you decide

for i, signal in enumerate(signals):
    print("CURRENT SIGNAL: %s" % signal)
    print("BALANCE: %s" % balance)
    if signal == 'Buy':
        buy_amount = allocation * balance
        if buy_amount >= prices[i]:
            buy_quantity = buy_amount // prices[i]
            balance -= buy_quantity * prices[i]
            stock_quantity += buy_quantity
    elif signal == 'Sell' and stock_quantity > 0:
        balance += stock_quantity * prices[i]
        stock_quantity = 0


# Final value
final_balance = balance + stock_quantity * (prices[-1] if prices[-1] else 0)

profit_or_loss = final_balance - initial_balance
print("P/L: %s" % profit_or_loss) """



# Define initial capital and other variables for backtesting
initial_capital = float(10000.0)  # Starting capital
position_size = 1  # Number of shares to buy/sell on each trade
balance = initial_capital
positions = pd.Series(index=df.index).fillna(0.0)  # Series to hold the positions
portfolio = pd.DataFrame(index=df.index).fillna(0.0)  # DataFrame to hold the value of the portfolio

# Assume we buy/sell on the close of the same day the signal is generated
portfolio['positions'] = positions

# Go through the signals and execute trades
for i in range(1, len(df)):
    signal = df['signals'].iloc[i]
    print("CURRENT SIGNAL: %s" % signal)
    print("BALANCE: %s" % balance)
    if signal == 'BUY' and balance >= df['close'].iloc[i] * position_size:
        positions.iloc[i] = position_size  # Buy
        balance -= df['close'].iloc[i] * position_size  # Update balance
    elif signal == 'SELL' and positions.iloc[i-1] > 0:
        positions.iloc[i] = 0  # Sell
        balance += df['close'].iloc[i] * position_size  # Update balance
    # Update the current position
    else:
        positions.iloc[i] = positions.iloc[i-1]

# Update portfolio to reflect buys/sells
portfolio['holdings'] = positions * df['close']
portfolio['cash'] = balance - (positions.diff() * df['close']).cumsum()
portfolio['total'] = portfolio['holdings'] + portfolio['cash']

# Calculate portfolio performance
portfolio['returns'] = portfolio['total'].pct_change()
total_return = portfolio['total'].iloc[-1] - initial_capital

print(f'Total Return: {total_return:.2f}')
print(f'Final Portfolio Value: {portfolio["total"].iloc[-1]:.2f}')