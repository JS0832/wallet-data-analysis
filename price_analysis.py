import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
# Load the CSV data into a DataFrame
csv_file = "example_data.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)
# Parse the "date" column as datetime
df["date"] = pd.to_datetime(df["date"])
# Set the "date" column as the index
df.set_index("date", inplace=True)
# Calculate a 40 EMA
df["EMA_40"] = df["close"].ewm(span=20, adjust=False).mean()

# Parameters for peak detection
x_pct_reversal = 10  # Minimum reversal percentage to reset peak search
y_pct_drop = 40  # Minimum trailing drop percentage for a valid peak
# Plot the results
#fig, ax = plt.subplots(figsize=(18, 9))

# Plot the close prices
#.plot(df.index, df["close"], label="Close Price", color="gray", linewidth=1)

def load_ndjson_to_dataframe(ndjson_file):
    """
    Load an NDJSON file into a Pandas DataFrame.

    Parameters:
    - ndjson_file: str - Path to the NDJSON file.

    Returns:
    - DataFrame containing wallet and transaction details.
    """
    # Initialize a list to store each line as a dictionary
    data = []

    # Read the NDJSON file line by line
    with open(ndjson_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Normalize the data into a DataFrame
    df = pd.json_normalize(data)

    # Expand the 'transactions' list into separate rows
    transactions = df.explode('transactions')

    # Normalize the transactions column into a DataFrame
    transactions_df = pd.json_normalize(transactions['transactions'])

    # Merge with the original DataFrame to include wallet info
    final_df = pd.concat([transactions.reset_index(drop=True), transactions_df.reset_index(drop=True)], axis=1)

    return final_df
ndjson_file = 'chads.ndjson'
wallets_df = load_ndjson_to_dataframe(ndjson_file)
# Function to find peaks
def find_peaks(data, reversal_pct, drop_pct):
    peaks = []
    trailing_min = data[0]
    current_peak = data[0]
    current_peak_index = 0
    in_reversal = False  # Track if we're in a reversal phase
    temp_bottom = 100000000 #this will keep the last low from the trailing min values from the last peak it marked
    temp_bottom_index = 0
    trade_closes = [] # will store the bottoms
    for i in range(1, len(data)):
        # Update trailing minimum if not in reversal
        if not in_reversal:
            trailing_min = min(trailing_min, data[i])
        if trailing_min < temp_bottom:
            temp_bottom_index = i
            temp_bottom = trailing_min

        # Check if the current point is a new peak
        if data[i] > current_peak and not in_reversal:
            current_peak = data[i]
            current_peak_index = i
            trailing_min = data[i]  # Reset trailing minimum

        # Check for trailing drop
        drop = (current_peak - trailing_min) / current_peak * 100
        if drop >= drop_pct and not in_reversal:
            peaks.append((current_peak_index, current_peak))  # Log the peak
            # Enter reversal phase
            in_reversal = True
            # Reversal threshold based on the last trailing minimum
            reversal_threshold = trailing_min * (1 + reversal_pct / 100)

        # Check if reversal condition is met from the last bottom
        if in_reversal and data[i] >= reversal_threshold:
            in_reversal = False  # Exit reversal phase
            current_peak = data[i]  # Reset peak to current value
            current_peak_index = i
            trailing_min = data[i]  # Reset trailing minimum
            trade_closes.append((current_peak_index, current_peak))

    return peaks , trade_closes



# Apply the function to the closing prices
peaks,trade_closes = find_peaks(df["close"].values, x_pct_reversal, y_pct_drop)


def calculate_pnl(peaks, trade_closes, data, ax):
    trades = []  # List to store trades with PnL
    starting_capital = 2000
    leverage = 4
    compound = True
    compound_win = 1
    # Ensure the peaks and trade closes align correctly
    for i in range(min(len(peaks), len(trade_closes))):
        peak_index, peak_price = peaks[i]
        close_index, close_price = trade_closes[i]
        # Calculate the PnL percentage
        pnl = -1 * ((close_price - peak_price) / peak_price) * 100
        multiplier = 1 + pnl / 100
        print(f"PNL nr.{i + 1}: {int(pnl)}%")
        compound_win *= (multiplier * leverage)
        # Store the trade information
        trades.append({
            "peak_index": peak_index,
            "peak_price": peak_price,
            "close_index": close_index,
            "close_price": close_price,
            "pnl": pnl
        })
    if compound:
        print(f"Starting equity: {starting_capital}$")
        print(f"Ending equity: {compound_win * starting_capital}$")
        print(f"ROI: {(compound_win - 1) * 100}%")
        print(f"Profit: {compound_win * starting_capital - starting_capital}$")
        roi = (compound_win - 1) * 100
        ending_equity = compound_win * starting_capital
        profit = compound_win * starting_capital - starting_capital
        metrics_text = (
            f"Starting Equity: ${starting_capital}\n"
            f"Ending Equity: ${ending_equity:.2f}\n"
            f"ROI: {roi:.2f}%\n"
            f"Profit: ${profit:.2f}\n"
            f"Leverage per trade: {leverage}x"
        )
        ax.text(
            0.98, 0.02, metrics_text,  # Bottom-right corner
            transform=ax.transAxes,
            fontsize=12,
            color="black",
            ha="right",  # Align text to the right
            va="bottom",  # Align text to the bottom
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5")
        )

    return trades




# Plot the EMA line
# Calculate PnL for trades
fig, ax = plt.subplots(figsize=(18, 9))
trades = calculate_pnl(peaks, trade_closes, df["close"].values,ax)

# Visualize the trades on the chart
#fig, ax = plt.subplots(figsize=(18, 9))

# Plot the close prices
ax.plot(df.index, df["close"], label="Close Price", color="gray", linewidth=1)

# Highlight peaks
for index, peak in peaks:
    ax.scatter(df.index[index], peak, color="red", label="Peak" if index == peaks[0][0] else "")

for index, close in trade_closes:
    ax.scatter(df.index[index], close, color="green", label="close short")

# Highlight peaks and trade closing points with PnL annotations
for trade in trades:
    peak_time = df.index[trade["peak_index"]]
    close_time = df.index[trade["close_index"]]
    peak_price = trade["peak_price"]
    close_price = trade["close_price"]
    pnl = trade["pnl"]

    # Plot arrows connecting the peak to the closing price
    ax.annotate(
        '',
        xy=(close_time, close_price),  # Arrow tip
        xytext=(peak_time, peak_price),  # Arrow tail
        arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='-|>', lw=1),
    )

    # Annotate the PnL percentage
    mid_time = peak_time + (close_time - peak_time) / 2  # Midpoint for annotation
    mid_price = peak_price + (close_price - peak_price) / 2  # Midpoint for annotation
    ax.annotate(
        f"{pnl:.2f}%",
        xy=(mid_time, mid_price),
        fontsize=10,
        color="green" if pnl >= 0 else "red",
        ha="center",
        va="center",
    )

# Plot buy and sell transactions from the wallet DataFrame
wallets_df['timeStamp'] = wallets_df['transactions'].apply(lambda x: pd.to_datetime(x['timeStamp']))

# Filter buy and sell transactions
buy_transactions = wallets_df[wallets_df['transactions'].apply(lambda x: x['transactionType'] == 'buy')]
sell_transactions = wallets_df[wallets_df['transactions'].apply(lambda x: x['transactionType'] == 'sell')]

# Set a fixed height for buy and sell markers
buy_height = df["close"].min() * 0.98  # Slightly below the minimum price
sell_height = df["close"].max() * 1.02  # Slightly above the maximum price

# Plot orange dots for buy transactions
ax.scatter(buy_transactions['timeStamp'], [buy_height] * len(buy_transactions), color='green', label='Buy Transactions', alpha=0.9)

# Plot pink dots for sell transactions
ax.scatter(sell_transactions['timeStamp'], [sell_height] * len(sell_transactions), color='pink', label='Sell Transactions', alpha=0.9)



# Formatting the chart
ax.set_title("Trades with PnL")
ax.set_xlabel("Time")
ax.set_ylabel("Price")

# Set the x-axis major locator to 1-hour intervals
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))

# Format the x-axis labels to show hours and minutes
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Rotate the x-axis labels for readability
plt.xticks(rotation=45)

ax.legend()
plt.grid()
plt.show()








