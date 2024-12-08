import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np
from scipy.signal import find_peaks
# Paths to CSV files
BOT_TX_VEL_CSV = "C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\botTxVel.csv"
TRANSACTION_CSV = "C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\transaction_velocity.csv"

# Initialize the plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    4, 1, figsize=(14, 20), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]}
)


def update(frame):
    # Read the botTxVel.csv data
    bot_data = pd.read_csv(BOT_TX_VEL_CSV)
    bot_data['datetime'] = pd.to_datetime(bot_data['timestamp'], unit='s')
    bot_data['walletcount_ema_40'] = bot_data['walletcount'].ewm(span=40, adjust=False).mean()
    bot_data['buys_ema_40'] = bot_data['buys'].ewm(span=40, adjust=False).mean()*10
    bot_data['sells_ema_40'] = bot_data['sells'].ewm(span=40, adjust=False).mean()
    bot_data['BsRatioPerTx_ema_40'] = bot_data['BsRatioPerTx'].ewm(span=10,
                                                                   adjust=False).mean()  # Add EMA for BsRatioPerTx

    # Read the transaction_velocity.csv data
    transaction_data = pd.read_csv(TRANSACTION_CSV)
    transaction_data['datetime'] = pd.to_datetime(transaction_data['timestamp'], unit='s')
    transaction_data['velocity_ema_40'] = transaction_data['velocity'].ewm(span=40, adjust=False).mean()
    transaction_data['marketcap_ema_40'] = transaction_data['marketcap'].ewm(span=10, adjust=False).mean()
    # Calculate Channel for Velocity
    transaction_data['velocity_std_40'] = transaction_data['velocity'].rolling(window=40).std()
    transaction_data['upper_channel'] = transaction_data['velocity_ema_40'] + 1.1 * transaction_data['velocity_std_40']
    transaction_data['lower_channel'] = transaction_data['velocity_ema_40'] - 1.1 * transaction_data['velocity_std_40']

    transaction_data['velocity_std_40'] = transaction_data['velocity'].rolling(window=40).std()
    transaction_data['breakout_threshold'] = transaction_data['velocity_ema_40'] + 3 * transaction_data[
        'velocity_std_40']

    # Identify Breakouts
    transaction_data['breakout'] = transaction_data['velocity'] > transaction_data['breakout_threshold']
    breakouts = transaction_data[transaction_data['breakout']]

    # Filter for the most recent 5-hour window
    recent_time = pd.Timestamp.now() - pd.Timedelta(hours=12)
    bot_data = bot_data[bot_data['datetime'] >= recent_time]
    transaction_data = transaction_data[transaction_data['datetime'] >= recent_time]

    # Clear the plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Plot Market Cap EMAs
    ax1.plot(transaction_data['datetime'], transaction_data['marketcap_ema_40'], label="Market Cap EMA (40)",
             color="purple")


    ax1.set_ylabel("Market Cap")
    ax1.legend(loc="upper left")
    ax1.grid()

    # Plot Wallet Count EMA
    ax2.plot(bot_data['datetime'], bot_data['walletcount_ema_40'], label="Wallet Count EMA (40)", color="blue")
    ax2.set_ylabel("Wallet Count")
    ax2.legend(loc="upper left")
    ax2.grid()

    # Plot Buys and Sells EMA
    ax3.plot(bot_data['datetime'], bot_data['buys_ema_40'], label="Buys EMA (40)", color="blue")
    ax3.plot(bot_data['datetime'], bot_data['sells_ema_40'], label="Sells EMA (40)", color="red")
    ax3.set_ylabel("Buys/Sells")
    ax3.legend(loc="upper left")
    ax3.grid()

    transaction_data['velocity_ema_40'] = transaction_data['velocity'].ewm(span=40, adjust=False).mean()
    # Plot the velocity, EMA, and polynomial fit
    ax4.clear()
    #ax4.plot(transaction_data['datetime'], transaction_data['velocity'], label="Velocity", color="blue", alpha=0.7)
    ax4.plot(transaction_data['datetime'], transaction_data['velocity_ema_40'], label="Velocity EMA (40)",
             color="orange")


    # Formatting and labeling
    ax4.set_title("Transaction Velocity with Polynomial Fit (Peaks)")
    ax4.set_ylabel("Velocity")
    ax4.legend(loc="upper left")
    ax4.grid()

    # Format the x-axis
    ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))  # Show labels every 1 minute
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.set_xlim([recent_time, pd.Timestamp.now()])

    # Rotate x-axis labels for all subplots
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()


# Animate
ani = animation.FuncAnimation(fig, update, interval=2000)
plt.show()
