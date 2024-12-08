import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Function to update the chart
def update_chart(frame):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure the CSV has the necessary columns
        if 'timestamp' not in df.columns or 'tokenAmount' not in df.columns:
            print("Error: CSV must contain 'timestamp' and 'tokenAmount' columns.")
            return

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Filter the data for the last 4 hours
        four_hours_ago = pd.Timestamp.now() - pd.Timedelta(hours=20)
        df = df[df['timestamp'] > four_hours_ago]

        # Clear the plot for updating
        ax.clear()

        # Plot the filtered data
        ax.scatter(df['timestamp'], df['tokenAmount'], alpha=0.7)

        # Set plot labels and title
        ax.set_title('Transaction Data Points (Last 4 Hours)', fontsize=14)
        ax.set_xlabel('Timestamp', fontsize=12)
        ax.set_ylabel('Token Amount', fontsize=12)
        ax.grid(True)

        # Rotate x-axis labels for readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
    except Exception as e:
        print(f"Error during chart update: {e}")

# Set up the file path to the CSV file
file_path = "C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\BottedWalletsAnalysis\\merged_transactions_batch_8.csv"

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create the animation object
ani = FuncAnimation(fig, update_chart, interval=10000)  # Update every 10 seconds

# Show the plot
plt.tight_layout()
plt.show()
