import json
import pandas as pd
import matplotlib.pyplot as plt


# Function to read NDJSON and parse it into a list of dictionaries
def read_ndjson(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Function to process data and extract the top 5 most active wallets per timestamp
def process_top_wallets(data):
    records = []
    for record in data:
        timestamp = record['timestamp']
        wallet_map = record['walletMap']

        # Sort wallets by count (descending) and take the top 5
        sorted_wallets = sorted(wallet_map.items(), key=lambda x: x[1], reverse=True)[:5]

        # Format the top wallets with their first 4 letters and their counts
        top_wallets = [f"{wallet[:4]}: {count}" for wallet, count in sorted_wallets]
        total_count = sum(wallet_map.values())  # Total activity count at the timestamp

        records.append({
            'timestamp': timestamp,
            'top_wallets': ', '.join(top_wallets),
            'total_count': total_count
        })
    return pd.DataFrame(records)


# Function to plot total activity count over time and display top wallets
def plot_top_wallets(dataframe):
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['timestamp'], dataframe['total_count'], marker='o', linestyle='-', label='Total Activity Count')

    # Annotate top wallets
    for i, row in dataframe.iterrows():
        plt.text(row['timestamp'], row['total_count'], row['top_wallets'], fontsize=8, ha='right')

    plt.title('Total Activity and Top Wallets Per Timestamp')
    plt.xlabel('Timestamp')
    plt.ylabel('Total Activity Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.show()


# Example usage
file_path = 'C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\seen_wallets.ndjson'  # Path to your NDJSON file
data = read_ndjson(file_path)  # Read the NDJSON file
processed_data = process_top_wallets(data)  # Process data into a DataFrame
plot_top_wallets(processed_data)  # Plot the chart

# Example us
