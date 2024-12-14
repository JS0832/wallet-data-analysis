#preprocess wallets and use that data to create another piece fo code to map their activity within the charts timestrame prepkea and psot peak
import json
import pandas as pd
WALLET_ACTIVITY_NDJSON = "C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\seen_wallets3.ndjson"

def read_wallet_activity(file_path): #uses the extra buying and selling data
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            timestamp = pd.to_datetime(record['timestamp'])  # Convert timestamp to datetime
            wallet_map = record['walletMap']

            # Extract wallets and their activity
            wallets = []
            activities = []  # Buying or selling

            for wallet, value in wallet_map.items():
                wallets.append(wallet)
                # Determine activity based on value
                activities.append(str(value))

            # Combine data into a single row
            data.append({
                'timestamp': timestamp,
                'wallets': ', '.join(wallets),  # Comma-separated wallets
                'activity': ', '.join(activities)  # Comma-separated activities
            })
    return pd.DataFrame(data)


#flatten the dataframe and take wallet that commonly appear in the flattened DF

wallet_activity_df = read_wallet_activity(WALLET_ACTIVITY_NDJSON)
wallet_activity = wallet_activity_df.copy()
flattened_activity = wallet_activity.assign(
        flattened_wallets=wallet_activity['wallets'].str.split(', ')
    ).explode('flattened_wallets')

wallet_counts = flattened_activity['flattened_wallets'].value_counts()

# Plot the wallet counts from top to lowest

# Determine the threshold for the top 50% wallets
total_wallets = len(wallet_counts)
top_50_percent_count = int(total_wallets * 0.05)  # Integer division to get the top 50% count

# Select the top 95% wallets
top_50_percent_wallets = wallet_counts.head(top_50_percent_count)

# Print the result
print(top_50_percent_wallets)
print(len(top_50_percent_wallets)) #about 250 for this data

# Convert the top 50% wallets into a DataFrame
top_wallets_df = top_50_percent_wallets.reset_index()
top_wallets_df.columns = ['wallet_address', 'occurrences']  # Rename columns

# Save the top wallets to a CSV file
output_csv_path = "top_wallets.csv"
top_wallets_df.to_csv(output_csv_path, index=False)

print(f"Top 95% wallets have been saved to {output_csv_path}")
