from tvDatafeed import TvDatafeed, Interval
import pandas as pd
from datetime import datetime, timedelta

# Your TradingView credentials
username = 'YourTradingViewUsername'
password = 'YourTradingViewPassword'

# Initialize tvDatafeed
tv = TvDatafeed()

# Fetch the historical data
data = tv.get_hist(symbol='CHILLGUYUSDT', exchange='MEXC', interval=Interval.in_5_minute, n_bars=10000)
df = pd.DataFrame(data)

# Drop unwanted column
df = df.drop(columns=["symbol"])

# Generate the date column
current_time = datetime.now()
time_deltas = [current_time - timedelta(minutes=5* i) for i in range(len(df))]
df['date'] = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in reversed(time_deltas)]

# Save to CSV
csv_filename = "example_data.csv"
df.to_csv(csv_filename, index=False)
