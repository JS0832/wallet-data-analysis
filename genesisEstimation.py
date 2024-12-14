from datetime import datetime

# Genesis timestamp
genesis_date = datetime(2020, 3, 16, 0, 0, 0)

# Target timestamp
target_date = datetime(2024, 11, 25, 0, 3, 50)

# Time difference in seconds
time_diff_seconds = (target_date - genesis_date).total_seconds()
print("Seconds since genesis:", time_diff_seconds)
# Average block time in seconds
average_block_time = 0.4

# Estimated block number
estimated_block = int(time_diff_seconds / average_block_time)
print("Estimated Block Number:", estimated_block)
