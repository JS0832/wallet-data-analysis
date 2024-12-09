import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider
from scipy.signal import find_peaks
import json

# Paths to files
TRANSACTION_CSV = "C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\transaction_velocity.csv"
#TRANSACTION_CSV = "C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\transaction_velocity2.csv"
WALLET_ACTIVITY_NDJSON = "C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\seen_wallets.ndjson"  # Path to your NDJSON file
#WALLET_ACTIVITY_NDJSON = "C:\\Users\\RICHCEL.SOL\\Desktop\\MoneyMaker\\seen_wallets2.ndjson"

#todo also for each wallet check if its overall buying or selling.

def read_wallet_activity_old(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            timestamp = pd.to_datetime(record['timestamp'])  # Convert timestamp to datetime
            wallet_map = record['walletMap']
            # Extract all wallets
            wallets = [wallet for wallet in wallet_map.keys()]
            data.append({'timestamp': timestamp, 'wallets': ', '.join(wallets)})#here I want to also add if the wallet was sellign or buying at thsi timestamp
    return pd.DataFrame(data)

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


# Initialize the plot
fig, (ax1) = plt.subplots(
    1, 1, figsize=(16, 9), sharex=True, gridspec_kw={'height_ratios': [2]}
)

plt.subplots_adjust(bottom=0.2)  # Reserve 20% of the figure height for sliders
fig.patch.set_facecolor('#121212')  # Set figure background to dark gray
fig.patch.set_alpha(1.0)

def find_tops_and_bottoms(data, column='marketcap_ema_40', distance=150, min_difference_percent=4,
                          retracement_percent=20, min_time_distance=pd.Timedelta(minutes=2)):
    """
    Finds local tops (peaks) and bottoms (troughs) in the data with:
    1. Minimum percentage difference between tops and bottoms.
    2. A trailing mechanism to capture significant price rallies or drops.
    3. Filters out consecutive peaks or troughs that are too close.
    4. Enforces alternation between peaks and troughs.

    Args:
        data (pd.DataFrame): The data containing the column to analyze.
        column (str): The column to find peaks and troughs in.
        distance (int): Minimum horizontal distance between adjacent peaks/troughs.
        min_difference_percent (float): Minimum percentage difference between tops and bottoms.
        retracement_percent (float): Minimum retracement percentage to confirm a top after a rally.
        min_time_distance (pd.Timedelta): Minimum time distance between consecutive peaks or troughs.

    Returns:
        tuple: Indices of significant peaks and troughs.
    """
    # Calculate prominence threshold
    max_prominence = data[column].max() - data[column].min()
    prominence = 0.05 * max_prominence  # 5% of the range

    # Detect local maxima (peaks)
    peaks, _ = find_peaks(data[column], distance=distance, prominence=prominence)

    # Detect local minima (troughs)
    troughs, _ = find_peaks(-data[column], distance=distance, prominence=prominence)

    # Combine peaks and troughs in chronological order
    events = sorted([(idx, 'peak') for idx in peaks] + [(idx, 'trough') for idx in troughs])

    # Ensure alternation between peaks and troughs
    filtered_events = []
    last_event_type = None

    for idx, event_type in events:
        if last_event_type is None or event_type != last_event_type:
            # Enforce minimum time distance
            if not filtered_events or abs((data.iloc[idx]['datetime'] - data.iloc[filtered_events[-1][0]]['datetime']).total_seconds()) >= min_time_distance.total_seconds():
                filtered_events.append((idx, event_type))
                last_event_type = event_type

    # Separate back into peaks and troughs
    filtered_peaks = [idx for idx, event_type in filtered_events if event_type == 'peak']
    filtered_troughs = [idx for idx, event_type in filtered_events if event_type == 'trough']

    # Further filter based on minimum difference percentage and retracement
    significant_peaks = []
    significant_troughs = []
    trailing_high = None
    trailing_low = None

    for i in range(len(filtered_peaks) - 1):
        peak_idx = filtered_peaks[i]
        trough_idx = next((t for t in filtered_troughs if t > peak_idx), None)

        if trough_idx is not None:
            peak_value = data.iloc[peak_idx][column]
            trough_value = data.iloc[trough_idx][column]

            # Calculate percentage difference
            difference_percent = abs((peak_value - trough_value) / ((peak_value + trough_value) / 2)) * 100

            if difference_percent >= min_difference_percent:
                significant_peaks.append(peak_idx)
                significant_troughs.append(trough_idx)

                # Initialize trailing variables
                trailing_high = peak_value
                trailing_low = trough_value

        # Check for retracement mechanism after a rally
        if trailing_high is not None and trailing_low is not None:
            for j in range(peak_idx, len(data)):
                current_value = data.iloc[j][column]

                # Update trailing high
                if current_value > trailing_high:
                    trailing_high = current_value
                    peak_idx = j

                # Detect retracement
                retracement_threshold = trailing_high * (1 - retracement_percent / 100)
                if current_value < retracement_threshold:
                    significant_peaks.append(peak_idx)
                    break

            # Reset trailing variables for the next cycle
            trailing_high = None
            trailing_low = None

    return significant_peaks, significant_troughs
def annotate_peaks_and_troughs(ax, data, peaks, troughs, column='marketcap_ema_40', label_offset=0.02):
    """
    Annotates each detected peak and trough in chronological order.

    Args:
        ax: Matplotlib Axes object.
        data (pd.DataFrame): The data containing the column to analyze.
        peaks (list): Indices of detected peaks.
        troughs (list): Indices of detected troughs.
        column (str): The column to use for y-values.
        label_offset (float): Vertical offset for the labels.
    """
    # Combine peaks and troughs into a single sorted list
    events = sorted(peaks + troughs)
    labels = []

    # Determine chronological labels
    for i, idx in enumerate(events, start=1):
        x_value = data['datetime'].iloc[idx]
        y_value = data.iloc[idx][column]

        # Determine label type
        label_type = "Peak" if idx in peaks else "Trough"
        labels.append((label_type, i))
        ax.annotate(
            f"{label_type} {i}",
            xy=(x_value, y_value),
            xytext=(x_value, y_value + label_offset * (data[column].max() - data[column].min())),
            fontsize=8,
            color='white' if label_type == "Peak" else 'cyan',
            ha='center',
            va='bottom' if label_type == "Peak" else 'top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='white' if label_type == "Peak" else 'cyan',
                      facecolor='#333333', alpha=0.8)
        )

    return labels  # Return the labels for potential debugging or reference

def connect_tops_and_bottoms(ax, data, peaks, troughs, column='marketcap_ema_40', alpha=0.5, label_offset=0.02):
    """
    Connects each detected top (peak) to the next bottom (trough) and each bottom to the next top
    with semi-transparent lines. Adds percentage change labels along the connections.

    Args:
        ax: Matplotlib Axes object.
        data (pd.DataFrame): The data containing the column to analyze.
        peaks (list): Indices of detected peaks.
        troughs (list): Indices of detected troughs.
        column (str): The column to use for y-values.
        alpha (float): Transparency level of the lines.
        label_offset (float): Vertical offset for the percentage change labels.
    """
    # Combine peaks and troughs into a single sorted list
    events = sorted(peaks + troughs)

    for i in range(len(events) - 1):
        # Current event and the next event
        current_idx = events[i]
        next_idx = events[i + 1]

        # Get x and y values for current and next event
        x_current, y_current = data['datetime'].iloc[current_idx], data.iloc[current_idx][column]
        x_next, y_next = data['datetime'].iloc[next_idx], data.iloc[next_idx][column]

        # Draw the line
        ax.plot(
            [x_current, x_next], [y_current, y_next],
            linestyle='--', color='gray', alpha=alpha
        )

        # Calculate percentage change (absolute value)
        percentage_change = abs((y_next - y_current) / y_current) * 100

        # Add label for percentage change at the midpoint
        x_mid = (x_current + (x_next - x_current) / 2)  # Midpoint for datetime
        y_mid = (y_current + y_next) / 2  # Midpoint for marketcap_ema_40

        ax.annotate(
            f"{percentage_change:.2f}%",
            xy=(x_mid, y_mid + label_offset * (data[column].max() - data[column].min())),
            fontsize=8,
            color='yellow',
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='yellow', facecolor='#333333', alpha=0.8)
        )

def annotate_wallet_rankings(ax, wallet_activity, qualifying_wallets_list, peak_trough_timestamps, time_window):

    # Flatten wallet activity within neighborhoods and count unique occurrences
    wallet_counts = {wallet: 0 for wallet in qualifying_wallets_list}

    for timestamp in peak_trough_timestamps:
        # Get all wallets in the neighborhood
        nearby_wallets = wallet_activity[
            (wallet_activity['timestamp'] >= timestamp - time_window) &
            (wallet_activity['timestamp'] <= timestamp + time_window)
        ]

        # Flatten and deduplicate wallets
        unique_wallets_in_neighborhood = set(
            wallet for wallets_str in nearby_wallets['wallets'] for wallet in wallets_str.split(', ')
        )

        # Increment counts for qualifying wallets seen in this neighborhood
        for wallet in unique_wallets_in_neighborhood:
            if wallet in wallet_counts:
                wallet_counts[wallet] += 1

    # Prepare annotation text
    annotation_text = "Wallet Rankings:\n"
    for wallet, count in sorted(wallet_counts.items(), key=lambda x: -x[1]):
        if count > 0:  # Only display wallets with counts
            annotation_text += f"{wallet[:4]}: {count}\n"

    # Add the annotation to the top-right corner of the plot
    ax.annotate(
        annotation_text,
        xy=(1, 1),  # Position in top-right corner of the plot
        xycoords='axes fraction',
        fontsize=10,
        color='white',
        ha='right', va='top',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='#444444', alpha=0.7)
    )

def draw_neighborhood_ellipses(ax, timestamps, time_window, data, column='marketcap_ema_40', alpha=0.9):
    """
    Draws semi-transparent yellow ellipses to indicate the neighborhood around each peak or trough.

    Args:
        ax: Matplotlib Axes object.
        timestamps (list): List of peak and trough timestamps.
        time_window (pd.Timedelta): Time window around each peak/trough.
        data (pd.DataFrame): The data containing the column to analyze.
        column (str): The column to use for y-values.
        alpha (float): Transparency level of the ellipses.
    """
    for timestamp in timestamps:
        # Get the corresponding y-value for the timestamp
        nearest_idx = (data['datetime'] - timestamp).abs().idxmin()
        y_value = data.iloc[nearest_idx][column]

        # Define the width (time range) and height (price range)
        width = time_window.total_seconds() / 3600  # Convert to hours for plotting
        height = 0.1 * (data[column].max() - data[column].min())  # 10% of the y-range

        # Create and add the ellipse
        ellipse = Ellipse(
            xy=(timestamp, y_value),
            width=width,
            height=height,
            facecolor='yellow',
            alpha=alpha
        )
        ax.add_patch(ellipse)


#TODO add another parameter where the wallets dont necessarily have to be exclusively within the
# neighbourhoods of peaks and troths but "mostly" like 80% of the time ect...
def filter_wallets_for_neighborhoods_old(wallet_activity, peak_trough_timestamps, time_window, min_occurrences,
                                     within_hours=200):

    start_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=within_hours)
    # Filter wallet activity by start_time
    wallet_activity = wallet_activity[wallet_activity['timestamp'] >= start_time]
    # Identify wallets in the neighborhoods of peaks and troughs
    wallet_neighborhood_counts = {}
    for timestamp in peak_trough_timestamps:
        # Find wallets in the current neighborhood
        nearby_wallets = wallet_activity[
            (wallet_activity['timestamp'] >= timestamp - time_window) &
            (wallet_activity['timestamp'] <= timestamp + time_window)
        ]

        # Count each wallet only once per neighborhood using a set
        unique_wallets = set(
            wallet for wallets_str in nearby_wallets['wallets'] for wallet in wallets_str.split(', ')
        )

        for wallet in unique_wallets:
            if wallet not in wallet_neighborhood_counts:
                wallet_neighborhood_counts[wallet] = 0
            wallet_neighborhood_counts[wallet] += 1  # Increment by 1 for this neighborhood
    # Filter wallets that meet the minimum occurrence requirement
    qualifying_wallets = [
        wallet for wallet, count in wallet_neighborhood_counts.items() if count >= min_occurrences
    ]
    # Ensure wallets only appear in neighborhoods
    exclusive_wallets = []
    for wallet in qualifying_wallets:
        # Flatten the wallets column and filter for the specific wallet
        flattened_activity = wallet_activity.assign(
            flattened_wallets=wallet_activity['wallets'].str.split(', ')
        ).explode('flattened_wallets')

        # Filter for timestamps where the current wallet appears
        wallet_timestamps = flattened_activity[
            flattened_activity['flattened_wallets'] == wallet
            ]['timestamp']

        # Check if all appearances of the wallet are within the neighborhoods
        exclusively_in_neighborhoods = all(
            any(
                abs((ts - peak_or_trough).total_seconds()) <= time_window.total_seconds()
                for peak_or_trough in peak_trough_timestamps
            )
            for ts in wallet_timestamps
        )

        if exclusively_in_neighborhoods:
            exclusive_wallets.append(wallet)

    return exclusive_wallets


def filter_wallets_for_neighborhoods(
        wallet_activity, peak_trough_timestamps, time_window,
        min_occurrences, min_total_occurrences=8, min_percentage_in_neighborhood=30, within_hours=200
):
    """
    Filters wallets to include only those that:
    1. Appear at least `min_occurrences` times in the neighborhoods of peaks/troughs.
    2. Appear at least `min_total_occurrences` times in the entire dataset.
    3. Appear within neighborhoods at least `min_percentage_in_neighborhood`% of their total occurrences.

    Args:
        wallet_activity (pd.DataFrame): Wallet activity DataFrame.
        peak_trough_timestamps (list): List of peak and trough timestamps.
        time_window (pd.Timedelta): Time window around each peak/trough.
        min_occurrences (int): Minimum occurrences in neighborhoods.
        min_total_occurrences (int): Minimum total occurrences in the dataset.
        min_percentage_in_neighborhood (float): Minimum percentage of neighborhood occurrences.
        within_hours (int): Number of hours to consider for filtering the wallet activity.

    Returns:
        list: Wallets meeting the criteria.
    """
    start_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=within_hours)

    # Filter wallet activity by start_time
    wallet_activity = wallet_activity[wallet_activity['timestamp'] >= start_time]

    # Flatten the wallets column into individual wallet entries
    flattened_activity = wallet_activity.assign(
        flattened_wallets=wallet_activity['wallets'].str.split(', ')
    ).explode('flattened_wallets')

    # Count total occurrences of each wallet across the entire dataset
    total_occurrences = flattened_activity['flattened_wallets'].value_counts()

    # Identify wallets in the neighborhoods of peaks and troughs
    wallet_neighborhood_counts = {}
    for timestamp in peak_trough_timestamps:
        # Find wallets in the current neighborhood
        nearby_wallets = wallet_activity[
            (wallet_activity['timestamp'] >= timestamp - time_window) &
            (wallet_activity['timestamp'] <= timestamp + time_window)
            ]

        # Count each wallet only once per neighborhood using a set
        unique_wallets = set(
            wallet for wallets_str in nearby_wallets['wallets'] for wallet in wallets_str.split(', ')
        )
        for wallet in unique_wallets:
            if wallet not in wallet_neighborhood_counts:
                wallet_neighborhood_counts[wallet] = 0
            wallet_neighborhood_counts[wallet] += 1  # Increment by 1 for this neighborhood

    # Filter wallets that meet the minimum occurrence requirement in neighborhoods
    qualifying_wallets = [
        wallet for wallet, count in wallet_neighborhood_counts.items() if count >= min_occurrences
    ]

    # Apply additional criteria (total occurrences and percentage in neighborhoods)
    strict_qualifying_wallets = []
    for wallet in qualifying_wallets:
        total_count = total_occurrences.get(wallet, 0)
        neighborhood_count = wallet_neighborhood_counts.get(wallet, 0)

        # Check if wallet meets total occurrences and percentage criteria
        if total_count >= min_total_occurrences:
            percentage_in_neighborhood = (neighborhood_count / total_count) * 100
            if percentage_in_neighborhood >= min_percentage_in_neighborhood:
                strict_qualifying_wallets.append(wallet)

    return strict_qualifying_wallets


def annotate_deviation(ax, timestamp, y_value, wallet_activity, qualifying_wallets_list, time_window, amount_of_different_wallets):

    # Filter wallets in the neighborhood of the given timestamp
    nearby_wallets = wallet_activity[
        (wallet_activity['timestamp'] >= timestamp - time_window) &
        (wallet_activity['timestamp'] <= timestamp + time_window)
    ]

    if not nearby_wallets.empty:
        # Flatten the comma-separated wallets into individual entries
        flattened_wallets = nearby_wallets['wallets'].str.split(', ').explode().str.strip()
        qualifying_wallets_in_neighborhood = flattened_wallets[flattened_wallets.isin(qualifying_wallets_list)]
        # Count unique qualifying wallets
        unique_wallets_count = qualifying_wallets_in_neighborhood.nunique()
        #print(unique_wallets_count)
        # Check if the number of unique qualifying wallets meets the threshold
        if unique_wallets_count >= amount_of_different_wallets:
            # Adjust the annotation position by adding an offset to y_value
            offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 5% of the y-axis range
            ax.annotate(
                "Deviation Expected",
                (timestamp, y_value + offset),  # Add offset to y_value
                fontsize=8, color='cyan', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='cyan', facecolor='white', alpha=0.1)
            )

def annotate_wallet_activity(ax, timestamp, y_value, wallet_activity, qualifying_wallets_list, time_window):
    # Filter wallets in the neighborhood of the given timestamp
    nearby_wallets = wallet_activity[
        (wallet_activity['timestamp'] >= timestamp - time_window) &
        (wallet_activity['timestamp'] <= timestamp + time_window)
    ]

    if not nearby_wallets.empty:
        # Flatten the comma-separated wallets and their activities into individual rows
        flattened = nearby_wallets.assign(
            wallets=nearby_wallets['wallets'].str.split(', '),
            activity=nearby_wallets['activity'].str.split(', ').apply(lambda x: [float(a) for a in x])
        ).explode(['wallets', 'activity'])

        # Filter only qualifying wallets
        qualifying_data = flattened[flattened['wallets'].isin(qualifying_wallets_list)]

        if not qualifying_data.empty:
            # Calculate net activity per wallet
            net_activity = qualifying_data.groupby('wallets')['activity'].sum().reset_index()

            # Format the annotation text based on net activity
            annotation_text = ', '.join(
                f"{row['wallets'][:4]} {'buy' if row['activity'] > 0 else 'sell'}({abs(row['activity']):.0f})"
                for _, row in net_activity.iterrows() if row['activity'] != 0
            )

            if annotation_text:  # Only annotate if there is meaningful data
                # Adjust the annotation position by adding an offset to y_value
                percent = 0.05 #move Down 5%
                offset = percent * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 5% of the y-axis range
                ax.annotate(
                    annotation_text,
                    (timestamp, y_value + offset),  # Add offset to y_value
                    fontsize=8, color='yellow', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='yellow', facecolor='black', alpha=0.8)
                )


#silders
slider_neighborhood_size_window = plt.axes([0.2, 0.03, 0.3, 0.03], facecolor='#333333')
slider_neighborhood_size = Slider(slider_neighborhood_size_window, 'neighborhood size', valmin=1, valmax=30, valinit=5, valstep=1)

min_occurrences_slider_window = plt.axes([0.2, 0.01, 0.3, 0.03], facecolor='#333333')
min_occurrences_slider= Slider(min_occurrences_slider_window, 'Min occurrences', valmin=1, valmax=20, valinit=3, valstep=1)

amout_of_different_wallets_window = plt.axes([0.2, 0.05, 0.3, 0.03], facecolor='#333333')
amout_of_different_wallets_slider = Slider(amout_of_different_wallets_window, 'Minimum good wallets', valmin=1, valmax=10, valinit=2, valstep=1)


def update(none):
    transaction_data = pd.read_csv(TRANSACTION_CSV)
    transaction_data['datetime'] = pd.to_datetime(transaction_data['timestamp'], unit='s', utc=True)
    transaction_data['marketcap_ema_40'] = transaction_data['marketcap'].ewm(span=10, adjust=False).mean()
    # Read wallet activity once (static data for this example)
    wallet_activity_df = read_wallet_activity(WALLET_ACTIVITY_NDJSON)
    # Filter wallet activity across the entire dataset
    wallet_activity = wallet_activity_df.copy()

    # Ensure recent_time is timezone-aware
    recent_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=20000)

    # Filter recent transaction data
    transaction_data_recent = transaction_data[transaction_data['datetime'] >= recent_time]

    # Find peaks and troughs
    peaks, troughs = find_tops_and_bottoms(transaction_data)
    peak_trough_timestamps = transaction_data.iloc[peaks]['datetime'].tolist() + transaction_data.iloc[troughs]['datetime'].tolist()

    # Filter wallets for neighborhoods
    time_window = pd.Timedelta(minutes=slider_neighborhood_size.val)  # Define the neighborhood range

    qualifying_wallets_list = filter_wallets_for_neighborhoods(wallet_activity, peak_trough_timestamps, time_window, int(min_occurrences_slider.val))

    ax1.clear()
    annotate_wallet_rankings(ax1, wallet_activity, qualifying_wallets_list, peak_trough_timestamps, time_window)

    if not transaction_data_recent.empty:
        ax1.plot(transaction_data_recent['datetime'], transaction_data_recent['marketcap_ema_40'],
                 label="Market Cap EMA (40)", color="purple")

        for idx in peaks:
            timestamp = transaction_data.iloc[idx]['datetime']
            y_value = transaction_data.iloc[idx]['marketcap_ema_40']

            # Get all wallets in the neighborhood
            nearby_wallets = wallet_activity[
                (wallet_activity['timestamp'] >= timestamp - time_window) &
                (wallet_activity['timestamp'] <= timestamp + time_window)
                ]

            # Flatten and deduplicate wallets
            unique_wallets_in_neighborhood = set(
                wallet for wallets_str in nearby_wallets['wallets'] for wallet in wallets_str.split(', ')
            )

            # Count only qualifying wallets
            wallets_in_neighborhood = [
                wallet for wallet in unique_wallets_in_neighborhood if wallet in qualifying_wallets_list
            ]

            # Prepare wallets for annotation
            wallets_text = ', '.join(
                wallet[:4] for wallet in wallets_in_neighborhood) if wallets_in_neighborhood else ""

            # Annotate the peak
            ax1.annotate(
                f"{wallets_text}", (timestamp, y_value),
                fontsize=6, color='white', ha='center', va='bottom'
            )
            # Annotate deviation if multiple unique wallets are found
            annotate_deviation(
                ax1,
                timestamp,
                y_value,
                wallet_activity,
                qualifying_wallets_list,
                time_window,
                amout_of_different_wallets_slider.val
            )
            ax1.plot(timestamp, y_value, 'ro')
            annotate_wallet_activity(ax1, timestamp, y_value, wallet_activity, qualifying_wallets_list, time_window)

        for idx in troughs:
            timestamp = transaction_data.iloc[idx]['datetime']
            y_value = transaction_data.iloc[idx]['marketcap_ema_40']

            # Get all wallets in the neighborhood
            nearby_wallets = wallet_activity[
                (wallet_activity['timestamp'] >= timestamp - time_window) &
                (wallet_activity['timestamp'] <= timestamp + time_window)
                ]

            # Flatten and deduplicate wallets
            unique_wallets_in_neighborhood = set(
                wallet for wallets_str in nearby_wallets['wallets'] for wallet in wallets_str.split(', ')
            )

            # Count only qualifying wallets
            wallets_in_neighborhood = [
                wallet for wallet in unique_wallets_in_neighborhood if wallet in qualifying_wallets_list
            ]

            # Prepare wallets for annotation
            wallets_text = ', '.join(
                wallet[:4] for wallet in wallets_in_neighborhood) if wallets_in_neighborhood else ""

            # Annotate the trough
            ax1.annotate(
                f"{wallets_text}", (timestamp, y_value),
                fontsize=6, color='white', ha='center', va='top'
            )
            # Annotate deviation if multiple unique wallets are found
            annotate_deviation(
                ax1,
                timestamp,
                y_value,
                wallet_activity,
                qualifying_wallets_list,
                time_window,
                amout_of_different_wallets_slider.val
            )
            ax1.plot(timestamp, y_value, 'bo')
            annotate_wallet_activity(ax1, timestamp, y_value, wallet_activity, qualifying_wallets_list, time_window)

    connect_tops_and_bottoms(ax1, transaction_data, peaks, troughs, column='marketcap_ema_40', alpha=0.5)

    draw_neighborhood_ellipses(ax1, peak_trough_timestamps, time_window, transaction_data, column='marketcap_ema_40',
                               alpha=0.1)

    # Set x-axis limits
    start_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=20)
    end_time = pd.Timestamp.now(tz="UTC")
    #ax1.set_xlim([start_time, end_time])

    # Format the x-axis
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    ax1.set_facecolor('#1e1e1e')  # Darker gray for the plot area
    ax1.tick_params(colors='white')  # White ticks
    ax1.xaxis.label.set_color('white')  # White x-axis label
    ax1.yaxis.label.set_color('white')  # White y-axis label
    ax1.title.set_color('white')  # White title
    ax1.grid(color='#444444', linestyle='--', linewidth=0.5)  # Darker gridlines

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')


# Animate
ani = animation.FuncAnimation(fig, update, interval=2000)
plt.show()

