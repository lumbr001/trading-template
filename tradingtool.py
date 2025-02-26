import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Loading stock data from CSV and preprocess it."""
    def convert_volume(value):
        """Converting volume values from K, M, B to numerical format."""
        value = value.strip()
        if value == "":
            return 0  # Handle empty values
        elif value.endswith('M'):
            return float(value.strip('M')) * 1e6
        elif value.endswith('B'):
            return float(value.strip('B')) * 1e9
        elif value.endswith('K'):
            return float(value.strip('K')) * 1e3
        else:
            return float(value)
    
    df = pd.read_csv(file_path, 
                     parse_dates=['Date'],
                     dayfirst=True,
                     thousands=',',
                     converters={'Vol.': convert_volume,
                                 'Change %': lambda x: float(x.strip('%')) if x.strip() else 0})
    return df.sort_values('Date').reset_index(drop=True)

def calculate_moving_averages(df, short_window=50, long_window=200):
    """Calculateing moving averages for given window sizes."""
    df[f'{short_window}_day_MA'] = df['Price'].rolling(window=short_window).mean()
    df[f'{long_window}_day_MA'] = df['Price'].rolling(window=long_window).mean()
    return df

def plot_stock_data(df):
    """Ploting up the stock price with moving averages and trading volume."""
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Price'], label='Daily Price', alpha=0.5)
    plt.plot(df['Date'], df['50_day_MA'], label='50-Day Moving Average', color='orange')
    plt.plot(df['Date'], df['200_day_MA'], label='200-Day Moving Average', color='purple')
    plt.title('Stock Price Performance')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(14, 4))
    plt.bar(df['Date'], df['Vol.'] / 1e6, color='gray', alpha=0.5)
    plt.title('Trading Volume (Millions)')
    plt.xlabel('Date')
    plt.ylabel('Volume (M)')
    plt.tight_layout()
    plt.show()

def display_latest_statistics(df):
    """Printing up the latest stock statistics."""
    latest_data = {
        "Last Date": df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        "Latest Price": f"{df['Price'].iloc[-1]:,.2f}",
        "50-Day MA": f"{df['50_day_MA'].iloc[-1]:,.2f}",
        "200-Day MA": f"{df['200_day_MA'].iloc[-1]:,.2f}",
        "Volume (M)": f"{df['Vol.'].iloc[-1] / 1e6:.2f}M"
    }
    for key, value in latest_data.items():
        print(f"{key}: {value}")

# Main execution
if __name__ == "__main__":
    file_path = "Nifty Financial Services Historical Data.csv"  # Change as needed
    df = load_data(file_path)
    df = calculate_moving_averages(df)
    plot_stock_data(df)
    display_latest_statistics(df)
