import numpy as np
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import datetime

class FinancialDataAnalyzer:
    def __init__(self):
        """
        Initialize the Financial Data Analyzer
        """
        self.data = None
        self.processed_data = None
        self.clustered_data = None
        
    def fetch_financial_data(self):
        """
        Fetch financial data using alternative sources
        
        :return: Pandas DataFrame with financial data
        """
        # Method 1: Using NSE India's historical data CSV
        def fetch_nse_data(symbol, series='EQ'):
            """
            Fetch historical data from NSE India
            
            :param symbol: Stock symbol
            :param series: Series type (default 'EQ' for equity)
            :return: DataFrame with historical prices
            """
            try:
                # Construct URL for NSE historical data
                end_date = datetime.datetime.now().strftime('%d-%m-%Y')
                start_date = (datetime.datetime.now() - datetime.timedelta(days=365*5)).strftime('%d-%m-%Y')
                
                url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series={series}&from={start_date}&to={end_date}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data)
                    df['Date'] = pd.to_datetime(df['mTIMESTAMP'])
                    df.set_index('Date', inplace=True)
                    return df['close']
                else:
                    print(f"Failed to fetch data for {symbol}")
                    return None
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return None
        
        # Method 2: Synthetic data generation as a fallback
        def generate_synthetic_data(length=1260):  # 5 years of trading days
            """
            Generate synthetic financial data
            
            :param length: Number of data points
            :return: DataFrame with synthetic price data
            """
            np.random.seed(42)
            dates = pd.date_range(end=datetime.datetime.now(), periods=length, freq='B')
            
            # Generate synthetic price series with some market-like characteristics
            def generate_price_series(start_price=100, volatility=0.01):
                series = [start_price]
                for _ in range(length - 1):
                    change = np.random.normal(0, volatility)
                    next_price = series[-1] * (1 + change)
                    series.append(next_price)
                return pd.Series(series, index=dates)
            
            # Generate multiple synthetic data series
            data_dict = {
                'NIFTY_Close': generate_price_series(start_price=15000, volatility=0.02),
                'Bond_Close': generate_price_series(start_price=100, volatility=0.005),
                'SBIN_Close': generate_price_series(start_price=500, volatility=0.03),
                'RELIANCE_Close': generate_price_series(start_price=2000, volatility=0.025),
                'TCS_Close': generate_price_series(start_price=3500, volatility=0.015)
            }
            
            return pd.DataFrame(data_dict)
        
        # Attempt to fetch real data, fallback to synthetic
        # Try fetching for key stocks
        stocks = ['SBIN', 'RELIANCE', 'TCS']
        real_data = {}
        
        for stock in stocks:
            stock_data = fetch_nse_data(stock)
            if stock_data is not None:
                real_data[f'{stock}_Close'] = stock_data
        
        # If real data fetch fails, use synthetic data
        if not real_data:
            print("Falling back to synthetic financial data")
            self.data = generate_synthetic_data()
        else:
            # Combine real and synthetic data
            nifty_data = fetch_nse_data('NIFTY', series='INDEX')
            if nifty_data is not None:
                real_data['NIFTY_Close'] = nifty_data
            
            self.data = pd.DataFrame(real_data)
        
        return self.data
    
    def process_features(self):
        """
        Calculate features for clustering and prediction
        
        :return: Processed features DataFrame
        """
        if self.data is None:
            self.fetch_financial_data()
        
        # Calculate daily returns
        returns = self.data.pct_change().dropna()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=30).std().fillna(0)
        
        # Calculate moving averages
        moving_averages = self.data.rolling(window=30).mean().fillna(0)
        
        # Combine features
        processed_features = pd.concat([
            returns,
            volatility.add_suffix('_Volatility'),
            moving_averages.add_suffix('_MA30')
        ], axis=1).dropna()
        
        self.processed_data = processed_features
        return processed_features
    
    def cluster_market_days(self, n_clusters=4):
        """
        Perform K-Means clustering on market days
        
        :param n_clusters: Number of clusters to create
        :return: Clustered data
        """
        if self.processed_data is None:
            self.process_features()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.processed_data)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Add clusters to processed data
        self.processed_data['Cluster'] = clusters
        self.clustered_data = self.processed_data.copy()
        
        return self.clustered_data
    
    def train_prediction_models(self):
        """
        Train machine learning models to predict market day clusters
        
        :return: Trained models and their performance metrics
        """
        if self.clustered_data is None:
            self.cluster_market_days()
        
        # Prepare features and target
        X = self.clustered_data.drop('Cluster', axis=1)
        y = self.clustered_data['Cluster']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        
        self.trained_models = results
        return results
    
    def visualize_clusters(self):
        """
        Visualize market day clusters
        """
        if self.clustered_data is None:
            raise ValueError("Please perform clustering first")
        
        plt.figure(figsize=(12, 6))
        for cluster in self.clustered_data['Cluster'].unique():
            cluster_data = self.clustered_data[self.clustered_data['Cluster'] == cluster]
            plt.scatter(
                cluster_data.index, 
                cluster_data['NIFTY_Close'], 
                label=f'Cluster {cluster}'
            )
        
        plt.title('Market Day Clusters based on NIFTY Close')
        plt.xlabel('Date')
        plt.ylabel('NIFTY Close Price')
        plt.legend()
        plt.show()

# Example usage
def main():
    # Initialize analyzer
    analyzer = FinancialDataAnalyzer()
    
    try:
        # Fetch financial data
        data = analyzer.fetch_financial_data()
        
        # Process features
        processed_data = analyzer.process_features()
        
        # Cluster market days
        clustered_data = analyzer.cluster_market_days()
        
        # Train prediction models
        models = analyzer.train_prediction_models()
        
        # Print model performance
        for name, result in models.items():
            print(f"\n{name} Model Performance:")
            print(result['classification_report'])
        
        # Visualize clusters
        analyzer.visualize_clusters()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
