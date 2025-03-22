import ccxt
import pandas as pd
import yfinance as yf
import argparse
from datetime import datetime, timezone
import time
import os

def get_bitcoin_data(start_date):
    """
    Fetch Bitcoin OHLCV data from Coinbase using CCXT
    """
    print("Fetching Bitcoin data from Coinbase...")
    
    # Initialize the exchange
    exchange = ccxt.coinbase()
    
    # Define the symbol and timeframe
    symbol = 'BTC/USD'
    timeframe = '1d'  # Daily data
    
    # Convert the start date to a timestamp in milliseconds
    since = exchange.parse8601(start_date)
    
    # Fetch OHLCV data starting from the given date
    ohlcv_data = []
    limit = 1000  # Number of candles per fetch
    
    try:
        while True:
            # Fetch data
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not data:
                break
            
            ohlcv_data.extend(data)
            print(f"Fetched {len(data)} BTC candles, total: {len(ohlcv_data)}")
            
            # Update 'since' to the last fetched timestamp + 1 millisecond to avoid duplicate data
            since = data[-1][0] + 1
            
            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)  # rateLimit is in milliseconds
            
            # Break if we've reached recent data
            if data[-1][0] > int(datetime.now(timezone.utc).timestamp() * 1000) - (24 * 60 * 60 * 1000):
                break
    except Exception as e:
        print(f"Error fetching Bitcoin data: {e}")
        if len(ohlcv_data) == 0:
            raise
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp to a readable datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.index = df.index.normalize()  # Normalize to remove time component
    
    # Rename columns to be more descriptive
    df = df.rename(columns={
        'open': 'btc_open',
        'high': 'btc_high',
        'low': 'btc_low',
        'close': 'btc_close',
        'volume': 'btc_volume'
    })
    
    return df

def get_yfinance_data(start_date, end_date=None):
    """
    Fetch Gold, Oil, and 3-month Treasury yield data from Yahoo Finance
    """
    print("Fetching data from Yahoo Finance...")
    
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Tickers:
    # GLD - SPDR Gold Shares ETF
    # USO - United States Oil Fund
    # ^IRX - 13-week (3-month) Treasury yield
    # ^GSPC - S&P 500 Index
    tickers = ['GLD', 'USO', '^IRX', '^GSPC', 'BITW']
    
    # Dictionary to store dataframes
    dfs = {}
    
    for ticker in tickers:
        try:
            print(f"Fetching {ticker} data...")
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Keep only the 'Close' and 'Volume' columns
            if ticker == '^IRX':  # Treasury yields don't have meaningful volume
                data = data[['Close']]
                data.columns = ['treasury_3m_yield']
            elif ticker == '^GSPC':  # S&P 500 Index
                data = data[['Close', 'Volume']]
                data.columns = ['sp500_price', 'sp500_volume']
            elif ticker == 'BITW':  # Bitwise 10 Crypto Index
                data = data[['Close', 'Volume']]
                data.columns = ['crypto_index_price', 'crypto_index_volume']
            else:
                data = data[['Close', 'Volume']]
                data.columns = [f'{ticker.lower()}_price', f'{ticker.lower()}_volume']
            
            dfs[ticker] = data
        except Exception as e:
            print(f"Error fetching {ticker} data: {e}")
    
    # Merge the dataframes on index (date)
    result_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    
    for df in dfs.values():
        result_df = result_df.join(df)
    
    # Forward fill missing values (weekends/holidays)
    result_df = result_df.fillna(method='ffill')
    
    return result_df

def merge_data(btc_df, market_df):
    """
    Merge Bitcoin data with other market data
    """
    print("Merging datasets...")
    
    # Ensure the indices are DatetimeIndex
    if not isinstance(btc_df.index, pd.DatetimeIndex):
        btc_df.index = pd.to_datetime(btc_df.index)
    
    if not isinstance(market_df.index, pd.DatetimeIndex):
        market_df.index = pd.to_datetime(market_df.index)
    
    # Merge the dataframes on index (date)
    merged_df = btc_df.join(market_df, how='inner')
    
    print(f"Final dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    return merged_df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch financial data starting from a specific date')
    parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--output', type=str, default='financial_data.csv', 
                      help='Output CSV filename (default: financial_data.csv)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate start date format
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD")
    
    # Add time component to start date for CCXT
    start_date_ccxt = f"{args.start_date}T00:00:00Z"
    
    # Get Bitcoin data
    btc_data = get_bitcoin_data(start_date_ccxt)
    
    # Get market data
    market_data = get_yfinance_data(args.start_date)
    
    # Merge the datasets
    final_data = merge_data(btc_data, market_data)
    
    # Save to CSV
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    final_data.to_csv(args.output)
    print(f"Data saved to {args.output}")
    
    # Display sample
    print("\nSample of collected data:")
    print(final_data.head())
    
    # Display some statistics
    print("\nData statistics:")
    print(f"Date range: {final_data.index.min()} to {final_data.index.max()}")
    print(f"Bitcoin price range: ${final_data['btc_close'].min():.2f} to ${final_data['btc_close'].max():.2f}")
    
    # Check for missing values
    missing = final_data.isnull().sum()
    if missing.sum() > 0:
        print("\nWarning: Dataset contains missing values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values detected in the dataset.")

if __name__ == "__main__":
    main()