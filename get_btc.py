import ccxt
import pandas as pd
import time
from datetime import datetime, timezone

def get_bitcoin_data(start_date, timeframe):
    """
    Fetch Bitcoin hourly OHLCV data from Coinbase using CCXT
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing hourly Bitcoin OHLCV data
    """
    print("Fetching hourly Bitcoin data from Coinbase...")
    
    # Initialize the exchange
    exchange = ccxt.coinbase()
    
    # Define the symbol and timeframe
    symbol = 'BTC/USD'
    
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
            print(f"Fetched {len(data)} hourly BTC candles, total: {len(ohlcv_data)}")
            
            # Update 'since' to the last fetched timestamp + 1 millisecond to avoid duplicate data
            since = data[-1][0] + 1
            
            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)  # rateLimit is in milliseconds
            
            # Break if we've reached recent data
            if data[-1][0] > int(datetime.now(timezone.utc).timestamp() * 1000) - (60 * 60 * 1000):
                break
    except Exception as e:
        print(f"Error fetching hourly Bitcoin data: {e}")
        if len(ohlcv_data) == 0:
            raise
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Convert timestamp to a readable datetime format
    df[''] = pd.to_datetime(df[''], unit='ms')
    df.set_index('', inplace=True)
    
    # Rename columns to be more descriptive
    
    return df


if __name__ == "__main__":
    start_date = '2015-01-01T00:00:00Z'  
    df = get_bitcoin_data(start_date, '1d')
    print(df.head())
    # Save to CSV

    import os
    path = os.path.join(os.path.dirname(__file__), 'btc_daily.csv')
    df.to_csv(path)
    print("Hourly Bitcoin data saved to btc_daily.csv")