import ccxt
import pandas as pd
import time
from datetime import datetime, timezone
import yfinance as yf


def get_crypto_data(ticker, start_date, timeframe='1d'):
    """
    Fetch cryptocurrency OHLCV data from Coinbase using CCXT
    
    Parameters:
    -----------
    ticker : str
        Cryptocurrency ticker symbol (e.g., 'BTC', 'ETH', 'SOL')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    timeframe : str
        Timeframe for the data (e.g., '1d' for daily, '1h' for hourly)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing cryptocurrency OHLCV data
    """
    print(f"Fetching {timeframe} {ticker} data from Coinbase...")
    
    # Initialize the exchange
    exchange = ccxt.coinbase()
    
    # Define the symbol and timeframe
    symbol = f'{ticker}/USD'
    
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
            print(f"Fetched {len(data)} {timeframe} {ticker} candles, total: {len(ohlcv_data)}")
            
            # Update 'since' to the last fetched timestamp + 1 millisecond to avoid duplicate data
            since = data[-1][0] + 1
            
            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)  # rateLimit is in milliseconds
            
            # Break if we've reached recent data
            if data[-1][0] > int(datetime.now(timezone.utc).timestamp() * 1000) - (60 * 60 * 1000):
                break
    except Exception as e:
        print(f"Error fetching {timeframe} {ticker} data: {e}")
        if len(ohlcv_data) == 0:
            raise
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Convert timestamp to a readable datetime format
    df[''] = pd.to_datetime(df[''], unit='ms')
    df.set_index('', inplace=True)
    
    return df

def get_risk_free_rate(start_date):
    # get todays date
    today = datetime.now().strftime('%Y-%m-%d')

    # get 15 week risk free rate
    rf_data = yf.download('^TNX', start='2015-01-01', end=today)['Close'] / 100
    rf_data.columns = ['Risk Free Rate']
    return rf_data


if __name__ == "__main__":
    # get crypto data
    start_date = '2017-01-01T00:00:00Z'  
    ticker = 'BTC' # 'BTC
    df = get_crypto_data(ticker, start_date, '1d')
    
    # Get risk free rate
    rf_data = get_risk_free_rate(start_date)
    
    # Merge with risk free rate data
    merged_df = df.merge(rf_data, left_index=True, right_index=True, how='left')
    merged_df['Risk Free Rate'] = merged_df['Risk Free Rate'].ffill()

    # reset index and name index as Date
    merged_df.reset_index(inplace=True)
    # name first column as Date
    merged_df.rename(columns={merged_df.columns[0]: 'Date'}, inplace=True)
    
    # Save to CSV
    import os
    path = os.path.join(os.path.dirname(__file__), f'Data/{ticker}.csv')
    merged_df.to_csv(path, index=False)
    print(f"Cryptocurrency and risk free rate data saved to {ticker}.csv")

    print(merged_df.columns)