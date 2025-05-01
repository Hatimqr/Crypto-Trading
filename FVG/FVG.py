import pandas as pd
from tqdm import tqdm


class FVGStrategy:
    """
    Fair Value Gap (FVG) strategy for Bitcoin trading.
    
    This class encapsulates the logic to identify Fair Value Gaps, detect key 
    price levels, and generate trading signals with corresponding position sizes.
    """
    
    def __init__(self, lookback_period=20 , body_multiplier=1.5, backcandles=50, test_candles=10):
        """
        Initialize the FVG strategy with the specified parameters.
        
        Args:
            lookback_period (int): Number of candles to look back for average body size.
            body_multiplier (float): Multiplier to determine significant body size.
            backcandles (int): Number of candles to look back for key level detection.
            test_candles (int): Number of candles before/after for key level validation.
            responsiveness (float in [0,1]) : how responsive are we going to be to buy/sell signals
        """
        self.lookback_period = int(lookback_period)
        self.body_multiplier = float(body_multiplier)
        self.backcandles = int(backcandles)
        self.test_candles = int(test_candles)

    def detect_fvg(self, data):
        """
        Detects Fair Value Gaps (FVGs) in historical price data.
        
        Args:
            data (DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close'].
            
        Returns:
            list of tuples: Each tuple contains ('type', start, end, index).
        """
        fvg_list = [None, None]  # First two candles can't form an FVG

        for i in tqdm(range(2, len(data)),
                      desc="Detecting Fair Value Gaps"):
            first_high = data['High'].iloc[i-2]
            first_low = data['Low'].iloc[i-2]
            middle_open = data['Open'].iloc[i-1]
            middle_close = data['Close'].iloc[i-1]
            third_low = data['Low'].iloc[i]
            third_high = data['High'].iloc[i]

            # Calculate the average absolute body size over the lookback period
            prev_bodies = (data['Close'].iloc[max(0, i-1-self.lookback_period):i-1] - 
                          data['Open'].iloc[max(0, i-1-self.lookback_period):i-1]).abs()
            avg_body_size = prev_bodies.mean()
            
            # Ensure avg_body_size is nonzero to avoid false positives
            avg_body_size = avg_body_size if avg_body_size > 0 else 0.001

            middle_body = abs(middle_close - middle_open)

            # Check for Bullish FVG
            if third_low > first_high and middle_body > avg_body_size * self.body_multiplier:
                fvg_list.append(('bullish', first_high, third_low, i))

            # Check for Bearish FVG
            elif third_high < first_low and middle_body > avg_body_size * self.body_multiplier:
                fvg_list.append(('bearish', first_low, third_high, i))
            else:
                fvg_list.append(None)

        return fvg_list
    
    def detect_key_levels(self, df, current_candle):
        """
        Detects key support and resistance levels in a given backcandles window.
        
        Args:
            df (DataFrame): DataFrame containing 'High' and 'Low' columns.
            current_candle (int): The index of the current candle.
            
        Returns:
            dict: A dictionary with detected 'support' and 'resistance' levels.
        """
        key_levels = {"support": [], "resistance": []}

        # Define the last candle that can be tested to avoid lookahead bias
        last_testable_candle = current_candle - self.test_candles

        # Ensure we have enough data
        if last_testable_candle < self.backcandles + self.test_candles:
            return key_levels  # Not enough historical data

        # Iterate through the backcandles window
        for i in range(current_candle - self.backcandles, last_testable_candle):
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]

            # Get surrounding window of test_candles before and after
            before = df.iloc[max(0, i - self.test_candles):i]
            after = df.iloc[i + 1: min(len(df), i + self.test_candles + 1)]

            # Check if current high is the highest among before & after candles
            if high > before['High'].max() and high > after['High'].max():
                key_levels["resistance"].append((i, high))

            # Check if current low is the lowest among before & after candles
            if low < before['Low'].min() and low < after['Low'].min():
                key_levels["support"].append((i, low))

        return key_levels
    
    def fill_key_levels(self, df):
        """
        Adds a 'key_levels' column to the DataFrame with detected support/resistance levels.
        
        Args:
            df (DataFrame): DataFrame containing 'High' and 'Low' columns.
            
        Returns:
            DataFrame: Updated DataFrame with the new 'key_levels' column.
        """
        df["key_levels"] = None  # Initialize the column
        
        for current_candle in tqdm(range(self.backcandles + self.test_candles, len(df)), 
                                  desc="Detecting Key Levels"):
            # Detect key levels for the current candle
            key_levels = self.detect_key_levels(df, current_candle)

            # Collect support and resistance levels (with their indices) up to current_candle
            support_levels = [(idx, level) for (idx, level) in key_levels["support"] 
                             if idx < current_candle]
            resistance_levels = [(idx, level) for (idx, level) in key_levels["resistance"] 
                                if idx < current_candle]

            # Store the levels along with the originating candle index
            if support_levels or resistance_levels:
                df.at[current_candle, "key_levels"] = {
                    "support": support_levels,
                    "resistance": resistance_levels
                }
                
        return df
    
    def detect_break_signal(self, df):
        """
        Detects if a candle has an FVG signal coinciding with price crossing a key level.
        
        Args:
            df (DataFrame): DataFrame with 'FVG' and 'key_levels' columns.
            
        Returns:
            DataFrame: Updated DataFrame with the new 'break_signal' column.
        """
        # Initialize the new signal column to 0
        df["break_signal"] = 0

        # We start at 1 because we compare candle i with its previous candle (i-1)
        for i in tqdm(range(1, len(df)),
                      desc="Detecting Break Signals"):
            fvg = df.loc[i, "FVG"]
            key_levels = df.loc[i, "key_levels"]

            # We only proceed if there's an FVG tuple and some key_levels dict
            if isinstance(fvg, tuple) and isinstance(key_levels, dict):
                fvg_type = fvg[0]  # "bullish" or "bearish"

                # Previous candle's OHLC
                prev_open = df.loc[i-1, "Open"]
                prev_close = df.loc[i-1, "Close"]

                # Bullish FVG check
                if fvg_type == "bullish":
                    # Check crossing a "resistance" level (from below -> above)
                    resistance_levels = key_levels.get("resistance", [])
                    
                    for (lvl_idx, lvl_price) in resistance_levels:
                        # Condition: previously below, ended above
                        if prev_open < lvl_price and prev_close > lvl_price:
                            df.loc[i, "break_signal"] = 1  # Buy signal
                            break  # No need to check more levels

                # Bearish FVG check
                elif fvg_type == "bearish":
                    # Check crossing a "support" level (from above -> below)
                    support_levels = key_levels.get("support", [])
                    
                    for (lvl_idx, lvl_price) in support_levels:
                        # Condition: previously above, ended below
                        if prev_open > lvl_price and prev_close < lvl_price:
                            df.loc[i, "break_signal"] = -1  # Sell signal
                            break  # No need to check more levels

        return df
    

    
    def calculate_position(self, df):
        # df has signal column

        df["position"] = 1
        for i in range(len(df)):
            if df.loc[i, "break_signal"] == 1:
                df.loc[i, "position"] = 1
            elif df.loc[i, "break_signal"] == -1:
                df.loc[i, "position"] = 0
            else:
                try:
                    df.loc[i, "position"] = df.loc[i-1, "position"]
                except:
                    pass # on day 0, we have no previous day, so we assume we are long (initialised everything to 1)

        
        return df



    
    def generate_signals(self, df):
        """
        Process the DataFrame and generate all signals and position sizes.
        This is the main method to call for implementing the strategy.
        
        Args:
            df (DataFrame): DataFrame with OHLC data.
            
        Returns:
            DataFrame: Processed DataFrame with all signals and position information.
        """
        processed_df = df.copy()
        
        # Ensure column names are standardized
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in processed_df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # If the dataframe has a DatetimeIndex, convert it for processing
        if isinstance(processed_df.index, pd.DatetimeIndex):
            processed_df = processed_df.reset_index()
            datetime_index = True
            datetime_column = processed_df.columns[0]  # Store the name of the datetime column
        else:
            datetime_index = False
        

        processed_df['FVG'] = self.detect_fvg(processed_df)
        
        processed_df = self.fill_key_levels(processed_df)
        
        processed_df = self.detect_break_signal(processed_df)
        
        # processed_df = self.calculate_position(processed_df)
        
        
        print("Signal Generation Complete")

        processed_df.rename(columns={'Unnamed: 0': 'Date', 'break_signal': 'Signal'}, inplace=True)
        processed_df['Date'] = pd.to_datetime(processed_df['Date'])
        return processed_df[['Date', 'Close', 'Signal']]
