import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math


# class Trader:
#     """
#     Evaluates trading signals and calculates performance metrics.
    
#     This class simulates trading based on provided signals and calculates
#     performance metrics like returns, Sharpe ratio, and drawdown. It also
#     provides benchmark comparisons. Cash not invested in Bitcoin earns the risk-free rate.
#     """
    
#     def __init__(self, initial_cash=100000.0, commission_rate=0.001, risk_free_rate=0.0):
#         """
#         Initialize the strategy evaluator.
        
#         Args:
#             initial_cash (float): Starting capital for the strategy.
#             commission_rate (float): Transaction fee as a decimal (e.g., 0.001 for 0.1%).
#             risk_free_rate (float): Annual risk-free rate for Sharpe ratio calculation.
#                                    If a pandas Series is provided, it will use the time-varying rates.
#         """
#         self.initial_cash = initial_cash
#         self.commission_rate = commission_rate
        
#         # Risk-free rate handling - can be a constant or a Series
#         self.risk_free_rate = risk_free_rate
        
#         # If risk_free_rate is a scalar, calculate daily equivalent
#         if not isinstance(risk_free_rate, pd.Series):
#             self.daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1  # Daily equivalent
#         else:
#             # Will handle daily conversion when using the Series
#             self.daily_risk_free = None
        
#         # Results storage
#         self.strategy_results = None
#         self.buy_hold_results = None
    
#     def get_daily_risk_free_rate(self, date_or_idx, df=None):
#         """
#         Get the appropriate daily risk-free rate for a given date/index.
        
#         Args:
#             date_or_idx: The date or index to look up the rate for
#             df: DataFrame containing dates if risk_free_rate is a Series with different indexing
            
#         Returns:
#             float: Daily risk-free rate
#         """
#         if isinstance(self.risk_free_rate, pd.Series):
#             # Try to get the rate for the specific date/index
#             try:
#                 # If risk_free_rate Series has date index
#                 if isinstance(self.risk_free_rate.index, pd.DatetimeIndex):
#                     # If date_or_idx is a date, use it directly
#                     if isinstance(date_or_idx, (pd.Timestamp, datetime)):
#                         date = date_or_idx
#                     # If it's an index, convert to date using passed DataFrame
#                     elif df is not None and isinstance(df.index, pd.DatetimeIndex):
#                         date = df.index[date_or_idx]
#                     else:
#                         # Default to first rate if we can't match
#                         return (1 + self.risk_free_rate.iloc[0]) ** (1/365) - 1
                    
#                     # Get the closest date in risk_free_rate Series
#                     closest_date = self.risk_free_rate.index[
#                         self.risk_free_rate.index.get_indexer([date], method='nearest')[0]
#                     ]
#                     annual_rate = self.risk_free_rate.loc[closest_date]
                    
#                 # If risk_free_rate Series has numeric index
#                 else:
#                     # If date_or_idx is numeric, use it to index (with bounds checking)
#                     if isinstance(date_or_idx, (int, np.integer)):
#                         idx = min(max(0, date_or_idx), len(self.risk_free_rate) - 1)
#                         annual_rate = self.risk_free_rate.iloc[idx]
#                     else:
#                         # Default to first rate if we can't match
#                         annual_rate = self.risk_free_rate.iloc[0]
                
#                 # Convert annual rate to daily
#                 return (1 + annual_rate) ** (1/365) - 1
                
#             except (KeyError, IndexError):
#                 # If rate not found, use the first available rate
#                 annual_rate = self.risk_free_rate.iloc[0]
#                 return (1 + annual_rate) ** (1/365) - 1
#         else:
#             # If risk_free_rate is a scalar, use the pre-computed daily rate
#             return self.daily_risk_free
    
#     def backtest_strategy(self, df):
#         """
#         Backtest the trading strategy based on signals in the DataFrame.
#         Cash not invested in Bitcoin earns the risk-free rate.
        
#         Args:
#             df (DataFrame): DataFrame with OHLC data, 'break_signal', 'buy_pct', and 'sell_pct' columns.
            
#         Returns:
#             DataFrame: Results of the strategy with portfolio values and position data.
#         """
#         # Create a copy of the DataFrame to avoid modifying the original
#         results = df.copy()
        
#         # Initialize portfolio tracking columns
#         results['cash'] = self.initial_cash
#         results['cash_interest'] = 0.0
#         results['risk_free_value'] = 0.0  # Value of cash + interest
#         results['bitcoin_qty'] = 0.0
#         results['bitcoin_value'] = 0.0
#         results['portfolio_value'] = self.initial_cash
#         results['trade_type'] = ''  # 'buy', 'sell', or ''
#         results['trade_amount'] = 0.0
#         results['trade_price'] = 0.0
#         results['commission'] = 0.0
        
#         # Simulate trading for each day
#         for i in range(1, len(results)):
#             # Get daily risk-free rate for this period
#             daily_rf_rate = self.get_daily_risk_free_rate(i, df)
            
#             # Default: carry forward previous values
#             results.loc[i, 'bitcoin_qty'] = results.loc[i-1, 'bitcoin_qty']
            
#             # Calculate interest earned on cash
#             interest_earned = results.loc[i-1, 'cash'] * daily_rf_rate
#             results.loc[i, 'cash_interest'] = results.loc[i-1, 'cash_interest'] + interest_earned
            
#             # Update cash with interest
#             results.loc[i, 'cash'] = results.loc[i-1, 'cash'] + interest_earned
#             results.loc[i, 'risk_free_value'] = results.loc[i, 'cash']
            
#             signal = results.loc[i, 'break_signal']
#             close_price = results.loc[i, 'Close']

#             # Process buy signal (signal == 2)
#             if signal == 2:
#                 buy_pct = results.loc[i, 'buy_pct']
#                 cash_to_use = results.loc[i, 'cash'] * buy_pct  # Use current cash (with interest)
                
#                 if cash_to_use > 0 and results.loc[i, 'cash'] > 0:
#                     # Calculate commission
#                     commission = cash_to_use * self.commission_rate
                    
#                     # Calculate actual BTC amount to buy after commission
#                     actual_cash_used = cash_to_use - commission
#                     btc_bought = actual_cash_used / close_price
                    
#                     # Update portfolio
#                     results.loc[i, 'cash'] = results.loc[i, 'cash'] - cash_to_use
#                     results.loc[i, 'risk_free_value'] = results.loc[i, 'cash']  # Update risk-free value
#                     results.loc[i, 'bitcoin_qty'] = results.loc[i, 'bitcoin_qty'] + btc_bought
                    
#                     # Record trade details
#                     results.loc[i, 'trade_type'] = 'buy'
#                     results.loc[i, 'trade_amount'] = btc_bought
#                     results.loc[i, 'trade_price'] = close_price
#                     results.loc[i, 'commission'] = commission
            
#             # Process sell signal (signal == 1)
#             elif signal == 1:
#                 sell_pct = results.loc[i, 'sell_pct']
#                 btc_to_sell = results.loc[i, 'bitcoin_qty'] * sell_pct
                
#                 if btc_to_sell > 0 and results.loc[i, 'bitcoin_qty'] > 0:
#                     # Calculate sale value and commission
#                     sale_value = btc_to_sell * close_price
#                     commission = sale_value * self.commission_rate
                    
#                     # Update portfolio
#                     results.loc[i, 'cash'] = results.loc[i, 'cash'] + (sale_value - commission)
#                     results.loc[i, 'risk_free_value'] = results.loc[i, 'cash']  # Update risk-free value
#                     results.loc[i, 'bitcoin_qty'] = results.loc[i, 'bitcoin_qty'] - btc_to_sell
                    
#                     # Record trade details
#                     results.loc[i, 'trade_type'] = 'sell'
#                     results.loc[i, 'trade_amount'] = btc_to_sell
#                     results.loc[i, 'trade_price'] = close_price
#                     results.loc[i, 'commission'] = commission
            
#             # Update portfolio valuation
#             results.loc[i, 'bitcoin_value'] = results.loc[i, 'bitcoin_qty'] * close_price
#             results.loc[i, 'portfolio_value'] = results.loc[i, 'risk_free_value'] + results.loc[i, 'bitcoin_value']
        
#         # Calculate daily returns
#         results['daily_return'] = results['portfolio_value'].pct_change()
        
#         # Store results
#         self.strategy_results = results
        
#         return results
    
#     def calculate_buy_and_hold(self, df):
#         """
#         Calculate the performance of a buy-and-hold strategy for comparison.
#         Cash not invested in Bitcoin earns the risk-free rate.
        
#         Args:
#             df (DataFrame): DataFrame with OHLC data.
            
#         Returns:
#             DataFrame: Results of the buy-and-hold strategy.
#         """
#         # Create a copy for buy-and-hold simulation
#         bh_results = df.copy()
        
#         # Initialize portfolio tracking columns
#         bh_results['bh_cash'] = 0.0
#         bh_results['bh_cash_interest'] = 0.0
#         bh_results['bh_risk_free_value'] = 0.0
#         bh_results['bh_bitcoin_qty'] = 0.0
#         bh_results['bh_bitcoin_value'] = 0.0
#         bh_results['bh_portfolio_value'] = self.initial_cash
        
#         # Buy as much Bitcoin as possible on the first day
#         first_price = bh_results.loc[0, 'Close']
#         commission = self.initial_cash * self.commission_rate
#         cash_after_commission = self.initial_cash - commission
#         btc_bought = cash_after_commission / first_price
        
#         # Set initial portfolio state
#         bh_results.loc[0, 'bh_cash'] = self.initial_cash - cash_after_commission
#         bh_results.loc[0, 'bh_risk_free_value'] = bh_results.loc[0, 'bh_cash']
#         bh_results.loc[0, 'bh_bitcoin_qty'] = btc_bought
#         bh_results.loc[0, 'bh_bitcoin_value'] = btc_bought * first_price
#         bh_results.loc[0, 'bh_portfolio_value'] = bh_results.loc[0, 'bh_risk_free_value'] + bh_results.loc[0, 'bh_bitcoin_value']
        
#         # Calculate daily values, accumulating interest on cash
#         for i in range(1, len(bh_results)):
#             # Get daily risk-free rate for this period
#             daily_rf_rate = self.get_daily_risk_free_rate(i, df)
#             close_price = bh_results.loc[i, 'Close']
            
#             # Calculate interest on cash
#             interest_earned = bh_results.loc[i-1, 'bh_cash'] * daily_rf_rate
#             bh_results.loc[i, 'bh_cash_interest'] = bh_results.loc[i-1, 'bh_cash_interest'] + interest_earned
            
#             # Update cash with interest
#             bh_results.loc[i, 'bh_cash'] = bh_results.loc[i-1, 'bh_cash'] + interest_earned
#             bh_results.loc[i, 'bh_risk_free_value'] = bh_results.loc[i, 'bh_cash']
            
#             # Keep the same BTC quantity throughout
#             bh_results.loc[i, 'bh_bitcoin_qty'] = bh_results.loc[0, 'bh_bitcoin_qty']
#             bh_results.loc[i, 'bh_bitcoin_value'] = bh_results.loc[i, 'bh_bitcoin_qty'] * close_price
            
#             # Update total portfolio value
#             bh_results.loc[i, 'bh_portfolio_value'] = bh_results.loc[i, 'bh_risk_free_value'] + bh_results.loc[i, 'bh_bitcoin_value']
        
#         # Calculate daily returns
#         bh_results['bh_daily_return'] = bh_results['bh_portfolio_value'].pct_change()
        
#         # Store results
#         self.buy_hold_results = bh_results
        
#         return bh_results
    
#     def calculate_sharpe_ratio(self, returns, annualization_factor=365):
#         """
#         Calculate the Sharpe Ratio for a series of returns.
        
#         Args:
#             returns (Series): Daily return series.
#             annualization_factor (int, optional): Factor to annualize the ratio. Defaults to 365 for daily returns.
            
#         Returns:
#             float: Annualized Sharpe Ratio.
#         """
#         # Remove NaN values
#         returns = returns.dropna()
        
#         if len(returns) == 0:
#             return 0.0
        
#         # For Sharpe ratio, we'll use the first risk-free rate if it's a Series
#         if isinstance(self.risk_free_rate, pd.Series):
#             annual_rf_rate = self.risk_free_rate.iloc[0]
#             daily_rf = (1 + annual_rf_rate) ** (1/365) - 1
#         else:
#             daily_rf = self.daily_risk_free
        
#         # Calculate excess returns
#         excess_returns = returns - daily_rf
        
#         # Calculate Sharpe ratio (if std is 0, return 0 to avoid division by zero)
#         std_dev = returns.std()
#         if std_dev == 0:
#             return 0.0
            
#         sharpe = (excess_returns.mean() / std_dev) * math.sqrt(annualization_factor)
#         return sharpe
    
#     def calculate_drawdown(self, values):
#         """
#         Calculate the drawdown series and maximum drawdown.
        
#         Args:
#             values (Series): Portfolio value series.
            
#         Returns:
#             tuple: (drawdown_series, max_drawdown_percentage)
#         """
#         # Calculate cumulative maximum
#         cumulative_max = values.cummax()
        
#         # Calculate drawdown series
#         drawdown_series = (values - cumulative_max) / cumulative_max
        
#         # Get maximum drawdown (as a positive percentage)
#         max_drawdown = -drawdown_series.min() * 100 if not drawdown_series.empty else 0.0
        
#         return drawdown_series, max_drawdown
    
#     def calculate_win_rate(self):
#         """
#         Calculate the win rate for all completed trades.
        
#         Returns:
#             float: Percentage of profitable trades.
#         """
#         if self.strategy_results is None:
#             return 0.0
            
#         # Find all trades
#         trades = self.strategy_results[self.strategy_results['trade_type'] != ''].copy()
        
#         if len(trades) == 0:
#             return 0.0
            
#         # Extract buy and sell pairs
#         buys = trades[trades['trade_type'] == 'buy'].reset_index()
#         sells = trades[trades['trade_type'] == 'sell'].reset_index()
        
#         wins = 0
#         total_pairs = min(len(buys), len(sells))
        
#         for buy_idx in range(total_pairs):
#             buy_price = buys.loc[buy_idx, 'trade_price']
#             sell_idx = np.searchsorted(sells['index'], buys.loc[buy_idx, 'index'])
            
#             # If we found a matching sell after this buy
#             if sell_idx < len(sells):
#                 sell_price = sells.loc[sell_idx, 'trade_price']
#                 if sell_price > buy_price:
#                     wins += 1
        
#         return (wins / total_pairs * 100) if total_pairs > 0 else 0.0
    
#     def generate_performance_report(self):
#         """
#         Generate a detailed performance report for the strategy and benchmark.
        
#         Returns:
#             dict: Dictionary containing all performance metrics.
#         """
#         if self.strategy_results is None or self.buy_hold_results is None:
#             raise ValueError("Must run backtest_strategy and calculate_buy_and_hold first")
        
#         # Overall return calculation
#         strategy_start = self.strategy_results['portfolio_value'].iloc[0]
#         strategy_end = self.strategy_results['portfolio_value'].iloc[-1]
#         strategy_return = ((strategy_end / strategy_start) - 1) * 100
        
#         bh_start = self.buy_hold_results['bh_portfolio_value'].iloc[0]
#         bh_end = self.buy_hold_results['bh_portfolio_value'].iloc[-1]
#         bh_return = ((bh_end / bh_start) - 1) * 100
        
#         # Sharpe ratio calculation
#         strategy_sharpe = self.calculate_sharpe_ratio(self.strategy_results['daily_return'])
#         bh_sharpe = self.calculate_sharpe_ratio(self.buy_hold_results['bh_daily_return'])
        
#         # Drawdown calculation
#         _, strategy_max_dd = self.calculate_drawdown(self.strategy_results['portfolio_value'])
#         _, bh_max_dd = self.calculate_drawdown(self.buy_hold_results['bh_portfolio_value'])
        
#         # Win rate calculation
#         win_rate = self.calculate_win_rate()
        
#         # Trade count
#         total_trades = len(self.strategy_results[self.strategy_results['trade_type'] != ''])
#         buy_trades = len(self.strategy_results[self.strategy_results['trade_type'] == 'buy'])
#         sell_trades = len(self.strategy_results[self.strategy_results['trade_type'] == 'sell'])
        
#         # Interest earned
#         interest_earned = self.strategy_results['cash_interest'].iloc[-1]
#         bh_interest_earned = self.buy_hold_results['bh_cash_interest'].iloc[-1]
        
#         # Compile the performance report
#         report = {
#             'strategy_return': strategy_return,
#             'bh_return': bh_return,
#             'outperformance': strategy_return - bh_return,
#             'strategy_sharpe': strategy_sharpe,
#             'bh_sharpe': bh_sharpe,
#             'sharpe_improvement': strategy_sharpe - bh_sharpe,
#             'strategy_max_drawdown': strategy_max_dd,
#             'bh_max_drawdown': bh_max_dd,
#             'drawdown_improvement': bh_max_dd - strategy_max_dd,
#             'win_rate': win_rate,
#             'total_trades': total_trades,
#             'buy_trades': buy_trades,
#             'sell_trades': sell_trades,
#             'interest_earned': interest_earned,
#             'bh_interest_earned': bh_interest_earned,
#             'final_portfolio_value': strategy_end,
#             'final_bh_value': bh_end,
#         }
        
#         return report
    
#     def plot_results(self, output_path=None):
#         """
#         Generate plots comparing strategy and buy-and-hold performance.
#         Also shows cash and equity composition over time.
        
#         Args:
#             output_path (str, optional): Path to save the plot. If None, the plot is displayed instead.
            
#         Returns:
#             None
#         """
#         if self.strategy_results is None or self.buy_hold_results is None:
#             raise ValueError("Must run backtest_strategy and calculate_buy_and_hold first")
        
#         # Create a figure with 3 subplots
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), 
#                                           gridspec_kw={'height_ratios': [3, 1, 2]})
        
#         # Plot 1: Portfolio Value Comparison
#         ax1.plot(self.strategy_results.index, self.strategy_results['portfolio_value'], 
#                 label='FVG Strategy', color='blue')
#         ax1.plot(self.buy_hold_results.index, self.buy_hold_results['bh_portfolio_value'], 
#                 label='Buy & Hold', color='orange', linestyle='--')
        
#         # Mark buy signals
#         buys = self.strategy_results[self.strategy_results['trade_type'] == 'buy']
#         ax1.scatter(buys.index, buys['portfolio_value'], color='green', marker='^', s=100, label='Buy')
        
#         # Mark sell signals
#         sells = self.strategy_results[self.strategy_results['trade_type'] == 'sell']
#         ax1.scatter(sells.index, sells['portfolio_value'], color='red', marker='v', s=100, label='Sell')
        
#         ax1.set_title('Portfolio Value Comparison', fontsize=14)
#         ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # Plot 2: Drawdown
#         strategy_dd, _ = self.calculate_drawdown(self.strategy_results['portfolio_value'])
#         bh_dd, _ = self.calculate_drawdown(self.buy_hold_results['bh_portfolio_value'])
        
#         ax2.plot(self.strategy_results.index, strategy_dd * 100, label='FVG Strategy', color='blue')
#         ax2.plot(self.buy_hold_results.index, bh_dd * 100, label='Buy & Hold', color='orange', linestyle='--')
        
#         ax2.set_title('Drawdown Comparison', fontsize=14)
#         ax2.set_ylabel('Drawdown (%)', fontsize=12)
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         # Invert y-axis for drawdown (negative is down)
#         ax2.invert_yaxis()
        
#         # Plot 3: Cash (with interest) and Bitcoin Holdings Over Time
#         ax3.stackplot(self.strategy_results.index, 
#                      [self.strategy_results['risk_free_value'], self.strategy_results['bitcoin_value']],
#                      labels=['Cash + Interest', 'Bitcoin Holdings'],
#                      colors=['lightgreen', 'lightblue'],
#                      alpha=0.7)
        
#         # Add a line for total portfolio value
#         ax3.plot(self.strategy_results.index, self.strategy_results['portfolio_value'],
#                 label='Total Value', color='darkblue', linewidth=2)
        
#         ax3.set_title('Strategy Allocation Over Time', fontsize=14)
#         ax3.set_ylabel('Value ($)', fontsize=12)
#         ax3.set_xlabel('Date', fontsize=12)
#         ax3.legend(loc='upper left')
#         ax3.grid(True, alpha=0.3)
        
#         # Add buy/sell annotations on the allocation chart too
#         ax3.scatter(buys.index, buys['portfolio_value'], color='green', marker='^', s=80)
#         ax3.scatter(sells.index, sells['portfolio_value'], color='red', marker='v', s=80)
        
#         plt.tight_layout()
        
#         # Save or display the plot
#         if output_path:
#             plt.savefig(output_path)
#             plt.close()
#         else:
#             plt.show()
    
#     def print_performance_summary(self):
#         """
#         Print a summary of the strategy's performance metrics.
        
#         Returns:
#             None
#         """
#         if self.strategy_results is None or self.buy_hold_results is None:
#             raise ValueError("Must run backtest_strategy and calculate_buy_and_hold first")
        
#         report = self.generate_performance_report()
        
#         print("=" * 60)
#         print(f"{'PERFORMANCE SUMMARY':^60}")
#         print("=" * 60)
        
#         print(f"\n{'RETURNS':^60}")
#         print("-" * 60)
#         print(f"Strategy Total Return:     {report['strategy_return']:>10.2f}%")
#         print(f"Buy & Hold Total Return:   {report['bh_return']:>10.2f}%")
#         print(f"Outperformance:            {report['outperformance']:>10.2f}%")
        
#         print(f"\n{'RISK METRICS':^60}")
#         print("-" * 60)
#         print(f"Strategy Sharpe Ratio:     {report['strategy_sharpe']:>10.2f}")
#         print(f"Buy & Hold Sharpe Ratio:   {report['bh_sharpe']:>10.2f}")
#         sharpe_improvement = (report['strategy_sharpe'] - report['bh_sharpe'])/report['bh_sharpe'] * 100
#         print(f"Sharpe Improvement:        {sharpe_improvement:>10.2f}%")
#         print(f"Strategy Max Drawdown:     {report['strategy_max_drawdown']:>10.2f}%")
#         print(f"Buy & Hold Max Drawdown:   {report['bh_max_drawdown']:>10.2f}%")
#         drawdown_improvement = (report['bh_max_drawdown'] - report['strategy_max_drawdown'])/report['bh_max_drawdown'] * 100
#         print(f"Drawdown Improvement:      {drawdown_improvement:>10.2f}%")
        
#         print(f"\n{'TRADE STATISTICS':^60}")
#         print("-" * 60)
#         print(f"Win Rate:                  {report['win_rate']:>10.2f}%")
#         print(f"Total Trades:              {report['total_trades']:>10}")
#         print(f"Buy Trades:                {report['buy_trades']:>10}")
#         print(f"Sell Trades:               {report['sell_trades']:>10}")
#         print(f"Interest Earned:           {report['interest_earned']:>10.2f}$")
#         print(f"B&H Interest Earned:       {report['bh_interest_earned']:>10.2f}$")
        
#         print(f"\n{'FINAL VALUES':^60}")
#         print("-" * 60)
#         print(f"Starting Capital:          {self.initial_cash:>10.2f}$")
#         print(f"Strategy Final Value:      {report['final_portfolio_value']:>10.2f}$")
#         print(f"Buy & Hold Final Value:    {report['final_bh_value']:>10.2f}$")
        
#         print("\n" + "=" * 60)






# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import math


class TraderWithStoploss:
    """
    Evaluates trading signals and calculates performance metrics.
    
    This class simulates trading based on provided signals and calculates
    performance metrics like returns, Sharpe ratio, and drawdown. It also
    provides benchmark comparisons. Cash not invested in Bitcoin earns the risk-free rate.
    """
    
    def __init__(self, initial_cash=100000.0, commission_rate=0.001, risk_free_rate=0.0, 
                 max_drawdown_threshold=None):
        """
        Initialize the strategy evaluator.
        
        Args:
            initial_cash (float): Starting capital for the strategy.
            commission_rate (float): Transaction fee as a decimal (e.g., 0.001 for 0.1%).
            risk_free_rate (float): Annual risk-free rate for Sharpe ratio calculation.
                                   If a pandas Series is provided, it will use the time-varying rates.
            max_drawdown_threshold (float, optional): Maximum allowed drawdown as a percentage (e.g., 20 for 20%).
                                                     If exceeded, positions will be liquidated.
        """
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.max_drawdown_threshold = max_drawdown_threshold
        
        # Risk-free rate handling - can be a constant or a Series
        self.risk_free_rate = risk_free_rate
        
        # If risk_free_rate is a scalar, calculate daily equivalent
        if not isinstance(risk_free_rate, pd.Series):
            self.daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1  # Daily equivalent
        else:
            # Will handle daily conversion when using the Series
            self.daily_risk_free = None
        
        # Results storage
        self.strategy_results = None
        self.buy_hold_results = None
    
    def get_daily_risk_free_rate(self, date_or_idx, df=None):
        """
        Get the appropriate daily risk-free rate for a given date/index.
        
        Args:
            date_or_idx: The date or index to look up the rate for
            df: DataFrame containing dates if risk_free_rate is a Series with different indexing
            
        Returns:
            float: Daily risk-free rate
        """
        if isinstance(self.risk_free_rate, pd.Series):
            # Try to get the rate for the specific date/index
            try:
                # If risk_free_rate Series has date index
                if isinstance(self.risk_free_rate.index, pd.DatetimeIndex):
                    # If date_or_idx is a date, use it directly
                    if isinstance(date_or_idx, (pd.Timestamp, datetime)):
                        date = date_or_idx
                    # If it's an index, convert to date using passed DataFrame
                    elif df is not None and isinstance(df.index, pd.DatetimeIndex):
                        date = df.index[date_or_idx]
                    else:
                        # Default to first rate if we can't match
                        return (1 + self.risk_free_rate.iloc[0]) ** (1/365) - 1
                    
                    # Get the closest date in risk_free_rate Series
                    closest_date = self.risk_free_rate.index[
                        self.risk_free_rate.index.get_indexer([date], method='nearest')[0]
                    ]
                    annual_rate = self.risk_free_rate.loc[closest_date]
                    
                # If risk_free_rate Series has numeric index
                else:
                    # If date_or_idx is numeric, use it to index (with bounds checking)
                    if isinstance(date_or_idx, (int, np.integer)):
                        idx = min(max(0, date_or_idx), len(self.risk_free_rate) - 1)
                        annual_rate = self.risk_free_rate.iloc[idx]
                    else:
                        # Default to first rate if we can't match
                        annual_rate = self.risk_free_rate.iloc[0]
                
                # Convert annual rate to daily
                return (1 + annual_rate) ** (1/365) - 1
                
            except (KeyError, IndexError):
                # If rate not found, use the first available rate
                annual_rate = self.risk_free_rate.iloc[0]
                return (1 + annual_rate) ** (1/365) - 1
        else:
            # If risk_free_rate is a scalar, use the pre-computed daily rate
            return self.daily_risk_free
    
    def backtest_strategy(self, df):
        """
        Backtest the trading strategy based on signals in the DataFrame.
        Cash not invested in Bitcoin earns the risk-free rate.
        Implements max drawdown protection if threshold is set.
        
        Args:
            df (DataFrame): DataFrame with OHLC data, 'break_signal', 'buy_pct', and 'sell_pct' columns.
            
        Returns:
            DataFrame: Results of the strategy with portfolio values and position data.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        results = df.copy()
        
        # Initialize portfolio tracking columns
        results['cash'] = self.initial_cash
        results['cash_interest'] = 0.0
        results['risk_free_value'] = 0.0  # Value of cash + interest
        results['bitcoin_qty'] = 0.0
        results['bitcoin_value'] = 0.0
        results['portfolio_value'] = self.initial_cash
        results['trade_type'] = ''  # 'buy', 'sell', or ''
        results['trade_amount'] = 0.0
        results['trade_price'] = 0.0
        results['commission'] = 0.0
        results['drawdown_pct'] = 0.0  # Track drawdown percentage
        results['stop_loss_triggered'] = False  # Track if stop loss was triggered
        results['equity_at_stop_loss'] = 0.0  # Track equity when stop loss triggered
        
        # Variables to track drawdown
        peak_value = self.initial_cash
        is_stopped_out = False
        equity_at_stop = 0.0
        
        # Simulate trading for each day
        for i in range(1, len(results)):
            # Get daily risk-free rate for this period
            daily_rf_rate = self.get_daily_risk_free_rate(i, df)
            
            # Default: carry forward previous values
            results.loc[i, 'bitcoin_qty'] = results.loc[i-1, 'bitcoin_qty']
            results.loc[i, 'stop_loss_triggered'] = is_stopped_out
            results.loc[i, 'equity_at_stop_loss'] = equity_at_stop
            
            # Calculate interest earned on cash
            interest_earned = results.loc[i-1, 'cash'] * daily_rf_rate
            results.loc[i, 'cash_interest'] = results.loc[i-1, 'cash_interest'] + interest_earned
            
            # Update cash with interest
            results.loc[i, 'cash'] = results.loc[i-1, 'cash'] + interest_earned
            results.loc[i, 'risk_free_value'] = results.loc[i, 'cash']
            
            # Update bitcoin value before processing any signals
            close_price = results.loc[i, 'Close']
            results.loc[i, 'bitcoin_value'] = results.loc[i, 'bitcoin_qty'] * close_price
            
            # Update portfolio value before processing signals
            results.loc[i, 'portfolio_value'] = results.loc[i, 'risk_free_value'] + results.loc[i, 'bitcoin_value']
            
            # Check if we have a new peak
            if results.loc[i, 'portfolio_value'] > peak_value:
                peak_value = results.loc[i, 'portfolio_value']
            
            # Calculate current drawdown
            if peak_value > 0:
                current_drawdown_pct = (peak_value - results.loc[i, 'portfolio_value']) / peak_value * 100
                results.loc[i, 'drawdown_pct'] = current_drawdown_pct
            else:
                current_drawdown_pct = 0
                results.loc[i, 'drawdown_pct'] = 0
            
            # Check if max drawdown threshold is exceeded
            if (self.max_drawdown_threshold is not None and 
                current_drawdown_pct > self.max_drawdown_threshold and 
                not is_stopped_out and 
                results.loc[i, 'bitcoin_qty'] > 0):
                
                # Trigger stop loss - sell all bitcoin
                btc_to_sell = results.loc[i, 'bitcoin_qty']
                sale_value = btc_to_sell * close_price
                commission = sale_value * self.commission_rate
                
                # Update portfolio
                results.loc[i, 'cash'] = results.loc[i, 'cash'] + (sale_value - commission)
                results.loc[i, 'risk_free_value'] = results.loc[i, 'cash']
                results.loc[i, 'bitcoin_qty'] = 0
                results.loc[i, 'bitcoin_value'] = 0
                
                # Record trade details
                results.loc[i, 'trade_type'] = 'stop_loss_sell'
                results.loc[i, 'trade_amount'] = btc_to_sell
                results.loc[i, 'trade_price'] = close_price
                results.loc[i, 'commission'] = commission
                
                # Update portfolio value after selling
                results.loc[i, 'portfolio_value'] = results.loc[i, 'risk_free_value']
                
                # Set the stop loss flag
                is_stopped_out = True
                results.loc[i, 'stop_loss_triggered'] = True
                equity_at_stop = results.loc[i, 'portfolio_value']
                results.loc[i, 'equity_at_stop_loss'] = equity_at_stop
                
                # Skip normal signal processing for this day
                continue
            
            # If we're stopped out, check if we can re-enter the market
            if is_stopped_out:
                # Calculate what the drawdown would be if we re-entered with the same equity position
                btc_potential_qty = equity_at_stop / close_price
                potential_portfolio_value = results.loc[i, 'risk_free_value'] - equity_at_stop + (btc_potential_qty * close_price)
                
                # Calculate what the drawdown would be with this position
                potential_drawdown = 0
                if peak_value > 0:
                    potential_drawdown = (peak_value - potential_portfolio_value) / peak_value * 100
                
                # If re-entering would keep us below the threshold, do it
                if potential_drawdown < self.max_drawdown_threshold:
                    cash_to_use = equity_at_stop
                    
                    # Calculate commission
                    commission = cash_to_use * self.commission_rate
                    
                    # Calculate actual BTC amount to buy after commission
                    actual_cash_used = cash_to_use - commission
                    btc_bought = actual_cash_used / close_price
                    
                    # Update portfolio
                    results.loc[i, 'cash'] = results.loc[i, 'cash'] - cash_to_use
                    results.loc[i, 'risk_free_value'] = results.loc[i, 'cash']
                    results.loc[i, 'bitcoin_qty'] = btc_bought
                    results.loc[i, 'bitcoin_value'] = btc_bought * close_price
                    
                    # Record trade details
                    results.loc[i, 'trade_type'] = 'stop_loss_rebuy'
                    results.loc[i, 'trade_amount'] = btc_bought
                    results.loc[i, 'trade_price'] = close_price
                    results.loc[i, 'commission'] = commission
                    
                    # Update portfolio value
                    results.loc[i, 'portfolio_value'] = results.loc[i, 'risk_free_value'] + results.loc[i, 'bitcoin_value']
                    
                    # Reset the stop loss flag
                    is_stopped_out = False
                    results.loc[i, 'stop_loss_triggered'] = False
                    equity_at_stop = 0
                    results.loc[i, 'equity_at_stop_loss'] = 0
                    
                    # Skip normal signal processing for this day
                    continue
            
            # Process normal trading signals if not handled by stop loss logic
            signal = results.loc[i, 'break_signal']

            # Process buy signal (signal == 2)
            if signal == 2:
                buy_pct = results.loc[i, 'buy_pct']
                cash_to_use = results.loc[i, 'cash'] * buy_pct  # Use current cash (with interest)
                
                if cash_to_use > 0 and results.loc[i, 'cash'] > 0:
                    # Calculate commission
                    commission = cash_to_use * self.commission_rate
                    
                    # Calculate actual BTC amount to buy after commission
                    actual_cash_used = cash_to_use - commission
                    btc_bought = actual_cash_used / close_price
                    
                    # Update portfolio
                    results.loc[i, 'cash'] = results.loc[i, 'cash'] - cash_to_use
                    results.loc[i, 'risk_free_value'] = results.loc[i, 'cash']  # Update risk-free value
                    results.loc[i, 'bitcoin_qty'] = results.loc[i, 'bitcoin_qty'] + btc_bought
                    
                    # Record trade details
                    results.loc[i, 'trade_type'] = 'buy'
                    results.loc[i, 'trade_amount'] = btc_bought
                    results.loc[i, 'trade_price'] = close_price
                    results.loc[i, 'commission'] = commission
            
            # Process sell signal (signal == 1)
            elif signal == 1:
                sell_pct = results.loc[i, 'sell_pct']
                btc_to_sell = results.loc[i, 'bitcoin_qty'] * sell_pct
                
                if btc_to_sell > 0 and results.loc[i, 'bitcoin_qty'] > 0:
                    # Calculate sale value and commission
                    sale_value = btc_to_sell * close_price
                    commission = sale_value * self.commission_rate
                    
                    # Update portfolio
                    results.loc[i, 'cash'] = results.loc[i, 'cash'] + (sale_value - commission)
                    results.loc[i, 'risk_free_value'] = results.loc[i, 'cash']  # Update risk-free value
                    results.loc[i, 'bitcoin_qty'] = results.loc[i, 'bitcoin_qty'] - btc_to_sell
                    
                    # Record trade details
                    results.loc[i, 'trade_type'] = 'sell'
                    results.loc[i, 'trade_amount'] = btc_to_sell
                    results.loc[i, 'trade_price'] = close_price
                    results.loc[i, 'commission'] = commission
            
            # Update portfolio valuation
            results.loc[i, 'bitcoin_value'] = results.loc[i, 'bitcoin_qty'] * close_price
            results.loc[i, 'portfolio_value'] = results.loc[i, 'risk_free_value'] + results.loc[i, 'bitcoin_value']
        
        # Calculate daily returns
        results['daily_return'] = results['portfolio_value'].pct_change()
        
        # Store results
        self.strategy_results = results
        
        return results
    
    def calculate_buy_and_hold(self, df):
        """
        Calculate the performance of a buy-and-hold strategy for comparison.
        Cash not invested in Bitcoin earns the risk-free rate.
        
        Args:
            df (DataFrame): DataFrame with OHLC data.
            
        Returns:
            DataFrame: Results of the buy-and-hold strategy.
        """
        # Create a copy for buy-and-hold simulation
        bh_results = df.copy()
        
        # Initialize portfolio tracking columns
        bh_results['bh_cash'] = 0.0
        bh_results['bh_cash_interest'] = 0.0
        bh_results['bh_risk_free_value'] = 0.0
        bh_results['bh_bitcoin_qty'] = 0.0
        bh_results['bh_bitcoin_value'] = 0.0
        bh_results['bh_portfolio_value'] = self.initial_cash
        
        # Buy as much Bitcoin as possible on the first day
        first_price = bh_results.loc[0, 'Close']
        commission = self.initial_cash * self.commission_rate
        cash_after_commission = self.initial_cash - commission
        btc_bought = cash_after_commission / first_price
        
        # Set initial portfolio state
        bh_results.loc[0, 'bh_cash'] = self.initial_cash - cash_after_commission
        bh_results.loc[0, 'bh_risk_free_value'] = bh_results.loc[0, 'bh_cash']
        bh_results.loc[0, 'bh_bitcoin_qty'] = btc_bought
        bh_results.loc[0, 'bh_bitcoin_value'] = btc_bought * first_price
        bh_results.loc[0, 'bh_portfolio_value'] = bh_results.loc[0, 'bh_risk_free_value'] + bh_results.loc[0, 'bh_bitcoin_value']
        
        # Calculate daily values, accumulating interest on cash
        for i in range(1, len(bh_results)):
            # Get daily risk-free rate for this period
            daily_rf_rate = self.get_daily_risk_free_rate(i, df)
            close_price = bh_results.loc[i, 'Close']
            
            # Calculate interest on cash
            interest_earned = bh_results.loc[i-1, 'bh_cash'] * daily_rf_rate
            bh_results.loc[i, 'bh_cash_interest'] = bh_results.loc[i-1, 'bh_cash_interest'] + interest_earned
            
            # Update cash with interest
            bh_results.loc[i, 'bh_cash'] = bh_results.loc[i-1, 'bh_cash'] + interest_earned
            bh_results.loc[i, 'bh_risk_free_value'] = bh_results.loc[i, 'bh_cash']
            
            # Keep the same BTC quantity throughout
            bh_results.loc[i, 'bh_bitcoin_qty'] = bh_results.loc[0, 'bh_bitcoin_qty']
            bh_results.loc[i, 'bh_bitcoin_value'] = bh_results.loc[i, 'bh_bitcoin_qty'] * close_price
            
            # Update total portfolio value
            bh_results.loc[i, 'bh_portfolio_value'] = bh_results.loc[i, 'bh_risk_free_value'] + bh_results.loc[i, 'bh_bitcoin_value']
        
        # Calculate daily returns
        bh_results['bh_daily_return'] = bh_results['bh_portfolio_value'].pct_change()
        
        # Store results
        self.buy_hold_results = bh_results
        
        return bh_results
    
    def calculate_sharpe_ratio(self, returns, annualization_factor=365):
        """
        Calculate the Sharpe Ratio for a series of returns.
        
        Args:
            returns (Series): Daily return series.
            annualization_factor (int, optional): Factor to annualize the ratio. Defaults to 365 for daily returns.
            
        Returns:
            float: Annualized Sharpe Ratio.
        """
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # For Sharpe ratio, we'll use the first risk-free rate if it's a Series
        if isinstance(self.risk_free_rate, pd.Series):
            annual_rf_rate = self.risk_free_rate.iloc[0]
            daily_rf = (1 + annual_rf_rate) ** (1/365) - 1
        else:
            daily_rf = self.daily_risk_free
        
        # Calculate excess returns
        excess_returns = returns - daily_rf
        
        # Calculate Sharpe ratio (if std is 0, return 0 to avoid division by zero)
        std_dev = returns.std()
        if std_dev == 0:
            return 0.0
            
        sharpe = (excess_returns.mean() / std_dev) * math.sqrt(annualization_factor)
        return sharpe
    
    def calculate_drawdown(self, values):
        """
        Calculate the drawdown series and maximum drawdown.
        
        Args:
            values (Series): Portfolio value series.
            
        Returns:
            tuple: (drawdown_series, max_drawdown_percentage)
        """
        # Calculate cumulative maximum
        cumulative_max = values.cummax()
        
        # Calculate drawdown series
        drawdown_series = (values - cumulative_max) / cumulative_max
        
        # Get maximum drawdown (as a positive percentage)
        max_drawdown = -drawdown_series.min() * 100 if not drawdown_series.empty else 0.0
        
        return drawdown_series, max_drawdown
    
    def calculate_win_rate(self):
        """
        Calculate the win rate for all completed trades.
        
        Returns:
            float: Percentage of profitable trades.
        """
        if self.strategy_results is None:
            return 0.0
            
        # Find all trades
        trades = self.strategy_results[self.strategy_results['trade_type'] != ''].copy()
        
        if len(trades) == 0:
            return 0.0
            
        # Extract buy and sell pairs
        buys = trades[trades['trade_type'].isin(['buy', 'stop_loss_rebuy'])].reset_index()
        sells = trades[trades['trade_type'].isin(['sell', 'stop_loss_sell'])].reset_index()
        
        wins = 0
        total_pairs = min(len(buys), len(sells))
        
        for buy_idx in range(total_pairs):
            buy_price = buys.loc[buy_idx, 'trade_price']
            sell_idx = np.searchsorted(sells['index'], buys.loc[buy_idx, 'index'])
            
            # If we found a matching sell after this buy
            if sell_idx < len(sells):
                sell_price = sells.loc[sell_idx, 'trade_price']
                if sell_price > buy_price:
                    wins += 1
        
        return (wins / total_pairs * 100) if total_pairs > 0 else 0.0
    
    def generate_performance_report(self):
        """
        Generate a detailed performance report for the strategy and benchmark.
        
        Returns:
            dict: Dictionary containing all performance metrics.
        """
        if self.strategy_results is None or self.buy_hold_results is None:
            raise ValueError("Must run backtest_strategy and calculate_buy_and_hold first")
        
        # Overall return calculation
        strategy_start = self.strategy_results['portfolio_value'].iloc[0]
        strategy_end = self.strategy_results['portfolio_value'].iloc[-1]
        strategy_return = ((strategy_end / strategy_start) - 1) * 100
        
        bh_start = self.buy_hold_results['bh_portfolio_value'].iloc[0]
        bh_end = self.buy_hold_results['bh_portfolio_value'].iloc[-1]
        bh_return = ((bh_end / bh_start) - 1) * 100
        
        # Sharpe ratio calculation
        strategy_sharpe = self.calculate_sharpe_ratio(self.strategy_results['daily_return'])
        bh_sharpe = self.calculate_sharpe_ratio(self.buy_hold_results['bh_daily_return'])
        
        # Drawdown calculation
        _, strategy_max_dd = self.calculate_drawdown(self.strategy_results['portfolio_value'])
        _, bh_max_dd = self.calculate_drawdown(self.buy_hold_results['bh_portfolio_value'])
        
        # Win rate calculation
        win_rate = self.calculate_win_rate()
        
        # Trade count
        total_trades = len(self.strategy_results[self.strategy_results['trade_type'] != ''])
        buy_trades = len(self.strategy_results[self.strategy_results['trade_type'] == 'buy'])
        sell_trades = len(self.strategy_results[self.strategy_results['trade_type'] == 'sell'])
        stop_loss_sells = len(self.strategy_results[self.strategy_results['trade_type'] == 'stop_loss_sell'])
        stop_loss_rebuys = len(self.strategy_results[self.strategy_results['trade_type'] == 'stop_loss_rebuy'])
        
        # Interest earned
        interest_earned = self.strategy_results['cash_interest'].iloc[-1]
        bh_interest_earned = self.buy_hold_results['bh_cash_interest'].iloc[-1]
        
        # Compile the performance report
        report = {
            'strategy_return': strategy_return,
            'bh_return': bh_return,
            'outperformance': strategy_return - bh_return,
            'strategy_sharpe': strategy_sharpe,
            'bh_sharpe': bh_sharpe,
            'sharpe_improvement': strategy_sharpe - bh_sharpe,
            'strategy_max_drawdown': strategy_max_dd,
            'bh_max_drawdown': bh_max_dd,
            'drawdown_improvement': bh_max_dd - strategy_max_dd,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'stop_loss_sells': stop_loss_sells,
            'stop_loss_rebuys': stop_loss_rebuys,
            'interest_earned': interest_earned,
            'bh_interest_earned': bh_interest_earned,
            'final_portfolio_value': strategy_end,
            'final_bh_value': bh_end,
        }
        
        return report
    
    def plot_results(self, output_path=None):
        """
        Generate plots comparing strategy and buy-and-hold performance.
        Also shows cash and equity composition over time.
        
        Args:
            output_path (str, optional): Path to save the plot. If None, the plot is displayed instead.
            
        Returns:
            None
        """
        if self.strategy_results is None or self.buy_hold_results is None:
            raise ValueError("Must run backtest_strategy and calculate_buy_and_hold first")
        
        # Create a figure with 4 subplots (added a subplot for drawdown)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20), 
                                          gridspec_kw={'height_ratios': [3, 1, 1, 2]})
        
        # Plot 1: Portfolio Value Comparison
        ax1.plot(self.strategy_results.index, self.strategy_results['portfolio_value'], 
                label='FVG Strategy', color='blue')
        ax1.plot(self.buy_hold_results.index, self.buy_hold_results['bh_portfolio_value'], 
                label='Buy & Hold', color='orange', linestyle='--')
        
        # Mark buy signals
        buys = self.strategy_results[self.strategy_results['trade_type'] == 'buy']
        ax1.scatter(buys.index, buys['portfolio_value'], color='green', marker='^', s=100, label='Buy')
        
        # Mark sell signals
        sells = self.strategy_results[self.strategy_results['trade_type'] == 'sell']
        ax1.scatter(sells.index, sells['portfolio_value'], color='red', marker='v', s=100, label='Sell')
        
        # Mark stop loss sells
        stop_loss_sells = self.strategy_results[self.strategy_results['trade_type'] == 'stop_loss_sell']
        if not stop_loss_sells.empty:
            ax1.scatter(stop_loss_sells.index, stop_loss_sells['portfolio_value'], 
                      color='purple', marker='X', s=120, label='Stop Loss Sell')
        
        # Mark stop loss rebuys
        stop_loss_rebuys = self.strategy_results[self.strategy_results['trade_type'] == 'stop_loss_rebuy']
        if not stop_loss_rebuys.empty:
            ax1.scatter(stop_loss_rebuys.index, stop_loss_rebuys['portfolio_value'], 
                       color='blue', marker='*', s=120, label='Stop Loss Rebuy')
        
        ax1.set_title('Portfolio Value Comparison', fontsize=14)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        strategy_dd, _ = self.calculate_drawdown(self.strategy_results['portfolio_value'])
        bh_dd, _ = self.calculate_drawdown(self.buy_hold_results['bh_portfolio_value'])
        
        ax2.plot(self.strategy_results.index, strategy_dd * 100, label='FVG Strategy', color='blue')
        ax2.plot(self.buy_hold_results.index, bh_dd * 100, label='Buy & Hold', color='orange', linestyle='--')
        
        # Add a horizontal line at the max drawdown threshold
        if self.max_drawdown_threshold is not None:
            ax2.axhline(y=-self.max_drawdown_threshold, color='r', linestyle='-', 
                       label=f'Max DD Threshold ({self.max_drawdown_threshold}%)')
        
        ax2.set_title('Drawdown Comparison', fontsize=14)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Invert y-axis for drawdown (negative is down)
        ax2.invert_yaxis()
        
        # Plot 3: Stop Loss Triggered
        # Add a new plot to show when stop loss was triggered
        if self.max_drawdown_threshold is not None:
            stop_loss_flags = self.strategy_results['stop_loss_triggered'].astype(int)
            ax3.plot(self.strategy_results.index, stop_loss_flags, 
                    color='red', label='Stop Loss Active', linewidth=2)
            ax3.set_title('Stop Loss Status', fontsize=14)
            ax3.set_ylabel('Status (1=Active)', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.1, 1.1)  # Set y-axis limits for boolean values
        else:
            ax3.set_visible(False)  # Hide this subplot if no max drawdown threshold
        
        # Plot 4: Cash (with interest) and Bitcoin Holdings Over Time
        ax4.stackplot(self.strategy_results.index, 
                     [self.strategy_results['risk_free_value'], self.strategy_results['bitcoin_value']],
                     labels=['Cash + Interest', 'Bitcoin Holdings'],
                     colors=['lightgreen', 'lightblue'],
                     alpha=0.7)
        
        # Add a line for total portfolio value
        ax4.plot(self.strategy_results.index, self.strategy_results['portfolio_value'],
                label='Total Value', color='darkblue', linewidth=2)
        
        ax4.set_title('Strategy Allocation Over Time', fontsize=14)
        ax4.set_ylabel('Value ($)', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Add buy/sell annotations on the allocation chart too
        ax4.scatter(buys.index, buys['portfolio_value'], color='green', marker='^', s=80)
        ax4.scatter(sells.index, sells['portfolio_value'], color='red', marker='v', s=80)
        
        # Add stop loss sells and rebuys if any
        if not stop_loss_sells.empty:
            ax4.scatter(stop_loss_sells.index, stop_loss_sells['portfolio_value'], 
                       color='purple', marker='X', s=100)
        if not stop_loss_rebuys.empty:
            ax4.scatter(stop_loss_rebuys.index, stop_loss_rebuys['portfolio_value'], 
                       color='blue', marker='*', s=100)
        
        plt.tight_layout()
        
        # Save or display the plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def print_performance_summary(self):
        """
        Print a summary of the strategy's performance metrics.
        
        Returns:
            None
        """
        if self.strategy_results is None or self.buy_hold_results is None:
            raise ValueError("Must run backtest_strategy and calculate_buy_and_hold first")
        
        report = self.generate_performance_report()
        
        print("=" * 60)
        print(f"{'PERFORMANCE SUMMARY':^60}")
        print("=" * 60)
        
        print(f"\n{'RETURNS':^60}")
        print("-" * 60)
        print(f"Strategy Total Return:     {report['strategy_return']:>10.2f}%")
        print(f"Buy & Hold Total Return:   {report['bh_return']:>10.2f}%")
        print(f"Outperformance:            {report['outperformance']:>10.2f}%")
        
        print(f"\n{'RISK METRICS':^60}")
        print("-" * 60)
        print(f"Strategy Sharpe Ratio:     {report['strategy_sharpe']:>10.2f}")
        print(f"Buy & Hold Sharpe Ratio:   {report['bh_sharpe']:>10.2f}")
        sharpe_improvement = (report['strategy_sharpe'] - report['bh_sharpe'])/report['bh_sharpe'] * 100
        print(f"Sharpe Improvement:        {sharpe_improvement:>10.2f}%")
        print(f"Strategy Max Drawdown:     {report['strategy_max_drawdown']:>10.2f}%")
        if self.max_drawdown_threshold is not None:
            print(f"Drawdown Threshold:        {self.max_drawdown_threshold:>10.2f}%")
        print(f"Buy & Hold Max Drawdown:   {report['bh_max_drawdown']:>10.2f}%")
        drawdown_improvement = (report['bh_max_drawdown'] - report['strategy_max_drawdown'])/report['bh_max_drawdown'] * 100
        print(f"Drawdown Improvement:      {drawdown_improvement:>10.2f}%")
        
        print(f"\n{'TRADE STATISTICS':^60}")
        print("-" * 60)
        print(f"Win Rate:                  {report['win_rate']:>10.2f}%")
        print(f"Total Trades:              {report['total_trades']:>10}")
        print(f"Buy Trades:                {report['buy_trades']:>10}")
        print(f"Sell Trades:               {report['sell_trades']:>10}")
        print(f"Stop Loss Sells:           {report['stop_loss_sells']:>10}")
        print(f"Stop Loss Rebuys:          {report['stop_loss_rebuys']:>10}")
        print(f"Interest Earned:           {report['interest_earned']:>10.2f}$")
        print(f"B&H Interest Earned:       {report['bh_interest_earned']:>10.2f}$")
        
        print(f"\n{'FINAL VALUES':^60}")
        print("-" * 60)
        print(f"Starting Capital:          {self.initial_cash:>10.2f}$")
        print(f"Strategy Final Value:      {report['final_portfolio_value']:>10.2f}$")
        print(f"Buy & Hold Final Value:    {report['final_bh_value']:>10.2f}$")
        
        print("\n" + "=" * 60)