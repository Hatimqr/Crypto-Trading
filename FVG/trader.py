import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math


class TraderWithStoploss:
    
    def __init__(self, initial_cash=100000.0, commission_rate=0.005, risk_free_rate=0.0, max_drawdown_threshold=0.4):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.max_drawdown_threshold = max_drawdown_threshold
        
        # Risk-free rate handling - sereies of daily rates. need to convert to daily equivalent
        self.daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1
        
        
        # Results storage
        self.strategy_results = None
        self.buy_hold_results = None  # [Date, Bitcoin Price, Bitcoin Quantity (constant), Portfolio Value, Returns, Risk Free Rate]

    def buy_and_hold(self, df): # df = [Date, Open, High, Low, Close, Volume]
        # Initialize results dataframe
        results = pd.DataFrame(index=df['Date'])
        
        # Get bitcoin price
        results['Bitcoin Price'] = df['Close']
        
        # Get bitcoin quantity (constant)
        results['Bitcoin Quantity'] = self.initial_cash / df['Close']

        # Get portfolio value
        results['Portfolio Value'] = results['Bitcoin Quantity'] * results['Bitcoin Price']

        # Get returns
        results['Returns'] = results['Portfolio Value'].pct_change()

        # Get risk free rate
        results['Risk Free Rate'] = self.daily_risk_free

        # get drawdown
        results['Drawdown'] = (results['Portfolio Value'] - results['Portfolio Value'].cummax()) / results['Portfolio Value'].cummax()


        self.buy_hold_results = results



    # simulation methods
    def simulate_trade(self, df): # df = [Date, Close, break_signal ,Position]

        """
        ALGORITHM SimulateTradingStrategy

        INPUT:
        - price_series: Array of daily closing prices for Bitcoin
        - signal_series: Array of trading signals (1: Buy, -1: Sell, 0: Hold)
        - position_series: Array of target position sizes (as % of portfolio)
        - risk_free_rate: Daily risk-free interest rate
        - initial_capital: Initial capital to start trading
        - drawdown_threshold: Maximum acceptable Bitcoin price drawdown
        - drawdown_buffer: Additional buffer before re-entering after drawdown

        OUTPUT:
        - portfolio_values: Daily portfolio values
        - returns: Daily returns
        - metrics: Performance metrics including Sharpe ratio and max drawdown

        INITIALIZATION:
        Set portfolio_value = initial_capital
        Set bitcoin_units = 0
        Set cash = initial_capital
        Set portfolio_values = empty array
        Set returns = empty array
        Set bitcoin_peak_price = price_series[0]
        Set in_drawdown_protection_mode = false

        FOR each day i from 0 to length(price_series) - 1:
            
            // Update Bitcoin peak price
            IF price_series[i] > bitcoin_peak_price:
                bitcoin_peak_price = price_series[i]
            
            // Calculate current Bitcoin drawdown
            bitcoin_drawdown = (bitcoin_peak_price - price_series[i]) / bitcoin_peak_price
            
            // Check if we need to trigger drawdown protection
            IF NOT in_drawdown_protection_mode AND bitcoin_drawdown > drawdown_threshold:
                // Drawdown exceeds threshold - move to all cash
                in_drawdown_protection_mode = true
                
                // Sell all Bitcoin
                cash = cash + (bitcoin_units * price_series[i])
                bitcoin_units = 0
                
            ELSE IF in_drawdown_protection_mode AND bitcoin_drawdown < (drawdown_threshold - drawdown_buffer):
                // Drawdown has improved - exit protection mode
                in_drawdown_protection_mode = false
            
            // Only proceed with normal signal-based trading if not in drawdown protection mode
            IF NOT in_drawdown_protection_mode:
                // Apply signal and rebalance portfolio
                IF signal_series[i] != 0 OR position_series[i] has changed from previous day:
                    target_bitcoin_value = portfolio_value * position_series[i]
                    target_bitcoin_units = target_bitcoin_value / price_series[i]
                    
                    // Execute the trade
                    IF target_bitcoin_units > bitcoin_units:  // Buy more Bitcoin
                        amount_to_buy = target_bitcoin_units - bitcoin_units
                        cash_needed = amount_to_buy * price_series[i]
                        
                        IF cash >= cash_needed:
                            bitcoin_units = target_bitcoin_units
                            cash = cash - cash_needed
                        ELSE:
                            // Partial fill if not enough cash
                            bitcoin_units = bitcoin_units + (cash / price_series[i])
                            cash = 0
                    
                    ELSE IF target_bitcoin_units < bitcoin_units:  // Sell some Bitcoin
                        amount_to_sell = bitcoin_units - target_bitcoin_units
                        cash_gained = amount_to_sell * price_series[i]
                        
                        bitcoin_units = target_bitcoin_units
                        cash = cash + cash_gained
            
            // Apply daily risk-free return to cash position
            cash = cash * (1 + risk_free_rate)
            
            // Calculate portfolio value at end of day
            current_bitcoin_value = bitcoin_units * price_series[i]
            current_portfolio_value = current_bitcoin_value + cash
            
            // Store portfolio value
            Append current_portfolio_value to portfolio_values
            
            // Calculate daily return
            IF i > 0:
                daily_return = (current_portfolio_value / portfolio_values[i-1]) - 1
                Append daily_return to returns

        // Calculate performance metrics
        average_return = Mean(returns)
        std_return = StandardDeviation(returns)
        sharpe_ratio = (average_return - risk_free_rate) / std_return

        // Calculate maximum drawdown
        max_drawdown = 0
        peak = portfolio_values[0]

        FOR each value in portfolio_values:
            IF value > peak:
                peak = value
            drawdown = (peak - value) / peak
            
            IF drawdown > max_drawdown:
                max_drawdown = drawdown

        RETURN portfolio_values, returns, sharpe_ratio, max_drawdown       
        """
    
        data = df.copy()
        data['Risk Free Rate'] = self.daily_risk_free


        
