import numpy as np
import plotly.graph_objects as go
import pandas as pd

class OptStrat:
    def __init__(self, df, transaction_cost=0.005):
        self.df = df
        self.prices = df['Close'].values
        self.rf_array = ((1+ df['Risk Free Rate'])**(1/365) - 1).values
        self.transaction_cost = transaction_cost

        # get optimal strategy
        decisions = self.optimize_trading_strategy()
        
        self.df['Signals'] = self.transform_signals(decisions)
        
        # Calculate strategy returns
        self.df['Strategy Returns'] = self.calculate_strategy_returns(decisions)

        self.plot()

    def calculate_strategy_returns(self, decisions):
        """
        Calculate daily returns from following the optimal strategy
        
        Args:
            decisions (array-like): Array of 0s and 1s representing out/in market
            
        Returns:
            numpy.ndarray: Array of daily returns under the strategy
        """
        # Calculate BTC returns
        btc_daily_returns = np.zeros(len(self.prices))
        btc_daily_returns[1:] = (self.prices[1:] - self.prices[:-1]) / self.prices[:-1]
        
        # Initialize strategy returns array
        strategy_returns = np.zeros(len(self.prices))
        
        # Apply transaction costs on signal changes
        signals = self.transform_signals(decisions)
        
        # For each day, apply the correct return based on position
        for i in range(1, len(decisions)):
            if decisions[i-1] == 1:  # In BTC
                strategy_returns[i] = btc_daily_returns[i]
            else:  # In risk-free
                strategy_returns[i] = self.rf_array[i-1]
                
            # Subtract transaction cost when we have a buy or sell signal
            if signals[i] != 0:
                strategy_returns[i] -= self.transaction_cost
                
        return strategy_returns

    def optimize_trading_strategy(self):
        """
        Optimize a binary Bitcoin trading strategy using dynamic programming.
        
        Args:
            prices (list): Daily Bitcoin closing prices
            risk_free_rate (float): Daily risk-free rate (as a decimal)
            transaction_cost (float): Transaction cost rate (default: 0.0005 for 0.05%)
            
        Returns:
            tuple: (optimal_policy, expected_return)
        """
        prices = self.prices
        transaction_cost = self.transaction_cost
        risk_free_rate = self.rf_array


        T = len(prices)-1  # Number of trading days
        
        # Calculate Bitcoin daily returns
        btc_returns = [(prices[t+1] - prices[t])/prices[t] for t in range(T)]
        
        # Initialize memoization table and decisions
        memo = np.zeros((T+1, 2))
        decisions = np.zeros((T, 2), dtype=int)
        
        # Fill table bottom-up
        for t in range(T-1, -1, -1):
            for prev_action in [0, 1]:
                # Option 1: Bitcoin
                btc_return = btc_returns[t]
                if prev_action == 0:  # Switch to Bitcoin
                    value_bitcoin = btc_return - transaction_cost + memo[t+1, 1]
                elif prev_action == 1:  # Stay in Bitcoin
                    value_bitcoin = btc_return + memo[t+1, 1]
                
                # Option 2: Risk-free
                if prev_action == 1:  # Switch to risk-free
                    value_risk_free = risk_free_rate[t] - transaction_cost + memo[t+1, 0]
                elif prev_action == 0:  # Stay in risk-free
                    value_risk_free = risk_free_rate[t] + memo[t+1, 0]
                
                # Select better option
                if value_bitcoin > value_risk_free:
                    memo[t, prev_action] = value_bitcoin
                    decisions[t, prev_action] = 1
                else:
                    memo[t, prev_action] = value_risk_free
                    decisions[t, prev_action] = 0
        
        # Reconstruct optimal policy
        optimal_policy = []
        current_action = 0  # Assume we start with no position
        
        for t in range(T):
            current_action = decisions[t, current_action]
            optimal_policy.append(current_action)
        optimal_policy.append(np.nan)
        
        

        
        return optimal_policy
    


    
    def transform_signals(self, decisions):
        """
        Transform binary decisions (0,1) into trading signals (-1,0,1)
        
        Args:
            decisions (array-like): Array of 0s and 1s representing out/in market
        
        Returns:
            array-like: Array of -1 (sell), 0 (hold), 1 (buy)
        """
        # Convert to numpy array if not already
        signals = np.array(decisions)
        
        # Get the differences between consecutive elements
        changes = np.diff(signals, prepend=0)
        
        # Initialize output array
        trading_signals = np.zeros_like(signals)
        
        # 1 when changing from 0 to 1 (buy signal)
        trading_signals[changes == 1] = 1
        
        # -1 when changing from 1 to 0 (sell signal)
        trading_signals[changes == -1] = -1
        
        return trading_signals
    
    def calculate_strat_results(self, initial_capital=100):
        """
        Calculate the expected daily return of the strategy
        calculate the expected annual return of the strategy

        Calculate sharpe ratio

        Get a series of portfoilio values given the initial capital
        """
        returns = self.df['Strategy Returns']

        # daily and annual returns
        daily_return = returns.mean()
        annualized_return = (1+daily_return)**365 - 1

        # sharpe ratio
        sharpe_ratio = (returns-self.rf_array).mean() / returns.std()

        # portfoilio values
        # sequence of cumulative returns at each day 
        cum_returns = (1 + returns).cumprod() - 1
        # sequence of portfoilio values at each day
        portfolio_values = initial_capital * (1 + cum_returns)

        return {
            'E_daily_return': daily_return,
            'E_annual_return': annualized_return,
            'Sharpe_ratio': sharpe_ratio,
            'Portfolio_values': portfolio_values
        }
        
        



    def plot(self, initial_capital=100):

        strat_results = self.calculate_strat_results(initial_capital=initial_capital)


        # Create figure
        fig = go.Figure()

        # Add base price line
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['Close'],
                name='BTC Price',
                line=dict(color='lightgrey'),
            )
        )

        # Add buy signals (green dots)
        buy_signals = self.df[self.df['Signals'] == 1]  
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                name='Buy Signal',
                mode='markers',
                marker=dict(
                    color='green',
                    size=2,
                    symbol='circle'
                )
            )
        )

        # Add sell signals (red dots)
        sell_signals = self.df[self.df['Signals'] == -1]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                name='Sell Signal',
                mode='markers',
                marker=dict(
                    color='red',
                    size=2,
                    symbol='circle'
                )
            )
        )

        # add potfolio values
        fig.add_trace(
            go.Scatter(
                x=strat_results['Portfolio_values'].index,
                y=strat_results['Portfolio_values'],
                name=f'Portfolio Values (initial capital: {initial_capital})',
                line=dict(color='blue'),
            )
        )



        # Update layout with range slider
        fig.update_layout(
            title='BTC Price with Optimal Trading Signals',
            yaxis_title='Price',
            xaxis_title='Date',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        fig.show()


