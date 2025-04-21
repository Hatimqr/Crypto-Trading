import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TradingSimulator:
    def __init__(self, df, initial_capital=10000, transaction_cost=0.005):
        """
        Initialize the trading simulator with a DataFrame containing trading signals/probabilities.
        
        Args:
            df: DataFrame with at least Sell, Hold, Buy, Close, and Risk Free Rate columns
            initial_capital: Starting capital for simulation
            transaction_cost: Cost per transaction as a decimal (e.g., 0.005 for 0.5%)
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Add necessary columns for simulation results
        self.df['Action'] = None
        self.df['Position'] = None
        self.df['Capital'] = None
        self.df['Crypto_Holdings'] = None
        self.df['Portfolio_Value'] = None
        self.df['Return'] = None
        
        # Performance metrics
        self.metrics = {}
    
    def get_action_from_probabilities(self, method='highest_prob'):
        """
        Determine trading action based on probability values.
        
        Args:
            method: Strategy to use - 'highest_prob' or 'random_weighted'
        
        Returns:
            Series with actions: 'buy', 'hold', 'sell'
        """
        if method == 'highest_prob':
            # Choose action with highest probability
            actions = pd.DataFrame({
                'sell': self.df['Sell'],
                'hold': self.df['Hold'],
                'buy': self.df['Buy']
            }).idxmax(axis=1)
            
        elif method == 'random_weighted':
            # Randomly select action weighted by probabilities
            actions = []
            for _, row in self.df.iterrows():
                probs = [row['Sell'], row['Hold'], row['Buy']]
                action = np.random.choice(['sell', 'hold', 'buy'], p=probs)
                actions.append(action)
            actions = pd.Series(actions, index=self.df.index)
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return actions
    
    def simulate(self, decision_method='highest_prob', threshold=0.5):
        """
        Run trading simulation based on the signals/probabilities.
        
        Args:
            decision_method: 'highest_prob', 'random_weighted', or 'threshold'
            threshold: Probability threshold to take action (if using 'threshold' method)
            
        Returns:
            DataFrame with simulation results
        """
        df = self.df.copy()
        
        # Determine actions based on probabilities
        if decision_method in ['highest_prob', 'random_weighted']:
            df['Action'] = self.get_action_from_probabilities(method=decision_method)
        elif decision_method == 'threshold':
            # Use thresholds to determine actions
            df['Action'] = 'hold'  # Default action
            df.loc[df['Buy'] > threshold, 'Action'] = 'buy'
            df.loc[df['Sell'] > threshold, 'Action'] = 'sell'
        
        # Initialize simulation variables
        capital = self.initial_capital
        crypto_holdings = 0
        position = 'cash'  # Start with cash position
        
        # Daily risk-free return (convert annual to daily)
        df['Daily_RF_Return'] = ((1 + df['Risk Free Rate'])**(1/365) - 1)
        
        # Simulate trading
        results = []
        
        for i, row in df.iterrows():
            action = row['Action']
            price = row['Close']
            
            # Apply action
            if action == 'buy' and position != 'crypto':
                # Buy crypto
                crypto_holdings = capital * (1 - self.transaction_cost) / price
                capital = 0
                position = 'crypto'
            elif action == 'sell' and position != 'cash':
                # Sell crypto
                capital = crypto_holdings * price * (1 - self.transaction_cost)
                crypto_holdings = 0
                position = 'cash'
            elif position == 'cash':
                # Apply risk-free return to capital
                capital *= (1 + row['Daily_RF_Return'])
            
            # Calculate portfolio value
            portfolio_value = capital + (crypto_holdings * price)
            
            # Store results
            results.append({
                'date': i,
                'action': action,
                'position': position,
                'capital': capital,
                'crypto_holdings': crypto_holdings,
                'portfolio_value': portfolio_value,
                'price': price
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        
        # Calculate returns
        results_df['return'] = results_df['portfolio_value'].pct_change()
        
        # Update simulation DataFrame
        self.results = results_df
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        return results_df
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics from simulation results"""
        if not hasattr(self, 'results'):
            raise ValueError("Must run simulate() first")
        
        results = self.results
        
        # Calculate metrics
        total_return = (results['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = len(results)
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Volatility (annualized)
        daily_returns = results['return'].dropna()
        volatility = daily_returns.std() * np.sqrt(365)
        
        # Sharpe ratio
        risk_free_rate = self.df['Risk Free Rate'].mean()
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + results['return'].fillna(0)).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        # Win ratio
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_ratio = winning_days / total_days if total_days > 0 else 0
        
        # Trading activity
        position_changes = (results['position'] != results['position'].shift(1)).sum()
        trades_per_year = position_changes * (365 / days)
        
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_ratio': win_ratio,
            'position_changes': position_changes,
            'trades_per_year': trades_per_year
        }
        
        return self.metrics
    
    def plot_portfolio_performance(self):
        """Plot portfolio value over time"""
        if not hasattr(self, 'results'):
            raise ValueError("Must run simulate() first")
        
        plt.figure(figsize=(12, 6))
        
        # Portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(self.results.index, self.results['portfolio_value'], label='Portfolio Value')
        
        # Add buy/sell markers
        buys = self.results[self.results['action'] == 'buy']
        sells = self.results[self.results['action'] == 'sell']
        
        plt.scatter(buys.index, buys['portfolio_value'], 
                   marker='^', color='green', label='Buy', alpha=0.7, s=100)
        plt.scatter(sells.index, sells['portfolio_value'], 
                   marker='v', color='red', label='Sell', alpha=0.7, s=100)
        
        plt.title('Portfolio Performance')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Bitcoin price for context
        plt.subplot(2, 1, 2)
        plt.plot(self.df.index, self.df['Close'], label='Bitcoin Price', color='orange')
        plt.title('Bitcoin Price')
        plt.ylabel('Price ($)')
        plt.grid(True)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_performance_metrics(self):
        """Plot performance metrics and comparisons"""
        if not hasattr(self, 'results') or not self.metrics:
            raise ValueError("Must run simulate() first")
        
        # Create a buy and hold benchmark
        benchmark = pd.DataFrame(index=self.df.index)
        benchmark['Close'] = self.df['Close']
        benchmark['portfolio_value'] = self.initial_capital * (benchmark['Close'] / benchmark['Close'].iloc[0])
        benchmark['return'] = benchmark['portfolio_value'].pct_change()
        
        # Calculate benchmark metrics
        benchmark_total_return = (benchmark['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        days = len(benchmark)
        benchmark_annual_return = (1 + benchmark_total_return) ** (365 / days) - 1
        benchmark_volatility = benchmark['return'].std() * np.sqrt(365)
        risk_free_rate = self.df['Risk Free Rate'].mean()
        benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_volatility
        
        benchmark_cum_returns = (1 + benchmark['return'].fillna(0)).cumprod()
        benchmark_peak = benchmark_cum_returns.expanding().max()
        benchmark_drawdown = (benchmark_cum_returns / benchmark_peak) - 1
        benchmark_max_drawdown = benchmark_drawdown.min()
        
        # Strategy cumulative returns
        strategy_cum_returns = (1 + self.results['return'].fillna(0)).cumprod()
        
        # Plot cumulative returns comparison
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(strategy_cum_returns.index, strategy_cum_returns, 
                label='Strategy', color='blue')
        plt.plot(benchmark_cum_returns.index, benchmark_cum_returns, 
                label='Buy & Hold', color='orange', linestyle='--')
        plt.title('Cumulative Returns Comparison')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        strategy_drawdown = (strategy_cum_returns / strategy_cum_returns.expanding().max()) - 1
        plt.plot(strategy_drawdown.index, strategy_drawdown, 
                label='Strategy Drawdown', color='blue')
        plt.plot(benchmark_drawdown.index, benchmark_drawdown, 
                label='Buy & Hold Drawdown', color='orange', linestyle='--')
        plt.title('Drawdown Comparison')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Create a bar chart for key metrics comparison
        metrics_comp = pd.DataFrame({
            'Strategy': [
                self.metrics['annual_return'], 
                self.metrics['volatility'],
                self.metrics['sharpe_ratio'],
                self.metrics['max_drawdown']
            ],
            'Buy & Hold': [
                benchmark_annual_return,
                benchmark_volatility,
                benchmark_sharpe,
                benchmark_max_drawdown
            ]
        }, index=['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'])
        
        plt.figure(figsize=(10, 6))
        metrics_comp.plot(kind='bar', color=['blue', 'orange'])
        plt.title('Performance Metrics Comparison')
        plt.ylabel('Value')
        plt.grid(True, axis='y')
        plt.xticks(rotation=0)
        
        for i, v in enumerate(metrics_comp['Strategy']):
            plt.text(i-0.15, v/2, f'{v:.2f}', color='white', fontweight='bold')
            
        for i, v in enumerate(metrics_comp['Buy & Hold']):
            plt.text(i+0.15, v/2, f'{v:.2f}', color='white', fontweight='bold')
        
        return metrics_comp
    
    def get_performance_summary(self):
        """Get a text summary of performance metrics"""
        if not self.metrics:
            raise ValueError("Must run simulate() first")
        
        summary = [
            f"Trading Simulation Summary:",
            f"-----------------------------",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Portfolio Value: ${self.results['portfolio_value'].iloc[-1]:,.2f}",
            f"Total Return: {self.metrics['total_return']*100:.2f}%",
            f"Annualized Return: {self.metrics['annual_return']*100:.2f}%",
            f"Volatility (Annual): {self.metrics['volatility']*100:.2f}%",
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}",
            f"Maximum Drawdown: {self.metrics['max_drawdown']*100:.2f}%",
            f"Win Ratio: {self.metrics['win_ratio']*100:.2f}%",
            f"Total Trades: {self.metrics['position_changes']}",
            f"Trades Per Year: {self.metrics['trades_per_year']:.2f}"
        ]
        
        return "\n".join(summary)

