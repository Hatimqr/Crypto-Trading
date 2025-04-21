import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureEngineer:
    def __init__(self, df):
        """
        Initialize the FeatureEngineer with a DataFrame containing price data.
        
        Args:
            df: DataFrame with at least Open, High, Low, Close, Volume columns
                If it has a 'Signals' column, it will be used as the target variable
        """
        # Store the original dataframe
        self.original_df = df.copy()
        # Create features dataframe without signals
        self.X = df.drop(columns=['Signals'], errors='ignore')
        # Calculate returns if not already present
        if 'Return' not in self.X.columns:
            self.X['Return'] = self.X['Close'].pct_change()
        # Store target variable
        self.y = df['Signals'] if 'Signals' in df.columns else None
        # Keep track of added features
        self.feature_names = []
    
    def reset(self):
        """Reset to initial state, removing all added features"""
        self.X = self.original_df.drop(columns=['Signals'], errors='ignore')
        if 'Return' not in self.X.columns:
            self.X['Return'] = self.X['Close'].pct_change()
        self.feature_names = []
        return self
    
    def add_MA_price(self, window=[3, 5, 10, 30], column='Close', add_to_df=True): # 1 is just the close price
        """
        Add Simple Moving Averages (SMA) for a given column
        
        Args:
            window: List of window sizes for SMA calculation
            column: Column to calculate SMA on (default 'Close')
            add_to_df: Whether to add the features to the instance's DataFrame
            
        Returns:
            DataFrame with added features
        """
        df = self.X.copy()
        added_features = []
        
        for w in window:
            feature_name = f'SMA_{column}_{w}' if column != 'Close' else f'SMA_{w}'
            df.ta.sma(length=w, close=column, append=True)
            
            # Rename for clarity when using non-Close columns
            if column != 'Close':
                df.rename(columns={f'SMA_{w}': feature_name}, inplace=True)
            
            added_features.append(feature_name)
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_EMA_price(self, window=[3, 5, 10, 30], column='Close', add_to_df=True): # 1 is just the close price
        """Add Exponential Moving Averages (EMA) for a given column"""
        df = self.X.copy()
        added_features = []
        
        for w in window:
            feature_name = f'EMA_{column}_{w}' if column != 'Close' else f'EMA_{w}'
            df.ta.ema(length=w, close=column, append=True)
            
            # Rename for clarity
            if column != 'Close':
                df.rename(columns={f'EMA_{w}': feature_name}, inplace=True)
            
            added_features.append(feature_name)
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_MA_returns(self, window=[3, 5, 10, 30], add_to_df=True): # 1 is just the close price
        """Add Moving Averages calculated on returns"""
        df = self.X.copy()
        added_features = []
        
        # Make sure Return column exists
        if 'Return' not in df.columns:
            df['Return'] = df['Close'].pct_change()
        
        for w in window:
            feature_name = f'Return_SMA_{w}'
            # Calculate SMA on returns
            sma_series = df['Return'].rolling(window=w).mean()
            df[feature_name] = sma_series
            added_features.append(feature_name)
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_price_momentum(self, window=[1, 3, 5, 10, 30], add_to_df=True):
        """Add price momentum indicators (ROC - Rate of Change)"""
        df = self.X.copy()
        added_features = []
        
        for w in window:
            feature_name = f'ROC_{w}'
            df.ta.roc(length=w, append=True)
            added_features.append(feature_name)
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_volatility_indicators(self, window=[5, 10, 20], add_to_df=True):
        """Add volatility indicators (Bollinger Bands, ATR)"""
        df = self.X.copy()
        added_features = []
        
        # Add Bollinger Bands
        for w in window:
            # Standard Bollinger Bands with 2.0 standard deviations
            df.ta.bbands(length=w, append=True)
            # remove the middle band as it is just the SMA which we already have
            df.drop(columns=[f'BBM_{w}_2.0'], inplace=True)
            
            # Add features to our tracking list
            band_features = [f'BBL_{w}_2.0', f'BBU_{w}_2.0', 
                             f'BBB_{w}_2.0', f'BBP_{w}_2.0']
            added_features.extend(band_features)
        
        # Add Average True Range (ATR)
        for w in window:
            df.ta.atr(length=w, append=True)
            added_features.append(f'ATR_{w}')
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_oscillators(self, add_to_df=True):
        """Add oscillator indicators (RSI, MACD, Stochastic)"""
        df = self.X.copy()
        added_features = []
        
        # Add RSI with different lengths
        for length in [7, 14, 21]:
            df.ta.rsi(length=length, append=True)
            added_features.append(f'RSI_{length}')
        
        # Add MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        macd_features = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
        added_features.extend(macd_features)
        
        # Add Stochastic oscillator
        df.ta.stoch(append=True)
        stoch_features = ['STOCHk_14_3_3', 'STOCHd_14_3_3']
        added_features.extend(stoch_features)
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_trend_indicators(self, add_to_df=True):
        """Add trend indicators (ADX, Aroon)"""
        df = self.X.copy()
        added_features = []
        
        # Add ADX (Average Directional Index)
        df.ta.adx(length=14, append=True)
        adx_features = ['ADX_14', 'DMP_14', 'DMN_14']
        added_features.extend(adx_features)
        
        # Add Aroon indicator
        df.ta.aroon(length=14, append=True)
        aroon_features = ['AROOND_14', 'AROONU_14', 'AROONOSC_14']
        added_features.extend(aroon_features)
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_risk_free_features(self, window=[5, 10, 20], add_to_df=True):
        """Add features related to risk-free rate"""
        df = self.X.copy()
        added_features = []
        
        if 'Risk Free Rate' not in df.columns:
            print("Warning: Risk Free Rate column not found. Skipping risk-free features.")
            return df
        
        # Calculate daily risk-free rate
        df['Daily_RF'] = ((1 + df['Risk Free Rate'])**(1/365) - 1)
        added_features.append('Daily_RF')
        
        # Risk-free rate trends
        for w in window:
            # Change in risk-free rate
            feature_name = f'RF_Change_{w}d'
            df[feature_name] = df['Risk Free Rate'].diff(w)
            added_features.append(feature_name)
            
            # Percentage change in risk-free rate
            feature_name = f'RF_PctChange_{w}d'
            df[feature_name] = df['Risk Free Rate'].pct_change(w)
            added_features.append(feature_name)
        
        # Spread between BTC returns and risk-free rate
        df['Excess_Return'] = df['Return'] - df['Daily_RF']
        added_features.append('Excess_Return')
        
        # Rolling average excess return
        for w in window:
            feature_name = f'Avg_Excess_Return_{w}d'
            df[feature_name] = df['Excess_Return'].rolling(window=w).mean()
            added_features.append(feature_name)
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_lagged_signals(self, lags=[1, 2, 3, 4, 5], add_to_df=True):
        """Add lagged signals as features"""
        if self.y is None:
            print("Warning: No signals data available. Skipping lagged signals.")
            return self.X.copy()
        
        df = self.X.copy()
        added_features = []
        
        # Add previous signals
        for lag in lags:
            feature_name = f'Signal_Lag_{lag}'
            df[feature_name] = self.y.shift(lag)
            added_features.append(feature_name)

        # add average of previous signals
        for lag in lags:
            lag*=10
            feature_name = f'Signal_Avg_Lag_{lag}'
            df[feature_name] = self.y.rolling(window=lag).mean()
            added_features.append(feature_name)
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_price_levels(self, window=[20, 50, 200], add_to_df=True):
        """Add price level features (support/resistance indicators)"""
        df = self.X.copy()
        added_features = []
        
        for w in window:
            # Rolling highs and lows
            high_feature = f'Rolling_High_{w}'
            low_feature = f'Rolling_Low_{w}'
            
            df[high_feature] = df['High'].rolling(window=w).max()
            df[low_feature] = df['Low'].rolling(window=w).min()
            
            # Distance from current price to support/resistance
            df[f'Dist_To_Support_{w}'] = (df['Close'] / df[low_feature] - 1) * 100
            df[f'Dist_To_Resistance_{w}'] = (df[high_feature] / df['Close'] - 1) * 100
            
            added_features.extend([high_feature, low_feature, 
                                  f'Dist_To_Support_{w}', f'Dist_To_Resistance_{w}'])
        
        if add_to_df:
            self.X = df
            self.feature_names.extend(added_features)
        return df
    
    def add_all_features(self):
        """Add all available features"""
        self.add_MA_price()
        self.add_EMA_price()
        self.add_MA_returns()
        self.add_price_momentum()
        self.add_volatility_indicators()
        self.add_oscillators()
        self.add_trend_indicators()
        self.add_risk_free_features()
        self.add_price_levels()
        
        # Only add lagged signals if we have target data
        if self.y is not None:
            self.add_lagged_signals()
        
        return self.X
    
    def get_features(self):
        """Get the current feature DataFrame"""
        return self.X
    
    def get_feature_target_split(self, dropna=True):
        """Get X and y for model training"""
        X = self.X.copy()
        y = self.y.copy() if self.y is not None else None
        
        if dropna:
            # Drop rows with NaN values
            if y is not None:
                # Align X and y
                X, y = X.align(y, join='inner', axis=0)
                # Drop NaNs from both
                mask = X.notna().all(axis=1) & y.notna()
                X = X[mask]
                y = y[mask]
            else:
                # Just drop NaNs from X
                X = X.dropna()
        
        return X, y
    
    def select_best_features(self, k=20, method='f_classif', plot=True):
        """
        Select the k best features using statistical tests
        
        Args:
            k: Number of top features to select
            method: 'f_classif' or 'mutual_info' for selection method
            plot: Whether to plot feature importance
            
        Returns:
            DataFrame with only the best features
        """
        if self.y is None:
            print("Warning: No target variable available. Cannot select features.")
            return self.X
        
        X, y = self.get_feature_target_split(dropna=True)
        
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:  # mutual_info
            selector = SelectKBest(mutual_info_classif, k=k)
            
        selector.fit(X, y)
        
        # Get feature importance scores
        scores = selector.scores_
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores
        })
        
        # Sort by importance
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        # Get selected feature names
        mask = selector.get_support()
        selected_features = X.columns[mask]
        
        if plot:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Score', y='Feature', data=feature_scores.head(k))
            plt.title(f'Top {k} Features by Importance')
            plt.tight_layout()
            plt.show()
        
        # Return DataFrame with only selected features
        return self.X[selected_features], feature_scores, selected_features
    
    def analyze_feature_correlation(self, threshold=0.8, plot=True):
        """
        Analyze correlation between features and with target
            
        Args:
            threshold: Correlation threshold to identify highly correlated features
            plot: Whether to plot correlation matrix
            
        Returns:
            List of highly correlated feature pairs
        """
        X, y = self.get_feature_target_split(dropna=True)
        
        # Add target to correlation analysis if available
        if y is not None:
            data = X.copy()
            data['Signals'] = y
        else:
            data = X.copy()
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        
        # Upper triangle of correlation matrix (excluding diagonal)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        # Sort by absolute correlation (highest first)
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if plot:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                        center=0, square=True, linewidths=.5)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.show()
            
            # If target is available, plot correlation with target
            if y is not None:
                target_corr = corr_matrix['Signals'].drop('Signals').sort_values(ascending=False)
                plt.figure(figsize=(10, 8))
                sns.barplot(x=target_corr.values, y=target_corr.index)
                plt.title('Feature Correlation with Target (Signals)')
                plt.tight_layout()
                plt.show()
        
        return high_corr_pairs
 