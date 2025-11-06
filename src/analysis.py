"""
Analysis utilities for exploring trading performance vs sentiment relationships.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


class SentimentPerformanceAnalyzer:
    """Analyze relationships between market sentiment and trading performance."""
    
    def __init__(self, merged_df: pd.DataFrame):
        """
        Initialize analyzer with merged trading and sentiment data.
        
        Args:
            merged_df: DataFrame with both trading metrics and sentiment features
        """
        self.df = merged_df.copy()
        
    def performance_by_sentiment(self, 
                                 pnl_col: str,
                                 sentiment_col: str = 'sentiment') -> pd.DataFrame:
        """
        Calculate performance metrics grouped by sentiment category.
        
        Args:
            pnl_col: Name of PnL column to analyze
            sentiment_col: Name of sentiment column
            
        Returns:
            DataFrame with performance stats by sentiment
        """
        grouped = self.df.groupby(sentiment_col)[pnl_col].agg([
            ('count', 'count'),
            ('mean_pnl', 'mean'),
            ('median_pnl', 'median'),
            ('std_pnl', 'std'),
            ('total_pnl', 'sum'),
            ('min_pnl', 'min'),
            ('max_pnl', 'max')
        ])
        
        # Calculate win rate - ensure it returns a Series
        if pnl_col in self.df.columns:
            win_rate = self.df.groupby(sentiment_col)[pnl_col].apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0
            )
            grouped['win_rate'] = win_rate
        
        return grouped.round(4)
    
    def correlation_analysis(self, 
                            metrics: List[str],
                            sentiment_numeric_col: str = 'sentiment_numeric') -> pd.DataFrame:
        """
        Calculate correlations between sentiment and performance metrics.
        
        Args:
            metrics: List of metric columns to correlate with sentiment
            sentiment_numeric_col: Numeric encoding of sentiment
            
        Returns:
            DataFrame with correlation coefficients and p-values
        """
        results = []
        
        for metric in metrics:
            if metric in self.df.columns and sentiment_numeric_col in self.df.columns:
                # Remove NaN values
                valid_data = self.df[[metric, sentiment_numeric_col]].dropna()
                
                if len(valid_data) > 2:
                    corr, p_value = stats.pearsonr(
                        valid_data[sentiment_numeric_col], 
                        valid_data[metric]
                    )
                    
                    results.append({
                        'metric': metric,
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': 'Yes' if p_value < 0.05 else 'No',
                        'n_samples': len(valid_data)
                    })
        
        return pd.DataFrame(results)
    
    def lag_analysis(self,
                    pnl_col: str,
                    max_lag: int = 7) -> pd.DataFrame:
        """
        Analyze correlation between lagged sentiment and current performance.
        
        Args:
            pnl_col: Name of PnL column
            max_lag: Maximum lag days to analyze
            
        Returns:
            DataFrame with correlations for each lag
        """
        results = []
        
        for lag in range(max_lag + 1):
            sentiment_col = f'sentiment_lag_{lag}' if lag > 0 else 'sentiment_numeric'
            
            if sentiment_col in self.df.columns and pnl_col in self.df.columns:
                valid_data = self.df[[pnl_col, sentiment_col]].dropna()
                
                if len(valid_data) > 2:
                    corr, p_value = stats.pearsonr(valid_data[sentiment_col], valid_data[pnl_col])
                    
                    results.append({
                        'lag_days': lag,
                        'correlation': corr,
                        'p_value': p_value,
                        'n_samples': len(valid_data)
                    })
        
        return pd.DataFrame(results)
    
    def leverage_by_sentiment(self, 
                             leverage_col: str = 'leverage_mean',
                             sentiment_col: str = 'sentiment') -> pd.DataFrame:
        """
        Analyze leverage usage patterns across different sentiment regimes.
        
        Args:
            leverage_col: Name of leverage column
            sentiment_col: Name of sentiment column
            
        Returns:
            DataFrame with leverage stats by sentiment
        """
        if leverage_col not in self.df.columns:
            return pd.DataFrame()
        
        grouped = self.df.groupby(sentiment_col)[leverage_col].agg([
            ('count', 'count'),
            ('mean_leverage', 'mean'),
            ('median_leverage', 'median'),
            ('std_leverage', 'std'),
            ('max_leverage', 'max')
        ])
        
        return grouped.round(4)
    
    def trading_volume_by_sentiment(self,
                                    volume_col: str,
                                    sentiment_col: str = 'sentiment') -> pd.DataFrame:
        """
        Analyze trading volume patterns across sentiment regimes.
        
        Args:
            volume_col: Name of volume/size column
            sentiment_col: Name of sentiment column
            
        Returns:
            DataFrame with volume stats by sentiment
        """
        if volume_col not in self.df.columns:
            return pd.DataFrame()
        
        grouped = self.df.groupby(sentiment_col)[volume_col].agg([
            ('total_volume', 'sum'),
            ('avg_volume', 'mean'),
            ('median_volume', 'median')
        ])
        
        return grouped.round(4)
    
    def detect_outliers(self, 
                       col: str,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in a given column.
        
        Args:
            col: Column name to check for outliers
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier (1.5) or z-score threshold (3.0)
            
        Returns:
            DataFrame with outlier records
        """
        if col not in self.df.columns:
            return pd.DataFrame()
        
        data = self.df[col].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outlier_mask = (self.df[col] < lower) | (self.df[col] > upper)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.df[outlier_mask]
    
    def sentiment_transition_analysis(self, 
                                     pnl_col: str,
                                     sentiment_col: str = 'sentiment') -> pd.DataFrame:
        """
        Analyze performance during sentiment transitions (e.g., Fear -> Greed).
        
        Args:
            pnl_col: Name of PnL column
            sentiment_col: Name of sentiment column
            
        Returns:
            DataFrame with transition patterns
        """
        df_sorted = self.df.sort_values(['account', 'date']) if 'account' in self.df.columns else self.df.sort_values('date')
        
        # Create previous sentiment column
        if 'account' in df_sorted.columns:
            df_sorted['prev_sentiment'] = df_sorted.groupby('account')[sentiment_col].shift(1)
        else:
            df_sorted['prev_sentiment'] = df_sorted[sentiment_col].shift(1)
        
        # Create transition label
        df_sorted['transition'] = df_sorted['prev_sentiment'] + ' → ' + df_sorted[sentiment_col]
        
        # Analyze performance by transition
        transitions = df_sorted.groupby('transition')[pnl_col].agg([
            ('count', 'count'),
            ('mean_pnl', 'mean'),
            ('median_pnl', 'median')
        ]).round(4)
        
        return transitions.sort_values('count', ascending=False)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a return series.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (default 0)
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from cumulative returns.
    
    Args:
        cumulative_returns: Series of cumulative returns
        
    Returns:
        Maximum drawdown as a percentage
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()
