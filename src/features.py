"""
Feature engineering utilities for creating predictive features.
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def create_rolling_features(df: pd.DataFrame, 
                            columns: List[str],
                            windows: List[int] = [3, 7, 14, 30],
                            group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create rolling window features (mean, std, min, max) for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to create rolling features for
        windows: List of window sizes (days)
        group_col: Optional column to group by (e.g., 'account')
        
    Returns:
        DataFrame with added rolling features
    """
    df = df.copy()
    df = df.sort_values('date')
    
    for col in columns:
        if col not in df.columns:
            continue
        
        for window in windows:
            if group_col and group_col in df.columns:
                # Grouped rolling
                df[f'{col}_rolling_mean_{window}d'] = df.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_{window}d'] = df.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
            else:
                # Simple rolling
                df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}d'] = df[col].rolling(window, min_periods=1).std()
    
    return df


def create_sentiment_features(df: pd.DataFrame,
                              sentiment_col: str = 'sentiment',
                              windows: List[int] = [3, 7]) -> pd.DataFrame:
    """
    Create sentiment-based features including streaks and rolling counts.
    
    Args:
        df: Input DataFrame with sentiment column
        sentiment_col: Name of sentiment column
        windows: Rolling window sizes for sentiment aggregation
        
    Returns:
        DataFrame with added sentiment features
    """
    df = df.copy()
    df = df.sort_values('date')
    
    # Sentiment streak (consecutive days of same sentiment)
    df['sentiment_change'] = (df[sentiment_col] != df[sentiment_col].shift(1)).astype(int)
    df['sentiment_streak'] = df.groupby('sentiment_change').cumcount() + 1
    
    # Rolling sentiment counts
    for window in windows:
        for sentiment_val in df[sentiment_col].dropna().unique():
            col_name = f'{sentiment_val.lower().replace(" ", "_")}_count_{window}d'
            df[col_name] = (df[sentiment_col] == sentiment_val).rolling(window, min_periods=1).sum()
    
    # Volatility of sentiment (using numeric encoding)
    if 'sentiment_numeric' in df.columns:
        for window in windows:
            df[f'sentiment_volatility_{window}d'] = df['sentiment_numeric'].rolling(window, min_periods=1).std()
    
    return df


def create_performance_features(df: pd.DataFrame,
                                pnl_col: str,
                                group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create performance-based features like cumulative PnL, win streaks, etc.
    
    Args:
        df: Input DataFrame
        pnl_col: Name of PnL column
        group_col: Optional grouping column (e.g., 'account')
        
    Returns:
        DataFrame with added performance features
    """
    df = df.copy()
    df = df.sort_values('date')
    
    if pnl_col not in df.columns:
        return df
    
    # Cumulative PnL
    if group_col and group_col in df.columns:
        df['cumulative_pnl'] = df.groupby(group_col)[pnl_col].cumsum()
    else:
        df['cumulative_pnl'] = df[pnl_col].cumsum()
    
    # Win/Loss indicator
    df['is_win'] = (df[pnl_col] > 0).astype(int)
    df['is_loss'] = (df[pnl_col] < 0).astype(int)
    
    # Win/Loss streaks
    if group_col and group_col in df.columns:
        df['win_streak'] = df.groupby(group_col)['is_win'].apply(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        ).values
        df['loss_streak'] = df.groupby(group_col)['is_loss'].apply(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        ).values
    else:
        df['win_streak'] = df['is_win'] * (df['is_win'].groupby((df['is_win'] != df['is_win'].shift()).cumsum()).cumcount() + 1)
        df['loss_streak'] = df['is_loss'] * (df['is_loss'].groupby((df['is_loss'] != df['is_loss'].shift()).cumsum()).cumcount() + 1)
    
    # Days since last win/loss
    if group_col and group_col in df.columns:
        df['days_since_win'] = df.groupby(group_col).apply(
            lambda x: x.reset_index(drop=True).index.to_series() - x[x['is_win'] == 1].reset_index(drop=True).index.to_series().reindex(x.index, method='ffill')
        ).values
    
    return df


def create_time_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create time-based features from date column.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    if date_col not in df.columns:
        return df
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    
    return df


def create_account_features(df: pd.DataFrame, 
                           account_col: str = 'account',
                           pnl_col: str = 'closedpnl_sum') -> pd.DataFrame:
    """
    Create account-level aggregate features.
    
    Args:
        df: Input DataFrame
        account_col: Name of account column
        pnl_col: Name of PnL column
        
    Returns:
        DataFrame with added account features
    """
    df = df.copy()
    
    if account_col not in df.columns:
        return df
    
    # Account lifetime stats
    account_stats = df.groupby(account_col).agg({
        pnl_col: ['sum', 'mean', 'std', 'count'],
    })
    
    account_stats.columns = ['_'.join(col).strip('_') for col in account_stats.columns.values]
    account_stats = account_stats.add_prefix('account_lifetime_')
    
    df = df.merge(account_stats, left_on=account_col, right_index=True, how='left')
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between sentiment and other variables.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    # Sentiment x Leverage
    if 'sentiment_numeric' in df.columns and 'leverage_mean' in df.columns:
        df['sentiment_x_leverage'] = df['sentiment_numeric'] * df['leverage_mean']
    
    # Sentiment x Volume
    if 'sentiment_numeric' in df.columns and 'size_sum' in df.columns:
        df['sentiment_x_volume'] = df['sentiment_numeric'] * df['size_sum']
    
    # Sentiment x Day of Week
    if 'sentiment_numeric' in df.columns and 'day_of_week' in df.columns:
        df['sentiment_x_dow'] = df['sentiment_numeric'] * df['day_of_week']
    
    return df
