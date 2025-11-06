"""
Preprocessing utilities for cleaning and aligning trading and sentiment data.
"""
import pandas as pd
import numpy as np
from typing import Optional, List
import warnings


class DataPreprocessor:
    """Preprocess and align trading and sentiment datasets."""
    
    def __init__(self):
        self.historical_processed = None
        self.sentiment_processed = None
    
    def preprocess_sentiment(self, df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
        """
        Clean and standardize the Fear/Greed Index dataset.
        
        Args:
            df: Raw sentiment DataFrame
            date_col: Name of the date column (auto-detected if None)
            
        Returns:
            Processed DataFrame with standardized date and sentiment columns
        """
        df = df.copy()
        
        # Auto-detect date column
        if date_col is None:
            date_candidates = ['Date', 'date', 'timestamp', 'Timestamp', 'TIME', 'time']
            for col in date_candidates:
                if col in df.columns:
                    date_col = col
                    break
        
        if date_col is None:
            raise ValueError(f"Could not find date column. Available columns: {list(df.columns)}")
        
        # Parse dates
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Standardize sentiment column
        sentiment_candidates = ['Classification', 'classification', 'sentiment', 'Sentiment', 'label']
        sentiment_col = None
        for col in sentiment_candidates:
            if col in df.columns:
                sentiment_col = col
                break
        
        if sentiment_col:
            df['sentiment'] = df[sentiment_col].str.strip().str.title()
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates (keep last)
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        print(f"✓ Preprocessed sentiment data: {len(df)} records")
        if 'sentiment' in df.columns:
            print(f"  Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        self.sentiment_processed = df
        return df
    
    def preprocess_historical(self, df: pd.DataFrame, time_col: str = None) -> pd.DataFrame:
        """
        Clean and standardize the Hyperliquid historical trading dataset.
        
        Args:
            df: Raw historical trading DataFrame
            time_col: Name of the time column (auto-detected if None)
            
        Returns:
            Processed DataFrame with standardized columns
        """
        df = df.copy()
        
        # First, standardize column names to avoid duplicates
        column_mapping = {}
        for col in df.columns:
            new_col = col.lower().replace(' ', '_').replace('-', '_')
            if new_col != col:
                column_mapping[col] = new_col
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Remove duplicate columns if any exist
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Auto-detect time column
        if time_col is None:
            time_candidates = ['time', 'timestamp', 'date', 'datetime']
            for col in time_candidates:
                if col in df.columns:
                    time_col = col
                    break
        
        if time_col is None:
            raise ValueError(f"Could not find time column. Available columns: {list(df.columns)}")
        
        # Parse timestamps - only create if doesn't exist
        if time_col != 'timestamp':
            df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        df['date'] = df['timestamp'].dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns
        numeric_candidates = ['size', 'execution_price', 'closedpnl', 'closed_pnl', 
                             'leverage', 'start_position', 'startposition']
        
        for col in df.columns:
            if any(candidate in col.lower() for candidate in numeric_candidates):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize side column (buy/sell)
        if 'side' in df.columns:
            df['side'] = df['side'].str.strip().str.title()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Preprocessed historical data: {len(df):,} records")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        if 'account' in df.columns:
            print(f"  Unique accounts: {df['account'].nunique():,}")
        if 'symbol' in df.columns:
            print(f"  Unique symbols: {df['symbol'].nunique()}")
        
        self.historical_processed = df
        return df
    
    def aggregate_daily_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate trading data to daily account-level performance metrics.
        
        Args:
            df: Preprocessed historical trading DataFrame
            
        Returns:
            DataFrame with daily performance metrics per account
        """
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column. Run preprocess_historical first.")
        
        # Identify PnL column
        pnl_col = None
        for col in ['closedpnl', 'closed_pnl', 'pnl']:
            if col in df.columns:
                pnl_col = col
                break
        
        agg_dict = {}
        
        # Basic aggregations
        if pnl_col:
            agg_dict[pnl_col] = ['sum', 'mean', 'count']
        
        if 'size' in df.columns:
            agg_dict['size'] = ['sum', 'mean']
        
        if 'leverage' in df.columns:
            agg_dict['leverage'] = ['mean', 'max']
        
        # Group by account and date
        if 'account' in df.columns:
            daily = df.groupby(['account', 'date']).agg(agg_dict).reset_index()
        else:
            daily = df.groupby('date').agg(agg_dict).reset_index()
        
        # Flatten column names
        daily.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                        for col in daily.columns.values]
        
        # Calculate win rate if PnL available
        if pnl_col:
            pnl_sum_col = f"{pnl_col}_sum"
            if 'account' in df.columns:
                win_counts = df[df[pnl_col] > 0].groupby(['account', 'date']).size()
                total_counts = df.groupby(['account', 'date']).size()
            else:
                win_counts = df[df[pnl_col] > 0].groupby('date').size()
                total_counts = df.groupby('date').size()
            
            win_rate = (win_counts / total_counts).fillna(0)
            daily['win_rate'] = daily.apply(
                lambda row: win_rate.get((row['account'], row['date']) if 'account' in daily.columns else row['date'], 0),
                axis=1
            )
        
        print(f"✓ Aggregated to daily performance: {len(daily):,} account-days")
        
        return daily
    
    def merge_with_sentiment(self, 
                            trading_df: pd.DataFrame, 
                            sentiment_df: pd.DataFrame,
                            lag_days: List[int] = [0, 1, 3, 7]) -> pd.DataFrame:
        """
        Merge trading performance with sentiment data, including lagged features.
        
        Args:
            trading_df: Daily aggregated trading performance
            sentiment_df: Processed sentiment data
            lag_days: List of lag days to create for sentiment features
            
        Returns:
            Merged DataFrame with sentiment features
        """
        # Ensure both have date column
        if 'date' not in trading_df.columns or 'date' not in sentiment_df.columns:
            raise ValueError("Both DataFrames must have 'date' column")
        
        # Ensure dates are in datetime format for both
        trading_df = trading_df.copy()
        sentiment_df = sentiment_df.copy()
        
        trading_df['date'] = pd.to_datetime(trading_df['date']).dt.normalize()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
        
        # Start with trading data
        merged = trading_df.copy()
        
        # Select columns to merge from sentiment (include sentiment_numeric if available)
        sentiment_cols = ['date', 'sentiment']
        if 'sentiment_numeric' in sentiment_df.columns:
            sentiment_cols.append('sentiment_numeric')
        
        sentiment_minimal = sentiment_df[sentiment_cols].copy()
        sentiment_minimal = sentiment_minimal.drop_duplicates(subset=['date'], keep='last')
        
        # Merge current day sentiment
        merged = merged.merge(sentiment_minimal, on='date', how='left', suffixes=('', '_current'))
        
        # Create lagged sentiment features
        sentiment_minimal_base = sentiment_df[['date', 'sentiment']].copy()
        sentiment_minimal_base = sentiment_minimal_base.sort_values('date')
        
        for lag in lag_days:
            if lag > 0:
                lagged = sentiment_minimal_base.copy()
                lagged['date'] = lagged['date'] + pd.Timedelta(days=lag)
                lagged = lagged.rename(columns={'sentiment': f'sentiment_lag_{lag}'})
                merged = merged.merge(lagged, on='date', how='left')
        
        # Report merge quality
        matched_records = merged['sentiment'].notna().sum()
        match_pct = (matched_records / len(merged) * 100) if len(merged) > 0 else 0
        
        print(f"✓ Merged with sentiment: {len(merged):,} records")
        print(f"  Matched records: {matched_records:,} ({match_pct:.1f}%)")
        print(f"  Created {len([l for l in lag_days if l > 0])} lagged sentiment features")
        
        if matched_records == 0:
            print("\n⚠️  WARNING: No dates matched between datasets!")
            print(f"  Trading date range: {trading_df['date'].min()} to {trading_df['date'].max()}")
            print(f"  Sentiment date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
        
        return merged


def create_sentiment_numeric_encoding(df: pd.DataFrame, 
                                      sentiment_col: str = 'sentiment') -> pd.DataFrame:
    """
    Create numeric encodings for sentiment categories.
    
    Fear = -1, Neutral = 0, Greed = 1
    
    Args:
        df: DataFrame with sentiment column
        sentiment_col: Name of sentiment column
        
    Returns:
        DataFrame with added sentiment_numeric column
    """
    df = df.copy()
    
    sentiment_map = {
        'Fear': -1,
        'Extreme Fear': -2,
        'Neutral': 0,
        'Greed': 1,
        'Extreme Greed': 2
    }
    
    df['sentiment_numeric'] = df[sentiment_col].map(sentiment_map)
    
    # Handle any unmapped values
    if df['sentiment_numeric'].isnull().any():
        unique_sentiments = df[sentiment_col].unique()
        print(f"⚠️  Warning: Some sentiment values not mapped: {unique_sentiments}")
    
    return df
