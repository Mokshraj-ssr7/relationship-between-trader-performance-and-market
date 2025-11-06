
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import DataPreprocessor, create_sentiment_numeric_encoding


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_preprocess_sentiment(self):
        """Test sentiment data preprocessing."""
        preprocessor = DataPreprocessor()
        
        # Create sample sentiment data
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Classification': ['Fear', 'Greed', 'Fear']
        })
        
        result = preprocessor.preprocess_sentiment(df)
        
        assert 'date' in result.columns
        assert 'sentiment' in result.columns
        assert len(result) == 3
        assert result['sentiment'].iloc[0] == 'Fear'
    
    def test_preprocess_sentiment_deduplication(self):
        """Test that duplicate dates are handled correctly."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'Classification': ['Fear', 'Greed', 'Fear']
        })
        
        result = preprocessor.preprocess_sentiment(df)
        
        # Should keep last occurrence
        assert len(result) == 2
        assert result[result['date'] == '2024-01-01']['sentiment'].iloc[0] == 'Greed'
    
    def test_aggregate_daily_performance(self):
        """Test daily performance aggregation."""
        preprocessor = DataPreprocessor()
        
        # Create sample trading data
        df = pd.DataFrame({
            'account': ['acc1', 'acc1', 'acc1', 'acc2'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-01']),
            'closedpnl': [100, -50, 200, 150],
            'size': [10, 20, 15, 25],
            'leverage': [2.0, 3.0, 2.5, 1.5]
        })
        
        result = preprocessor.aggregate_daily_performance(df)
        
        assert 'account' in result.columns
        assert 'date' in result.columns
        assert len(result) == 3  # 2 days for acc1, 1 day for acc2
    
    def test_merge_with_sentiment(self):
        """Test merging trading data with sentiment."""
        preprocessor = DataPreprocessor()
        
        trading_df = pd.DataFrame({
            'account': ['acc1', 'acc1'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'pnl': [100, 200]
        })
        
        sentiment_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'sentiment': ['Fear', 'Greed', 'Fear']
        })
        
        result = preprocessor.merge_with_sentiment(trading_df, sentiment_df, lag_days=[0, 1])
        
        assert 'sentiment' in result.columns
        assert 'sentiment_lag_1' in result.columns
        assert len(result) == 2


class TestSentimentEncoding:
    """Test sentiment encoding functions."""
    
    def test_create_sentiment_numeric_encoding(self):
        """Test numeric encoding of sentiment."""
        df = pd.DataFrame({
            'sentiment': ['Fear', 'Greed', 'Neutral', 'Extreme Fear', 'Extreme Greed']
        })
        
        result = create_sentiment_numeric_encoding(df)
        
        assert 'sentiment_numeric' in result.columns
        assert result['sentiment_numeric'].iloc[0] == -1  # Fear
        assert result['sentiment_numeric'].iloc[1] == 1   # Greed
        assert result['sentiment_numeric'].iloc[2] == 0   # Neutral
        assert result['sentiment_numeric'].iloc[3] == -2  # Extreme Fear
        assert result['sentiment_numeric'].iloc[4] == 2   # Extreme Greed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
