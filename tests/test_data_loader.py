"""
Unit tests for data_loader module.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader("data")
        assert loader.data_dir == Path("data")
    
    def test_validate_data_basic(self):
        """Test basic data validation."""
        loader = DataLoader()
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3, None],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        results = loader.validate_data(df, "test_dataset")
        
        assert results['dataset'] == "test_dataset"
        assert results['rows'] == 4
        assert results['columns'] == 2
        assert results['duplicates'] == 0
        assert results['missing_values']['col1'] == 1
        assert results['missing_values']['col2'] == 0
    
    def test_validate_data_duplicates(self):
        """Test duplicate detection."""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        
        results = loader.validate_data(df, "test_dataset")
        assert results['duplicates'] == 1  # One duplicate row
    
    def test_missing_file_error_message(self):
        """Test that helpful error messages are raised for missing files."""
        loader = DataLoader("nonexistent_dir")
        
        with pytest.raises(FileNotFoundError) as excinfo:
            loader.load_fear_greed_index()
        
        assert "Fear/Greed index file not found" in str(excinfo.value)
        assert "drive.google.com" in str(excinfo.value)
    
    def test_missing_historical_data_error(self):
        """Test error message for missing historical data."""
        loader = DataLoader("nonexistent_dir")
        
        with pytest.raises(FileNotFoundError) as excinfo:
            loader.load_historical_data()
        
        assert "Historical data file not found" in str(excinfo.value)
        assert "drive.google.com" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
