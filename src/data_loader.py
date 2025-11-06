"""
Data loading utilities for Hyperliquid trading and sentiment analysis.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import warnings


class DataLoader:
    """Load and validate trading and sentiment datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def load_fear_greed_index(self, filename: str = "fear_greed_index.csv") -> pd.DataFrame:
        """
        Load the Fear/Greed Index dataset.
        
        Args:
            filename: Name of the CSV file containing sentiment data
            
        Returns:
            DataFrame with Date and Classification columns
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Fear/Greed index file not found at {filepath}\n"
                f"Please download from: "
                f"https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing"
            )
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded Fear/Greed Index: {len(df)} records")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.iloc[0, 0] if len(df) > 0 else 'N/A'} to {df.iloc[-1, 0] if len(df) > 0 else 'N/A'}")
        
        return df
    
    def load_historical_data(self, filename: str = "historical_data.csv") -> pd.DataFrame:
        """
        Load the Hyperliquid historical trader data.
        
        Args:
            filename: Name of the CSV file containing trading data
            
        Returns:
            DataFrame with trader execution records
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Historical data file not found at {filepath}\n"
                f"Please download from: "
                f"https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing"
            )
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded Historical Trading Data: {len(df):,} records")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Unique accounts: {df['account'].nunique() if 'account' in df.columns else 'N/A'}")
        print(f"  Date range: {df['time'].min() if 'time' in df.columns else 'N/A'} to {df['time'].max() if 'time' in df.columns else 'N/A'}")
        
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both datasets and return as a tuple.
        
        Returns:
            Tuple of (historical_data, fear_greed_index)
        """
        print("Loading datasets...")
        print("-" * 60)
        
        try:
            historical = self.load_historical_data()
            sentiment = self.load_fear_greed_index()
            print("-" * 60)
            print("✓ All datasets loaded successfully!")
            return historical, sentiment
        except FileNotFoundError as e:
            print("\n" + "=" * 60)
            print("⚠️  DATASET NOT FOUND")
            print("=" * 60)
            print(str(e))
            print("\n📥 Please download the datasets and place them in the data/ directory.")
            print("=" * 60)
            raise
    
    def validate_data(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Perform basic validation checks on a dataset.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'dataset': dataset_name,
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        print(f"\n📊 Validation: {dataset_name}")
        print(f"  Rows: {results['rows']:,}")
        print(f"  Columns: {results['columns']}")
        print(f"  Duplicates: {results['duplicates']}")
        print(f"  Memory: {results['memory_usage_mb']:.2f} MB")
        
        missing = {k: v for k, v in results['missing_values'].items() if v > 0}
        if missing:
            print(f"  Missing values:")
            for col, count in missing.items():
                pct = (count / len(df)) * 100
                print(f"    - {col}: {count} ({pct:.1f}%)")
        else:
            print(f"  ✓ No missing values")
        
        return results


# Convenience function for quick loading
def load_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick function to load both datasets.
    
    Returns:
        Tuple of (historical_data, fear_greed_index)
    """
    loader = DataLoader(data_dir)
    return loader.load_all()
