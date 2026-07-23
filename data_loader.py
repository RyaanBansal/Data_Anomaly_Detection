# data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Simplified Data Loader for CSV files only
    Handles encoding issues and common CSV variations
    """
    
    def __init__(self):
        self.df = None
        self.file_info = {}
        
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with automatic encoding and delimiter detection
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
        **kwargs : 
            Additional arguments passed to pd.read_csv
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != '.csv':
            raise ValueError("Only CSV files are supported")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    **kwargs
                )
                self.file_info = {
                    'file_name': file_path.name,
                    'file_size_kb': round(file_path.stat().st_size / 1024, 2),
                    'encoding': encoding,
                    'rows': len(self.df),
                    'columns': len(self.df.columns)
                }
                return self.df
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file with any supported encoding: {file_path}")
    
    def load_csv_auto(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV with automatic delimiter and header detection
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read first few lines for analysis
        with open(file_path, 'rb') as f:
            raw_head = f.read(5000)
        
        # Detect encoding
        import chardet
        encoding_result = chardet.detect(raw_head)
        encoding = encoding_result['encoding'] or 'utf-8'
        
        # Try to detect delimiter
        sample_lines = raw_head.decode(encoding, errors='ignore').split('\n')[:5]
        delimiters = [',', '\t', ';', '|']
        best_delim = ','
        max_cols = 1
        
        for delim in delimiters:
            cols = len(sample_lines[0].split(delim))
            if cols > max_cols:
                max_cols = cols
                best_delim = delim
        
        # Check if first row is header
        first_row = sample_lines[0].split(best_delim)
        second_row = sample_lines[1].split(best_delim) if len(sample_lines) > 1 else []
        
        has_header = True
        if second_row:
            # If first row looks like data (mostly numeric), no header
            numeric_in_first = sum(1 for x in first_row if self._is_numeric(x))
            numeric_in_second = sum(1 for x in second_row if self._is_numeric(x))
            
            if numeric_in_first > numeric_in_second:
                has_header = False
        
        # Load with detected settings
        self.df = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=best_delim,
            header=0 if has_header else None
        )
        
        # Generate column names if no header
        if not has_header:
            self.df.columns = [f'Column_{i}' for i in range(len(self.df.columns))]
        
        self.file_info = {
            'file_name': file_path.name,
            'file_size_kb': round(file_path.stat().st_size / 1024, 2),
            'encoding': encoding,
            'delimiter': best_delim,
            'has_header': has_header,
            'rows': len(self.df),
            'columns': len(self.df.columns)
        }
        
        return self.df
    
    def _is_numeric(self, value: str) -> bool:
        """Check if string represents a numeric value"""
        try:
            float(value.strip().replace(',', '').replace('$', ''))
            return True
        except (ValueError, AttributeError):
            return False
    
    def get_info(self) -> Dict:
        """Get file loading information"""
        return self.file_info
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the loaded DataFrame"""
        return self.df