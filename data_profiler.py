# data_profiler.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataProfiler:
    """
    Data Profiling Module
    Analyzes column types, statistics, and data quality at row level
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.profile = {}
        self.column_types = {}
        
    def generate_profile(self) -> Dict:
        """Generate complete data profile"""
        self.column_types = self._detect_column_types()
        
        self.profile = {
            'basic_info': self._basic_info(),
            'column_types': self.column_types,
            'numeric_stats': self._numeric_statistics(),
            'categorical_stats': self._categorical_statistics(),
            'missing_analysis': self._row_level_missing_analysis(),
            'data_quality_score': self._calculate_quality_score(),
            'row_features': self._extract_row_features()
        }
        
        return self.profile
    
    def _basic_info(self) -> Dict:
        """Basic dataset information"""
        return {
            'shape': self.df.shape,
            'total_cells': self.df.shape[0] * self.df.shape[1],
            'memory_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
            'duplicate_rows': int(self.df.duplicated().sum()),
            'duplicate_percentage': round(self.df.duplicated().sum() / len(self.df) * 100, 2)
        }
    
    def _detect_column_types(self) -> Dict[str, str]:
        """
        Detect column data types for clustering preparation
        Returns: 'numeric', 'categorical', 'datetime', 'text', 'identifier'
        """
        column_types = {}
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            non_null = self.df[col].dropna()
            
            if len(non_null) == 0:
                column_types[col] = 'empty'
                continue
                
            # Check datetime
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                column_types[col] = 'datetime'
                continue
            
            # Try to parse as datetime
            if dtype == 'object':
                try:
                    pd.to_datetime(non_null.head(100), errors='raise')
                    column_types[col] = 'datetime'
                    continue
                except:
                    pass
            
            # Check if identifier (unique values, likely ID column)
            if dtype == 'object' or pd.api.types.is_integer_dtype(self.df[col]):
                unique_ratio = non_null.nunique() / len(non_null)
                if unique_ratio > 0.95 and len(non_null) > 10:
                    column_types[col] = 'identifier'
                    continue
            
            # Check numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                unique_count = non_null.nunique()
                # If very few unique values, treat as categorical
                if unique_count <= 10 and unique_count < len(non_null) * 0.05:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'numeric'
                continue
            
            # Check categorical (object with limited unique values)
            if dtype == 'object':
                unique_ratio = non_null.nunique() / len(non_null)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    column_types[col] = 'categorical'
                elif len(non_null.str.len().mode()) > 50:  # Long text
                    column_types[col] = 'text'
                else:
                    column_types[col] = 'categorical'
                continue
            
            column_types[col] = 'other'
        
        return column_types
    
    def _numeric_statistics(self) -> Dict[str, Dict]:
        """Compute statistics for numeric columns"""
        statistics = {}
        
        for col, col_type in self.column_types.items():
            if col_type == 'numeric':
                series = pd.to_numeric(self.df[col], errors='coerce')
                non_null = series.dropna()
                
                if len(non_null) == 0:
                    continue
                
                statistics[col] = {
                    'count': int(non_null.count()),
                    'missing': int(series.isna().sum()),
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()),
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'q1': float(non_null.quantile(0.25)),
                    'q3': float(non_null.quantile(0.75)),
                    'iqr': float(non_null.quantile(0.75) - non_null.quantile(0.25)),
                    'skewness': float(non_null.skew()),
                    'kurtosis': float(non_null.kurtosis())
                }
        
        return statistics
    
    def _categorical_statistics(self) -> Dict[str, Dict]:
        """Compute statistics for categorical columns"""
        statistics = {}
        
        for col, col_type in self.column_types.items():
            if col_type == 'categorical':
                non_null = self.df[col].dropna()
                value_counts = non_null.value_counts()
                
                statistics[col] = {
                    'count': int(non_null.count()),
                    'missing': int(self.df[col].isna().sum()),
                    'unique': int(non_null.nunique()),
                    'top_values': value_counts.head(10).to_dict(),
                    'top_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'top_percentage': float(value_counts.iloc[0] / len(non_null) * 100) if len(value_counts) > 0 else 0
                }
        
        return statistics
    
    def _row_level_missing_analysis(self) -> Dict:
        """
        Analyze missing values at row level
        Critical for identifying rows that are anomalies due to missing data
        """
        missing_per_row = self.df.isnull().sum(axis=1)
        missing_ratio_per_row = missing_per_row / self.df.shape[1]
        
        return {
            'rows_with_any_missing': int((missing_per_row > 0).sum()),
            'percentage_rows_with_missing': round((missing_per_row > 0).sum() / len(self.df) * 100, 2),
            'rows_missing_per_column_threshold': {
                '10%': int((missing_ratio_per_row > 0.1).sum()),
                '25%': int((missing_ratio_per_row > 0.25).sum()),
                '50%': int((missing_ratio_per_row > 0.5).sum()),
                '75%': int((missing_ratio_per_row > 0.75).sum())
            },
            'max_missing_in_single_row': int(missing_per_row.max()),
            'avg_missing_per_row': round(missing_per_row.mean(), 2),
            'missing_distribution': missing_per_row.value_counts().sort_index().to_dict()
        }
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        completeness = 100 - (self.df.isnull().sum().sum() / self.df.shape[1] / len(self.df) * 100)
        uniqueness = 100 - (self.df.duplicated().sum() / len(self.df) * 100)
        
        # Weighted average
        quality_score = (completeness * 0.7) + (uniqueness * 0.3)
        return round(quality_score, 2)
    
    def _extract_row_features(self) -> Dict:
        """
        Extract features that describe each row
        These features help in clustering
        """
        features = {
            'missing_count': self.df.isnull().sum(axis=1).tolist(),
            'missing_ratio': (self.df.isnull().sum(axis=1) / self.df.shape[1]).tolist(),
            'is_duplicate': self.df.duplicated(keep=False).tolist(),
            'numeric_sum': [],
            'numeric_mean': [],
            'numeric_std': []
        }
        
        numeric_cols = [col for col, t in self.column_types.items() if t == 'numeric']
        
        if numeric_cols:
            df_numeric = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            features['numeric_sum'] = df_numeric.sum(axis=1).fillna(0).tolist()
            features['numeric_mean'] = df_numeric.mean(axis=1).fillna(0).tolist()
            features['numeric_std'] = df_numeric.std(axis=1).fillna(0).tolist()
        
        return features
    
    def get_columns_by_type(self, type_name: str) -> List[str]:
        """Get columns of a specific type"""
        return [col for col, t in self.column_types.items() if t == type_name]
    
    def get_clustering_columns(self) -> List[str]:
        """
        Get columns suitable for clustering
        Excludes identifiers, text, and empty columns
        """
        exclude_types = ['identifier', 'text', 'empty']
        return [col for col, t in self.column_types.items() if t not in exclude_types]

    # ------------------------------------------------------------------
    # SYNTHETIC DATETIME LABELING (Pre-processing step)
    # ------------------------------------------------------------------

    def add_datetime_labels(self,
                            reference_date=None,
                            start_hour: int = 9,
                            end_hour: int = 17) -> None:
        """
        Add synthetic datetime labels to the dataset as a pre-processing step.

        The dataset is divided into 7 contiguous parts, one for each day of
        the week (Monday - Sunday).  Within each day the data-points are
        spread evenly across the ``start_hour`` - ``end_hour`` window so that
        every row receives a unique-ish timestamp and a ``day_of_week`` label.

        Two new columns are appended **in-place** to ``self.df``:

        * ``synthetic_datetime`` - ``pd.Timestamp``
        * ``day_of_week``       - ``str`` (e.g. ``'Monday'``)

        Parameters
        ----------
        reference_date : datetime-like or str, optional
            The Monday of the reference week.  Defaults to ``2025-01-06``
            (a Monday).  If a non-Monday date is supplied it is rolled back
            to the preceding Monday automatically.
        start_hour : int
            Start of the daily business-hours window (default 9 = 9 AM).
        end_hour : int
            End of the daily business-hours window (default 17 = 5 PM).

        Notes
        -----
        * Rows are split **contiguously** - the first chunk becomes Monday,
          the second Tuesday, and so on.  This preserves any ordering
          already present in the CSV.
        * A small random jitter (+-2 min) is added to the timestamps for
          realism and then sorted so the column remains monotonically
          increasing within each day.
        * Days with zero rows (possible when ``n_rows < 7``) are simply
          skipped.
        """
        n_rows = len(self.df)
        if n_rows == 0:
            return

        days_of_week = [
            'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]
        n_days = 7

        # ---- resolve reference Monday -----------------------------------
        if reference_date is None:
            reference_date = pd.Timestamp('2025-01-06')  # a Monday
        else:
            reference_date = pd.Timestamp(reference_date)
            # Roll back to the previous Monday if needed
            days_since_monday = reference_date.dayofweek
            if days_since_monday != 0:
                reference_date = reference_date - pd.Timedelta(days=days_since_monday)

        # ---- split rows into 7 groups -----------------------------------
        base, remainder = divmod(n_rows, n_days)
        group_sizes = [
            base + (1 if i < remainder else 0)
            for i in range(n_days)
        ]

        # ---- generate timestamps ----------------------------------------
        total_minutes = (end_hour - start_hour) * 60  # e.g. 480 min for 8 h

        all_datetimes: List[pd.Timestamp] = []
        all_day_labels: List[str] = []

        for day_i, day_name in enumerate(days_of_week):
            n_group = group_sizes[day_i]
            if n_group == 0:
                continue

            # Evenly space timestamps; avoid endpoint so last point != end_hour
            if n_group == 1:
                minute_offsets = np.array([total_minutes / 2.0])
            else:
                minute_offsets = np.linspace(
                    0, total_minutes, n_group, endpoint=False
                )

            # Small random jitter for realism (+-2 min), then re-sort
            rng = np.random.default_rng(seed=42 + day_i)
            jitter = rng.uniform(-2.0, 2.0, n_group)
            minute_offsets = np.clip(minute_offsets + jitter, 0, total_minutes - 1)
            minute_offsets = np.sort(minute_offsets)

            current_date = reference_date + pd.Timedelta(days=day_i)

            for mo in minute_offsets:
                h = start_hour + int(mo // 60)
                m = int(mo % 60)
                dt = current_date.replace(hour=h, minute=m, second=0, microsecond=0)
                all_datetimes.append(dt)
                all_day_labels.append(day_name)

        # ---- assign to DataFrame (positional alignment) ------------------
        self.df = self.df.copy()  # avoid SettingWithCopyWarning
        self.df['synthetic_datetime'] = all_datetimes
        self.df['day_of_week'] = all_day_labels

    def get_day_labels(self) -> pd.Series:
        """
        Return the ``day_of_week`` column (or an empty Series if datetime
        labeling has not been applied yet).
        """
        if 'day_of_week' in self.df.columns:
            return self.df['day_of_week']
        return pd.Series(dtype=str)
