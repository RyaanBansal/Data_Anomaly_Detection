# feature_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class FeaturePreprocessor:
    """
    Feature Preprocessing Module
    Prepares data for clustering by encoding, scaling, and handling missing values
    """

    def __init__(self, df: pd.DataFrame, profiler):
        self.df = df.copy()
        self.profiler = profiler
        self.processed_df = None
        self.scaler = None
        self.encoders = {}
        self.imputer = None
        self.feature_info = {}

    def preprocess_for_clustering(self,
                                   handle_missing: str = 'knn',
                                   scaling: str = 'robust',
                                   encode_categorical: str = 'label') -> pd.DataFrame:
        """
        Complete preprocessing pipeline for clustering

        Parameters:
        -----------
        handle_missing : str
            'knn', 'mean', 'median', 'most_frequent', 'indicator'
        scaling : str
            'standard', 'robust', 'minmax', 'none'
        encode_categorical : str
            'label', 'onehot', 'frequency'

        Returns:
        --------
        pd.DataFrame
            Preprocessed features ready for clustering
        """
        # Get columns by type
        numeric_cols = self.profiler.get_columns_by_type('numeric')
        categorical_cols = self.profiler.get_columns_by_type('categorical')
        datetime_cols = self.profiler.get_columns_by_type('datetime')

        # Initialize processed dataframe with row-level features
        processed_parts = []

        # 1. Process numeric columns
        if numeric_cols:
            numeric_processed = self._process_numeric(numeric_cols, handle_missing)
            processed_parts.append(numeric_processed)

        # 2. Process categorical columns
        if categorical_cols:
            categorical_processed = self._process_categorical(categorical_cols, encode_categorical)
            processed_parts.append(categorical_processed)

        # 3. Process datetime columns (extract features)
        if datetime_cols:
            datetime_processed = self._process_datetime(datetime_cols)
            processed_parts.append(datetime_processed)

        # 4. Add row-level meta features
        meta_features = self._create_meta_features(handle_missing, numeric_cols)
        processed_parts.append(meta_features)

        # Combine all features
        if processed_parts:
            self.processed_df = pd.concat(processed_parts, axis=1)
        else:
            raise ValueError("No features available for clustering")

        # Handle any remaining NaN values
        self.processed_df = self.processed_df.fillna(0)

        # 5. Scale features
        if scaling != 'none':
            self._scale_features(scaling)

        # Store feature info
        self.feature_info = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'meta_features': list(meta_features.columns),
            'total_features': len(self.processed_df.columns),
            'handle_missing': handle_missing,
            'scaling': scaling,
            'encode_categorical': encode_categorical
        }

        return self.processed_df

    def _process_numeric(self, columns: List[str],
                         handle_missing: str) -> pd.DataFrame:
        """Process numeric columns"""
        df_numeric = self.df[columns].apply(pd.to_numeric, errors='coerce')

        # Handle missing values
        if df_numeric.isnull().any().any():
            if handle_missing == 'knn':
                # KNN Imputation
                imputer = KNNImputer(n_neighbors=5)
                df_imputed = pd.DataFrame(
                    imputer.fit_transform(df_numeric),
                    columns=columns,
                    index=df_numeric.index
                )
                self.imputer = imputer
            elif handle_missing == 'mean':
                df_imputed = df_numeric.fillna(df_numeric.mean())
            elif handle_missing == 'median':
                df_imputed = df_numeric.fillna(df_numeric.median())
            elif handle_missing == 'most_frequent':
                df_imputed = df_numeric.fillna(df_numeric.mode().iloc[0])
            elif handle_missing == 'indicator':
                # Add indicator columns for missing
                missing_indicators = df_numeric.isnull().astype(int)
                missing_indicators.columns = [f"{col}_missing" for col in columns]
                df_imputed = df_numeric.fillna(df_numeric.median())
                df_imputed = pd.concat([df_imputed, missing_indicators], axis=1)
            else:
                df_imputed = df_numeric.fillna(0)
        else:
            df_imputed = df_numeric

        return df_imputed

    def _process_categorical(self, columns: List[str],
                             method: str) -> pd.DataFrame:
        """Process categorical columns"""
        df_categorical = self.df[columns].copy()

        if method == 'label':
            # Label encoding for each column
            for col in columns:
                le = LabelEncoder()
                # Handle NaN by filling with a placeholder
                df_categorical[col] = df_categorical[col].fillna('__MISSING__')
                df_categorical[col] = le.fit_transform(df_categorical[col].astype(str))
                self.encoders[col] = le

        elif method == 'onehot':
            # One-hot encoding (only for low cardinality)
            for col in columns:
                if df_categorical[col].nunique() <= 10:
                    dummies = pd.get_dummies(df_categorical[col].fillna('__MISSING__'),
                                            prefix=col, drop_first=True)
                    df_categorical = pd.concat([df_categorical.drop(col, axis=1), dummies], axis=1)
                else:
                    # Fall back to label encoding for high cardinality
                    le = LabelEncoder()
                    df_categorical[col] = df_categorical[col].fillna('__MISSING__')
                    df_categorical[col] = le.fit_transform(df_categorical[col].astype(str))
                    self.encoders[col] = le

        elif method == 'frequency':
            # Frequency encoding
            for col in columns:
                freq_map = df_categorical[col].value_counts(normalize=True).to_dict()
                df_categorical[col] = df_categorical[col].map(freq_map).fillna(0)

        return df_categorical

    def _process_datetime(self, columns: List[str]) -> pd.DataFrame:
        """Extract features from datetime columns"""
        features = pd.DataFrame(index=self.df.index)

        for col in columns:
            try:
                dt_series = pd.to_datetime(self.df[col], errors='coerce')

                # Extract numeric features
                features[f"{col}_year"] = dt_series.dt.year.fillna(0)
                features[f"{col}_month"] = dt_series.dt.month.fillna(0)
                features[f"{col}_day"] = dt_series.dt.day.fillna(0)
                features[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0)
                features[f"{col}_hour"] = dt_series.dt.hour.fillna(0)

                # Add missing indicator
                features[f"{col}_missing"] = dt_series.isnull().astype(int)

            except Exception as e:
                # If parsing fails, add missing indicator
                features[f"{col}_missing"] = pd.to_datetime(self.df[col], errors='coerce').isnull().astype(int)

        return features

    def _create_meta_features(self, handle_missing: str,
                              numeric_cols: List[str]) -> pd.DataFrame:
        """Create row-level meta features"""
        meta = pd.DataFrame(index=self.df.index)

        # Missing value features (always included)
        meta['row_missing_count'] = self.df.isnull().sum(axis=1)
        meta['row_missing_ratio'] = meta['row_missing_count'] / self.df.shape[1]
        meta['has_any_missing'] = (meta['row_missing_count'] > 0).astype(int)

        # Numeric summary features
        if numeric_cols:
            df_numeric = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            meta['numeric_row_mean'] = df_numeric.mean(axis=1).fillna(0)
            meta['numeric_row_std'] = df_numeric.std(axis=1).fillna(0)
            meta['numeric_row_min'] = df_numeric.min(axis=1).fillna(0)
            meta['numeric_row_max'] = df_numeric.max(axis=1).fillna(0)
            meta['numeric_row_range'] = meta['numeric_row_max'] - meta['numeric_row_min']

        # Duplicate indicator
        meta['is_duplicate'] = self.df.duplicated(keep=False).astype(int)

        return meta

    def _scale_features(self, method: str):
        """Scale features"""
        if method == 'standard':
            self.scaler = StandardScaler()
            self.processed_df = pd.DataFrame(
                self.scaler.fit_transform(self.processed_df),
                columns=self.processed_df.columns,
                index=self.processed_df.index
            )
        elif method == 'robust':
            self.scaler = RobustScaler()
            self.processed_df = pd.DataFrame(
                self.scaler.fit_transform(self.processed_df),
                columns=self.processed_df.columns,
                index=self.processed_df.index
            )
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            self.processed_df = pd.DataFrame(
                self.scaler.fit_transform(self.processed_df),
                columns=self.processed_df.columns,
                index=self.processed_df.index
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance using variance
        Higher variance = more informative for clustering
        """
        if self.processed_df is None:
            return {}

        variances = self.processed_df.var()
        total_variance = variances.sum()

        if total_variance == 0:
            return {col: 0 for col in self.processed_df.columns}

        importance = (variances / total_variance * 100).to_dict()
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_processed_data(self) -> pd.DataFrame:
        """Get preprocessed data"""
        return self.processed_df

    def get_feature_info(self) -> Dict:
        """Get feature processing information"""
        return self.feature_info
