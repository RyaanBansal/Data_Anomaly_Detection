# trend_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TrendAnalyzer:
    def __init__(self, cluster_analyzer, preprocessor, profiler):
        self.cluster_analyzer = cluster_analyzer
        self.preprocessor = preprocessor
        self.profiler = profiler
        self.trend_analysis = {}

    def analyze_trends(self) -> Dict:
        trend_info = self.cluster_analyzer.get_trend_clusters()
        if not trend_info['trend_clusters']:
            return {'trend_detected': False, 'message': 'No clear trend clusters identified'}

        self.trend_analysis = {
            'trend_detected': True,
            'trend_summary': trend_info,
            'normal_profile': None
        }
        self.trend_analysis['normal_profile'] = self._build_normal_profile(trend_info)
        return self.trend_analysis

    def _build_normal_profile(self, trend_info: Dict) -> Dict:
        processed_data = self.preprocessor.get_processed_data()
        all_trend_indices = [idx for c in trend_info['trend_clusters'] for idx in c['row_indices']]
        trend_data = processed_data.loc[all_trend_indices]

        bounds = {}
        for col in processed_data.columns:
            mean, std = trend_data[col].mean(), trend_data[col].std()
            bounds[col] = {'lower': mean - 2 * std, 'upper': mean + 2 * std, 'mean': mean, 'std': std}

        return {'centroid': trend_data.mean().values.tolist(), 'feature_bounds': bounds, 'trend_indices': all_trend_indices}

    def calculate_deviation_severity(self, anomaly_indices: List[int]) -> Dict[int, str]:
        """
        Calculates severity for clustered anomalies based strictly on how far
        they deviate from the normal trend cluster centroid.
        """
        processed_data = self.preprocessor.get_processed_data()
        normal_profile = self.trend_analysis['normal_profile']
        trend_centroid = np.array(normal_profile['centroid'])

        distances = {}
        valid_anomalies = [idx for idx in anomaly_indices if idx in processed_data.index]

        # Calculate Euclidean distance from the normal trend centroid for each anomaly
        for idx in valid_anomalies:
            row_values = processed_data.loc[idx].values
            distances[idx] = np.linalg.norm(row_values - trend_centroid)

        if not distances:
            return {idx: 'medium' for idx in anomaly_indices}

        # Assign severity based on distance percentiles
        dist_series = pd.Series(distances)
        p75 = dist_series.quantile(0.75)
        p50 = dist_series.quantile(0.50)

        severity_map = {}
        for idx, dist in distances.items():
            if dist >= p75:
                severity_map[idx] = 'high'      # Top 25% furthest from normal
            elif dist >= p50:
                severity_map[idx] = 'medium'    # 50th - 75th percentile
            else:
                severity_map[idx] = 'low'       # Closest to normal trend

        return severity_map

    def get_error_columns_bulk(self, anomaly_indices: List[int]) -> Dict[int, List[str]]:
        """Maps internal processed features back to the original dataframe column names."""
        processed_data = self.preprocessor.get_processed_data()
        original_data = self.cluster_analyzer.original_data
        normal_profile = self.trend_analysis['normal_profile']

        error_map = {}
        for idx in anomaly_indices:
            if idx not in processed_data.index: continue
            processed_row = processed_data.loc[idx]
            original_row = original_data.loc[idx]
            error_cols = set()

            for feat in processed_data.columns:
                if feat not in normal_profile['feature_bounds']: continue
                bounds = normal_profile['feature_bounds'][feat]
                z_score = (processed_row[feat] - bounds['mean']) / bounds['std'] if bounds['std'] > 0 else 0

                if abs(z_score) > 2:
                    original_col = self._translate_feature_to_column_name(feat, original_row)
                    if original_col: error_cols.add(original_col)

            error_map[idx] = list(error_cols)
        return error_map

    def _translate_feature_to_column_name(self, feat: str, original_row: pd.Series) -> Optional[str]:
        if feat in original_row.index: return feat
        if feat == 'is_duplicate': return "Duplicate Row"
        if feat == 'numeric_row_std': return "Row Variance"
        return None


# ======================================================================
# DAY-WISE TREND ANALYZER  (STL-inspired per-day profiling)
# ======================================================================

class DayWiseTrendAnalyzer:
    """
    Builds a **separate** normal-profile (centroid + 2-sigma bounds) for every
    day of the week, then scores anomalies against the profile of their
    *own* day.

    This mirrors the STL-decomposition idea: the "seasonal" component here
    is the day-of-week.  By learning what "normal" looks like for Monday
    independently of Tuesday, we avoid penalising legitimate day-to-day
    variation.
    """

    def __init__(self, day_cluster_analyzer, preprocessor):
        """
        Parameters
        ----------
        day_cluster_analyzer : DayWiseClusterAnalyzer
            The fitted day-wise clustering engine.
        preprocessor : FeaturePreprocessor
            Used to retrieve the processed feature matrix.
        """
        self.day_cluster_analyzer = day_cluster_analyzer
        self.preprocessor = preprocessor
        self.day_normal_profiles: Dict[str, Dict] = {}
        self.day_trend_info: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze_trends(self) -> Dict:
        """
        Build per-day normal profiles from the trend clusters of each day's
        ClusterAnalyzer.

        Returns
        -------
        dict
            ``{day: {trend_detected, n_trend_clusters, trend_pct, ...}}``
        """
        for day, clusterer in self.day_cluster_analyzer.day_models.items():
            trend_info = clusterer.get_trend_clusters()

            if trend_info['trend_clusters']:
                self.day_trend_info[day] = trend_info
                self.day_normal_profiles[day] = \
                    self._build_day_normal_profile(day, trend_info)
            else:
                self.day_trend_info[day] = {
                    'trend_detected': False,
                    'message': trend_info.get('message', 'No trend clusters')
                }

        return self.day_trend_info

    def calculate_deviation_severity(self,
                                     anomaly_records: List[Dict]) -> Dict[int, str]:
        """
        Score day-wise anomalies by their distance from the *per-day*
        normal centroid.

        Parameters
        ----------
        anomaly_records : list of dict
            Each dict must have ``'index'`` and ``'day_of_week'`` keys
            (as returned by ``DayWiseClusterAnalyzer.get_all_anomalies()``).

        Returns
        -------
        dict
            ``{dataframe_index: 'high'|'medium'|'low'}``
        """
        processed_data = self.preprocessor.get_processed_data()

        # Bucket distances by day so we can compute per-day percentiles
        day_distances: Dict[str, List[Tuple[int, float]]] = {}

        for record in anomaly_records:
            idx = record['index']
            day = record.get('day_of_week')

            if day is None or day not in self.day_normal_profiles:
                continue
            if idx not in processed_data.index:
                continue

            profile = self.day_normal_profiles[day]
            row_values = processed_data.loc[idx].values
            centroid = np.array(profile['centroid'])
            distance = float(np.linalg.norm(row_values - centroid))

            day_distances.setdefault(day, []).append((idx, distance))

        # Assign severity using per-day percentile thresholds
        severity_map: Dict[int, str] = {}

        for day, dist_list in day_distances.items():
            distances = np.array([d[1] for d in dist_list])
            if len(distances) == 0:
                continue

            p75 = float(np.percentile(distances, 75))
            p50 = float(np.percentile(distances, 50))

            for idx, dist in dist_list:
                if dist >= p75:
                    severity_map[idx] = 'high'
                elif dist >= p50:
                    severity_map[idx] = 'medium'
                else:
                    severity_map[idx] = 'low'

        # Fill defaults for any records that couldn't be scored
        for record in anomaly_records:
            if record['index'] not in severity_map:
                severity_map[record['index']] = 'medium'

        return severity_map

    def get_error_columns_bulk(self,
                                anomaly_records: List[Dict]) -> Dict[int, List[str]]:
        """
        Map processed features back to original column names using the
        **per-day** normal profile.

        Parameters
        ----------
        anomaly_records : list of dict
            Each with ``'index'`` and ``'day_of_week'``.

        Returns
        -------
        dict
            ``{dataframe_index: [column_name, ...]}``
        """
        processed_data = self.preprocessor.get_processed_data()
        original_data = self.day_cluster_analyzer.original_data
        error_map: Dict[int, List[str]] = {}

        for record in anomaly_records:
            idx = record['index']
            day = record.get('day_of_week')

            if day is None or day not in self.day_normal_profiles:
                continue
            if idx not in processed_data.index:
                continue

            processed_row = processed_data.loc[idx]
            original_row = original_data.loc[idx]
            profile = self.day_normal_profiles[day]
            error_cols: set = set()

            for feat in processed_data.columns:
                if feat not in profile['feature_bounds']:
                    continue

                bounds = profile['feature_bounds'][feat]
                std = bounds['std']
                z = (processed_row[feat] - bounds['mean']) / std if std > 0 else 0

                if abs(z) > 2:
                    original_col = self._translate_feature_name(
                        feat, original_row
                    )
                    if original_col:
                        error_cols.add(original_col)

            error_map[idx] = list(error_cols)

        return error_map

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _build_day_normal_profile(self, day: str, trend_info: Dict) -> Dict:
        """Build centroid + 2-sigma feature bounds for *day*'s trend rows."""
        processed_data = self.preprocessor.get_processed_data()
        day_mask = self.day_cluster_analyzer.day_labels == day
        day_processed = processed_data[day_mask]

        all_trend_indices = [
            idx
            for cluster in trend_info['trend_clusters']
            for idx in cluster['row_indices']
        ]
        trend_data = day_processed.loc[all_trend_indices]

        if trend_data.empty:
            return {'centroid': [], 'feature_bounds': {}}

        bounds = {}
        for col in day_processed.columns:
            mean = float(trend_data[col].mean())
            std = float(trend_data[col].std())
            bounds[col] = {
                'lower': mean - 2 * std,
                'upper': mean + 2 * std,
                'mean': mean,
                'std': std,
            }

        return {
            'centroid': trend_data.mean().values.tolist(),
            'feature_bounds': bounds,
            'trend_indices': all_trend_indices,
        }

    @staticmethod
    def _translate_feature_name(feat: str, original_row: pd.Series) -> Optional[str]:
        """Attempt to map a processed feature name to its original column."""
        if feat in original_row.index:
            return feat
        if feat == 'is_duplicate':
            return 'Duplicate Row'
        if feat == 'numeric_row_std':
            return 'Row Variance'
        return None
