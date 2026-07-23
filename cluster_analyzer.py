# cluster_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class ClusterAnalyzer:
    """
    Clustering Engine optimized for Clean Data Subset.
    """

    def __init__(self, processed_data: pd.DataFrame, original_data_subset: pd.DataFrame, original_full_size: int):
        self.processed_data = processed_data.copy()
        self.original_data = original_data_subset.copy()
        self.original_full_size = original_full_size
        self.cluster_labels = None
        self.cluster_info = {}
        self.anomaly_info = {}

        # FIX: Create a safe mapping from DataFrame index to Numpy array position
        self.idx_to_pos = {idx: pos for pos, idx in enumerate(self.original_data.index)}

    def fit(self, method: str = 'auto', min_cluster_size: Optional[int] = None,
            min_samples: Optional[int] = None, cluster_selection_epsilon: float = 0.0) -> Dict:

        if min_cluster_size is None:
            min_cluster_size = max(5, int(len(self.processed_data) * 0.01))
        if min_samples is None:
            min_samples = max(2, min_cluster_size // 2)

        if method == 'auto':
            method = 'hdbscan' if HDBSCAN_AVAILABLE else 'dbscan'

        if method == 'hdbscan' and HDBSCAN_AVAILABLE:
            self._fit_hdbscan(min_cluster_size, min_samples, cluster_selection_epsilon)
        else:
            self._fit_dbscan(min_samples)
            method = 'dbscan (fallback)'

        self._analyze_clusters()
        self._detect_cluster_anomalies()

        return {
            'method': method,
            'parameters': {'min_cluster_size': min_cluster_size, 'min_samples': min_samples},
            'cluster_info': self.cluster_info,
            'anomaly_info': self.anomaly_info
        }

    def _fit_hdbscan(self, min_cluster_size: int, min_samples: int, epsilon: float):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            cluster_selection_epsilon=epsilon, metric='euclidean',
            cluster_selection_method='eom', prediction_data=True
        )
        self.cluster_labels = clusterer.fit_predict(self.processed_data)
        self.probabilities = clusterer.probabilities_
        self.clusterer = clusterer

    def _fit_dbscan(self, min_samples: int):
        epsilon = self._calculate_epsilon(min_samples)
        clusterer = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean')
        self.cluster_labels = clusterer.fit_predict(self.processed_data)
        self.clusterer = clusterer
        self._calculate_dbscan_probabilities(epsilon)

    def _calculate_epsilon(self, min_samples: int) -> float:
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(self.processed_data)
        distances, _ = neighbors.kneighbors(self.processed_data)
        k_distances = np.sort(distances[:, -1])
        epsilon = np.percentile(k_distances, 90) if len(k_distances) > 10 else np.mean(k_distances)
        return max(0.1, min(epsilon, np.percentile(k_distances, 99)))

    def _calculate_dbscan_probabilities(self, epsilon: float):
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors.fit(self.processed_data)
        distances, _ = neighbors.kneighbors(self.processed_data)
        max_dist = distances[:, 1].max()
        self.probabilities = 1 - (distances[:, 1] / max_dist) if max_dist > 0 else np.ones(len(self.processed_data))
        self.probabilities[self.cluster_labels == -1] = 0.1

    def _analyze_clusters(self):
        unique_labels = np.unique(self.cluster_labels)
        cluster_stats = {}

        for label in unique_labels:
            mask = self.cluster_labels == label
            cluster_size = mask.sum()
            cluster_data = self.processed_data[mask]
            centroid = cluster_data.mean().values

            if cluster_size > 1:
                distances_from_centroid = np.linalg.norm(cluster_data.values - centroid, axis=1)
                avg_spread = distances_from_centroid.mean()
            else:
                avg_spread = 0

            density = cluster_size / (avg_spread + 0.001)

            cluster_stats[label] = {
                'size': int(cluster_size),
                'percentage': round(cluster_size / self.original_full_size * 100, 2),
                'centroid': centroid.tolist(),
                'density': round(density, 4),
                'is_noise': label == -1
            }

        self.cluster_info = {
            'n_clusters': len([l for l in unique_labels if l != -1]),
            'n_noise': int((self.cluster_labels == -1).sum()),
            'noise_percentage': round((self.cluster_labels == -1).sum() / self.original_full_size * 100, 2),
            'cluster_stats': cluster_stats,
            'largest_cluster': max(cluster_stats.items(), key=lambda x: x[1]['size'])[0] if cluster_stats else None
        }

        non_noise_mask = self.cluster_labels != -1
        if non_noise_mask.sum() > 2 and len(unique_labels) > 2:
            try:
                self.cluster_info['silhouette_score'] = round(silhouette_score(self.processed_data[non_noise_mask], self.cluster_labels[non_noise_mask]), 4)
            except:
                self.cluster_info['silhouette_score'] = None

    def _detect_cluster_anomalies(self):
        cluster_stats = self.cluster_info['cluster_stats']
        total_rows = self.original_full_size

        # Use boolean mask on numpy array to get positional indices, then map back to df index
        noise_positions = np.where(self.cluster_labels == -1)[0]
        noise_df_indices = [self.original_data.index[pos] for pos in noise_positions]

        anomalies = {
            'noise_anomalies': {'indices': noise_df_indices},
            'clustered_anomalies': {'indices': [], 'details': []}
        }

        for label, stats in cluster_stats.items():
            if stats['is_noise']:
                continue
            size_ratio = stats['size'] / total_rows
            if size_ratio < 0.05:
                # Map positional labels back to dataframe indices safely
                label_positions = np.where(self.cluster_labels == label)[0]
                label_df_indices = [self.original_data.index[pos] for pos in label_positions]

                anomalies['clustered_anomalies']['indices'].extend(label_df_indices)
                anomalies['clustered_anomalies']['details'].append({
                    'label': int(label),
                    'size': stats['size'],
                    'reason': f"Structural outlier (Cluster covers {stats['percentage']}% of total data)"
                })

        all_anomaly_indices = set(anomalies['noise_anomalies']['indices'] + anomalies['clustered_anomalies']['indices'])
        anomaly_details = []

        for idx in all_anomaly_indices:
            # FIX: Safely look up the cluster label using our mapping dictionary
            pos = self.idx_to_pos[idx]
            label = self.cluster_labels[pos]

            if label == -1:
                reason = "Noise point (unique structural error)"
            else:
                cl_detail = next((c for c in anomalies['clustered_anomalies']['details'] if c['label'] == label), None)
                reason = cl_detail['reason'] if cl_detail else "Structural anomaly"

            anomaly_details.append({
                'index': idx,  # Keep the ORIGINAL dataframe index here
                'cluster_label': int(label),
                'cluster_size': cluster_stats[label]['size'],
                'reason': reason
            })

        self.anomaly_info = {
            'total_anomalies': len(all_anomaly_indices),
            'total_percentage': round(len(all_anomaly_indices) / total_rows * 100, 2),
            'anomaly_details': anomaly_details,
            'by_severity': {'high': 0, 'medium': 0, 'low': 0},
            'normal_indices': list(set(self.original_data.index) - all_anomaly_indices),
            'normal_count': len(self.original_data) - len(all_anomaly_indices)
        }

    def get_trend_clusters(self) -> Dict:
        cluster_stats = self.cluster_info['cluster_stats']
        all_sizes = [v['size'] for v in cluster_stats.values() if not v['is_noise']]
        if not all_sizes: return {'trend_clusters': [], 'message': 'No valid clusters found'}

        size_threshold = np.percentile(all_sizes, 25)
        all_densities = [v['density'] for v in cluster_stats.values() if not v['is_noise']]
        density_threshold = np.percentile(all_densities, 25) if all_densities else 0

        trend_clusters = []
        for label, stats in cluster_stats.items():
            if not stats['is_noise'] and stats['size'] >= size_threshold and stats['density'] >= density_threshold:
                # Safely map positional indices back to original df indices
                label_positions = np.where(self.cluster_labels == label)[0]
                label_df_indices = [self.original_data.index[pos] for pos in label_positions]

                trend_clusters.append({
                    'label': int(label), 'size': stats['size'], 'percentage': stats['percentage'],
                    'centroid': stats['centroid'], 'density': stats['density'],
                    'row_indices': label_df_indices
                })

        trend_clusters.sort(key=lambda x: x['size'], reverse=True)
        total_trend_rows = sum(c['size'] for c in trend_clusters)

        return {
            'trend_clusters': trend_clusters, 'total_trend_rows': total_trend_rows,
            'trend_percentage': round(total_trend_rows / self.original_full_size * 100, 2),
            'n_trend_clusters': len(trend_clusters),
            'message': f'{len(trend_clusters)} clusters representing usual trend, covering {total_trend_rows} rows ({round(total_trend_rows / self.original_full_size * 100, 2)}%)'
        }


class DayWiseClusterAnalyzer:
    """
    STL-inspired Day-Wise Clustering Engine.

    Instead of fitting a single cluster model on the entire dataset, this
    analyzer fits **independent** HDBSCAN / DBSCAN models for each day of
    the week.  This means:

    * Monday's data is clustered *only* against other Monday data.
    * A point arriving on Wednesday will be compared against Wednesday's
      cluster model exclusively.

    Benefits
    --------
    * Captures day-specific behavioural patterns (e.g. weekday vs weekend).
    * Reduces false-positive rate for days with naturally different
      distributions.
    * Provides a second "layer" of anomaly detection on top of the global
      ClusterAnalyzer - a point that looks normal globally may still be
      anomalous within its own day.
    """

    DAYS = [
        'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]

    def __init__(self,
                 processed_data: pd.DataFrame,
                 original_data: pd.DataFrame,
                 day_labels: pd.Series,
                 original_full_size: int):
        """
        Parameters
        ----------
        processed_data : pd.DataFrame
            Scaled / encoded feature matrix (output of FeaturePreprocessor).
        original_data : pd.DataFrame
            Original (clean) DataFrame *with* the ``day_of_week`` column.
        day_labels : pd.Series
            Series of day-name strings aligned with ``processed_data.index``.
        original_full_size : int
            (Kept for API compatibility; no longer used internally.
            Per-day sizes are computed automatically.)
        """
        self.processed_data = processed_data.copy()
        self.original_data = original_data.copy()
        self.day_labels = day_labels

        # Per-day artefacts
        self.day_models: Dict[str, 'ClusterAnalyzer'] = {}
        self.day_cluster_info: Dict[str, dict] = {}
        self.day_anomaly_info: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fit(self,
            method: str = 'auto',
            min_cluster_size: Optional[int] = None,
            min_samples: Optional[int] = None,
            min_day_rows: int = 10,
            cluster_selection_epsilon: float = 0.0) -> Dict:
        """
        Fit a separate ClusterAnalyzer for every day that has enough rows.

        Parameters
        ----------
        method : str
            ``'auto'``, ``'hdbscan'``, or ``'dbscan'``.
        min_cluster_size / min_samples : int, optional
            Base values.  They are automatically scaled down for smaller
            per-day slices when left as ``None``.
        min_day_rows : int
            Minimum number of rows a day must have to be clustered
            (default 10).
        cluster_selection_epsilon : float
            Passed through to HDBSCAN.

        Returns
        -------
        dict
            Summary with per-day cluster counts, anomaly counts, etc.
        """
        for day in self.DAYS:
            mask = self.day_labels == day
            day_indices = self.day_labels[mask].index

            n_day = len(day_indices)
            if n_day < min_day_rows:
                # Not enough data for this day - skip silently
                continue

            day_processed = self.processed_data.loc[day_indices]
            day_original = self.original_data.loc[day_indices]

            # Scale down min_cluster_size for smaller slices
            if min_cluster_size is None:
                day_mcs = max(3, int(n_day * 0.05))
            else:
                day_mcs = min(min_cluster_size, n_day // 2)

            if min_samples is None:
                day_ms = max(2, day_mcs // 2)
            else:
                day_ms = min(min_samples, day_mcs - 1) if day_mcs > 2 else 2

            # Pass n_day so that percentage calculations and anomaly
            # thresholds are relative to THIS day's data, not the full CSV.
            clusterer = ClusterAnalyzer(
                day_processed, day_original, n_day
            )
            clusterer.fit(
                method=method,
                min_cluster_size=day_mcs,
                min_samples=day_ms,
                cluster_selection_epsilon=cluster_selection_epsilon
            )

            self.day_models[day] = clusterer
            self.day_cluster_info[day] = clusterer.cluster_info
            self.day_anomaly_info[day] = clusterer.anomaly_info

        return self.get_day_summary()

    def get_day_summary(self) -> Dict[str, Dict]:
        """Return a compact summary for every day that was modelled."""
        summary = {}
        for day, model in self.day_models.items():
            ci = model.cluster_info
            ai = model.anomaly_info
            summary[day] = {
                'n_clusters': ci.get('n_clusters', 0),
                'n_noise': ci.get('n_noise', 0),
                'silhouette': ci.get('silhouette_score'),
                'total_points': len(model.original_data),
                'n_anomalies': ai.get('total_anomalies', 0),
                'anomaly_pct': ai.get('total_percentage', 0.0),
            }
        return summary

    def get_all_anomalies(self) -> List[Dict]:
        """
        Collect anomaly detail dicts from every day, tagging each with
        its ``day_of_week``.
        """
        all_anomalies: List[Dict] = []
        for day, model in self.day_models.items():
            details = model.anomaly_info.get('anomaly_details', [])
            for d in details:
                all_anomalies.append({**d, 'day_of_week': day})
        return all_anomalies

    def get_model_for_day(self, day: str) -> Optional['ClusterAnalyzer']:
        """Return the ClusterAnalyzer trained for *day*, or ``None``."""
        return self.day_models.get(day)

    def get_days_with_models(self) -> List[str]:
        """Return the list of days that actually had enough data to model."""
        return list(self.day_models.keys())

    # ------------------------------------------------------------------
    # RETRAINING WITH USER FEEDBACK
    # ------------------------------------------------------------------

    def retrain_with_corrections(self,
                                  user_corrections: Dict[int, bool],
                                  total_dataset_size: int,
                                  min_day_rows: int = 10) -> Tuple[Dict, Dict]:
        """
        Retrain all day-wise cluster models with parameter adjustments
        derived from user corrections (false-positive / false-negative
        feedback loop).

        The algorithm:

        1.  Compute FP rate = (user-unmarked anomalies) / (total anomalies).
        2.  Compute FN rate = (user-flagged normals) / (total normals).
        3.  If FP rate > 10 %, increase ``min_cluster_size`` per day
            (tighter clusters  → fewer noise points).
        4.  If FN rate > 10 %, decrease ``min_cluster_size`` per day
            (looser clusters  → more noise points caught).
        5.  Re-fit every day model with the adjusted ``min_cluster_size``,
            then rebuild day_cluster_info / day_anomaly_info.

        Parameters
        ----------
        user_corrections : dict
            ``{dataframe_index: True / False}``  where *True* means the
            user says the row IS an anomaly, *False* means NOT an anomaly.
        total_dataset_size : int
            Total number of rows in the original dataset (used to compute
            the false-negative rate denominator).
        min_day_rows : int
            Minimum rows a day needs to be included in retraining.

        Returns
        -------
        tuple (updated_day_summary, retrain_log_entry)

        ``updated_day_summary``
            Same structure as :meth:`get_day_summary`.
        ``retrain_log_entry``
            Dict with keys ``fp_count``, ``fn_count``, ``fp_rate``,
            ``fn_rate``, ``old_anomaly_count``, ``new_anomaly_count``,
            ``param_changes``, ``mcs_multiplier``.
        """
        # -- 1. Gather all model-detected anomaly indices ----
        all_model_anomaly_indices: set = set()
        for model in self.day_models.values():
            for detail in model.anomaly_info.get('anomaly_details', []):
                all_model_anomaly_indices.add(detail['index'])

        # -- 2. Count FP and FN from user corrections ---
        fp_indices = [
            idx for idx in all_model_anomaly_indices
            if user_corrections.get(idx) is False
        ]
        fn_indices = [
            idx for idx, val in user_corrections.items()
            if val and idx not in all_model_anomaly_indices
        ]

        total_anomalies = len(all_model_anomaly_indices)
        total_normals = total_dataset_size - total_anomalies
        old_anomaly_count = total_anomalies

        fp_rate = len(fp_indices) / total_anomalies if total_anomalies > 0 else 0
        fn_rate = len(fn_indices) / total_normals if total_normals > 0 else 0

        # -- 3. Determine cluster-size multiplier ---
        mcs_multiplier = 1.0
        param_changes: List[str] = []

        if fp_rate > 0.20:
            mcs_multiplier = 1.25
            param_changes.append(
                f"min_cluster_size increased by 25% (FP rate {fp_rate*100:.0f}% > 20%)"
            )
        elif fp_rate > 0.10:
            mcs_multiplier = 1.12
            param_changes.append(
                f"min_cluster_size increased by 12% (FP rate {fp_rate*100:.0f}% > 10%)"
            )

        if fn_rate > 0.20:
            mcs_multiplier *= 0.75
            param_changes.append(
                f"min_cluster_size decreased by 25% (FN rate {fn_rate*100:.0f}% > 20%)"
            )
        elif fn_rate > 0.10:
            mcs_multiplier *= 0.88
            param_changes.append(
                f"min_cluster_size decreased by 12% (FN rate {fn_rate*100:.0f}% > 10%)"
            )

        # -- 4. Re-fit each day model with adjusted parameters ---
        for day, model in self.day_models.items():
            n_day = len(model.processed_data)

            # Base mcs = what was originally computed by fit()
            base_mcs = max(3, int(n_day * 0.05))

            if mcs_multiplier != 1.0:
                new_mcs = max(2, int(base_mcs * mcs_multiplier))
            else:
                new_mcs = base_mcs
            new_ms = max(2, new_mcs // 2)

            model.fit(
                method='auto',
                min_cluster_size=new_mcs,
                min_samples=new_ms,
            )
            self.day_cluster_info[day] = model.cluster_info
            self.day_anomaly_info[day] = model.anomaly_info

        updated_summary = self.get_day_summary()

        # Count new anomalies
        new_anomaly_count = sum(
            info['n_anomalies'] for info in updated_summary.values()
        )

        # -- 5. Build log entry ---
        log_entry = {
            'fp_count': len(fp_indices),
            'fn_count': len(fn_indices),
            'fp_rate': round(fp_rate * 100, 1),
            'fn_rate': round(fn_rate * 100, 1),
            'old_anomaly_count': old_anomaly_count,
            'new_anomaly_count': new_anomaly_count,
            'param_changes': param_changes,
            'mcs_multiplier': round(mcs_multiplier, 3),
        }

        return updated_summary, log_entry
