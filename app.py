import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
import hashlib
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from data_profiler import DataProfiler
from feature_processor import FeaturePreprocessor
from cluster_analyzer import DayWiseClusterAnalyzer
from trend_analyzer import DayWiseTrendAnalyzer

DAY_ORDER = [
    'Monday', 'Tuesday', 'Wednesday',
    'Thursday', 'Friday', 'Saturday', 'Sunday'
]

MAX_DISPLAY_ROWS = 500
PAGE_SIZE_CLEAN = 100

st.set_page_config(page_title="Anomaly Detector", page_icon="🚨", layout="wide")


# ======================================================================
# UTILITY HELPERS
# ======================================================================

def _apply_filters(all_anomalies, severity_filter, source_filter, day_filter):
    """Apply severity, source, and day filters to the anomaly list."""
    result = []
    for a in all_anomalies:
        if severity_filter != 'All' and a['severity'] != severity_filter.lower():
            continue
        if source_filter == 'Missing' and a.get('source') != 'missing':
            continue
        if source_filter == 'Day Wise' and a.get('source') != 'day_wise':
            continue
        if source_filter == 'User Flagged' and a.get('source') != 'user_flagged':
            continue
        if day_filter and day_filter not in ('All Days', None):
            if a.get('day_of_week') != day_filter:
                continue
        result.append(a)
    return result


def _sort_days(day_list):
    """Sort day names in Mon->Sun order."""
    return sorted(
        day_list,
        key=lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99
    )


def _file_hash(uploaded_file) -> str:
    """Return an MD5 hash of the uploaded file bytes for change detection."""
    return hashlib.md5(uploaded_file.getbuffer().tobytes()).hexdigest()


# ======================================================================
# SESSION-STATE INITIALISATION
# ======================================================================

def _init_session():
    """Ensure all session-state keys exist with sensible defaults."""
    defaults = {
        'file_hash': None,
        'user_corrections': {},
        'analysis_cache': None,
        'retrain_log': [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ======================================================================
# FULL ANALYSIS PIPELINE  (cached per file)
# ======================================================================

def _run_full_analysis(uploaded_file) -> Optional[Dict]:
    """
    Execute Phases 1-3 of the pipeline and return a cacheable result dict.
    Returns None on error.
    """
    try:
        with open("temp_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = DataLoader()
        df = loader.load_csv_auto("temp_data.csv")

        # ---- PHASE 1: PRE-CLUSTERING DATA CLEANUP ----
        empty_cols = df.columns[df.isnull().all()].tolist()
        id_cols_to_ignore = ['test_case_id', 'RUN_ID', 'Batch_ID'] + \
            [c for c in df.columns if 'id' in c.lower() and c not in empty_cols]
        cols_to_drop = list(set(empty_cols + id_cols_to_ignore))
        valid_cols = [c for c in df.columns if c not in cols_to_drop]

        has_missing_mask = df[valid_cols].isnull().any(axis=1)
        missing_rows_df = df[has_missing_mask].copy()
        clean_rows_df = df[~has_missing_mask].copy()

        missing_anomalies = []
        if not missing_rows_df.empty:
            missing_rows_df['_missing_pattern'] = missing_rows_df[valid_cols].apply(
                lambda x: tuple(x[x.isnull()].index), axis=1
            )
            pattern_counts = missing_rows_df['_missing_pattern'].value_counts()
            total_dataset_size = len(df)

            for pattern, count in pattern_counts.items():
                freq_ratio = count / total_dataset_size
                if freq_ratio > 0.15:
                    severity = 'low'
                    reason = f"Systematic missing pattern ({freq_ratio*100:.1f}% of data)"
                elif freq_ratio > 0.05:
                    severity = 'medium'
                    reason = f"Noticeable missing pattern ({freq_ratio*100:.1f}% of data)"
                else:
                    severity = 'high'
                    reason = f"Rare missing value error ({freq_ratio*100:.1f}% of data)"

                row_indices = missing_rows_df[
                    missing_rows_df['_missing_pattern'] == pattern
                ].index.tolist()

                for idx in row_indices:
                    missing_anomalies.append({
                        'index': idx,
                        'severity': severity,
                        'reason': reason,
                        'error_columns': list(pattern),
                        'source': 'missing'
                    })

        # ---- PHASE 2: SYNTHETIC DATETIME LABELLING ----
        clean_valid_df = (
            clean_rows_df[valid_cols].copy()
            if not clean_rows_df.empty
            else pd.DataFrame(columns=valid_cols)
        )
        if len(clean_valid_df) > 0:
            profiler_label = DataProfiler(clean_valid_df)
            profiler_label.add_datetime_labels(reference_date=None, start_hour=9, end_hour=17)
            clean_valid_df = profiler_label.df

        # ---- PHASE 3: DAY-WISE CLUSTERING & TREND ANALYSIS ----
        day_wise_anomalies = []
        day_summary = {}
        day_cluster_analyzer = None
        day_trend_analyzer = None
        preprocessor = None
        processed_data = None
        day_labels = None
        profiler = None

        if len(clean_valid_df) > 10:
            profiler = DataProfiler(clean_valid_df)
            profile = profiler.generate_profile()

            preprocessor = FeaturePreprocessor(clean_valid_df, profiler)
            processed_data = preprocessor.preprocess_for_clustering(
                handle_missing='median', scaling='robust', encode_categorical='label'
            )

            day_labels = clean_valid_df['day_of_week']

            day_cluster_analyzer = DayWiseClusterAnalyzer(
                processed_data, clean_valid_df, day_labels, len(df)
            )
            day_summary = day_cluster_analyzer.fit(
                method='auto', min_cluster_size=None, min_samples=None, min_day_rows=10
            )

            if day_cluster_analyzer.day_models:
                day_trend_analyzer = DayWiseTrendAnalyzer(day_cluster_analyzer, preprocessor)
                day_trend_analyzer.analyze_trends()

                day_raw_anomalies = day_cluster_analyzer.get_all_anomalies()
                if day_raw_anomalies:
                    sev_map = day_trend_analyzer.calculate_deviation_severity(day_raw_anomalies)
                    err_map = day_trend_analyzer.get_error_columns_bulk(day_raw_anomalies)

                    for detail in day_raw_anomalies:
                        idx = detail['index']
                        errors = err_map.get(idx, [])
                        if not errors:
                            errors = [
                                f"Multi-dimensional deviation "
                                f"(Cluster {detail['cluster_label']}, "
                                f"size {detail['cluster_size']})"
                            ]
                        day_name = detail.get('day_of_week', 'Unknown')
                        day_wise_anomalies.append({
                            'index': idx,
                            'severity': sev_map.get(idx, 'medium'),
                            'reason': f"Day-specific anomaly ({day_name}: Cluster size: {detail['cluster_size']})",
                            'error_columns': errors,
                            'source': 'day_wise',
                            'day_of_week': day_name,
                        })

        return {
            'df': df,
            'valid_cols': valid_cols,
            'missing_anomalies': missing_anomalies,
            'day_wise_anomalies_raw': day_wise_anomalies,
            'all_anomalies_raw': missing_anomalies + day_wise_anomalies,
            'day_summary': day_summary,
            'clean_valid_df': clean_valid_df,
            'day_cluster_analyzer': day_cluster_analyzer,
            'day_trend_analyzer': day_trend_analyzer,
            'preprocessor': preprocessor,
            'processed_data': processed_data,
            'day_labels': day_labels,
            'profiler': profiler,
        }

    except Exception as e:
        st.error(f"Analysis error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ======================================================================
# HELPERS TO REBUILD ANOMALY LISTS AFTER RETRAIN
# ======================================================================

def _rebuild_day_wise_anomalies(day_cluster, day_trend) -> List[Dict]:
    """Re-read anomalies from a retrained DayWiseClusterAnalyzer + TrendAnalyzer."""
    anomalies = []
    if not day_cluster.day_models:
        return anomalies
    raw = day_cluster.get_all_anomalies()
    if not raw:
        return anomalies
    sev = day_trend.calculate_deviation_severity(raw)
    err = day_trend.get_error_columns_bulk(raw)
    for d in raw:
        idx = d['index']
        errors = err.get(idx, [])
        if not errors:
            errors = [
                f"Multi-dimensional deviation "
                f"(Cluster {d['cluster_label']}, size {d['cluster_size']})"
            ]
        day_name = d.get('day_of_week', 'Unknown')
        anomalies.append({
            'index': idx,
            'severity': sev.get(idx, 'medium'),
            'reason': f"Day-specific anomaly ({day_name}: Cluster size: {d['cluster_size']})",
            'error_columns': errors,
            'source': 'day_wise',
            'day_of_week': day_name,
        })
    return anomalies


# ======================================================================
# FRAGMENT: ALL ANOMALIES TABLE  (isolated rerun)
# ======================================================================

@st.fragment
def _render_anomaly_table():
    """
    Interactive anomaly table with per-row checkbox toggle.
    Only this fragment reruns when a checkbox changes — the rest of
    the page stays put.
    """
    cache = st.session_state.analysis_cache
    if cache is None:
        return

    raw_anomalies = cache['all_anomalies_raw']
    raw_idx_set = {a['index'] for a in raw_anomalies}
    user_corr = st.session_state.user_corrections

    # Rebuild effective list inside the fragment
    effective = _get_effective_anomalies(raw_anomalies, user_corr)
    user_flagged = _get_user_flagged_normals(raw_idx_set, user_corr)
    for idx in user_flagged:
        day_name = ''
        if cache.get('day_labels') is not None and idx in cache['day_labels'].index:
            day_name = cache['day_labels'].iloc[
                cache['day_labels'].index.get_loc(idx)
            ] if hasattr(cache['day_labels'], 'iloc') else str(cache['day_labels'].get(idx, ''))
        effective.append({
            'index': idx,
            'severity': 'medium',
            'reason': 'User-flagged anomaly (missed by model)',
            'error_columns': ['User correction'],
            'source': 'user_flagged',
            'day_of_week': day_name,
        })

    df = cache['df']
    final_indices = [a['index'] for a in effective]
    anomaly_df = df.loc[final_indices].copy()

    anomaly_df.insert(0, 'Severity', [a['severity'].capitalize() for a in effective])
    anomaly_df.insert(1, 'Source', [
        a.get('source', 'day_wise').replace('_', ' ').title() for a in effective
    ])
    anomaly_df.insert(2, 'Root Cause', [a['reason'] for a in effective])
    anomaly_df.insert(3, 'Error Columns', [
        ", ".join(a['error_columns']) for a in effective
    ])

    day_map = {}
    if cache.get('clean_valid_df') is not None and 'synthetic_datetime' in cache['clean_valid_df'].columns:
        day_map = cache['clean_valid_df']['day_of_week'].to_dict()
        anomaly_df.insert(4, 'Day', [day_map.get(idx, '') for idx in final_indices])

    # Store for use by other tabs
    st.session_state._anomaly_df = anomaly_df
    st.session_state._effective_anomalies = effective
    st.session_state._day_map = day_map

    # ---- Filters ----
    fc1, fc2 = st.columns([3, 1])
    sev_filter = fc1.radio(
        "Filter by Severity:",
        ['All', 'High', 'Medium', 'Low'],
        horizontal=True, key='frag_all_sev'
    )
    src_filter = fc2.selectbox(
        "Filter by Source:",
        ['All', 'Missing', 'Day Wise', 'User Flagged'],
        key='frag_all_src'
    )

    filtered = _apply_filters(effective, sev_filter, src_filter, None)
    f_indices = [a['index'] for a in filtered]
    display = anomaly_df.loc[f_indices].copy()

    if len(display) > MAX_DISPLAY_ROWS:
        st.warning(f"Showing top {MAX_DISPLAY_ROWS} of {len(display)} rows.")
        display = display.head(MAX_DISPLAY_ROWS)

    if len(display) > 0:
        st.subheader(f"Anomaly Table ({len(filtered)} rows)")
        display.insert(0, 'Row #', display.index)

        toggle_vals = [
            user_corr.get(idx, True) for idx in display.index
        ]
        display['Marked Anomaly'] = toggle_vals

        edited = st.data_editor(
            display,
            column_config={
                "Marked Anomaly": st.column_config.CheckboxColumn(
                    "Is Anomaly",
                    help="Uncheck to mark this row as NOT an anomaly",
                )
            },
            disabled=[c for c in display.columns if c != 'Marked Anomaly'],
            hide_index=True,
            use_container_width=True,
            height=600,
            key="frag_anomaly_editor",
        )

        for row_idx, row in edited.iterrows():
            df_idx = int(row['Row #'])
            new_val = bool(row['Marked Anomaly'])
            orig_is_anomaly = df_idx in raw_idx_set
            old_val = user_corr.get(df_idx, orig_is_anomaly)
            if new_val != old_val:
                st.session_state.user_corrections[df_idx] = new_val
    else:
        st.info("No anomalies match the selected filters.")


# ======================================================================
# FRAGMENT: ANOMALIES BY DAY TABLE
# ======================================================================

@st.fragment
def _render_by_day_table():
    """
    Day-filtered anomaly view with per-row toggle.
    Only this fragment reruns on interaction.
    """
    cache = st.session_state.analysis_cache
    if cache is None:
        return

    day_map = st.session_state.get('_day_map', {})
    available_days = _sort_days(list(set(day_map.values()))) if day_map else []

    raw_anomalies = cache['all_anomalies_raw']
    raw_idx_set = {a['index'] for a in raw_anomalies}
    user_corr = st.session_state.user_corrections

    effective = _get_effective_anomalies(raw_anomalies, user_corr)
    user_flagged = _get_user_flagged_normals(raw_idx_set, user_corr)
    for idx in user_flagged:
        day_name = ''
        if cache.get('day_labels') is not None and idx in cache['day_labels'].index:
            day_name = str(cache['day_labels'].get(idx, ''))
        effective.append({
            'index': idx, 'severity': 'medium',
            'reason': 'User-flagged anomaly (missed by model)',
            'error_columns': ['User correction'],
            'source': 'user_flagged', 'day_of_week': day_name,
        })

    anomaly_df = st.session_state.get('_anomaly_df')

    if not available_days:
        st.warning("Not enough clean data for day-wise analysis.")
        return

    dc1, dc2 = st.columns([3, 1])
    selected_day = dc1.selectbox(
        "Select Day:", ['All Days'] + available_days, key='frag_day_sel'
    )
    day_sev_filter = dc2.selectbox(
        "Severity:", ['All', 'High', 'Medium', 'Low'], key='frag_day_sev'
    )

    day_filtered = _apply_filters(effective, day_sev_filter, 'Day Wise', selected_day)

    if selected_day == 'All Days':
        st.markdown("**Per-day anomaly breakdown:**")
        per_day = {}
        for a in day_filtered:
            d = a.get('day_of_week', 'Unassigned')
            per_day[d] = per_day.get(d, 0) + 1

        if per_day:
            rows = []
            for dn in _sort_days(per_day.keys()):
                cnt = per_day[dn]
                h = sum(1 for a in day_filtered if a.get('day_of_week') == dn and a['severity'] == 'high')
                m = sum(1 for a in day_filtered if a.get('day_of_week') == dn and a['severity'] == 'medium')
                l = sum(1 for a in day_filtered if a.get('day_of_week') == dn and a['severity'] == 'low')
                rows.append({'Day': dn, 'Total': cnt, 'High': h, 'Medium': m, 'Low': l})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if anomaly_df is not None:
            day_indices = [a['index'] for a in day_filtered]
            day_display = anomaly_df.loc[day_indices]
            if len(day_display) > 0:
                st.subheader(f"All Day-Wise Anomalies ({len(day_display)} rows)")
                st.dataframe(day_display.head(MAX_DISPLAY_ROWS), use_container_width=True, height=500)

    else:
        day_sev = {'high': 0, 'medium': 0, 'low': 0}
        for a in day_filtered:
            day_sev[a['severity']] += 1

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{selected_day} Anomalies", len(day_filtered))
        m2.metric("High", day_sev['high'])
        m3.metric("Medium", day_sev['medium'])
        m4.metric("Low", day_sev['low'])

        day_summary = cache.get('day_summary', {})
        if selected_day in day_summary:
            ds = day_summary[selected_day]
            info = (
                f"**Clusters:** {ds['n_clusters']}  |  "
                f"**Noise:** {ds['n_noise']}  |  "
                f"**Total Points:** {ds['total_points']}  |  "
                f"**Anomaly Rate:** {ds['anomaly_pct']:.1f}%"
            )
            if ds.get('silhouette') is not None:
                info += f"  |  **Silhouette:** {ds['silhouette']:.3f}"
            st.caption(info)

        if anomaly_df is not None:
            day_indices = [a['index'] for a in day_filtered]
            day_display = anomaly_df.loc[day_indices]
            if len(day_display) > 0:
                st.subheader(f"{selected_day} - {len(day_display)} anomalies")
                dd = day_display.head(MAX_DISPLAY_ROWS).copy()
                dd.insert(0, 'Row #', dd.index)
                dd['Marked Anomaly'] = [user_corr.get(idx, True) for idx in dd.index]

                edited = st.data_editor(
                    dd,
                    column_config={
                        "Marked Anomaly": st.column_config.CheckboxColumn(
                            "Is Anomaly", help="Uncheck to mark as NOT an anomaly"
                        )
                    },
                    disabled=[c for c in dd.columns if c != 'Marked Anomaly'],
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                    key="frag_day_editor",
                )

                for row_idx, row in edited.iterrows():
                    df_idx = int(row['Row #'])
                    new_val = bool(row['Marked Anomaly'])
                    old_val = user_corr.get(df_idx, df_idx in raw_idx_set)
                    if new_val != old_val:
                        st.session_state.user_corrections[df_idx] = new_val
            else:
                st.info(f"No anomalies found for {selected_day}.")
        else:
            st.info(f"No anomalies found for {selected_day}.")


# ======================================================================
# FRAGMENT: NON-ANOMALOUS DATA TABLE
# ======================================================================

@st.fragment
def _render_clean_table():
    """
    Non-anomalous data with per-row "Flag as Anomaly" checkbox.
    Only this fragment reruns on interaction.
    """
    cache = st.session_state.analysis_cache
    if cache is None:
        return

    df = cache['df']
    effective = st.session_state.get('_effective_anomalies', [])
    raw_idx_set = {a['index'] for a in cache['all_anomalies_raw']}
    user_corr = st.session_state.user_corrections
    day_map = st.session_state.get('_day_map', {})

    if not effective:
        st.info("No anomalies detected. All data shown as non-anomalous.")
        return

    eff_idx_set = {a['index'] for a in effective}
    clean_indices = sorted(set(df.index) - eff_idx_set)

    st.subheader(f"Non-Anomalous Data ({len(clean_indices)} rows)")

    if len(clean_indices) == 0:
        st.info("All rows are classified as anomalies.")
        return

    total_pages = max(1, (len(clean_indices) + PAGE_SIZE_CLEAN - 1) // PAGE_SIZE_CLEAN)
    pc1, pc2 = st.columns([1, 4])
    page_num = pc1.number_input("Page", min_value=1, max_value=total_pages, value=1, key='frag_clean_page')
    pc2.caption(f"Page {page_num} of {total_pages} ({len(clean_indices)} total rows)")

    start = (page_num - 1) * PAGE_SIZE_CLEAN
    end = min(start + PAGE_SIZE_CLEAN, len(clean_indices))
    page_indices = clean_indices[start:end]

    clean_display = df.loc[page_indices].copy()
    clean_display.insert(0, 'Row #', page_indices)
    if day_map:
        clean_display.insert(1, 'Day', [day_map.get(idx, '') for idx in page_indices])

    clean_display['Flag as Anomaly'] = [
        user_corr.get(idx, False) for idx in page_indices
    ]

    edited = st.data_editor(
        clean_display,
        column_config={
            "Flag as Anomaly": st.column_config.CheckboxColumn(
                "Flag as Anomaly",
                help="Check to mark this row as an anomaly (model miss)",
            )
        },
        disabled=[c for c in clean_display.columns if c != 'Flag as Anomaly'],
        hide_index=True,
        use_container_width=True,
        height=500,
        key="frag_clean_editor",
    )

    for row_idx, row in edited.iterrows():
        df_idx = int(row['Row #'])
        new_val = bool(row['Flag as Anomaly'])
        old_val = user_corr.get(df_idx, df_idx in raw_idx_set)
        if new_val != old_val:
            st.session_state.user_corrections[df_idx] = new_val


# ======================================================================
# SHARED HELPERS (used by both fragments and main)
# ======================================================================

def _get_effective_anomalies(raw_anomalies, user_corrections):
    effective = []
    for a in raw_anomalies:
        idx = a['index']
        if idx in user_corrections and not user_corrections[idx]:
            continue
        effective.append(a)
    return effective


def _get_user_flagged_normals(raw_anomaly_indices, user_corrections):
    return [
        idx for idx, val in user_corrections.items()
        if val and idx not in raw_anomaly_indices
    ]


# ======================================================================
# MAIN
# ======================================================================

def main():
    _init_session()

    st.title("🚨 Automated Data Anomaly Detector")
    st.caption("Upload a CSV. Missing values are isolated by frequency, then day-wise ML clustering finds structural anomalies per weekday.")
    st.caption("Data is labelled across 7 days (9 AM - 5 PM) so each weekday gets its own independent cluster model and trend profile.")
    st.caption("Toggle any row's anomaly status. Corrections feed back into the model to improve future detections.")

    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

    if not uploaded_file:
        return

    # ---- File-change detection ----
    fh = _file_hash(uploaded_file)
    if st.session_state.file_hash != fh:
        st.session_state.file_hash = fh
        st.session_state.user_corrections = {}
        st.session_state.analysis_cache = None
        st.session_state.retrain_log = []

    # ---- Run / retrieve cached analysis ----
    if st.session_state.analysis_cache is None:
        with st.spinner("Loading and analysing CSV..."):
            st.session_state.analysis_cache = _run_full_analysis(uploaded_file)

    cache = st.session_state.analysis_cache
    if cache is None:
        return

    df: pd.DataFrame = cache['df']
    raw_anomalies: List[Dict] = cache['all_anomalies_raw']
    raw_anomaly_idx_set = {a['index'] for a in raw_anomalies}
    user_corr: Dict[int, bool] = st.session_state.user_corrections

    # ---- Compute effective anomaly list ----
    effective_anomalies = _get_effective_anomalies(raw_anomalies, user_corr)
    user_flagged = _get_user_flagged_normals(raw_anomaly_idx_set, user_corr)
    for idx in user_flagged:
        day_name = ''
        if cache.get('day_labels') is not None and idx in cache['day_labels'].index:
            day_name = str(cache['day_labels'].get(idx, ''))
        effective_anomalies.append({
            'index': idx, 'severity': 'medium',
            'reason': 'User-flagged anomaly (missed by model)',
            'error_columns': ['User correction'],
            'source': 'user_flagged', 'day_of_week': day_name,
        })

    # ---- Top-level metrics ----
    severity_counts = {'high': 0, 'medium': 0, 'low': 0}
    for a in effective_anomalies:
        severity_counts[a['severity']] += 1

    corrections_count = len(user_corr)
    fp_count = sum(1 for a in raw_anomalies if user_corr.get(a['index']) is False)
    fn_count = len(user_flagged)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Anomalies", len(effective_anomalies))
    col2.metric("High", severity_counts['high'])
    col3.metric("Medium", severity_counts['medium'])
    col4.metric("Low", severity_counts['low'])
    col5.metric("User Corrections", corrections_count)
    col6.metric("Removed / Added", f"{fp_count} / {fn_count}")

    st.success(
        f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns. "
        f"{len(effective_anomalies)} anomalies after corrections."
    )

    # ---- Day-wise cluster summary expander ----
    day_summary = cache['day_summary']
    if day_summary:
        with st.expander("Day-Wise Cluster Summary", expanded=False):
            summary_df = pd.DataFrame([
                {
                    'Day': day,
                    'Total Points': info['total_points'],
                    'Clusters': info['n_clusters'],
                    'Noise Points': info['n_noise'],
                    'Anomalies': info['n_anomalies'],
                    'Anomaly %': f"{info['anomaly_pct']:.1f}%",
                    'Silhouette': (
                        f"{info['silhouette']:.3f}" if info.get('silhouette') else 'N/A'
                    ),
                }
                for day, info in day_summary.items()
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # =================================================================
    # TABS — each tab body is a fragment (no full-page rerun)
    # =================================================================
    tab_all, tab_by_day, tab_clean = st.tabs(
        ["All Anomalies", "Anomalies by Day", "Non-Anomalous Data"]
    )

    with tab_all:
        _render_anomaly_table()

    with tab_by_day:
        _render_by_day_table()

    with tab_clean:
        _render_clean_table()

    # =================================================================
    # CORRECTION PANEL  (NOT a fragment — uses st.rerun intentionally)
    # =================================================================
    st.markdown("---")

    with st.expander("Model Correction Panel", expanded=len(user_corr) > 0):
        st.caption(
            "Review your corrections below.  Click **Apply Corrections & Retrain Model** "
            "to adjust clustering parameters based on your feedback and re-run "
            "the analysis with improved settings."
        )

        day_map: Dict[int, str] = {}
        if cache.get('clean_valid_df') is not None and 'synthetic_datetime' in cache['clean_valid_df'].columns:
            day_map = cache['clean_valid_df']['day_of_week'].to_dict()

        if not user_corr:
            st.info(
                "No corrections yet.  Use the checkboxes in the tables above to "
                "toggle anomaly status, then return here to apply."
            )
        else:
            # Summary table of corrections
            corr_details = []
            for idx, is_anomaly in user_corr.items():
                orig = idx in raw_anomaly_idx_set
                if orig and not is_anomaly:
                    direction = "Removed (was anomaly)"
                elif not orig and is_anomaly:
                    direction = "Added (was normal)"
                else:
                    continue
                corr_details.append({
                    'Row': idx,
                    'Direction': direction,
                    'Day': day_map.get(idx, ''),
                })

            if corr_details:
                st.dataframe(pd.DataFrame(corr_details), use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Corrections", len(corr_details))
            c2.metric("False Positives Removed", fp_count)
            c3.metric("False Negatives Added", fn_count)

            if st.button("Apply Corrections & Retrain Model", type="primary", key="retrain_btn"):
                with st.spinner("Retraining model with adjusted parameters..."):
                    day_cluster = cache['day_cluster_analyzer']
                    preprocessor = cache['preprocessor']

                    if day_cluster is not None:
                        # Delegate to model layer
                        new_summary, log_entry = day_cluster.retrain_with_corrections(
                            user_corr,
                            total_dataset_size=len(df),
                        )

                        # Rebuild trend analysis on updated clusters
                        day_trend = DayWiseTrendAnalyzer(day_cluster, preprocessor)
                        day_trend.analyze_trends()

                        # Rebuild anomaly list from updated models
                        new_day_anomalies = _rebuild_day_wise_anomalies(day_cluster, day_trend)

                        # Update cache
                        cache['day_summary'] = new_summary
                        cache['day_cluster_analyzer'] = day_cluster
                        cache['day_trend_analyzer'] = day_trend
                        cache['day_wise_anomalies_raw'] = new_day_anomalies
                        cache['all_anomalies_raw'] = cache['missing_anomalies'] + new_day_anomalies
                        st.session_state.analysis_cache = cache

                        st.session_state.retrain_log.append(log_entry)

                        # Clear stale fragment keys so editors reset
                        for key in list(st.session_state.keys()):
                            if key.startswith('frag_'):
                                del st.session_state[key]

                        st.rerun()
                    else:
                        st.warning("No day-wise clustering model to retrain.")

        # Retrain history
        if st.session_state.retrain_log:
            st.markdown("**Retrain History:**")
            for i, log in enumerate(st.session_state.retrain_log, 1):
                with st.container():
                    st.markdown(f"**Retrain #{i}**")
                    cols = st.columns(4)
                    cols[0].metric("FP Removed", log['fp_count'])
                    cols[1].metric("FN Added", log['fn_count'])
                    cols[2].metric("Old Anomalies", log['old_anomaly_count'])
                    cols[3].metric("New Anomalies", log['new_anomaly_count'])
                    if log['param_changes']:
                        for change in log['param_changes']:
                            st.info(change)
                    else:
                        st.caption("No parameter changes (correction rates below threshold).")
                    st.caption(
                        f"FP rate: {log['fp_rate']}% | FN rate: {log['fn_rate']}% | "
                        f"Cluster-size multiplier: {log['mcs_multiplier']}"
                    )
                    st.markdown("---")

        if user_corr and st.button("Reset All Corrections", key="reset_corr"):
            st.session_state.user_corrections = {}
            for key in list(st.session_state.keys()):
                if key.startswith('frag_'):
                    del st.session_state[key]
            st.rerun()

    # ---- Download ----
    st.markdown("---")

    anomaly_df = st.session_state.get('_anomaly_df')
    if anomaly_df is not None:
        csv_export = anomaly_df.to_csv(index=False)
        st.download_button(
            label="Download Full Anomaly Report (CSV)",
            data=csv_export,
            file_name='anomalies_report.csv',
            mime='text/csv'
        )


if __name__ == "__main__":
    main()
