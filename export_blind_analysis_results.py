#!/usr/bin/env python3
"""
Export Phase III Blind Analysis Results to CSV
Saves comprehensive blind pattern discovery findings for future analysis

Exports:
1. Natural cluster assignments with characteristics
2. Temporal pattern distributions 
3. Archaeological zone precision metrics
4. News proximity correlations
5. Energy-magnitude relationships
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path


class BlindAnalysisExporter:
    """Exports blind pattern discovery results to structured CSV files."""
    
    def __init__(self, data_path, output_dir):
        """Initialize exporter with data path and output directory."""
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.df = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load raw data and add blind analysis features."""
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Add temporal features
            self.df['timestamp_dt'] = pd.to_datetime(self.df['timestamp_et'], errors='coerce')
            self.df['hour'] = self.df['timestamp_dt'].dt.hour
            self.df['day_of_week_num'] = self.df['timestamp_dt'].dt.dayofweek
            self.df['minute'] = self.df['timestamp_dt'].dt.minute
            self.df['day_name'] = self.df['timestamp_dt'].dt.day_name()
            
            # Add archaeological zone precision metrics
            self.df['exact_40_percent'] = (self.df['range_position'] == 0.4).astype(int)
            self.df['within_0_5_pct'] = (abs(self.df['range_position'] - 0.4) <= 0.005).astype(int)
            self.df['within_1_pct'] = (abs(self.df['range_position'] - 0.4) <= 0.01).astype(int)
            self.df['within_2_5_pct'] = (abs(self.df['range_position'] - 0.4) <= 0.025).astype(int)
            self.df['distance_from_40'] = abs(self.df['range_position'] - 0.4)
            
            # Add clustering features
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Session clustering (7 natural clusters discovered)
            cluster_features = ['range_position', 'energy_density', 'magnitude', 'hour']
            cluster_data = self.df[cluster_features].fillna(0)
            
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
            self.df['natural_cluster'] = kmeans.fit_predict(cluster_data_scaled)
            
            # Energy-magnitude clustering (3 clusters)
            em_data = self.df[['energy_density', 'magnitude']].fillna(0)
            em_scaler = StandardScaler()
            em_scaled = em_scaler.fit_transform(em_data)
            
            em_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            self.df['energy_magnitude_cluster'] = em_kmeans.fit_predict(em_scaled)
            
            # Peak period indicators
            self.df['is_peak_hour'] = self.df['hour'].isin([12, 9, 19]).astype(int)
            self.df['is_lunch_peak'] = (self.df['hour'] == 12).astype(int)
            self.df['is_morning_peak'] = (self.df['hour'] == 9).astype(int)
            self.df['is_asia_peak'] = (self.df['hour'] == 19).astype(int)
            
            # News impact indicators
            self.df['has_high_news'] = (self.df['news_bucket'] == 'high±120m').astype(int)
            self.df['is_quiet_period'] = (self.df['news_bucket'] == 'quiet').astype(int)
            
            print(f"✅ Prepared {len(self.df)} events with blind analysis features")
            return True
            
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return False
    
    def export_main_results(self):
        """Export main blind analysis results with all discovered patterns."""
        print("\n📊 Exporting Main Blind Analysis Results...")
        
        # Select key columns for export
        export_columns = [
            # Original data
            'timestamp', 'timestamp_et', 'range_position', 'energy_density', 
            'price_level', 'magnitude', 'session_id', 'session_type', 
            'trading_day', 'news_bucket', 'has_news', 'compound_type',
            'in_macro_window', 'macro_window_name',
            
            # Temporal features
            'hour', 'day_of_week_num', 'day_name', 'minute',
            
            # Archaeological zone precision
            'exact_40_percent', 'within_0_5_pct', 'within_1_pct', 
            'within_2_5_pct', 'distance_from_40',
            
            # Natural clustering
            'natural_cluster', 'energy_magnitude_cluster',
            
            # Peak period indicators
            'is_peak_hour', 'is_lunch_peak', 'is_morning_peak', 'is_asia_peak',
            
            # News impact
            'has_high_news', 'is_quiet_period'
        ]
        
        # Export main results
        export_df = self.df[export_columns].copy()
        output_path = self.output_dir / "blind_analysis_main_results.csv"
        export_df.to_csv(output_path, index=False)
        
        print(f"✅ Exported main results to: {output_path}")
        return output_path
    
    def export_cluster_summary(self):
        """Export natural cluster characteristics summary."""
        print("\n📊 Exporting Cluster Summary...")
        
        cluster_summary = []
        
        # Natural session clusters (7 discovered)
        for cluster_id in range(7):
            cluster_data = self.df[self.df['natural_cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate cluster characteristics
            summary_row = {
                'cluster_id': cluster_id,
                'cluster_type': 'natural_session',
                'event_count': len(cluster_data),
                'avg_range_position': cluster_data['range_position'].mean(),
                'std_range_position': cluster_data['range_position'].std(),
                'avg_energy_density': cluster_data['energy_density'].mean(),
                'avg_magnitude': cluster_data['magnitude'].mean(),
                'std_magnitude': cluster_data['magnitude'].std(),
                'peak_hour': cluster_data['hour'].mode().iloc[0] if len(cluster_data['hour'].mode()) > 0 else None,
                'dominant_session': cluster_data['session_type'].mode().iloc[0] if len(cluster_data['session_type'].mode()) > 0 else None,
                'exact_40_pct_count': cluster_data['exact_40_percent'].sum(),
                'exact_40_pct_rate': cluster_data['exact_40_percent'].mean(),
                'high_news_count': cluster_data['has_high_news'].sum(),
                'quiet_period_count': cluster_data['is_quiet_period'].sum(),
                'macro_window_count': cluster_data['in_macro_window'].sum() if 'in_macro_window' in cluster_data else 0
            }
            
            cluster_summary.append(summary_row)
        
        # Energy-magnitude clusters (3 discovered)  
        for cluster_id in range(3):
            cluster_data = self.df[self.df['energy_magnitude_cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            summary_row = {
                'cluster_id': cluster_id,
                'cluster_type': 'energy_magnitude',
                'event_count': len(cluster_data),
                'avg_range_position': cluster_data['range_position'].mean(),
                'std_range_position': cluster_data['range_position'].std(),
                'avg_energy_density': cluster_data['energy_density'].mean(),
                'avg_magnitude': cluster_data['magnitude'].mean(),
                'std_magnitude': cluster_data['magnitude'].std(),
                'peak_hour': cluster_data['hour'].mode().iloc[0] if len(cluster_data['hour'].mode()) > 0 else None,
                'dominant_session': cluster_data['session_type'].mode().iloc[0] if len(cluster_data['session_type'].mode()) > 0 else None,
                'exact_40_pct_count': cluster_data['exact_40_percent'].sum(),
                'exact_40_pct_rate': cluster_data['exact_40_percent'].mean(),
                'high_news_count': cluster_data['has_high_news'].sum(),
                'quiet_period_count': cluster_data['is_quiet_period'].sum(),
                'macro_window_count': cluster_data['in_macro_window'].sum() if 'in_macro_window' in cluster_data else 0
            }
            
            cluster_summary.append(summary_row)
        
        # Export cluster summary
        cluster_df = pd.DataFrame(cluster_summary)
        output_path = self.output_dir / "blind_analysis_cluster_summary.csv"
        cluster_df.to_csv(output_path, index=False)
        
        print(f"✅ Exported cluster summary to: {output_path}")
        return output_path
    
    def export_temporal_patterns(self):
        """Export temporal distribution patterns."""
        print("\n📊 Exporting Temporal Patterns...")
        
        temporal_patterns = []
        
        # Hourly distribution
        hourly_counts = self.df['hour'].value_counts().sort_index()
        for hour, count in hourly_counts.items():
            temporal_patterns.append({
                'pattern_type': 'hourly',
                'time_value': int(hour),
                'time_label': f"{int(hour):02d}:00 ET",
                'event_count': count,
                'percentage': count / len(self.df) * 100,
                'exact_40_count': self.df[self.df['hour'] == hour]['exact_40_percent'].sum(),
                'high_news_count': self.df[self.df['hour'] == hour]['has_high_news'].sum(),
                'avg_magnitude': self.df[self.df['hour'] == hour]['magnitude'].mean()
            })
        
        # Day of week distribution
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = self.df['day_of_week_num'].value_counts().sort_index()
        
        for day_num, count in daily_counts.items():
            day_name = day_names[int(day_num)] if int(day_num) < 7 else 'Unknown'
            temporal_patterns.append({
                'pattern_type': 'daily',
                'time_value': int(day_num),
                'time_label': day_name,
                'event_count': count,
                'percentage': count / len(self.df) * 100,
                'exact_40_count': self.df[self.df['day_of_week_num'] == day_num]['exact_40_percent'].sum(),
                'high_news_count': self.df[self.df['day_of_week_num'] == day_num]['has_high_news'].sum(),
                'avg_magnitude': self.df[self.df['day_of_week_num'] == day_num]['magnitude'].mean()
            })
        
        # Minute clustering patterns  
        minute_counts = self.df['minute'].value_counts().head(10)
        for minute, count in minute_counts.items():
            temporal_patterns.append({
                'pattern_type': 'minute',
                'time_value': int(minute),
                'time_label': f":{int(minute):02d}",
                'event_count': count,
                'percentage': count / len(self.df) * 100,
                'exact_40_count': self.df[self.df['minute'] == minute]['exact_40_percent'].sum(),
                'high_news_count': self.df[self.df['minute'] == minute]['has_high_news'].sum(),
                'avg_magnitude': self.df[self.df['minute'] == minute]['magnitude'].mean()
            })
        
        # Export temporal patterns
        temporal_df = pd.DataFrame(temporal_patterns)
        output_path = self.output_dir / "blind_analysis_temporal_patterns.csv"
        temporal_df.to_csv(output_path, index=False)
        
        print(f"✅ Exported temporal patterns to: {output_path}")
        return output_path
    
    def export_precision_metrics(self):
        """Export archaeological zone precision metrics."""
        print("\n📊 Exporting Precision Metrics...")
        
        precision_data = []
        
        # Overall precision metrics
        total_events = len(self.df)
        
        precision_bands = [
            ('exact_40', 0.0, 'exactly_40_percent'),
            ('within_0_5_pct', 0.005, 'within_half_percent'), 
            ('within_1_pct', 0.01, 'within_one_percent'),
            ('within_2_5_pct', 0.025, 'within_two_five_percent')
        ]
        
        for band_name, tolerance, description in precision_bands:
            if tolerance == 0.0:
                count = (self.df['range_position'] == 0.4).sum()
            else:
                count = (abs(self.df['range_position'] - 0.4) <= tolerance).sum()
            
            precision_data.append({
                'precision_band': band_name,
                'tolerance': tolerance,
                'description': description,
                'event_count': count,
                'percentage': count / total_events * 100,
                'cumulative_percentage': count / total_events * 100  # Will recalculate cumulative
            })
        
        # Session-specific precision
        for session in self.df['session_type'].unique():
            if pd.isna(session):
                continue
                
            session_data = self.df[self.df['session_type'] == session]
            exact_40 = (session_data['range_position'] == 0.4).sum()
            
            precision_data.append({
                'precision_band': f'session_{session.lower()}',
                'tolerance': 0.0,
                'description': f'{session}_session_exact_40_percent',
                'event_count': exact_40,
                'percentage': exact_40 / len(session_data) * 100 if len(session_data) > 0 else 0,
                'cumulative_percentage': exact_40 / total_events * 100
            })
        
        # Export precision metrics
        precision_df = pd.DataFrame(precision_data)
        output_path = self.output_dir / "blind_analysis_precision_metrics.csv"
        precision_df.to_csv(output_path, index=False)
        
        print(f"✅ Exported precision metrics to: {output_path}")
        return output_path
    
    def export_summary_statistics(self):
        """Export summary statistics of blind analysis discoveries."""
        print("\n📊 Exporting Summary Statistics...")
        
        summary_stats = {
            # Data overview
            'total_events': len(self.df),
            'date_range_start': self.df['timestamp_dt'].min().strftime('%Y-%m-%d'),
            'date_range_end': self.df['timestamp_dt'].max().strftime('%Y-%m-%d'),
            'unique_trading_days': self.df['trading_day'].nunique() if 'trading_day' in self.df else 0,
            
            # Archaeological zone precision
            'exact_40_percent_count': self.df['exact_40_percent'].sum(),
            'exact_40_percent_rate': self.df['exact_40_percent'].mean(),
            'within_2_5_pct_count': self.df['within_2_5_pct'].sum(),
            'within_2_5_pct_rate': self.df['within_2_5_pct'].mean(),
            'avg_distance_from_40': self.df['distance_from_40'].mean(),
            'max_distance_from_40': self.df['distance_from_40'].max(),
            
            # Temporal concentrations
            'peak_hour': int(self.df['hour'].mode().iloc[0]),
            'peak_hour_count': self.df['hour'].value_counts().iloc[0],
            'peak_hour_percentage': self.df['hour'].value_counts().iloc[0] / len(self.df) * 100,
            'peak_day': self.df['day_name'].mode().iloc[0] if len(self.df['day_name'].mode()) > 0 else 'Unknown',
            'peak_day_count': self.df['day_name'].value_counts().iloc[0] if len(self.df['day_name'].value_counts()) > 0 else 0,
            
            # Session distributions
            'dominant_session': self.df['session_type'].mode().iloc[0] if len(self.df['session_type'].mode()) > 0 else 'Unknown',
            'dominant_session_count': self.df['session_type'].value_counts().iloc[0] if len(self.df['session_type'].value_counts()) > 0 else 0,
            'unique_sessions': self.df['session_type'].nunique(),
            
            # News impact
            'quiet_period_count': self.df['is_quiet_period'].sum(),
            'quiet_period_rate': self.df['is_quiet_period'].mean(),
            'high_news_count': self.df['has_high_news'].sum(),
            'high_news_rate': self.df['has_high_news'].mean(),
            
            # Natural clustering
            'natural_clusters_discovered': self.df['natural_cluster'].nunique(),
            'energy_magnitude_clusters': self.df['energy_magnitude_cluster'].nunique(),
            
            # Magnitude and energy
            'avg_magnitude': self.df['magnitude'].mean(),
            'std_magnitude': self.df['magnitude'].std(),
            'max_magnitude': self.df['magnitude'].max(),
            'avg_energy_density': self.df['energy_density'].mean(),
            
            # Macro window penetration (if available)
            'macro_window_events': self.df['in_macro_window'].sum() if 'in_macro_window' in self.df else 0,
            'macro_window_rate': self.df['in_macro_window'].mean() if 'in_macro_window' in self.df else 0,
        }
        
        # Convert to DataFrame for export
        summary_df = pd.DataFrame([summary_stats])
        output_path = self.output_dir / "blind_analysis_summary_statistics.csv"
        summary_df.to_csv(output_path, index=False)
        
        print(f"✅ Exported summary statistics to: {output_path}")
        return output_path
    
    def export_all_results(self):
        """Export all blind analysis results to CSV files."""
        print("\n🚀 EXPORTING ALL BLIND ANALYSIS RESULTS")
        print("=" * 60)
        
        if not self.load_and_prepare_data():
            return False
        
        # Export all result sets
        main_results = self.export_main_results()
        cluster_summary = self.export_cluster_summary()
        temporal_patterns = self.export_temporal_patterns()
        precision_metrics = self.export_precision_metrics()
        summary_stats = self.export_summary_statistics()
        
        # Summary of exported files
        print(f"\n✅ EXPORT COMPLETE - 5 CSV FILES CREATED:")
        print(f"   1. Main Results: {main_results.name}")
        print(f"   2. Cluster Summary: {cluster_summary.name}")
        print(f"   3. Temporal Patterns: {temporal_patterns.name}")
        print(f"   4. Precision Metrics: {precision_metrics.name}")
        print(f"   5. Summary Statistics: {summary_stats.name}")
        
        print(f"\n📂 All files saved to: {self.output_dir}")
        
        return True


def main():
    """Main execution for blind analysis export."""
    
    # Data paths
    data_path = "/Users/jack/IRONFORGE/runs/RUN_20250824_182221_NEWSCLUST_3P/artifacts/macro_window_confluence_analysis.csv"
    output_dir = "/Users/jack/IRONFORGE/runs/RUN_20250824_182221_NEWSCLUST_3P/artifacts/blind_analysis_exports"
    
    # Initialize exporter
    exporter = BlindAnalysisExporter(data_path, output_dir)
    
    # Export all results
    success = exporter.export_all_results()
    
    if success:
        print("\n🎉 Blind Analysis Export Complete!")
        return 0
    else:
        print("\n❌ Export failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())