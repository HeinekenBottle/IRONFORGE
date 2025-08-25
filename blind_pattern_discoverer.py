#!/usr/bin/env python3
"""
Phase III Blind Pattern Discovery
Pure data exploration of 209 RD@40 events without PO3 assumptions

NO PRECONCEPTIONS - Let the data reveal its own patterns:
- Session clustering analysis (natural groupings)
- Temporal distribution patterns (hour/day cycles)
- News proximity correlations (impact relationships)
- Energy density vs magnitude relationships (volatility patterns)

Pure discovery approach - patterns emerge from data, not theory.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from collections import defaultdict, Counter
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import sys


class BlindPatternDiscoverer:
    """
    Discovers natural patterns in RD@40 events without theoretical assumptions.
    
    Pure data-driven approach:
    - No PO3 phase assumptions
    - No ICT macro window preconceptions
    - Let clustering algorithms find natural groupings
    - Statistical correlation discovery
    """
    
    def __init__(self, data_path):
        """Initialize blind pattern discoverer with raw data."""
        self.data_path = data_path
        self.df = None
        self.patterns = {}
        
        # Raw numerical mappings for clustering (no interpretations)
        self.session_numeric = {
            'MIDNIGHT': 0, 'LONDON': 1, 'PREMARKET': 2, 'NY_AM': 3,
            'LUNCH': 4, 'NY_PM': 5, 'ASIA': 6, 'UNKNOWN': 7
        }
        
        self.news_numeric = {
            'quiet': 0, 'low±30m': 1, 'medium±60m': 2, 'high±120m': 3
        }
        
        self.compound_numeric = {
            'baseline': 0, 'macro_only': 1, 'news_only': 2
        }
    
    def load_raw_data(self):
        """Load raw confluence data for pattern discovery."""
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Add derived temporal features for discovery
            self.df['timestamp_dt'] = pd.to_datetime(self.df['timestamp_et'], errors='coerce')
            self.df['hour'] = self.df['timestamp_dt'].dt.hour
            self.df['day_of_week_num'] = self.df['timestamp_dt'].dt.dayofweek
            self.df['minute'] = self.df['timestamp_dt'].dt.minute
            
            # Add numerical encodings for clustering
            self.df['session_num'] = self.df['session_type'].map(self.session_numeric).fillna(7)
            self.df['news_num'] = self.df['news_bucket'].map(self.news_numeric).fillna(0)
            self.df['compound_num'] = self.df['compound_type'].map(self.compound_numeric).fillna(0)
            
            print(f"✅ Loaded {len(self.df)} RD@40 events for blind pattern discovery")
            print(f"📊 Raw data dimensions: {self.df.shape}")
            print(f"🔍 Temporal range: {self.df['timestamp_dt'].min()} to {self.df['timestamp_dt'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def discover_session_clusters(self):
        """Discover natural session clustering patterns without assumptions."""
        print("\n🔍 BLIND SESSION CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Prepare clustering features (no interpretations, just raw patterns)
        cluster_features = [
            'range_position', 'energy_density', 'magnitude', 'hour', 
            'session_num', 'news_num', 'compound_num'
        ]
        
        # Clean data for clustering
        cluster_data = self.df[cluster_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Try different cluster numbers to find natural groupings
        cluster_results = {}
        silhouette_scores = {}
        
        for n_clusters in range(2, 8):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(cluster_data_scaled)
            
            # Calculate silhouette score
            try:
                from sklearn.metrics import silhouette_score
                score = silhouette_score(cluster_data_scaled, cluster_labels)
                silhouette_scores[n_clusters] = score
            except:
                silhouette_scores[n_clusters] = 0
            
            cluster_results[n_clusters] = {
                'labels': cluster_labels,
                'centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            }
        
        # Find optimal cluster count
        optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
        best_labels = cluster_results[optimal_clusters]['labels']
        
        print(f"📊 Optimal cluster count discovered: {optimal_clusters}")
        print(f"   Silhouette score: {silhouette_scores[optimal_clusters]:.3f}")
        
        # Analyze discovered clusters
        self.df['discovered_cluster'] = best_labels
        
        cluster_analysis = {}
        for cluster_id in range(optimal_clusters):
            cluster_data = self.df[self.df['discovered_cluster'] == cluster_id]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'avg_range_position': cluster_data['range_position'].mean(),
                'avg_energy_density': cluster_data['energy_density'].mean(),
                'avg_magnitude': cluster_data['magnitude'].mean(),
                'peak_hours': cluster_data['hour'].value_counts().head(3).to_dict(),
                'dominant_sessions': cluster_data['session_type'].value_counts().head(3).to_dict(),
                'news_patterns': cluster_data['news_bucket'].value_counts().to_dict(),
                'compound_patterns': cluster_data['compound_type'].value_counts().to_dict()
            }
        
        self._print_cluster_analysis(cluster_analysis)
        self.patterns['session_clusters'] = cluster_analysis
        
        return cluster_analysis
    
    def discover_temporal_patterns(self):
        """Discover temporal distribution patterns by hour and day."""
        print("\n🔍 BLIND TEMPORAL PATTERN DISCOVERY")
        print("=" * 60)
        
        # Hour-by-hour distribution discovery
        hourly_dist = self.df['hour'].value_counts().sort_index()
        
        print(f"📊 Hourly Event Distribution (pure counts):")
        for hour, count in hourly_dist.items():
            pct = count / len(self.df) * 100
            print(f"   {int(hour):02d}:00 ET: {count:3d} events ({pct:4.1f}%)")
        
        # Day of week patterns
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_dist = self.df['day_of_week_num'].value_counts().sort_index()
        
        print(f"\n📊 Day of Week Distribution:")
        for day_num, count in daily_dist.items():
            day_name = day_names[int(day_num)] if int(day_num) < 7 else 'Unknown'
            pct = count / len(self.df) * 100
            print(f"   {day_name}: {count:3d} events ({pct:4.1f}%)")
        
        # Discover minute-level patterns
        minute_clusters = self.df['minute'].value_counts().head(10)
        
        print(f"\n📊 Top Minute Patterns (clustering without interpretation):")
        for minute, count in minute_clusters.items():
            print(f"   :{int(minute):02d} minute: {count} events")
        
        # Cross-temporal correlations
        print(f"\n🔬 Temporal Correlation Discovery:")
        temporal_features = ['hour', 'day_of_week_num', 'minute']
        
        for i, feature1 in enumerate(temporal_features):
            for feature2 in temporal_features[i+1:]:
                correlation = self.df[feature1].corr(self.df[feature2])
                print(f"   {feature1} ↔ {feature2}: correlation = {correlation:.3f}")
        
        # Peak activity periods (data-driven discovery)
        peak_periods = self._discover_peak_periods()
        
        self.patterns['temporal'] = {
            'hourly_distribution': dict(hourly_dist),
            'daily_distribution': dict(daily_dist),
            'minute_clusters': dict(minute_clusters),
            'peak_periods': peak_periods
        }
        
        return self.patterns['temporal']
    
    def discover_news_correlations(self):
        """Discover news proximity correlation patterns."""
        print("\n🔍 BLIND NEWS PROXIMITY CORRELATION DISCOVERY")
        print("=" * 60)
        
        # News bucket distribution (pure counts)
        news_dist = self.df['news_bucket'].value_counts()
        
        print(f"📊 News Proximity Distribution (raw counts):")
        for bucket, count in news_dist.items():
            pct = count / len(self.df) * 100
            print(f"   {bucket}: {count:3d} events ({pct:4.1f}%)")
        
        # Correlations with other variables
        print(f"\n🔬 News Proximity Correlations:")
        
        correlation_vars = ['range_position', 'energy_density', 'magnitude', 'hour']
        
        for var in correlation_vars:
            correlation = self.df['news_num'].corr(self.df[var])
            print(f"   News proximity ↔ {var}: {correlation:.3f}")
        
        # News impact on energy/magnitude patterns
        news_impact_analysis = {}
        
        for bucket in news_dist.index:
            news_data = self.df[self.df['news_bucket'] == bucket]
            
            news_impact_analysis[bucket] = {
                'count': len(news_data),
                'avg_energy': news_data['energy_density'].mean(),
                'avg_magnitude': news_data['magnitude'].mean(),
                'avg_range_position': news_data['range_position'].mean(),
                'energy_std': news_data['energy_density'].std(),
                'magnitude_std': news_data['magnitude'].std(),
                'temporal_concentration': news_data['hour'].value_counts().head(3).to_dict()
            }
        
        self._print_news_impact_analysis(news_impact_analysis)
        
        # Statistical significance tests
        print(f"\n📈 Statistical Significance Discovery:")
        self._discover_news_significance()
        
        self.patterns['news_correlations'] = news_impact_analysis
        
        return news_impact_analysis
    
    def discover_energy_magnitude_relationships(self):
        """Explore energy density vs magnitude relationships."""
        print("\n🔍 BLIND ENERGY-MAGNITUDE RELATIONSHIP DISCOVERY")
        print("=" * 60)
        
        # Basic correlation
        energy_mag_corr = self.df['energy_density'].corr(self.df['magnitude'])
        print(f"📊 Energy Density ↔ Magnitude Correlation: {energy_mag_corr:.3f}")
        
        # Range position correlations
        energy_range_corr = self.df['energy_density'].corr(self.df['range_position'])
        mag_range_corr = self.df['magnitude'].corr(self.df['range_position'])
        
        print(f"📊 Energy ↔ Range Position: {energy_range_corr:.3f}")
        print(f"📊 Magnitude ↔ Range Position: {mag_range_corr:.3f}")
        
        # Discover energy-magnitude clusters
        energy_mag_data = self.df[['energy_density', 'magnitude']].fillna(0)
        
        # Try clustering on energy-magnitude space
        em_scaler = StandardScaler()
        em_scaled = em_scaler.fit_transform(energy_mag_data)
        
        em_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        em_clusters = em_kmeans.fit_predict(em_scaled)
        
        self.df['em_cluster'] = em_clusters
        
        print(f"\n📊 Energy-Magnitude Natural Clusters:")
        for cluster_id in range(3):
            cluster_data = self.df[self.df['em_cluster'] == cluster_id]
            
            print(f"   Cluster {cluster_id}: {len(cluster_data)} events")
            print(f"     Avg Energy: {cluster_data['energy_density'].mean():.3f}")
            print(f"     Avg Magnitude: {cluster_data['magnitude'].mean():.3f}")
            print(f"     Avg Range Position: {cluster_data['range_position'].mean():.3f}")
            
            # Discover what characterizes each cluster
            top_sessions = cluster_data['session_type'].value_counts().head(2)
            print(f"     Top Sessions: {dict(top_sessions)}")
        
        # Range position precision analysis
        print(f"\n🎯 Range Position Precision Discovery:")
        
        # Events exactly at 0.4
        exact_40 = len(self.df[self.df['range_position'] == 0.4])
        
        # Events within tolerance bands
        tolerance_bands = [0.005, 0.01, 0.025, 0.05]
        
        print(f"   Exactly 0.4: {exact_40} events ({exact_40/len(self.df)*100:.1f}%)")
        
        for tolerance in tolerance_bands:
            within_tolerance = len(self.df[abs(self.df['range_position'] - 0.4) <= tolerance])
            pct = within_tolerance / len(self.df) * 100
            print(f"   Within ±{tolerance}: {within_tolerance} events ({pct:.1f}%)")
        
        self.patterns['energy_magnitude'] = {
            'correlation': energy_mag_corr,
            'energy_range_correlation': energy_range_corr,
            'magnitude_range_correlation': mag_range_corr,
            'clusters': em_clusters.tolist(),
            'exact_40_count': exact_40
        }
        
        return self.patterns['energy_magnitude']
    
    def _discover_peak_periods(self):
        """Discover peak activity periods from data."""
        hourly_counts = self.df['hour'].value_counts()
        mean_count = hourly_counts.mean()
        std_count = hourly_counts.std()
        
        peak_threshold = mean_count + std_count
        peak_hours = hourly_counts[hourly_counts > peak_threshold]
        
        return {
            'peak_hours': dict(peak_hours),
            'mean_hourly_events': mean_count,
            'peak_threshold': peak_threshold
        }
    
    def _discover_news_significance(self):
        """Discover statistical significance in news patterns."""
        # Test if news proximity affects energy density
        news_groups = []
        for bucket in self.df['news_bucket'].unique():
            if pd.notna(bucket):
                group_data = self.df[self.df['news_bucket'] == bucket]['energy_density']
                news_groups.append(group_data.values)
        
        if len(news_groups) > 1:
            try:
                f_stat, p_value = stats.f_oneway(*news_groups)
                print(f"   News impact on energy density: F={f_stat:.3f}, p={p_value:.4f}")
            except:
                print(f"   News significance test: insufficient data")
    
    def _print_cluster_analysis(self, cluster_analysis):
        """Print discovered cluster patterns."""
        print(f"\n📊 Discovered Natural Clusters (no assumptions):")
        
        for cluster_id, data in cluster_analysis.items():
            print(f"\n   🎯 Cluster {cluster_id}: {data['size']} events")
            print(f"      Range Position: {data['avg_range_position']:.3f}")
            print(f"      Energy Density: {data['avg_energy_density']:.3f}")
            print(f"      Magnitude: {data['avg_magnitude']:.3f}")
            
            # Top characteristics
            top_session = max(data['dominant_sessions'], key=data['dominant_sessions'].get)
            top_hour = max(data['peak_hours'], key=data['peak_hours'].get) if data['peak_hours'] else 'None'
            
            print(f"      Peak Session: {top_session} ({data['dominant_sessions'][top_session]} events)")
            print(f"      Peak Hour: {top_hour}")
    
    def _print_news_impact_analysis(self, analysis):
        """Print news impact analysis results."""
        print(f"\n📊 News Impact Patterns (pure data):")
        
        for bucket, data in analysis.items():
            print(f"\n   📰 {bucket}: {data['count']} events")
            print(f"      Avg Energy: {data['avg_energy']:.3f} (±{data['energy_std']:.3f})")
            print(f"      Avg Magnitude: {data['avg_magnitude']:.3f} (±{data['magnitude_std']:.3f})")
            print(f"      Range Position: {data['avg_range_position']:.3f}")
    
    def generate_blind_discovery_report(self):
        """Generate comprehensive blind pattern discovery report."""
        print("\n🚀 GENERATING PHASE III BLIND DISCOVERY REPORT")
        print("=" * 60)
        print("🔬 PURE DATA EXPLORATION - NO THEORETICAL ASSUMPTIONS")
        print("=" * 60)
        
        if not self.load_raw_data():
            return False
        
        # Run all discovery analyses
        session_clusters = self.discover_session_clusters()
        temporal_patterns = self.discover_temporal_patterns()
        news_correlations = self.discover_news_correlations()
        energy_magnitude = self.discover_energy_magnitude_relationships()
        
        # Generate pure discovery insights
        self._generate_blind_insights()
        
        return self.patterns
    
    def _generate_blind_insights(self):
        """Generate insights from pure data discovery."""
        print("\n💡 PURE DATA DISCOVERY INSIGHTS")
        print("=" * 60)
        print("🔬 What the DATA reveals (without theory):")
        
        # Most frequent patterns discovered
        most_common_hour = self.df['hour'].value_counts().index[0]
        most_common_session = self.df['session_type'].value_counts().index[0]
        most_common_news = self.df['news_bucket'].value_counts().index[0]
        
        print(f"\n📊 Most Frequent Patterns:")
        print(f"   Peak Hour: {most_common_hour}:00 ET ({self.df['hour'].value_counts().iloc[0]} events)")
        print(f"   Dominant Session: {most_common_session} ({self.df['session_type'].value_counts().iloc[0]} events)")
        print(f"   News Pattern: {most_common_news} ({self.df['news_bucket'].value_counts().iloc[0]} events)")
        
        # Range position precision
        exact_40_pct = len(self.df[self.df['range_position'] == 0.4]) / len(self.df) * 100
        near_40_pct = len(self.df[abs(self.df['range_position'] - 0.4) < 0.025]) / len(self.df) * 100
        
        print(f"\n🎯 Archaeological Precision Discovery:")
        print(f"   Exactly 40%: {exact_40_pct:.1f}% of all events")
        print(f"   Within ±2.5%: {near_40_pct:.1f}% of all events")
        
        # Natural clustering insights
        if 'session_clusters' in self.patterns:
            cluster_count = len(self.patterns['session_clusters'])
            largest_cluster_size = max([c['size'] for c in self.patterns['session_clusters'].values()])
            
            print(f"\n🎯 Natural Grouping Discovery:")
            print(f"   {cluster_count} distinct natural clusters found")
            print(f"   Largest cluster: {largest_cluster_size} events")
        
        # Energy-magnitude relationship
        if 'energy_magnitude' in self.patterns:
            em_corr = self.patterns['energy_magnitude']['correlation']
            print(f"\n⚡ Energy-Magnitude Relationship:")
            print(f"   Correlation strength: {em_corr:.3f}")
            
            if abs(em_corr) > 0.3:
                print(f"   Strong relationship detected!")
            elif abs(em_corr) > 0.1:
                print(f"   Moderate relationship detected")
            else:
                print(f"   Weak relationship detected")
        
        print(f"\n✅ Pure discovery complete. Patterns emerge without assumptions!")


def main():
    """Main execution for blind pattern discovery."""
    
    data_path = "/Users/jack/IRONFORGE/runs/RUN_20250824_182221_NEWSCLUST_3P/artifacts/macro_window_confluence_analysis.csv"
    
    # Initialize blind discoverer
    discoverer = BlindPatternDiscoverer(data_path)
    
    # Generate blind discovery report
    patterns = discoverer.generate_blind_discovery_report()
    
    if patterns:
        print("\n🎉 Phase III Blind Discovery Complete!")
        return 0
    else:
        print("\n❌ Discovery failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())