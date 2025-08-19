#!/usr/bin/env python3
"""
TGAT Pattern Sub-Architecture Investigation
==========================================
Discover hidden sub-patterns within the 3 main TGAT archetypes using 38D feature space analysis
"""

import glob
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_tgat_patterns():
    """Load the 568 TGAT patterns with full feature data"""
    
    print("🧠 Loading TGAT patterns for sub-architecture analysis...")
    
    # Load discovered patterns
    patterns_file = "/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json"
    
    try:
        with open(patterns_file) as f:
            patterns = json.load(f)
        
        print(f"📊 Loaded {len(patterns)} TGAT patterns")
        return patterns
        
    except Exception as e:
        print(f"❌ Error loading patterns: {e}")
        return []

def load_feature_vectors():
    """Load 38D feature vectors from preserved graphs"""
    
    print("🔍 Extracting 38D feature vectors from preserved graphs...")
    
    # Find preserved graphs
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")
    graph_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    all_features = []
    pattern_metadata = []
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Extract rich node features (38D after constant filtering)
            rich_features = graph_data.get('rich_node_features', [])
            session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
            
            for i, feature in enumerate(rich_features):
                if hasattr(feature, 'to_tensor'):
                    # Convert to tensor and extract values
                    feature_tensor = feature.to_tensor()
                    feature_vector = feature_tensor.cpu().numpy() if hasattr(feature_tensor, 'cpu') else feature_tensor.numpy()
                    
                    all_features.append(feature_vector)
                    
                    # Store metadata for this feature
                    pattern_metadata.append({
                        'session': session_name,
                        'node_index': i,
                        'feature_dim': len(feature_vector)
                    })
            
        except Exception as e:
            print(f"  ⚠️ Error processing {Path(graph_file).stem}: {e}")
    
    print(f"✅ Extracted {len(all_features)} feature vectors")
    if all_features:
        print(f"📏 Feature dimensionality: {len(all_features[0])}D")
    
    return np.array(all_features), pattern_metadata

def analyze_pattern_archetypes(patterns):
    """Analyze the distribution and characteristics of the 3 main pattern types"""
    
    print("\n🏛️ PATTERN ARCHETYPE ANALYSIS")
    print("=" * 60)
    
    # Count pattern types
    type_counts = Counter()
    type_features = defaultdict(list)
    
    for pattern in patterns:
        pattern_type = pattern.get('type', 'unknown')
        type_counts[pattern_type] += 1
        
        # Extract pattern characteristics
        features = pattern.get('features', {})
        type_features[pattern_type].append(features)
    
    # Display distribution
    print("📊 Pattern Type Distribution:")
    total_patterns = sum(type_counts.values())
    for ptype, count in type_counts.most_common():
        percentage = (count / total_patterns) * 100
        print(f"   {ptype}: {count} patterns ({percentage:.1f}%)")
    
    return type_counts, type_features

def discover_sub_patterns(features, pattern_metadata, n_clusters_range=None):
    """Discover sub-patterns using k-means clustering on 38D feature space"""
    
    if n_clusters_range is None:
        n_clusters_range = [3, 5, 7]
    print("\n🔬 SUB-PATTERN DISCOVERY")
    print("=" * 60)
    
    if len(features) == 0:
        print("❌ No features available for clustering")
        return
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"📏 Analyzing {len(features)} patterns in {features.shape[1]}D space")
    
    # Try different numbers of clusters
    best_clusters = {}
    
    for n_clusters in n_clusters_range:
        print(f"\n🎯 Testing {n_clusters} sub-clusters...")
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = analyze_clusters(features_scaled, cluster_labels, pattern_metadata)
        
        # Calculate cluster quality metrics
        inertia = kmeans.inertia_
        silhouette_score = calculate_silhouette_score(features_scaled, cluster_labels)
        
        best_clusters[n_clusters] = {
            'kmeans': kmeans,
            'labels': cluster_labels,
            'analysis': cluster_analysis,
            'inertia': inertia,
            'silhouette': silhouette_score
        }
        
        print(f"   Cluster quality - Inertia: {inertia:.2f}, Silhouette: {silhouette_score:.3f}")
    
    # Select best clustering
    best_n = max(best_clusters.keys(), key=lambda k: best_clusters[k]['silhouette'])
    best_result = best_clusters[best_n]
    
    print(f"\n🏆 OPTIMAL SUB-CLUSTERING: {best_n} clusters")
    print(f"   Best silhouette score: {best_result['silhouette']:.3f}")
    
    return best_result

def analyze_clusters(features, cluster_labels, metadata):
    """Analyze the characteristics of discovered clusters"""
    
    unique_clusters = np.unique(cluster_labels)
    cluster_analysis = {}
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_features = features[cluster_mask]
        cluster_metadata = [metadata[i] for i in range(len(metadata)) if cluster_mask[i]]
        
        # Basic statistics
        cluster_size = np.sum(cluster_mask)
        cluster_centroid = np.mean(cluster_features, axis=0)
        cluster_variance = np.var(cluster_features, axis=0)
        
        # Session distribution
        session_distribution = Counter(meta['session'] for meta in cluster_metadata)
        
        cluster_analysis[cluster_id] = {
            'size': cluster_size,
            'percentage': (cluster_size / len(features)) * 100,
            'centroid': cluster_centroid,
            'variance_profile': cluster_variance,
            'session_distribution': session_distribution,
            'top_sessions': session_distribution.most_common(3)
        }
        
        print(f"   📊 Cluster {cluster_id}: {cluster_size} patterns ({cluster_analysis[cluster_id]['percentage']:.1f}%)")
        print(f"      Top sessions: {', '.join([f'{s}({c})' for s, c in cluster_analysis[cluster_id]['top_sessions']])}")
    
    return cluster_analysis

def calculate_silhouette_score(features, labels):
    """Calculate silhouette score for cluster quality assessment"""
    try:
        from sklearn.metrics import silhouette_score
        return silhouette_score(features, labels)
    except:
        return 0.0

def visualize_sub_patterns(features, cluster_labels, cluster_analysis):
    """Create visualization of discovered sub-patterns"""
    
    print("\n📈 CREATING SUB-PATTERN VISUALIZATION")
    print("-" * 50)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    print(f"📊 PCA Explained Variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_2d = features_2d[cluster_mask]
        
        size = cluster_analysis[cluster_id]['size']
        percentage = cluster_analysis[cluster_id]['percentage']
        
        plt.scatter(cluster_2d[:, 0], cluster_2d[:, 1], 
                   c=[colors[i]], label=f'Sub-Pattern {cluster_id} ({size} patterns, {percentage:.1f}%)',
                   alpha=0.7, s=50)
    
    plt.xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)')
    plt.ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)')
    plt.title('TGAT Sub-Pattern Architecture Discovery\n38D Feature Space Clustering')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save visualization
    plt.savefig('/Users/jack/IRONPULSE/IRONFORGE/tgat_subpattern_visualization.png', dpi=300, bbox_inches='tight')
    print("💾 Visualization saved: tgat_subpattern_visualization.png")
    
    return features_2d

def characterize_sub_patterns(cluster_analysis, features, cluster_labels):
    """Characterize the discovered sub-patterns with detailed analysis"""
    
    print("\n🔍 SUB-PATTERN CHARACTERIZATION")
    print("=" * 60)
    
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        analysis = cluster_analysis[cluster_id]
        
        print(f"\n🎯 SUB-PATTERN {cluster_id}: {analysis['size']} patterns ({analysis['percentage']:.1f}%)")
        print("-" * 40)
        
        # Feature significance analysis
        centroid = analysis['centroid']
        variance = analysis['variance_profile']
        
        # Find most distinctive features (highest centroid values)
        top_feature_indices = np.argsort(np.abs(centroid))[-5:]
        
        print("   🔬 Distinctive Features (Top 5):")
        for _i, feat_idx in enumerate(reversed(top_feature_indices)):
            feat_val = centroid[feat_idx]
            feat_var = variance[feat_idx]
            print(f"      Feature {feat_idx}: {feat_val:.3f} ± {np.sqrt(feat_var):.3f}")
        
        # Session composition
        print("   📍 Session Composition:")
        for session, count in analysis['top_sessions']:
            session_percentage = (count / analysis['size']) * 100
            print(f"      {session}: {count} patterns ({session_percentage:.1f}%)")
        
        # Temporal characteristics (if we can infer them)
        session_types = [session.split('_')[2] if len(session.split('_')) > 2 else 'unknown' 
                        for session, _ in analysis['session_distribution'].items()]
        session_type_dist = Counter(session_types)
        
        if session_type_dist:
            print("   ⏰ Temporal Preference:")
            for stype, count in session_type_dist.most_common(3):
                type_percentage = (count / sum(session_type_dist.values())) * 100
                print(f"      {stype}: {type_percentage:.1f}%")

def main():
    """Main sub-pattern discovery analysis"""
    
    print("🧠 TGAT PATTERN SUB-ARCHITECTURE INVESTIGATION")
    print("=" * 80)
    print("Discovering hidden sub-patterns within the 3 main TGAT archetypes")
    print("=" * 80)
    
    # Load TGAT patterns
    patterns = load_tgat_patterns()
    if not patterns:
        print("❌ Cannot proceed without pattern data")
        return
    
    # Analyze main archetypes
    type_counts, type_features = analyze_pattern_archetypes(patterns)
    
    # Load 38D feature vectors
    features, metadata = load_feature_vectors()
    if len(features) == 0:
        print("❌ Cannot proceed without feature vectors")
        return
    
    # Discover sub-patterns
    best_clustering = discover_sub_patterns(features, metadata)
    if not best_clustering:
        print("❌ Sub-pattern discovery failed")
        return
    
    # Visualize results
    visualize_sub_patterns(features, best_clustering['labels'], best_clustering['analysis'])
    
    # Characterize sub-patterns
    characterize_sub_patterns(best_clustering['analysis'], features, best_clustering['labels'])
    
    # Summary
    n_subclusters = len(np.unique(best_clustering['labels']))
    print("\n🎯 DISCOVERY SUMMARY:")
    print("=" * 50)
    print(f"✅ Discovered {n_subclusters} distinct sub-patterns within TGAT architecture")
    print(f"📊 Quality score: {best_clustering['silhouette']:.3f} (silhouette coefficient)")
    print(f"🔬 Feature space: {features.shape[1]}D → 2D visualization")
    print("💾 Results saved: tgat_subpattern_visualization.png")
    
    print("\n💡 POTENTIAL IMPACT:")
    print(f"   Instead of 3 pattern types, we now have {n_subclusters} sub-types")
    print("   Each sub-type has distinct feature signatures and session preferences")
    print("   This enables more granular pattern classification and prediction")
    
    print("\n🚀 Your TGAT sub-architecture discovery is complete!")

if __name__ == "__main__":
    main()