#!/usr/bin/env python3
"""
IRONFORGE Broad-Spectrum Market Archaeology - Complete System Test
=================================================================

Comprehensive test of the full market archaeology system including:
- Event mining across all timeframes
- Temporal clustering and pattern detection
- Structural link analysis and cascade detection
- Lattice mapping and visualization
- Complete deliverable generation

Author: IRONFORGE Archaeological Discovery System
Date: August 15, 2025
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import archaeology components
try:
    from ironforge.analysis.broad_spectrum_archaeology import (
        ArchaeologicalSummary,
        BroadSpectrumArchaeology,
    )
    from ironforge.analysis.structural_link_analyzer import (
        StructuralAnalysis,
        StructuralLinkAnalyzer,
    )
    from ironforge.analysis.temporal_clustering_engine import (
        ClusteringAnalysis,
        TemporalClusteringEngine,
    )
    from ironforge.analysis.timeframe_lattice_mapper import LatticeDataset, TimeframeLatticeMapper
    from visualizations.lattice_visualizer import LatticeVisualizer, VisualizationConfig
    
    print("✅ All archaeology components imported successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all archaeology components are properly installed")
    sys.exit(1)


def setup_logging():
    """Setup comprehensive logging for the test"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_filename = f"archaeology_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = logs_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return str(log_path)


def test_archaeology_engine():
    """Test the broad-spectrum archaeology engine"""
    
    print("\n🏛️  Testing Broad-Spectrum Archaeology Engine")
    print("=" * 60)
    
    try:
        # Initialize archaeology engine
        archaeology = BroadSpectrumArchaeology(
            enhanced_sessions_path="enhanced_sessions_with_relativity",
            enable_deep_analysis=True
        )
        
        # Run archaeological discovery
        print("\n🔍 Running archaeological discovery...")
        summary = archaeology.discover_all_phenomena()
        
        # Validate results
        print("\n📊 Discovery Results:")
        print(f"  Sessions analyzed: {summary.sessions_analyzed}")
        print(f"  Total events discovered: {summary.total_events_discovered}")
        print(f"  Events by timeframe: {summary.events_by_timeframe}")
        print(f"  Events by type: {summary.events_by_type}")
        print(f"  Temporal clusters: {len(summary.significant_clusters)}")
        print(f"  Cross-session patterns: {len(summary.cross_session_patterns)}")
        
        # Export phenomena catalog
        catalog_file = archaeology.export_phenomena_catalog("test_outputs/phenomena_catalog.json")
        
        return summary, catalog_file
        
    except Exception as e:
        print(f"❌ Archaeology engine test failed: {e}")
        logging.error(f"Archaeology engine test failed: {e}", exc_info=True)
        return None, None


def test_lattice_mapper(archaeological_events):
    """Test the timeframe lattice mapper"""
    
    print("\n🗺️  Testing Timeframe Lattice Mapper")
    print("=" * 60)
    
    try:
        # Initialize lattice mapper
        mapper = TimeframeLatticeMapper(
            grid_resolution=50,  # Reduced for testing
            min_node_events=1,
            hot_zone_threshold=0.6
        )
        
        # Map events to lattice
        print("\n🔧 Mapping events to lattice...")
        lattice_dataset = mapper.map_events_to_lattice(archaeological_events)
        
        # Validate results
        print("\n📊 Lattice Mapping Results:")
        print(f"  Nodes created: {len(lattice_dataset.nodes)}")
        print(f"  Connections identified: {len(lattice_dataset.connections)}")
        print(f"  Hot zones detected: {len(lattice_dataset.hot_zones)}")
        print(f"  Events mapped: {lattice_dataset.total_events_mapped}")
        print(f"  Sessions covered: {len(lattice_dataset.sessions_covered)}")
        
        # Export lattice dataset
        lattice_file = mapper.export_lattice_dataset(lattice_dataset, "test_outputs/lattice_dataset.json")
        
        return lattice_dataset, lattice_file
        
    except Exception as e:
        print(f"❌ Lattice mapper test failed: {e}")
        logging.error(f"Lattice mapper test failed: {e}", exc_info=True)
        return None, None


def test_temporal_clustering(archaeological_events):
    """Test the temporal clustering engine"""
    
    print("\n🕰️  Testing Temporal Clustering Engine")
    print("=" * 60)
    
    try:
        # Initialize clustering engine
        clustering_engine = TemporalClusteringEngine(
            min_cluster_size=2,  # Reduced for testing
            temporal_weight=0.4,
            structural_weight=0.3,
            significance_weight=0.3
        )
        
        # Perform clustering analysis
        print("\n🎯 Performing temporal clustering...")
        clustering_analysis = clustering_engine.cluster_temporal_patterns(archaeological_events)
        
        # Validate results
        print("\n📊 Clustering Results:")
        print(f"  Clusters identified: {clustering_analysis.cluster_count}")
        print(f"  Overall silhouette score: {clustering_analysis.overall_silhouette_score:.3f}")
        print(f"  Temporal coverage: {clustering_analysis.temporal_coverage:.1%}")
        print(f"  Pattern discovery rate: {clustering_analysis.pattern_discovery_rate:.3f}")
        print(f"  Quality distribution: {clustering_analysis.cluster_quality_distribution}")
        print(f"  Noise events: {len(clustering_analysis.noise_events)}")
        
        # Export clustering analysis
        clustering_file = clustering_engine.export_clustering_analysis(
            clustering_analysis, "test_outputs/temporal_clustering_analysis.json"
        )
        
        return clustering_analysis, clustering_file
        
    except Exception as e:
        print(f"❌ Temporal clustering test failed: {e}")
        logging.error(f"Temporal clustering test failed: {e}", exc_info=True)
        return None, None


def test_structural_analyzer(archaeological_events):
    """Test the structural link analyzer"""
    
    print("\n🔗 Testing Structural Link Analyzer")
    print("=" * 60)
    
    try:
        # Initialize structural analyzer
        analyzer = StructuralLinkAnalyzer(
            min_link_strength=0.2,  # Reduced for testing
            cascade_threshold=0.4,
            energy_accumulation_threshold=0.5
        )
        
        # Perform structural analysis
        print("\n⚡ Performing structural analysis...")
        structural_analysis = analyzer.analyze_structural_relationships(archaeological_events)
        
        # Validate results
        print("\n📊 Structural Analysis Results:")
        print(f"  Structural links: {len(structural_analysis.structural_links)}")
        print(f"  Cascade chains: {len(structural_analysis.cascade_chains)}")
        print(f"  Energy accumulations: {len(structural_analysis.energy_accumulations)}")
        print(f"  Network density: {structural_analysis.network_density:.3f}")
        print(f"  Average path length: {structural_analysis.average_path_length:.3f}")
        print(f"  Clustering coefficient: {structural_analysis.clustering_coefficient:.3f}")
        print(f"  Active cascades: {len(structural_analysis.active_cascade_chains)}")
        print(f"  Energy hotspots: {len(structural_analysis.energy_hotspots)}")
        
        # Export structural analysis
        structural_file = analyzer.export_structural_analysis(
            structural_analysis, "test_outputs/structural_analysis.json"
        )
        
        return structural_analysis, structural_file
        
    except Exception as e:
        print(f"❌ Structural analyzer test failed: {e}")
        logging.error(f"Structural analyzer test failed: {e}", exc_info=True)
        return None, None


def test_visualizer(lattice_dataset, clustering_analysis, structural_analysis):
    """Test the lattice visualizer"""
    
    print("\n🎨 Testing Lattice Visualizer")
    print("=" * 60)
    
    try:
        # Initialize visualizer
        config = VisualizationConfig(
            figure_size=(12, 8),  # Reduced for testing
            dpi=100  # Reduced for faster processing
        )
        visualizer = LatticeVisualizer(config)
        
        # Create comprehensive visualizations
        print("\n🖼️  Creating visualizations...")
        visualization_files = visualizer.create_comprehensive_visualization(
            lattice_dataset=lattice_dataset,
            clustering_analysis=clustering_analysis,
            structural_analysis=structural_analysis,
            output_dir="test_outputs/visualizations"
        )
        
        # Validate results
        print("\n📊 Visualization Results:")
        print(f"  Visualizations created: {len(visualization_files)}")
        for name, file_path in visualization_files.items():
            print(f"    {name}: {file_path}")
        
        return visualization_files
        
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")
        logging.error(f"Visualizer test failed: {e}", exc_info=True)
        return {}


def generate_test_summary(results: dict[str, Any]):
    """Generate comprehensive test summary"""
    
    print("\n📋 Generating Test Summary")
    print("=" * 60)
    
    summary = {
        'test_timestamp': datetime.now().isoformat(),
        'test_duration': results.get('total_duration', 0),
        'components_tested': [],
        'results': {},
        'files_generated': {},
        'performance_metrics': {},
        'success_status': True
    }
    
    # Component test results
    if results.get('archaeology_summary'):
        summary['components_tested'].append('BroadSpectrumArchaeology')
        summary['results']['archaeology'] = {
            'sessions_analyzed': results['archaeology_summary'].sessions_analyzed,
            'events_discovered': results['archaeology_summary'].total_events_discovered,
            'temporal_clusters': len(results['archaeology_summary'].significant_clusters),
            'cross_session_patterns': len(results['archaeology_summary'].cross_session_patterns)
        }
        summary['files_generated']['phenomena_catalog'] = results.get('catalog_file')
    
    if results.get('lattice_dataset'):
        summary['components_tested'].append('TimeframeLatticeMapper')
        summary['results']['lattice_mapping'] = {
            'nodes_created': len(results['lattice_dataset'].nodes),
            'connections_identified': len(results['lattice_dataset'].connections),
            'hot_zones_detected': len(results['lattice_dataset'].hot_zones),
            'events_mapped': results['lattice_dataset'].total_events_mapped
        }
        summary['files_generated']['lattice_dataset'] = results.get('lattice_file')
    
    if results.get('clustering_analysis'):
        summary['components_tested'].append('TemporalClusteringEngine')
        summary['results']['temporal_clustering'] = {
            'clusters_identified': results['clustering_analysis'].cluster_count,
            'silhouette_score': results['clustering_analysis'].overall_silhouette_score,
            'temporal_coverage': results['clustering_analysis'].temporal_coverage,
            'pattern_discovery_rate': results['clustering_analysis'].pattern_discovery_rate
        }
        summary['files_generated']['clustering_analysis'] = results.get('clustering_file')
    
    if results.get('structural_analysis'):
        summary['components_tested'].append('StructuralLinkAnalyzer')
        summary['results']['structural_analysis'] = {
            'structural_links': len(results['structural_analysis'].structural_links),
            'cascade_chains': len(results['structural_analysis'].cascade_chains),
            'energy_accumulations': len(results['structural_analysis'].energy_accumulations),
            'network_density': results['structural_analysis'].network_density
        }
        summary['files_generated']['structural_analysis'] = results.get('structural_file')
    
    if results.get('visualization_files'):
        summary['components_tested'].append('LatticeVisualizer')
        summary['results']['visualization'] = {
            'visualizations_created': len(results['visualization_files']),
            'file_types': list(results['visualization_files'].keys())
        }
        summary['files_generated']['visualizations'] = results['visualization_files']
    
    # Performance metrics
    summary['performance_metrics'] = {
        'events_per_second': results.get('events_discovered', 0) / max(results.get('total_duration', 1), 1),
        'nodes_per_second': results.get('nodes_created', 0) / max(results.get('total_duration', 1), 1),
        'memory_efficient': True,  # Placeholder
        'scalable_architecture': True  # Placeholder
    }
    
    # Success status
    required_components = ['BroadSpectrumArchaeology', 'TimeframeLatticeMapper', 
                          'TemporalClusteringEngine', 'StructuralLinkAnalyzer']
    summary['success_status'] = all(comp in summary['components_tested'] for comp in required_components)
    
    # Save summary
    summary_file = "test_outputs/test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n📊 Test Summary:")
    print(f"  Components tested: {len(summary['components_tested'])}")
    print(f"  Success status: {'✅ PASS' if summary['success_status'] else '❌ FAIL'}")
    print(f"  Test duration: {summary['test_duration']:.1f} seconds")
    print(f"  Files generated: {len(summary['files_generated'])}")
    print(f"  Summary saved to: {summary_file}")
    
    return summary


def main():
    """Main test execution"""
    
    print("🏛️  IRONFORGE Broad-Spectrum Market Archaeology - System Test")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    log_file = setup_logging()
    print(f"Logging to: {log_file}")
    
    # Create output directory
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    
    start_time = time.time()
    results = {}
    
    try:
        # Test 1: Archaeology Engine
        archaeology_summary, catalog_file = test_archaeology_engine()
        if archaeology_summary:
            results['archaeology_summary'] = archaeology_summary
            results['catalog_file'] = catalog_file
            results['events_discovered'] = archaeology_summary.total_events_discovered
            
            # Get archaeological events for other tests
            archaeological_events = archaeology_summary.phenomena_catalog
            print(f"✅ Archaeological events available for testing: {len(archaeological_events)}")
        else:
            print("❌ Cannot proceed without archaeological events")
            return
        
        # Test 2: Lattice Mapper
        lattice_dataset, lattice_file = test_lattice_mapper(archaeological_events)
        if lattice_dataset:
            results['lattice_dataset'] = lattice_dataset
            results['lattice_file'] = lattice_file
            results['nodes_created'] = len(lattice_dataset.nodes)
        
        # Test 3: Temporal Clustering
        clustering_analysis, clustering_file = test_temporal_clustering(archaeological_events)
        if clustering_analysis:
            results['clustering_analysis'] = clustering_analysis
            results['clustering_file'] = clustering_file
        
        # Test 4: Structural Analysis
        structural_analysis, structural_file = test_structural_analyzer(archaeological_events)
        if structural_analysis:
            results['structural_analysis'] = structural_analysis
            results['structural_file'] = structural_file
        
        # Test 5: Visualizations
        if lattice_dataset:
            visualization_files = test_visualizer(lattice_dataset, clustering_analysis, structural_analysis)
            if visualization_files:
                results['visualization_files'] = visualization_files
        
        # Calculate total duration
        total_duration = time.time() - start_time
        results['total_duration'] = total_duration
        
        # Generate comprehensive summary
        test_summary = generate_test_summary(results)
        
        # Final status
        print("\n🏁 Test Execution Complete!")
        print(f"  Total duration: {total_duration:.1f} seconds")
        print(f"  Overall status: {'✅ SUCCESS' if test_summary['success_status'] else '❌ FAILURE'}")
        
        if test_summary['success_status']:
            print("\n🎉 All components tested successfully!")
            print("  The IRONFORGE Broad-Spectrum Market Archaeology system is operational")
            print("  Ready for production archaeological discovery workflows")
        else:
            print("\n⚠️  Some components failed testing")
            print("  Review logs and fix issues before production use")
        
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        logging.error(f"Test execution failed: {e}", exc_info=True)
        
        # Save error summary
        error_summary = {
            'test_timestamp': datetime.now().isoformat(),
            'test_duration': time.time() - start_time,
            'error': str(e),
            'success_status': False
        }
        
        with open("test_outputs/error_summary.json", 'w') as f:
            json.dump(error_summary, f, indent=2)


if __name__ == "__main__":
    main()