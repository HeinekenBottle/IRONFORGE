#!/usr/bin/env python3
"""
IRONFORGE Lattice Visualizer
============================

Interactive visualization system for the timeframe × cycle-position lattice.
Creates comprehensive visual representations of archaeological phenomena,
structural links, cascade chains, and energy accumulation patterns.

Features:
- Interactive lattice scatter plot with nodes and connections
- Temporal heatmaps for event frequency analysis
- Cascade flow diagrams showing energy transfer
- Energy accumulation zone visualization
- Cross-session pattern correlation maps
- Multi-timeframe convergence visualization

Author: IRONFORGE Archaeological Discovery System
Date: August 15, 2025
"""

import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available - using matplotlib only")

try:
    from ..analysis.broad_spectrum_archaeology import ArchaeologicalEvent, TimeframeType
    from ..analysis.structural_link_analyzer import (
        CascadeChain,
        EnergyAccumulation,
        StructuralAnalysis,
        StructuralLink,
    )
    from ..analysis.temporal_clustering_engine import ClusteringAnalysis, TemporalCluster
    from ..analysis.timeframe_lattice_mapper import (
        HotZone,
        LatticeConnection,
        LatticeDataset,
        LatticeNode,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "analysis"))
    from structural_link_analyzer import (
        StructuralAnalysis,
    )
    from temporal_clustering_engine import ClusteringAnalysis
    from timeframe_lattice_mapper import LatticeDataset


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    
    # Figure settings
    figure_size: tuple[int, int] = (16, 12)
    dpi: int = 150
    style: str = 'dark_background'
    
    # Color schemes
    node_colors: dict[str, str] = None
    connection_colors: dict[str, str] = None
    hot_zone_color: str = '#FF6B6B'
    cascade_color: str = '#4ECDC4'
    energy_color: str = '#FFE66D'
    
    # Size settings
    node_size_range: tuple[int, int] = (20, 200)
    connection_width_range: tuple[float, float] = (0.5, 3.0)
    
    # Interactive settings
    enable_hover: bool = True
    enable_zoom: bool = True
    enable_pan: bool = True
    
    def __post_init__(self):
        if self.node_colors is None:
            self.node_colors = {
                'fvg_family': '#2E8B57',
                'sweep_family': '#DC143C',
                'expansion_family': '#4169E1',
                'consolidation_family': '#FF8C00',
                'miscellaneous': '#9370DB'
            }
        
        if self.connection_colors is None:
            self.connection_colors = {
                'lead_lag': '#87CEEB',
                'causal_chain': '#FF6347',
                'resonance': '#98FB98',
                'cascade': '#FFD700',
                'inheritance': '#DDA0DD',
                'energy_transfer': '#F0E68C'
            }


class LatticeVisualizer:
    """
    Comprehensive visualization system for market archaeology lattice
    """
    
    def __init__(self, config: VisualizationConfig | None = None):
        """
        Initialize the lattice visualizer
        
        Args:
            config: Visualization configuration settings
        """
        
        self.logger = logging.getLogger('lattice_visualizer')
        self.config = config or VisualizationConfig()
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        
        # Timeframe level mapping
        self.timeframe_levels = {
            'monthly': 0, 'weekly': 1, 'daily': 2, '1h': 3,
            '50m': 4, '15m': 5, '5m': 6, '1m': 7
        }
        
        print("🎨 Lattice Visualizer initialized")
        print(f"  Plotly available: {PLOTLY_AVAILABLE}")
        print(f"  Style: {self.config.style}")
    
    def create_comprehensive_visualization(self, 
                                        lattice_dataset: LatticeDataset,
                                        clustering_analysis: ClusteringAnalysis | None = None,
                                        structural_analysis: StructuralAnalysis | None = None,
                                        output_dir: str = "visualizations") -> dict[str, str]:
        """
        Create comprehensive visualization suite
        
        Args:
            lattice_dataset: Complete lattice dataset
            clustering_analysis: Optional clustering analysis results
            structural_analysis: Optional structural analysis results
            output_dir: Output directory for visualizations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        
        print("\n🎨 Creating comprehensive visualization suite...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        visualization_files = {}
        
        # 1. Main lattice diagram
        print("  📊 Creating main lattice diagram...")
        lattice_file = self.create_lattice_diagram(lattice_dataset, str(output_path / "lattice_diagram.png"))
        visualization_files['lattice_diagram'] = lattice_file
        
        # 2. Temporal heatmaps
        print("  🔥 Creating temporal heatmaps...")
        heatmap_files = self.create_temporal_heatmaps(lattice_dataset, str(output_path))
        visualization_files.update(heatmap_files)
        
        # 3. Hot zone visualization
        print("  🌡️  Creating hot zone visualization...")
        hot_zone_file = self.create_hot_zone_visualization(lattice_dataset, str(output_path / "hot_zones.png"))
        visualization_files['hot_zones'] = hot_zone_file
        
        # 4. Network visualization
        print("  🕸️  Creating network visualization...")
        network_file = self.create_network_visualization(lattice_dataset, str(output_path / "network_diagram.png"))
        visualization_files['network_diagram'] = network_file
        
        # 5. Clustering visualizations
        if clustering_analysis:
            print("  🎯 Creating clustering visualizations...")
            clustering_files = self.create_clustering_visualizations(clustering_analysis, str(output_path))
            visualization_files.update(clustering_files)
        
        # 6. Structural analysis visualizations
        if structural_analysis:
            print("  🔗 Creating structural analysis visualizations...")
            structural_files = self.create_structural_visualizations(structural_analysis, str(output_path))
            visualization_files.update(structural_files)
        
        # 7. Interactive dashboard (if Plotly available)
        if PLOTLY_AVAILABLE:
            print("  📱 Creating interactive dashboard...")
            dashboard_file = self.create_interactive_dashboard(
                lattice_dataset, clustering_analysis, structural_analysis, 
                str(output_path / "interactive_dashboard.html")
            )
            visualization_files['interactive_dashboard'] = dashboard_file
        
        print("\n✅ Visualization suite complete!")
        print(f"  Generated {len(visualization_files)} visualizations")
        
        return visualization_files
    
    def create_lattice_diagram(self, lattice_dataset: LatticeDataset, output_path: str) -> str:
        """Create main lattice diagram with nodes and connections"""
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Extract node positions and properties
        x_positions = []
        y_positions = []
        node_sizes = []
        node_colors = []
        node_labels = []
        
        for node_id, node in lattice_dataset.nodes.items():
            x_positions.append(node.coordinate.cycle_position)
            y_positions.append(node.coordinate.timeframe_level)
            node_sizes.append(node.size)
            
            # Get color from pattern family
            color = self.config.node_colors.get(node.dominant_event_type, '#9370DB')
            node_colors.append(color)
            
            node_labels.append(f"{node_id}\n{node.event_count} events")
        
        # Plot nodes
        ax.scatter(x_positions, y_positions, s=node_sizes, c=node_colors, 
                           alpha=0.7, edgecolors='white', linewidth=1)
        
        # Plot connections
        for _conn_id, conn in lattice_dataset.connections.items():
            source_node = lattice_dataset.nodes[conn.source_node_id]
            target_node = lattice_dataset.nodes[conn.target_node_id]
            
            x1, y1 = source_node.coordinate.cycle_position, source_node.coordinate.timeframe_level
            x2, y2 = target_node.coordinate.cycle_position, target_node.coordinate.timeframe_level
            
            # Connection color based on type
            conn_color = self.config.connection_colors.get(conn.connection_type, '#87CEEB')
            line_width = conn.line_width
            
            # Plot connection
            ax.plot([x1, x2], [y1, y2], color=conn_color, linewidth=line_width, 
                   alpha=0.6, linestyle='-')
            
            # Add arrow for direction
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops={'arrowstyle': '->', 'color': conn_color, 'lw': 1, 'alpha': 0.8})
        
        # Highlight hot zones
        for _zone_id, zone in lattice_dataset.hot_zones.items():
            y_min, y_max = zone.timeframe_range
            x_min, x_max = zone.position_range
            
            # Create hot zone rectangle
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor=self.config.hot_zone_color,
                                   facecolor=self.config.hot_zone_color, alpha=0.2)
            ax.add_patch(rect)
        
        # Customize plot
        ax.set_xlabel('Cycle Position (0.0 = Start, 1.0 = End)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Timeframe Level', fontsize=12, fontweight='bold')
        ax.set_title('IRONFORGE Market Archaeology Lattice\nTimeframe × Cycle Position', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set timeframe level labels
        timeframe_labels = ['Monthly', 'Weekly', 'Daily', '1H', '50M', '15M', '5M', '1M']
        ax.set_yticks(range(len(timeframe_labels)))
        ax.set_yticklabels(timeframe_labels)
        
        # Grid
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.5, len(timeframe_labels) - 0.5)
        
        # Legend for node colors
        legend_elements = []
        for pattern_family, color in self.config.node_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, 
                                            label=pattern_family.replace('_', ' ').title()))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                 title='Event Families', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"    Lattice diagram saved to {output_path}")
        return output_path
    
    def create_temporal_heatmaps(self, lattice_dataset: LatticeDataset, output_dir: str) -> dict[str, str]:
        """Create temporal heatmaps for different dimensions"""
        
        output_path = Path(output_dir)
        heatmap_files = {}
        
        # 1. Absolute time heatmap
        absolute_time_data = defaultdict(int)
        relative_position_data = defaultdict(int)
        timeframe_position_data = defaultdict(lambda: defaultdict(int))
        
        # Collect data from nodes
        for _node_id, node in lattice_dataset.nodes.items():
            for event in node.events:
                # Absolute time (session minute buckets)
                time_bucket = int(event.session_minute / 10) * 10
                absolute_time_data[time_bucket] += 1
                
                # Relative position
                pos_bucket = int(event.relative_cycle_position * 20) / 20
                relative_position_data[pos_bucket] += 1
                
                # Timeframe vs position
                timeframe_position_data[event.timeframe.value][pos_bucket] += 1
        
        # Create absolute time heatmap
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.config.dpi)
        
        time_buckets = sorted(absolute_time_data.keys())
        event_counts = [absolute_time_data[bucket] for bucket in time_buckets]
        
        bars = ax.bar(time_buckets, event_counts, width=8, alpha=0.7, 
                     color=self.config.energy_color, edgecolor='white')
        
        ax.set_xlabel('Session Minute', fontsize=12, fontweight='bold')
        ax.set_ylabel('Event Count', fontsize=12, fontweight='bold')
        ax.set_title('Temporal Distribution - Absolute Time Heatmap', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        absolute_time_file = str(output_path / "absolute_time_heatmap.png")
        plt.tight_layout()
        plt.savefig(absolute_time_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        heatmap_files['absolute_time_heatmap'] = absolute_time_file
        
        # Create relative position heatmap
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.config.dpi)
        
        pos_buckets = sorted(relative_position_data.keys())
        event_counts = [relative_position_data[bucket] for bucket in pos_buckets]
        
        bars = ax.bar(pos_buckets, event_counts, width=0.04, alpha=0.7,
                     color=self.config.cascade_color, edgecolor='white')
        
        ax.set_xlabel('Relative Cycle Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Event Count', fontsize=12, fontweight='bold')
        ax.set_title('Temporal Distribution - Relative Position Heatmap', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        relative_position_file = str(output_path / "relative_position_heatmap.png")
        plt.tight_layout()
        plt.savefig(relative_position_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        heatmap_files['relative_position_heatmap'] = relative_position_file
        
        # Create timeframe vs position 2D heatmap
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.config.dpi)
        
        # Prepare 2D data
        timeframes = sorted(timeframe_position_data.keys())
        positions = sorted({pos for tf_data in timeframe_position_data.values() for pos in tf_data})
        
        heatmap_matrix = np.zeros((len(timeframes), len(positions)))
        
        for i, tf in enumerate(timeframes):
            for j, pos in enumerate(positions):
                heatmap_matrix[i, j] = timeframe_position_data[tf].get(pos, 0)
        
        # Create heatmap
        im = ax.imshow(heatmap_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Set labels
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([f'{pos:.1f}' for pos in positions], rotation=45)
        ax.set_yticks(range(len(timeframes)))
        ax.set_yticklabels(timeframes)
        
        ax.set_xlabel('Relative Cycle Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Timeframe', fontsize=12, fontweight='bold')
        ax.set_title('Event Density Heatmap - Timeframe × Position', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Event Count', fontsize=10)
        
        # Add text annotations
        for i in range(len(timeframes)):
            for j in range(len(positions)):
                if heatmap_matrix[i, j] > 0:
                    ax.text(j, i, int(heatmap_matrix[i, j]), ha="center", va="center",
                                 color="white" if heatmap_matrix[i, j] > np.max(heatmap_matrix) * 0.5 else "black")
        
        timeframe_position_file = str(output_path / "timeframe_position_heatmap.png")
        plt.tight_layout()
        plt.savefig(timeframe_position_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        heatmap_files['timeframe_position_heatmap'] = timeframe_position_file
        
        print(f"    Created {len(heatmap_files)} temporal heatmaps")
        return heatmap_files
    
    def create_hot_zone_visualization(self, lattice_dataset: LatticeDataset, output_path: str) -> str:
        """Create hot zone visualization with detailed analysis"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=self.config.dpi)
        
        # Left plot: Hot zones on lattice
        # Plot all nodes as background
        for _node_id, node in lattice_dataset.nodes.items():
            x, y = node.coordinate.cycle_position, node.coordinate.timeframe_level
            color = 'gray' if not node.hot_zone_member else self.config.hot_zone_color
            alpha = 0.3 if not node.hot_zone_member else 0.8
            ax1.scatter(x, y, s=node.size, c=color, alpha=alpha, edgecolors='white')
        
        # Highlight hot zones
        for zone_id, zone in lattice_dataset.hot_zones.items():
            y_min, y_max = zone.timeframe_range
            x_min, x_max = zone.position_range
            
            # Hot zone boundary
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=3, edgecolor=self.config.hot_zone_color,
                                   facecolor='none', linestyle='--')
            ax1.add_patch(rect)
            
            # Zone label
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            ax1.text(center_x, center_y, f'HZ-{zone_id[-1]}', 
                    ha='center', va='center', fontweight='bold', 
                    bbox={'boxstyle': 'round,pad=0.3', 'facecolor': self.config.hot_zone_color, 'alpha': 0.7})
        
        # Customize left plot
        ax1.set_xlabel('Cycle Position', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Timeframe Level', fontsize=12, fontweight='bold')
        ax1.set_title('Hot Zone Identification', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        timeframe_labels = ['Monthly', 'Weekly', 'Daily', '1H', '50M', '15M', '5M', '1M']
        ax1.set_yticks(range(len(timeframe_labels)))
        ax1.set_yticklabels(timeframe_labels)
        
        # Right plot: Hot zone statistics
        if lattice_dataset.hot_zones:
            zone_names = []
            event_densities = []
            significance_scores = []
            recurrence_rates = []
            
            for zone_id, zone in lattice_dataset.hot_zones.items():
                zone_names.append(f'HZ-{zone_id[-3:]}')
                event_densities.append(zone.event_density)
                significance_scores.append(zone.average_significance)
                recurrence_rates.append(zone.recurrence_frequency)
            
            x_pos = np.arange(len(zone_names))
            width = 0.25
            
            bars1 = ax2.bar(x_pos - width, event_densities, width, label='Event Density', 
                           color=self.config.hot_zone_color, alpha=0.7)
            bars2 = ax2.bar(x_pos, significance_scores, width, label='Avg Significance', 
                           color=self.config.cascade_color, alpha=0.7)
            bars3 = ax2.bar(x_pos + width, recurrence_rates, width, label='Recurrence Rate', 
                           color=self.config.energy_color, alpha=0.7)
            
            ax2.set_xlabel('Hot Zones', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax2.set_title('Hot Zone Statistics', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(zone_names, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No Hot Zones Detected', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=16, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"    Hot zone visualization saved to {output_path}")
        return output_path
    
    def create_network_visualization(self, lattice_dataset: LatticeDataset, output_path: str) -> str:
        """Create network visualization showing connections between nodes"""
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Use force-directed layout for better network visualization
        import networkx as nx
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        pos = {}
        node_sizes = []
        node_colors = []
        
        for node_id, node in lattice_dataset.nodes.items():
            G.add_node(node_id)
            pos[node_id] = (node.coordinate.cycle_position, node.coordinate.timeframe_level)
            node_sizes.append(node.size)
            
            # Color based on pattern family
            color = self.config.node_colors.get(node.dominant_event_type, '#9370DB')
            node_colors.append(color)
        
        # Add edges
        edge_colors = []
        edge_widths = []
        
        for _conn_id, conn in lattice_dataset.connections.items():
            G.add_edge(conn.source_node_id, conn.target_node_id)
            
            conn_color = self.config.connection_colors.get(conn.connection_type, '#87CEEB')
            edge_colors.append(conn_color)
            edge_widths.append(conn.line_width)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                              alpha=0.8, ax=ax)
        
        # Draw edges with different colors and widths
        edges = G.edges()
        for i, (u, v) in enumerate(edges):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color=edge_colors[i], 
                   linewidth=edge_widths[i], alpha=0.6)
            
            # Add arrow
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops={'arrowstyle': '->', 'color': edge_colors[i], 
                                         'lw': edge_widths[i], 'alpha': 0.8})
        
        # Customize plot
        ax.set_xlabel('Cycle Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Timeframe Level', fontsize=12, fontweight='bold')
        ax.set_title('Network Structure - Node Connections and Information Flow', 
                    fontsize=14, fontweight='bold')
        
        timeframe_labels = ['Monthly', 'Weekly', 'Daily', '1H', '50M', '15M', '5M', '1M']
        ax.set_yticks(range(len(timeframe_labels)))
        ax.set_yticklabels(timeframe_labels)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.5, len(timeframe_labels) - 0.5)
        
        # Create legend for connection types
        legend_elements = []
        for conn_type, color in self.config.connection_colors.items():
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                            label=conn_type.replace('_', ' ').title()))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                 title='Connection Types', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"    Network visualization saved to {output_path}")
        return output_path
    
    def create_clustering_visualizations(self, clustering_analysis: ClusteringAnalysis, output_dir: str) -> dict[str, str]:
        """Create visualizations for clustering analysis results"""
        
        output_path = Path(output_dir)
        clustering_files = {}
        
        # 1. Cluster distribution pie chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.config.dpi)
        
        # Cluster types distribution
        cluster_types = [cluster.cluster_type.value for cluster in clustering_analysis.clusters]
        type_counts = Counter(cluster_types)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
        wedges, texts, autotexts = ax1.pie(type_counts.values(), labels=type_counts.keys(), 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Cluster Types Distribution', fontsize=12, fontweight='bold')
        
        # Cluster quality distribution
        quality_ranges = clustering_analysis.cluster_quality_distribution
        ax2.bar(quality_ranges.keys(), quality_ranges.values(), 
               color=[self.config.energy_color, self.config.cascade_color, 
                     self.config.hot_zone_color, '#9370DB'])
        ax2.set_title('Cluster Quality Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Quality Level')
        ax2.set_ylabel('Number of Clusters')
        ax2.tick_params(axis='x', rotation=45)
        
        # Recurrence rate histogram
        recurrence_rates = [cluster.recurrence_rate for cluster in clustering_analysis.clusters]
        ax3.hist(recurrence_rates, bins=20, alpha=0.7, color=self.config.cascade_color, 
                edgecolor='white')
        ax3.set_title('Cluster Recurrence Rates', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Recurrence Rate')
        ax3.set_ylabel('Number of Clusters')
        ax3.grid(True, alpha=0.3)
        
        # Temporal stability vs significance scatter
        significances = [cluster.average_significance for cluster in clustering_analysis.clusters]
        stabilities = [cluster.temporal_stability for cluster in clustering_analysis.clusters]
        
        ax4.scatter(significances, stabilities, s=50, alpha=0.6, 
                            c=self.config.hot_zone_color, edgecolors='white')
        ax4.set_xlabel('Average Significance')
        ax4.set_ylabel('Temporal Stability')
        ax4.set_title('Cluster Quality Metrics', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        clustering_overview_file = str(output_path / "clustering_overview.png")
        plt.tight_layout()
        plt.savefig(clustering_overview_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        clustering_files['clustering_overview'] = clustering_overview_file
        
        # 2. Temporal patterns visualization
        if clustering_analysis.temporal_heatmap_data:
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=self.config.dpi)
            
            # Absolute time patterns
            abs_time_data = clustering_analysis.temporal_heatmap_data.get('absolute_time', {})
            if abs_time_data:
                times = sorted(abs_time_data.keys())
                counts = [abs_time_data[t] for t in times]
                axes[0].bar(times, counts, width=8, alpha=0.7, color=self.config.energy_color)
                axes[0].set_title('Temporal Clustering - Absolute Time Patterns', fontweight='bold')
                axes[0].set_xlabel('Session Minute')
                axes[0].set_ylabel('Cluster Count')
                axes[0].grid(True, alpha=0.3)
            
            # Relative position patterns
            rel_pos_data = clustering_analysis.temporal_heatmap_data.get('relative_position', {})
            if rel_pos_data:
                positions = sorted(rel_pos_data.keys())
                counts = [rel_pos_data[p] for p in positions]
                axes[1].bar(positions, counts, width=0.04, alpha=0.7, color=self.config.cascade_color)
                axes[1].set_title('Temporal Clustering - Relative Position Patterns', fontweight='bold')
                axes[1].set_xlabel('Relative Cycle Position')
                axes[1].set_ylabel('Cluster Count')
                axes[1].grid(True, alpha=0.3)
            
            # Session phase patterns
            phase_data = clustering_analysis.temporal_heatmap_data.get('session_phase', {})
            if phase_data:
                phases = list(phase_data.keys())
                counts = list(phase_data.values())
                axes[2].bar(phases, counts, alpha=0.7, color=self.config.hot_zone_color)
                axes[2].set_title('Temporal Clustering - Session Phase Patterns', fontweight='bold')
                axes[2].set_xlabel('Session Phase')
                axes[2].set_ylabel('Cluster Count')
                axes[2].tick_params(axis='x', rotation=45)
                axes[2].grid(True, alpha=0.3)
            
            temporal_patterns_file = str(output_path / "temporal_clustering_patterns.png")
            plt.tight_layout()
            plt.savefig(temporal_patterns_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            clustering_files['temporal_patterns'] = temporal_patterns_file
        
        print(f"    Created {len(clustering_files)} clustering visualizations")
        return clustering_files
    
    def create_structural_visualizations(self, structural_analysis: StructuralAnalysis, output_dir: str) -> dict[str, str]:
        """Create visualizations for structural analysis results"""
        
        output_path = Path(output_dir)
        structural_files = {}
        
        # 1. Cascade analysis visualization
        if structural_analysis.cascade_chains:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.config.dpi)
            
            # Cascade length distribution
            cascade_lengths = [len(cascade.events) for cascade in structural_analysis.cascade_chains]
            ax1.hist(cascade_lengths, bins=min(10, max(cascade_lengths)), alpha=0.7, 
                    color=self.config.cascade_color, edgecolor='white')
            ax1.set_title('Cascade Chain Length Distribution', fontweight='bold')
            ax1.set_xlabel('Number of Events in Cascade')
            ax1.set_ylabel('Number of Cascades')
            ax1.grid(True, alpha=0.3)
            
            # Energy efficiency vs cascade length
            efficiencies = [cascade.energy_efficiency for cascade in structural_analysis.cascade_chains]
            ax2.scatter(cascade_lengths, efficiencies, s=50, alpha=0.6, 
                       c=self.config.cascade_color, edgecolors='white')
            ax2.set_xlabel('Cascade Length')
            ax2.set_ylabel('Energy Efficiency')
            ax2.set_title('Cascade Efficiency Analysis', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Cascade coherence histogram
            coherences = [cascade.chain_coherence for cascade in structural_analysis.cascade_chains]
            ax3.hist(coherences, bins=15, alpha=0.7, color=self.config.hot_zone_color, 
                    edgecolor='white')
            ax3.set_title('Cascade Coherence Distribution', fontweight='bold')
            ax3.set_xlabel('Chain Coherence')
            ax3.set_ylabel('Number of Cascades')
            ax3.grid(True, alpha=0.3)
            
            # Completion probability vs risk
            completion_probs = [cascade.completion_probability for cascade in structural_analysis.cascade_chains]
            risk_scores = [cascade.risk_assessment for cascade in structural_analysis.cascade_chains]
            ax4.scatter(completion_probs, risk_scores, s=50, alpha=0.6, 
                       c=self.config.energy_color, edgecolors='white')
            ax4.set_xlabel('Completion Probability')
            ax4.set_ylabel('Risk Assessment')
            ax4.set_title('Cascade Risk Analysis', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            cascade_analysis_file = str(output_path / "cascade_analysis.png")
            plt.tight_layout()
            plt.savefig(cascade_analysis_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            structural_files['cascade_analysis'] = cascade_analysis_file
        
        # 2. Energy accumulation visualization
        if structural_analysis.energy_accumulations:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.config.dpi)
            
            # Energy density distribution
            densities = [ea.energy_density for ea in structural_analysis.energy_accumulations]
            ax1.hist(densities, bins=15, alpha=0.7, color=self.config.energy_color, 
                    edgecolor='white')
            ax1.set_title('Energy Density Distribution', fontweight='bold')
            ax1.set_xlabel('Energy Density')
            ax1.set_ylabel('Number of Accumulations')
            ax1.grid(True, alpha=0.3)
            
            # Release probability vs energy level
            release_probs = [ea.release_probability for ea in structural_analysis.energy_accumulations]
            energy_levels = [ea.current_energy_level for ea in structural_analysis.energy_accumulations]
            ax2.scatter(energy_levels, release_probs, s=50, alpha=0.6, 
                       c=self.config.energy_color, edgecolors='white')
            ax2.set_xlabel('Current Energy Level')
            ax2.set_ylabel('Release Probability')
            ax2.set_title('Energy Release Risk', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Accumulation rate histogram
            acc_rates = [ea.accumulation_rate for ea in structural_analysis.energy_accumulations]
            ax3.hist(acc_rates, bins=15, alpha=0.7, color=self.config.cascade_color, 
                    edgecolor='white')
            ax3.set_title('Accumulation Rate Distribution', fontweight='bold')
            ax3.set_xlabel('Accumulation Rate')
            ax3.set_ylabel('Number of Zones')
            ax3.grid(True, alpha=0.3)
            
            # Release efficiency vs magnitude
            release_effs = [ea.release_efficiency for ea in structural_analysis.energy_accumulations if ea.release_efficiency > 0]
            release_mags = [ea.release_magnitude for ea in structural_analysis.energy_accumulations if ea.release_efficiency > 0]
            
            if release_effs and release_mags:
                ax4.scatter(release_mags, release_effs, s=50, alpha=0.6, 
                           c=self.config.hot_zone_color, edgecolors='white')
                ax4.set_xlabel('Release Magnitude')
                ax4.set_ylabel('Release Efficiency')
                ax4.set_title('Energy Release Characteristics', fontweight='bold')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Release Events', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=14, fontweight='bold')
            
            energy_analysis_file = str(output_path / "energy_analysis.png")
            plt.tight_layout()
            plt.savefig(energy_analysis_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            structural_files['energy_analysis'] = energy_analysis_file
        
        # 3. Timeframe interaction matrix
        if hasattr(structural_analysis, 'timeframe_interaction_matrix'):
            fig, ax = plt.subplots(figsize=(10, 8), dpi=self.config.dpi)
            
            interaction_matrix = structural_analysis.timeframe_interaction_matrix
            timeframe_labels = ['Monthly', 'Weekly', 'Daily', '1H', '50M', '15M', '5M', '1M']
            
            im = ax.imshow(interaction_matrix, cmap='viridis', aspect='auto')
            
            ax.set_xticks(range(len(timeframe_labels)))
            ax.set_xticklabels(timeframe_labels, rotation=45)
            ax.set_yticks(range(len(timeframe_labels)))
            ax.set_yticklabels(timeframe_labels)
            
            ax.set_xlabel('Target Timeframe', fontsize=12, fontweight='bold')
            ax.set_ylabel('Source Timeframe', fontsize=12, fontweight='bold')
            ax.set_title('Timeframe Interaction Matrix\n(Structural Link Strength)', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Total Link Strength', fontsize=10)
            
            # Add text annotations for non-zero values
            for i in range(interaction_matrix.shape[0]):
                for j in range(interaction_matrix.shape[1]):
                    if interaction_matrix[i, j] > 0.1:
                        ax.text(j, i, f'{interaction_matrix[i, j]:.1f}', 
                                     ha="center", va="center", color="white", fontweight='bold')
            
            interaction_matrix_file = str(output_path / "timeframe_interaction_matrix.png")
            plt.tight_layout()
            plt.savefig(interaction_matrix_file, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            structural_files['interaction_matrix'] = interaction_matrix_file
        
        print(f"    Created {len(structural_files)} structural visualizations")
        return structural_files
    
    def create_interactive_dashboard(self, 
                                   lattice_dataset: LatticeDataset,
                                   clustering_analysis: ClusteringAnalysis | None = None,
                                   structural_analysis: StructuralAnalysis | None = None,
                                   output_path: str = "interactive_dashboard.html") -> str:
        """Create interactive dashboard using Plotly"""
        
        if not PLOTLY_AVAILABLE:
            print("    Plotly not available - skipping interactive dashboard")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Main Lattice View', 'Temporal Heatmap', 
                          'Network Analysis', 'Statistical Overview'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Main lattice scatter plot
        x_positions = []
        y_positions = []
        node_sizes = []
        node_colors = []
        hover_texts = []
        
        for node_id, node in lattice_dataset.nodes.items():
            x_positions.append(node.coordinate.cycle_position)
            y_positions.append(node.coordinate.timeframe_level)
            node_sizes.append(node.size)
            node_colors.append(node.average_significance)
            
            hover_text = (f"Node: {node_id}<br>"
                         f"Events: {node.event_count}<br>"
                         f"Significance: {node.average_significance:.3f}<br>"
                         f"Type: {node.dominant_event_type}<br>"
                         f"Hot Zone: {node.hot_zone_member}")
            hover_texts.append(hover_text)
        
        lattice_scatter = go.Scatter(
            x=x_positions, y=y_positions,
            mode='markers',
            marker={'size': node_sizes, 'color': node_colors, 'colorscale': 'viridis',
                       'showscale': True, 'colorbar': {'title': "Significance"}},
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name='Lattice Nodes'
        )
        
        fig.add_trace(lattice_scatter, row=1, col=1)
        
        # 2. Temporal heatmap
        if clustering_analysis and clustering_analysis.temporal_heatmap_data:
            abs_time_data = clustering_analysis.temporal_heatmap_data.get('absolute_time', {})
            times = sorted(abs_time_data.keys())
            counts = [abs_time_data[t] for t in times]
            
            temporal_bar = go.Bar(x=times, y=counts, name='Event Frequency', 
                                marker_color=self.config.energy_color)
            fig.add_trace(temporal_bar, row=1, col=2)
        
        # 3. Network visualization (simplified)
        if lattice_dataset.connections:
            edge_x = []
            edge_y = []
            
            for _conn_id, conn in lattice_dataset.connections.items():
                if conn.source_node_id in lattice_dataset.nodes and conn.target_node_id in lattice_dataset.nodes:
                    source_node = lattice_dataset.nodes[conn.source_node_id]
                    target_node = lattice_dataset.nodes[conn.target_node_id]
                    
                    edge_x.extend([source_node.coordinate.cycle_position, 
                                  target_node.coordinate.cycle_position, None])
                    edge_y.extend([source_node.coordinate.timeframe_level, 
                                  target_node.coordinate.timeframe_level, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                  line={'width': 1, 'color': 'lightblue'},
                                  hoverinfo='none', showlegend=False)
            fig.add_trace(edge_trace, row=2, col=1)
            
            # Add nodes again for network view
            network_scatter = go.Scatter(
                x=x_positions, y=y_positions,
                mode='markers',
                marker={'size': 10, 'color': 'red'},
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_trace(network_scatter, row=2, col=1)
        
        # 4. Statistical overview
        if structural_analysis:
            stats_labels = ['Network Density', 'Avg Path Length', 'Clustering Coeff']
            stats_values = [structural_analysis.network_density,
                           structural_analysis.average_path_length / 10,  # Normalize
                           structural_analysis.clustering_coefficient]
            
            stats_bar = go.Bar(x=stats_labels, y=stats_values, name='Network Stats',
                             marker_color=self.config.cascade_color)
            fig.add_trace(stats_bar, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="IRONFORGE Market Archaeology - Interactive Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update subplot titles and axes
        fig.update_xaxes(title_text="Cycle Position", row=1, col=1)
        fig.update_yaxes(title_text="Timeframe Level", row=1, col=1)
        
        fig.update_xaxes(title_text="Session Minute", row=1, col=2)
        fig.update_yaxes(title_text="Event Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Cycle Position", row=2, col=1)
        fig.update_yaxes(title_text="Timeframe Level", row=2, col=1)
        
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        # Save interactive dashboard
        pyo.plot(fig, filename=output_path, auto_open=False)
        
        print(f"    Interactive dashboard saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Test the lattice visualizer
    print("🎨 Testing Lattice Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = LatticeVisualizer()
    
    print("✅ Lattice visualizer initialized and ready for use")
    print("   Use create_comprehensive_visualization() with lattice dataset to generate visualizations")