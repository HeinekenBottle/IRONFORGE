#!/usr/bin/env python3
"""
IRONFORGE Pattern Correlation Visualizer
========================================

Advanced visualization system for pattern correlation analysis.
Replaces generic statistical charts with actionable temporal relationship visualizations.

Features:
- Temporal clustering analysis
- Session boundary pattern detection  
- Sequential pattern visualization
- Co-occurrence heatmaps
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PatternCorrelationVisualizer:
    """Advanced pattern correlation visualization system"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up advanced plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'fvg': '#FF6B6B',           # Red for FVG events
            'expansion': '#4ECDC4',      # Teal for expansion
            'consolidation': '#45B7D1',  # Blue for consolidation
            'retracement': '#FFA07A',    # Light salmon for retracement
            'reversal': '#9B59B6',       # Purple for reversal
            'session_boundary': '#2C3E50' # Dark blue for boundaries
        }
    
    def generate_correlation_visualizations(self, analysis_results: list[dict]) -> None:
        """Generate all pattern correlation visualizations"""
        
        print("🎨 Generating pattern correlation visualizations...")
        
        # Convert results to DataFrame for analysis
        df = self._prepare_correlation_dataframe(analysis_results)
        
        if df.empty:
            print("⚠️ No data available for correlation analysis")
            return
        
        # 1. Temporal Clustering Analysis
        self._create_temporal_clustering_chart(df)
        
        # 2. Session Boundary Pattern Detection
        self._create_session_boundary_analysis(df)
        
        # 3. Sequential Pattern Visualization
        self._create_sequential_pattern_chart(df)
        
        # 4. Co-occurrence Heatmaps
        self._create_cooccurrence_heatmaps(df)
        
        # 5. Market Cycle Progression Chart
        self._create_market_cycle_progression(df)
        
        print(f"✅ Pattern correlation visualizations saved to {self.output_dir}/")
    
    def _prepare_correlation_dataframe(self, analysis_results: list[dict]) -> pd.DataFrame:
        """Prepare data for correlation analysis"""
        
        correlation_data = []
        
        for result in analysis_results:
            session_name = result['session_metadata'].get('session_name', 'unknown')
            session_date = result['session_metadata'].get('session_date', 'unknown')
            
            # Extract semantic events with timing
            semantic_patterns = result.get('semantic_patterns', {})
            semantic_events = semantic_patterns.get('semantic_events', {})
            
            # Create time-based event records
            base_record = {
                'session_name': session_name,
                'session_date': session_date,
                'session_file': result['session_file'],
                'total_nodes': result['graph_statistics']['total_nodes']
            }
            
            # Add semantic event counts
            for event_type, count in semantic_events.items():
                base_record[event_type] = count
                base_record[f'{event_type}_rate'] = (count / base_record['total_nodes']) * 100 if base_record['total_nodes'] > 0 else 0
            
            correlation_data.append(base_record)
        
        return pd.DataFrame(correlation_data)
    
    def _create_temporal_clustering_chart(self, df: pd.DataFrame) -> None:
        """Create temporal clustering analysis showing FVG-Expansion relationships"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Clustering Analysis: Event Co-occurrence Patterns', fontsize=16, fontweight='bold')
        
        # 1. FVG vs Expansion Scatter with Clustering
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df['fvg_redelivery'], df['expansion_phase'], 
                            c=df['total_nodes'], cmap='viridis', alpha=0.7, s=100)
        ax1.set_xlabel('FVG Redelivery Events')
        ax1.set_ylabel('Expansion Phase Events')
        ax1.set_title('FVG-Expansion Event Clustering')
        plt.colorbar(scatter, ax=ax1, label='Total Nodes')
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['fvg_redelivery'], df['expansion_phase'], 1)
            p = np.poly1d(z)
            ax1.plot(df['fvg_redelivery'], p(df['fvg_redelivery']), "r--", alpha=0.8)
        
        # 2. Retracement vs Reversal Correlation
        ax2 = axes[0, 1]
        if 'retracement' in df.columns and 'reversal' in df.columns:
            scatter2 = ax2.scatter(df['retracement'], df['reversal'], 
                                 c=df['consolidation'], cmap='plasma', alpha=0.7, s=100)
            ax2.set_xlabel('Retracement Events')
            ax2.set_ylabel('Reversal Events')
            ax2.set_title('Retracement-Reversal Correlation')
            plt.colorbar(scatter2, ax=ax2, label='Consolidation Events')
        else:
            ax2.text(0.5, 0.5, 'Retracement/Reversal data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Retracement-Reversal Correlation (Pending)')
        
        # 3. Session Type Event Distribution
        ax3 = axes[1, 0]
        session_summary = df.groupby('session_name').agg({
            'fvg_redelivery': 'mean',
            'expansion_phase': 'mean',
            'consolidation': 'mean'
        }).reset_index()
        
        x = np.arange(len(session_summary))
        width = 0.25
        
        ax3.bar(x - width, session_summary['fvg_redelivery'], width, 
               label='FVG Redelivery', color=self.colors['fvg'], alpha=0.8)
        ax3.bar(x, session_summary['expansion_phase'], width, 
               label='Expansion', color=self.colors['expansion'], alpha=0.8)
        ax3.bar(x + width, session_summary['consolidation'], width, 
               label='Consolidation', color=self.colors['consolidation'], alpha=0.8)
        
        ax3.set_xlabel('Session Types')
        ax3.set_ylabel('Average Events per Session')
        ax3.set_title('Event Distribution by Session Type')
        ax3.set_xticks(x)
        ax3.set_xticklabels(session_summary['session_name'], rotation=45)
        ax3.legend()
        
        # 4. Temporal Density Heatmap
        ax4 = axes[1, 1]
        
        # Create correlation matrix for semantic events
        semantic_cols = ['fvg_redelivery', 'expansion_phase', 'consolidation']
        if 'retracement' in df.columns:
            semantic_cols.append('retracement')
        if 'reversal' in df.columns:
            semantic_cols.append('reversal')
        
        correlation_matrix = df[semantic_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, ax=ax4, cbar_kws={'label': 'Correlation Coefficient'})
        ax4.set_title('Semantic Event Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_session_boundary_analysis(self, df: pd.DataFrame) -> None:
        """Create session boundary pattern detection visualization"""
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Session Boundary Pattern Detection', fontsize=16, fontweight='bold')
        
        # 1. Session Start/End Pattern Analysis
        ax1 = axes[0]
        
        # Group by session type and calculate boundary patterns
        session_groups = df.groupby('session_name')
        
        session_types = []
        start_patterns = []
        end_patterns = []
        
        for session_type, group in session_groups:
            session_types.append(session_type)
            # Simulate boundary pattern detection (would use actual timing data in real implementation)
            start_pattern_strength = group['expansion_phase'].mean() + group['fvg_redelivery'].mean() * 0.5
            end_pattern_strength = group['consolidation'].mean() + (group.get('retracement', pd.Series([0])).mean() * 0.3)
            
            start_patterns.append(start_pattern_strength)
            end_patterns.append(end_pattern_strength)
        
        x = np.arange(len(session_types))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, start_patterns, width, label='Session Start Patterns', 
                       color=self.colors['expansion'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, end_patterns, width, label='Session End Patterns', 
                       color=self.colors['consolidation'], alpha=0.8)
        
        ax1.set_xlabel('Session Types')
        ax1.set_ylabel('Pattern Strength Score')
        ax1.set_title('Session Boundary Pattern Strength by Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(session_types, rotation=45)
        ax1.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Time-based Pattern Evolution
        ax2 = axes[1]
        
        # Convert session_date to datetime for time series analysis
        df['session_datetime'] = pd.to_datetime(df['session_date'], errors='coerce')
        df_time = df.dropna(subset=['session_datetime']).sort_values('session_datetime')
        
        if len(df_time) > 0:
            # Plot evolution of different event types over time
            ax2.plot(df_time['session_datetime'], df_time['fvg_redelivery'], 
                    marker='o', label='FVG Redelivery', color=self.colors['fvg'], linewidth=2)
            ax2.plot(df_time['session_datetime'], df_time['expansion_phase'], 
                    marker='s', label='Expansion Phase', color=self.colors['expansion'], linewidth=2)
            ax2.plot(df_time['session_datetime'], df_time['consolidation'], 
                    marker='^', label='Consolidation', color=self.colors['consolidation'], linewidth=2)
            
            if 'retracement' in df_time.columns:
                ax2.plot(df_time['session_datetime'], df_time['retracement'], 
                        marker='d', label='Retracement', color=self.colors['retracement'], linewidth=2)
            
            if 'reversal' in df_time.columns:
                ax2.plot(df_time['session_datetime'], df_time['reversal'], 
                        marker='*', label='Reversal', color=self.colors['reversal'], linewidth=2)
            
            ax2.set_xlabel('Session Date')
            ax2.set_ylabel('Event Count')
            ax2.set_title('Pattern Evolution Over Time')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No valid date data available for time series analysis', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'session_boundary_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sequential_pattern_chart(self, df: pd.DataFrame) -> None:
        """Create sequential pattern visualization showing market cycle progression"""
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle('Sequential Market Cycle Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Create a flow diagram showing typical sequence patterns
        # This would ideally use actual temporal sequence data
        
        # Calculate average progression patterns
        avg_consolidation = df['consolidation'].mean()
        avg_expansion = df['expansion_phase'].mean()
        avg_fvg = df['fvg_redelivery'].mean()
        
        # Add retracement and reversal if available
        avg_retracement = df.get('retracement', pd.Series([0])).mean()
        avg_reversal = df.get('reversal', pd.Series([0])).mean()
        
        # Create flow visualization
        stages = ['Consolidation', 'Expansion', 'FVG Redelivery', 'Retracement', 'Reversal']
        values = [avg_consolidation, avg_expansion, avg_fvg, avg_retracement, avg_reversal]
        colors = [self.colors['consolidation'], self.colors['expansion'], 
                 self.colors['fvg'], self.colors['retracement'], self.colors['reversal']]
        
        # Create horizontal bar chart showing cycle progression
        y_pos = np.arange(len(stages))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stages)
        ax.set_xlabel('Average Events per Session')
        ax.set_title('Market Cycle Stage Frequency')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values, strict=False)):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}', ha='left', va='center', fontweight='bold')
        
        # Add flow arrows between stages
        for i in range(len(stages) - 1):
            ax.annotate('', xy=(max(values) * 0.8, i), xytext=(max(values) * 0.8, i + 1),
                       arrowprops={"arrowstyle": '->', "lw": 2, "color": 'gray', "alpha": 0.6})
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequential_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cooccurrence_heatmaps(self, df: pd.DataFrame) -> None:
        """Create co-occurrence heatmaps for semantic events"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Semantic Event Co-occurrence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Event Count Co-occurrence
        ax1 = axes[0]
        
        semantic_cols = ['fvg_redelivery', 'expansion_phase', 'consolidation']
        if 'retracement' in df.columns:
            semantic_cols.append('retracement')
        if 'reversal' in df.columns:
            semantic_cols.append('reversal')
        
        # Calculate co-occurrence matrix (correlation of event counts)
        cooccurrence_matrix = df[semantic_cols].corr()
        
        sns.heatmap(cooccurrence_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, ax=ax1, cbar_kws={'label': 'Co-occurrence Strength'})
        ax1.set_title('Event Count Co-occurrence Matrix')
        
        # 2. Session Type Co-occurrence Patterns
        ax2 = axes[1]
        
        # Create session type vs event type matrix
        session_event_matrix = df.groupby('session_name')[semantic_cols].mean()
        
        sns.heatmap(session_event_matrix.T, annot=True, cmap='YlOrRd', 
                   ax=ax2, cbar_kws={'label': 'Average Events'})
        ax2.set_title('Session Type vs Event Type Patterns')
        ax2.set_xlabel('Session Types')
        ax2.set_ylabel('Event Types')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cooccurrence_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_market_cycle_progression(self, df: pd.DataFrame) -> None:
        """Create market cycle progression visualization"""
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Complete Market Cycle Progression Analysis', fontsize=16, fontweight='bold')
        
        # Calculate cycle completeness for each session
        cycle_data = []
        
        for _, row in df.iterrows():
            cycle_score = 0
            cycle_components = []
            
            # Check for each cycle component
            if row['consolidation'] > 0:
                cycle_score += 1
                cycle_components.append('Consolidation')
            
            if row['expansion_phase'] > 0:
                cycle_score += 1
                cycle_components.append('Expansion')
            
            if row['fvg_redelivery'] > 0:
                cycle_score += 1
                cycle_components.append('FVG')
            
            if row.get('retracement', 0) > 0:
                cycle_score += 1
                cycle_components.append('Retracement')
            
            if row.get('reversal', 0) > 0:
                cycle_score += 1
                cycle_components.append('Reversal')
            
            cycle_data.append({
                'session': f"{row['session_name']}_{row['session_date']}",
                'cycle_completeness': cycle_score / 5.0,  # Normalize to 0-1
                'components': cycle_components,
                'total_events': row['fvg_redelivery'] + row['expansion_phase'] + row['consolidation']
            })
        
        cycle_df = pd.DataFrame(cycle_data)
        
        # Create scatter plot of cycle completeness vs total events
        scatter = ax.scatter(cycle_df['total_events'], cycle_df['cycle_completeness'], 
                           s=100, alpha=0.7, c=cycle_df['cycle_completeness'], 
                           cmap='viridis')
        
        ax.set_xlabel('Total Semantic Events')
        ax.set_ylabel('Market Cycle Completeness (0-1)')
        ax.set_title('Market Cycle Completeness vs Event Activity')
        
        plt.colorbar(scatter, ax=ax, label='Cycle Completeness')
        
        # Add trend line
        if len(cycle_df) > 1:
            z = np.polyfit(cycle_df['total_events'], cycle_df['cycle_completeness'], 1)
            p = np.poly1d(z)
            ax.plot(cycle_df['total_events'], p(cycle_df['total_events']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'market_cycle_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Generated 5 pattern correlation visualizations")
        print("   - Temporal clustering analysis")
        print("   - Session boundary pattern detection")
        print("   - Sequential pattern analysis")
        print("   - Co-occurrence heatmaps")
        print("   - Market cycle progression")
