#!/usr/bin/env python3
"""
IRONFORGE Temporal Visualization Manager
Display, plotting, and reporting functionality for temporal analysis
"""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class VisualizationManager:
    """Manages visualization, plotting, and reporting for temporal analysis"""
    
    def __init__(self):
        # Set up plotting style with fallback
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                # Use default style if seaborn not available
                pass

        try:
            sns.set_palette("husl")
        except Exception:
            # Continue without seaborn palette if not available
            pass
        
    def display_query_results(self, results: dict[str, Any]) -> None:
        """Display query results in a formatted manner"""
        query_type = results.get("query_type", "unknown")
        
        print(f"\n📊 {query_type.replace('_', ' ').title()} Results")
        print("=" * 60)
        
        if query_type == "temporal_sequence_with_relativity":
            self._display_temporal_sequence_results(results)
        elif query_type == "archaeological_zones":
            self._display_archaeological_zone_results(results)
        elif query_type == "theory_b_patterns":
            self._display_theory_b_results(results)
        elif query_type == "post_rd40_sequences":
            self._display_rd40_results(results)
        elif query_type == "opening_patterns_with_relativity":
            self._display_opening_pattern_results(results)
        else:
            self._display_generic_results(results)
            
    def _display_temporal_sequence_results(self, results: dict[str, Any]) -> None:
        """Display temporal sequence analysis results"""
        total_sessions = results.get("total_sessions", 0)
        matches = results.get("matches", [])
        probabilities = results.get("probabilities", {})
        
        print(f"📈 Analyzed {total_sessions} sessions")
        print(f"🎯 Found {len(matches)} matching patterns")
        
        if probabilities:
            print("\n📊 Pattern Probabilities:")
            for pattern, prob_data in probabilities.items():
                if isinstance(prob_data, dict):
                    prob = prob_data.get("probability", 0)
                    count = prob_data.get("count", 0)
                    print(f"  • {pattern}: {prob:.1%} ({count} occurrences)")
                else:
                    print(f"  • {pattern}: {prob_data:.1%}")
                    
        # Display top matches
        if matches:
            print(f"\n🔍 Top {min(5, len(matches))} Matches:")
            for i, match in enumerate(matches[:5]):
                session_id = match.get("session_id", "unknown")
                event_idx = match.get("event_index", 0)
                confidence = match.get("pattern_match", {}).get("confidence", 0)
                print(f"  {i+1}. {session_id} (Event {event_idx}) - Confidence: {confidence:.1%}")
                
        # Display insights
        insights = results.get("insights", [])
        if insights:
            print("\n💡 Key Insights:")
            for insight in insights[:3]:
                print(f"  • {insight}")
                
    def _display_archaeological_zone_results(self, results: dict[str, Any]) -> None:
        """Display archaeological zone analysis results"""
        total_sessions = results.get("total_sessions", 0)
        zone_analysis = results.get("zone_analysis", {})
        theory_b_events = results.get("theory_b_events", [])
        
        print(f"🏛️ Archaeological Zone Analysis ({total_sessions} sessions)")
        print(f"⚡ Found {len(theory_b_events)} Theory B events")
        
        # Zone distribution summary
        zone_counts = {}
        for _session_id, analysis in zone_analysis.items():
            zone_events = analysis.get("zone_events", {})
            for zone, count in zone_events.items():
                zone_counts[zone] = zone_counts.get(zone, 0) + count
                
        if zone_counts:
            print("\n📍 Zone Event Distribution:")
            for zone in sorted(zone_counts.keys()):
                count = zone_counts[zone]
                print(f"  • {zone} zone: {count} events")
                
        # Theory B event summary
        if theory_b_events:
            print("\n⚡ Theory B Events by Zone:")
            zone_theory_b = {}
            for event in theory_b_events:
                zone = f"{event.get('zone_percentage', 0)}%"
                zone_theory_b[zone] = zone_theory_b.get(zone, 0) + 1
                
            for zone, count in sorted(zone_theory_b.items()):
                print(f"  • {zone} zone: {count} Theory B events")
                
    def _display_theory_b_results(self, results: dict[str, Any]) -> None:
        """Display Theory B pattern analysis results"""
        precision_events = results.get("precision_events", [])
        non_locality_patterns = results.get("non_locality_patterns", [])
        temporal_correlations = results.get("temporal_correlations", {})
        
        print("⚡ Theory B Temporal Non-Locality Analysis")
        print(f"🎯 Precision Events: {len(precision_events)}")
        print(f"🌀 Non-Locality Patterns: {len(non_locality_patterns)}")
        
        # Precision event distribution
        if precision_events:
            precision_scores = [event.get("precision_score", 0) for event in precision_events]
            avg_precision = np.mean(precision_scores)
            print(f"\n📊 Average Precision Score: {avg_precision:.3f}")
            
            # Zone distribution of precision events
            zone_distribution = {}
            for event in precision_events:
                zone = f"{event.get('zone_percentage', 0)}%"
                zone_distribution[zone] = zone_distribution.get(zone, 0) + 1
                
            print("📍 Precision Events by Zone:")
            for zone, count in sorted(zone_distribution.items()):
                print(f"  • {zone}: {count} events")
                
        # Temporal correlations
        if temporal_correlations:
            print("\n🕒 Temporal Correlations:")
            for correlation_type, value in temporal_correlations.items():
                if isinstance(value, int | float):
                    print(f"  • {correlation_type}: {value:.3f}")
                    
    def _display_rd40_results(self, results: dict[str, Any]) -> None:
        """Display RD@40% sequence analysis results"""
        rd40_events = results.get("rd40_events", [])
        sequence_paths = results.get("sequence_paths", {})
        path_probabilities = results.get("path_probabilities", {})
        
        print("🎯 RD@40% Sequence Analysis")
        print(f"📍 RD@40% Events Found: {len(rd40_events)}")
        
        # Path distribution
        total_paths = sum(len(paths) for paths in sequence_paths.values())
        if total_paths > 0:
            print(f"🛤️ Total Classified Paths: {total_paths}")
            
            print("\n📊 Path Distribution:")
            for path_type, paths in sequence_paths.items():
                count = len(paths)
                percentage = (count / total_paths) * 100 if total_paths > 0 else 0
                print(f"  • {path_type}: {count} ({percentage:.1f}%)")
                
        # Path probabilities with confidence intervals
        if path_probabilities:
            print("\n📈 Path Probabilities (95% CI):")
            for path_type, prob_data in path_probabilities.items():
                prob = prob_data.get("probability", 0)
                ci = prob_data.get("confidence_interval", [0, 0])
                print(f"  • {path_type}: {prob:.1%} [{ci[0]:.1%} - {ci[1]:.1%}]")
                
    def _display_opening_pattern_results(self, results: dict[str, Any]) -> None:
        """Display opening pattern analysis results"""
        total_sessions = results.get("total_sessions", 0)
        opening_analysis = results.get("opening_analysis", {})
        pattern_distribution = results.get("pattern_distribution", {})
        
        print(f"🌅 Opening Pattern Analysis ({total_sessions} sessions)")
        
        # Pattern distribution
        if pattern_distribution:
            print("\n📊 Opening Pattern Distribution:")
            for pattern_type, data in pattern_distribution.items():
                if isinstance(data, dict):
                    count = data.get("count", 0)
                    percentage = data.get("percentage", 0)
                    print(f"  • {pattern_type}: {count} ({percentage:.1f}%)")
                else:
                    print(f"  • {pattern_type}: {data}")
                    
        # Session type breakdown
        session_types = {}
        for _session_id, analysis in opening_analysis.items():
            session_type = analysis.get("session_type", "UNKNOWN")
            session_types[session_type] = session_types.get(session_type, 0) + 1
            
        if session_types:
            print("\n🕒 Session Type Breakdown:")
            for session_type, count in sorted(session_types.items()):
                print(f"  • {session_type}: {count} sessions")
                
    def _display_generic_results(self, results: dict[str, Any]) -> None:
        """Display generic results for unknown query types"""
        print("📋 Analysis Results:")
        
        # Display key metrics
        for key, value in results.items():
            if key in ["total_sessions", "event_count", "match_count"]:
                print(f"  • {key.replace('_', ' ').title()}: {value}")
                
        # Display insights if available
        insights = results.get("insights", [])
        if insights:
            print("\n💡 Insights:")
            for insight in insights[:5]:
                print(f"  • {insight}")
                
    def plot_temporal_sequence(self, results: dict[str, Any], save_path: str | None = None) -> None:
        """Plot temporal sequence analysis results"""
        matches = results.get("matches", [])
        
        if not matches:
            print("No matches to plot")
            return
            
        # Extract data for plotting
        session_ids = []
        event_indices = []
        confidences = []
        zone_percentages = []
        
        for match in matches:
            session_ids.append(match.get("session_id", "unknown"))
            event_indices.append(match.get("event_index", 0))
            
            pattern_match = match.get("pattern_match", {})
            confidences.append(pattern_match.get("confidence", 0))
            
            event_context = match.get("event_context", {})
            zone_percentages.append(event_context.get("archaeological_zone_pct", 0))
            
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Sequence Analysis Results', fontsize=16)
        
        # Plot 1: Confidence distribution
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Pattern Match Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Pattern Match Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Archaeological zone distribution
        ax2.hist(zone_percentages, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Archaeological Zone Percentage')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Archaeological Zone Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Event index vs confidence
        scatter = ax3.scatter(event_indices, confidences, c=zone_percentages, 
                            cmap='viridis', alpha=0.6, s=50)
        ax3.set_xlabel('Event Index in Session')
        ax3.set_ylabel('Pattern Match Confidence')
        ax3.set_title('Event Position vs Confidence')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Zone %')
        
        # Plot 4: Session distribution
        session_counts = pd.Series(session_ids).value_counts().head(10)
        ax4.bar(range(len(session_counts)), session_counts.values, color='coral')
        ax4.set_xlabel('Session (Top 10)')
        ax4.set_ylabel('Match Count')
        ax4.set_title('Matches by Session')
        ax4.set_xticks(range(len(session_counts)))
        ax4.set_xticklabels([s[:10] + '...' if len(s) > 10 else s for s in session_counts.index], 
                           rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_archaeological_zones(self, results: dict[str, Any], save_path: str | None = None) -> None:
        """Plot archaeological zone analysis results"""
        zone_analysis = results.get("zone_analysis", {})
        theory_b_events = results.get("theory_b_events", [])
        
        if not zone_analysis:
            print("No zone analysis data to plot")
            return
            
        # Aggregate zone data across sessions
        zone_data = {"20%": [], "40%": [], "60%": [], "80%": []}
        session_ranges = []
        
        for _session_id, analysis in zone_analysis.items():
            zone_events = analysis.get("zone_events", {})
            session_range = analysis.get("session_range", 0)
            session_ranges.append(session_range)
            
            for zone in zone_data:
                zone_data[zone].append(zone_events.get(zone, 0))
                
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Archaeological Zone Analysis', fontsize=16)
        
        # Plot 1: Zone event distribution
        zones = list(zone_data.keys())
        avg_events = [np.mean(zone_data[zone]) for zone in zones]
        
        ax1.bar(zones, avg_events, color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
        ax1.set_xlabel('Archaeological Zone')
        ax1.set_ylabel('Average Events per Session')
        ax1.set_title('Average Events by Archaeological Zone')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Session range distribution
        ax2.hist(session_ranges, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax2.set_xlabel('Session Range (Price Points)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Session Range Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Theory B events by zone
        if theory_b_events:
            theory_b_zones = [event.get("zone_percentage", 0) for event in theory_b_events]
            ax3.hist(theory_b_zones, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax3.set_xlabel('Zone Percentage')
            ax3.set_ylabel('Theory B Events')
            ax3.set_title('Theory B Events by Zone')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Theory B Events', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Theory B Events by Zone')
            
        # Plot 4: Zone correlation heatmap
        zone_matrix = np.array([zone_data[zone] for zone in zones])
        correlation_matrix = np.corrcoef(zone_matrix)
        
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(zones)))
        ax4.set_yticks(range(len(zones)))
        ax4.set_xticklabels(zones)
        ax4.set_yticklabels(zones)
        ax4.set_title('Zone Event Correlation Matrix')
        
        # Add correlation values to heatmap
        for i in range(len(zones)):
            for j in range(len(zones)):
                ax4.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black")
                              
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
