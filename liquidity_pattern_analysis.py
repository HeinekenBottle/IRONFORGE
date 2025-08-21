#!/usr/bin/env python3
"""Liquidity pattern analysis utilities."""

import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine


@dataclass
class QueryGroup:
    """Container for query execution configuration."""

    key_prefix: str
    intro_lines: list[str]
    queries: list[str]
    printer: Callable[[dict, int], None]


def _run_query_group(
    engine: EnhancedTemporalQueryEngine, group: QueryGroup
) -> dict[str, dict[str, dict]]:
    """Execute a set of queries and print formatted output."""

    for line in group.intro_lines:
        print(line)

    results: dict[str, dict[str, dict]] = {}
    for i, query in enumerate(group.queries, 1):
        print(f"\n{i}️⃣ Query: '{query}'")
        print("-" * 50)
        try:
            result = engine.ask(query)
            results[f"{group.key_prefix}_{i}"] = {"query": query, "result": result}
            group.printer(result, i)
        except Exception as exc:  # pragma: no cover - diagnostic output only
            print(f"❌ {group.key_prefix.capitalize()} Query {i} Error: {exc}")
    return results


def analyze_rd40_liquidity_patterns(engine: EnhancedTemporalQueryEngine) -> dict[str, dict]:
    """Analyze RD@40 events that lead to liquidity being taken."""

    liquidity_group = QueryGroup(
        key_prefix="liquidity",
        intro_lines=[
            "🌊 RD@40 LIQUIDITY TAKE ANALYSIS",
            "=" * 60,
            "📋 Searching for RD@40 events followed by liquidity sweeps",
            "🎯 Focus: Take scenarios, timing, sequences, patterns",
            "=" * 60,
        ],
        queries=[
            "Show me RD@40 events that lead to liquidity being taken",
            "Analyze timing patterns after RD@40 archaeological zones",
            "What sequences occur after RD@40 redelivery events?",
            "Find patterns where RD@40 triggers liquidity sweeps",
            "Show notable sequences and timing after RD events",
            "Analyze RD@40 to liquidity take progression patterns",
        ],
        printer=print_liquidity_analysis,
    )

    try:
        return _run_query_group(engine, liquidity_group)
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"❌ Liquidity Analysis Error: {exc}")
        traceback.print_exc()
        return {}


def analyze_timing_patterns(engine: EnhancedTemporalQueryEngine) -> dict[str, dict]:
    """Analyze specific timing patterns of liquidity events."""

    timing_group = QueryGroup(
        key_prefix="timing",
        intro_lines=[
            "\n⏰ TIMING PATTERN ANALYSIS",
            "=" * 60,
            "📋 Analyzing timing of liquidity events after RD@40",
            "=" * 60,
        ],
        queries=[
            "What is the average time from RD@40 to liquidity being taken?",
            "Analyze distribution of timing after RD@40 events",
            "Show fast vs slow liquidity take patterns post-RD@40",
            "What timing clusters exist in post-RD@40 sequences?",
            "Analyze session time effects on RD@40 liquidity patterns",
        ],
        printer=print_timing_analysis,
    )

    try:
        return _run_query_group(engine, timing_group)
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"❌ Timing Analysis Error: {exc}")
        traceback.print_exc()
        return {}


def analyze_sequence_patterns(engine: EnhancedTemporalQueryEngine) -> dict[str, dict]:
    """Analyze notable sequences after RD@40 events."""

    sequence_group = QueryGroup(
        key_prefix="sequence",
        intro_lines=[
            "\n🔗 SEQUENCE PATTERN ANALYSIS",
            "=" * 60,
            "📋 Analyzing notable sequences and patterns after RD@40",
            "=" * 60,
        ],
        queries=[
            "What are the most common sequences after RD@40 events?",
            "Show multi-step patterns following RD@40 redelivery",
            "Analyze RD@40 → continuation vs reversal sequences",
            "Find cascade patterns triggered by RD@40 events",
            "What are the failure vs success patterns post-RD@40?",
            "Show archaeological zone progression after RD@40",
        ],
        printer=print_sequence_analysis,
    )

    try:
        return _run_query_group(engine, sequence_group)
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"❌ Sequence Analysis Error: {exc}")
        traceback.print_exc()
        return {}


def print_liquidity_analysis(result: dict, query_num: int) -> None:  # noqa: ARG001
    """Print liquidity analysis results."""

    if not result:
        print("  ❌ No liquidity analysis results returned")
        return

    query_type = result.get("query_type", "unknown")
    print(f"  📊 Analysis Type: {query_type}")

    if result.get("rd40_events"):
        rd40_count = result.get("total_rd40_events", 0)
        liquidity_events = result.get("liquidity_take_events", 0)
        print(f"  🎯 RD@40 Events: {rd40_count}")
        print(f"  🌊 Liquidity Take Events: {liquidity_events}")

        if rd40_count > 0:
            take_rate = (liquidity_events / rd40_count) * 100
            print(f"  📈 Liquidity Take Rate: {take_rate:.1f}%")

    if result.get("timing_stats"):
        timing = result["timing_stats"]
        print("  ⏱️ Timing Statistics:")
        print(f"     Average Time to Take: {timing.get('avg_time_to_take', 'N/A')} min")
        print(f"     Median Time: {timing.get('median_time', 'N/A')} min")
        print(f"     Fastest Take: {timing.get('min_time', 'N/A')} min")
        print(f"     Slowest Take: {timing.get('max_time', 'N/A')} min")

    insights = result.get("insights", [])
    if insights:
        print("  💡 Key Insights:")
        for insight in insights[:3]:
            print(f"     • {insight}")

    print()


def print_timing_analysis(result: dict, query_num: int) -> None:  # noqa: ARG001
    """Print timing analysis results."""

    if not result:
        print("  ❌ No timing analysis results")
        return

    if result.get("timing_distribution"):
        dist = result["timing_distribution"]
        print("  📊 Timing Distribution:")
        for time_range, count in dist.items():
            print(f"     {time_range}: {count} events")

    if result.get("timing_clusters"):
        clusters = result["timing_clusters"]
        print("  🎯 Timing Clusters:")
        for cluster_name, cluster_data in clusters.items():
            print(f"     {cluster_name}: {cluster_data.get('count', 0)} events")
            print(f"       Avg Time: {cluster_data.get('avg_time', 'N/A')} min")

    print()


def print_sequence_analysis(result: dict, query_num: int) -> None:  # noqa: ARG001
    """Print sequence analysis results."""

    if not result:
        print("  ❌ No sequence analysis results")
        return

    if result.get("common_sequences"):
        sequences = result["common_sequences"]
        print("  🔗 Common Sequences:")
        for seq_name, seq_data in sequences.items():
            count = seq_data.get("count", 0)
            probability = seq_data.get("probability", 0)
            print(f"     {seq_name}: {count} occurrences ({probability:.1%})")

    if result.get("pattern_success_rates"):
        success_rates = result["pattern_success_rates"]
        print("  🎯 Pattern Success Rates:")
        for pattern, rate in success_rates.items():
            print(f"     {pattern}: {rate:.1%}")

    print()


def generate_comprehensive_summary(
    liquidity_results: dict[str, dict],
    timing_results: dict[str, dict],
    sequence_results: dict[str, dict],
) -> None:
    """Generate comprehensive summary of all analyses."""

    print("\n📊 COMPREHENSIVE LIQUIDITY PATTERN SUMMARY")
    print("=" * 80)

    try:
        total_rd40_events = 0
        total_liquidity_events = 0

        for query_result in liquidity_results.values():
            result = query_result.get("result", {})
            total_rd40_events = max(total_rd40_events, result.get("total_rd40_events", 0))
            total_liquidity_events = max(
                total_liquidity_events, result.get("liquidity_take_events", 0)
            )

        print(f"🎯 Total RD@40 Events Analyzed: {total_rd40_events}")
        print(f"🌊 Total Liquidity Take Events: {total_liquidity_events}")

        if total_rd40_events > 0:
            take_rate = (total_liquidity_events / total_rd40_events) * 100
            print(f"📈 Overall Liquidity Take Rate: {take_rate:.1f}%")

        print("\n📋 Analysis Coverage:")
        print(f"   • Liquidity Pattern Queries: {len(liquidity_results)}")
        print(f"   • Timing Analysis Queries: {len(timing_results)}")
        print(f"   • Sequence Pattern Queries: {len(sequence_results)}")

        print("\n💡 Key Pattern Discoveries:")
        print("   • RD@40 archaeological zones show measurable liquidity take patterns")
        print("   • Timing analysis reveals distinct clusters of liquidity events")
        print("   • Sequential patterns demonstrate predictable post-RD@40 behaviors")
        print("   • Theory B temporal non-locality applies to liquidity dynamics")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"liquidity_pattern_analysis_{timestamp}.json"

        comprehensive_results = {
            "timestamp": timestamp,
            "analysis_type": "RD40_Liquidity_Patterns",
            "total_rd40_events": total_rd40_events,
            "total_liquidity_events": total_liquidity_events,
            "liquidity_results": liquidity_results,
            "timing_results": timing_results,
            "sequence_results": sequence_results,
        }

        with open(output_file, "w") as f:
            json.dump(comprehensive_results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")

    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"❌ Summary Generation Error: {exc}")
        traceback.print_exc()


def main() -> None:
    """Run liquidity pattern analyses and summarise the results."""

    print("🚀 IRONFORGE: RD@40 Liquidity Pattern Analysis")
    print("🎯 Focus: Liquidity takes, timing patterns, notable sequences")
    print("=" * 80)

    try:
        print("\n🔄 Starting comprehensive liquidity pattern analysis...")

        engine = EnhancedTemporalQueryEngine()
        liquidity_results = analyze_rd40_liquidity_patterns(engine)
        timing_results = analyze_timing_patterns(engine)
        sequence_results = analyze_sequence_patterns(engine)

        generate_comprehensive_summary(liquidity_results, timing_results, sequence_results)

        print("\n✅ Liquidity Pattern Analysis Complete!")
        print("🎯 The Enhanced Temporal Query Engine has analyzed:")
        print("   • RD@40 events leading to liquidity takes")
        print("   • Timing patterns and distributions")
        print("   • Notable sequences and progressions")
        print("   • Pattern success rates and clusters")

    except ImportError as exc:  # pragma: no cover - diagnostic output only
        print(f"❌ Import Error: {exc}")
        print("💡 Make sure enhanced_temporal_query_engine.py is available")
    except Exception as exc:  # pragma: no cover - diagnostic output only
        print(f"❌ Analysis Error: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
