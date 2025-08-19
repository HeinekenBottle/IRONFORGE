#!/usr/bin/env python3
"""Test the watchlist panel in minidash"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from ironforge.reporting.minidash import build_minidash, load_watchlist_data


def test_watchlist_panel():
    """Test watchlist panel functionality."""
    
    # Use working run with watchlist
    run_dir = Path("runs/real-tgat-fixed-2025-08-18")
    
    if not run_dir.exists():
        print("❌ Working run not found")
        return False
    
    # Test loading watchlist data
    watchlist_data = load_watchlist_data(run_dir)
    
    if watchlist_data:
        print(f"✅ Loaded watchlist: {len(watchlist_data)} zones")
        for zone in watchlist_data:
            print(f"  - {zone['zone_id']}: score={zone.get('trading_score', 0):.3f}")
    else:
        print("❌ No watchlist data found")
        return False
    
    # Create minimal test data for minidash
    activity_data = pd.DataFrame({
        'ts': pd.date_range('2025-08-05', periods=10, freq='H'),
        'count': range(10)
    })
    
    confluence_data = pd.DataFrame({
        'ts': pd.date_range('2025-08-05', periods=10, freq='H'),
        'score': [50 + i*2 for i in range(10)]
    })
    
    motifs_data = [
        {'name': 'test_motif_1', 'support': 5, 'ppv': 0.75},
        {'name': 'test_motif_2', 'support': 3, 'ppv': 0.60}
    ]
    
    # Generate minidash with watchlist
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    html_path = output_dir / "test_minidash.html"
    png_path = output_dir / "test_minidash.png"
    
    try:
        result_html, result_png = build_minidash(
            activity_data,
            confluence_data,
            motifs_data,
            html_path,
            png_path,
            run_dir=run_dir
        )
        
        print(f"✅ Generated minidash: {result_html}")
        
        # Check if watchlist is in HTML
        html_content = result_html.read_text()
        if "🎯 Watchlist" in html_content:
            print("✅ Watchlist panel included in minidash")
            print("✅ Test passed!")
            return True
        else:
            print("❌ Watchlist panel not found in minidash")
            return False
            
    except Exception as e:
        print(f"❌ Error generating minidash: {e}")
        return False

if __name__ == "__main__":
    success = test_watchlist_panel()
    sys.exit(0 if success else 1)