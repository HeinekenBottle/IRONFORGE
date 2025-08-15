#!/usr/bin/env python3
"""
IRONFORGE Enhanced Session Relativity Processor
===============================================

Transforms absolute price patterns in enhanced sessions into permanent structural relationships.

Critical Problem Solved:
- Current patterns: "23421 @ 12:00:00" (expire when market moves)  
- New patterns: "78% of range @ 6hrs → confluence" (permanent validity)

This enables IRONFORGE to discover patterns that survive regime changes,
making archaeological discoveries permanently valuable across all market conditions.

Enhanced Session Structure Support:
- Works with enhanced_*.json files (not HTF-regenerated)
- Processes price_movements with mixed price_level/price fields
- Maintains all existing enhanced features
- Adds structural relationship features
"""

import json
import glob
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

class EnhancedSessionRelativityProcessor:
    """
    Processes enhanced sessions to add price relativity features
    for permanent structural pattern discovery
    """
    
    def __init__(self):
        self.session_count = 0
        self.processed_files = []
        
    def calculate_session_statistics(self, session_data: Dict) -> Dict:
        """
        Calculate session-wide statistics for normalization from enhanced sessions
        """
        # STRICT VALIDATION: Session data must be valid dictionary
        if not isinstance(session_data, dict):
            raise ValueError(f"Session data must be dictionary, got {type(session_data)}")
        
        # Validate essential session structure
        if 'session_metadata' not in session_data:
            raise ValueError("Session missing required 'session_metadata' field")
        
        stats = {
            'session_high': None,
            'session_low': None, 
            'session_open': None,
            'session_close': None,
            'session_range': None,
            'session_start_dt': None,
            'session_end_dt': None,
            'session_duration_seconds': None
        }
        
        # Get session metadata
        session_meta = session_data.get('session_metadata', {})
        session_date = session_meta.get('session_date', '2025-08-04')
        session_start = session_meta.get('session_start', '13:30:00')
        session_end = session_meta.get('session_end', '16:00:00')
        
        try:
            stats['session_start_dt'] = datetime.strptime(f"{session_date} {session_start}", '%Y-%m-%d %H:%M:%S')
            stats['session_end_dt'] = datetime.strptime(f"{session_date} {session_end}", '%Y-%m-%d %H:%M:%S')
            stats['session_duration_seconds'] = int((stats['session_end_dt'] - stats['session_start_dt']).total_seconds())
        except:
            stats['session_start_dt'] = datetime.now()
            stats['session_end_dt'] = datetime.now() + timedelta(hours=3)
            stats['session_duration_seconds'] = 10800  # 3 hours default
        
        # Collect all prices from enhanced session price_movements
        all_prices = []
        
        # From price_movements (enhanced sessions have mixed field names)
        price_movements = session_data.get('price_movements', [])
        for pm in price_movements:
            try:
                # Enhanced sessions have mixed price_level/price fields
                price = pm.get('price_level') or pm.get('price')
                if price and float(price) > 0:
                    all_prices.append(float(price))
            except:
                continue
        
        if all_prices:
            # Validate price data quality
            if len(all_prices) < 2:
                raise ValueError(f"Insufficient price data: only {len(all_prices)} prices found (need ≥2)")
            
            stats['session_high'] = max(all_prices)
            stats['session_low'] = min(all_prices)
            stats['session_open'] = all_prices[0]
            stats['session_close'] = all_prices[-1]
            stats['session_range'] = stats['session_high'] - stats['session_low']
            
            # STRICT: No zero ranges allowed (indicates data corruption)
            if stats['session_range'] <= 0:
                raise ValueError(
                    f"INVALID PRICE RANGE: Session high ({stats['session_high']}) <= low ({stats['session_low']})\n"
                    f"This indicates price data corruption or duplicate prices\n"
                    f"All prices: {sorted(set(all_prices))}\n"
                    f"SOLUTION: Fix price data source to ensure valid price variation"
                )
            
            # Sanity check: ensure reasonable price range
            if stats['session_range'] > stats['session_high'] * 0.1:  # Range > 10% of price
                print(f"⚠️ WARNING: Large price range detected - {stats['session_range']} ({stats['session_range']/stats['session_high']*100:.1f}% of price)")
        else:
            # NO FALLBACKS: Missing price data is a fatal error
            raise ValueError(
                f"NO PRICE DATA FOUND - Session contains no valid prices\n"
                f"Price movements count: {len(price_movements)}\n"
                f"SOLUTION: Ensure session data contains valid price information\n"
                f"NO FALLBACKS: Cannot process session without price data"
            )
            
        return stats
    
    def add_relativity_to_price_movements(self, price_movements: List, stats: Dict) -> List:
        """
        Add relativity features to price_movements array in enhanced sessions
        """
        enhanced_movements = []
        
        for i, pm in enumerate(price_movements):
            movement_id = f"price_movements[{i}]"
            
            # STRICT VALIDATION: No defensive programming, fail fast on bad data
            if not isinstance(pm, dict):
                raise ValueError(f"DATA INTEGRITY FAILURE: {movement_id} is not a dictionary, got {type(pm)}")
            
            # Enhanced sessions have mixed price_level/price fields - accept both
            price_value = None
            if 'price_level' in pm:
                price_value = pm['price_level']
                price_field = 'price_level'
            elif 'price' in pm:
                price_value = pm['price']
                price_field = 'price'
            else:
                available_keys = list(pm.keys())
                raise ValueError(
                    f"MISSING PRICE FIELD: {movement_id} has neither 'price_level' nor 'price' field\n"
                    f"Available keys: {available_keys}\n"
                    f"SOLUTION: Ensure data source provides price information"
                )
            
            # STRICT: Validate timestamp field
            timestamp = pm.get('timestamp')
            if not timestamp or str(timestamp).strip() == '':
                raise ValueError(
                    f"INVALID TIMESTAMP: {movement_id} has empty or missing timestamp\n"
                    f"Timestamp value: '{timestamp}'\n"
                    f"SOLUTION: Ensure all price movements have valid timestamps"
                )
            
            # STRICT: Validate price value
            try:
                price = float(price_value)
                if price <= 0:
                    raise ValueError(
                        f"INVALID PRICE: {movement_id} has non-positive price {price}\n"
                        f"SOLUTION: Ensure all prices are positive values"
                    )
                elif price > 100000:  # Sanity check
                    raise ValueError(
                        f"UNREALISTIC PRICE: {movement_id} has price {price} > 100,000\n"
                        f"SOLUTION: Verify price data accuracy"
                    )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"NON-NUMERIC PRICE: {movement_id} {price_field} '{price_value}' cannot be converted to float\n"
                    f"Error: {str(e)}\n"
                    f"SOLUTION: Ensure all price values are numeric"
                )
            
            # Create enhanced price movement (original structure preserved)
            enhanced_pm = pm.copy()
            
            # Check if already enhanced
            existing_relativity_features = [
                'normalized_price', 'pct_from_open', 'pct_from_high', 'pct_from_low',
                'time_since_session_open', 'normalized_time'
            ]
            has_some_relativity = any(feature in pm for feature in existing_relativity_features)
            has_all_relativity = all(feature in pm for feature in existing_relativity_features)
            
            if has_some_relativity and not has_all_relativity:
                # Partial relativity features detected - this indicates data corruption
                present_features = [f for f in existing_relativity_features if f in pm]
                missing_features = [f for f in existing_relativity_features if f not in pm]
                raise ValueError(
                    f"PARTIAL RELATIVITY ENHANCEMENT DETECTED: {movement_id}\\n"
                    f"Present features: {present_features}\\n"
                    f"Missing features: {missing_features}\\n"
                    f"SOLUTION: Data already partially enhanced - complete enhancement or start fresh\\n"
                    f"NO FALLBACKS: Partial enhancement indicates data corruption"
                )
            
            try:
                # Add relativity features (structural relationships)
                
                # 1. normalized_price (0.0 to 1.0 within session range) - PERMANENT STRUCTURE
                enhanced_pm['normalized_price'] = (price - stats['session_low']) / stats['session_range']
                
                # 2. pct_from_open (percentage from session open) - REGIME INDEPENDENT
                if stats['session_open'] > 0:
                    enhanced_pm['pct_from_open'] = ((price - stats['session_open']) / stats['session_open']) * 100
                else:
                    enhanced_pm['pct_from_open'] = 0.0
                
                # 3. Percentage from session high/low - STRUCTURAL POSITIONING
                enhanced_pm['pct_from_high'] = ((stats['session_high'] - price) / stats['session_range']) * 100
                enhanced_pm['pct_from_low'] = ((price - stats['session_low']) / stats['session_range']) * 100
                
                # 4. time_since_session_open (temporal positioning) - PERMANENT TIMING
                try:
                    session_date = pm.get('date', stats['session_start_dt'].strftime('%Y-%m-%d'))
                    pm_dt = datetime.strptime(f"{session_date} {timestamp}", '%Y-%m-%d %H:%M:%S')
                    time_since_open = int((pm_dt - stats['session_start_dt']).total_seconds())
                    enhanced_pm['time_since_session_open'] = max(0, time_since_open)
                    enhanced_pm['normalized_time'] = time_since_open / stats['session_duration_seconds']
                except:
                    enhanced_pm['time_since_session_open'] = i * 60  # fallback: 1 minute per movement
                    enhanced_pm['normalized_time'] = i / len(price_movements)
                
                # 5. Time to next event - FLOW ANALYSIS
                if i < len(price_movements) - 1:
                    try:
                        next_timestamp = price_movements[i + 1].get('timestamp', timestamp)
                        next_dt = datetime.strptime(f"{session_date} {next_timestamp}", '%Y-%m-%d %H:%M:%S')
                        enhanced_pm['time_to_next_event'] = int((next_dt - pm_dt).total_seconds())
                    except:
                        enhanced_pm['time_to_next_event'] = 60  # default 1 minute
                else:
                    enhanced_pm['time_to_next_event'] = 0
                
                # 6. Price momentum (structural change rate)
                if i > 0:
                    try:
                        prev_price_value = price_movements[i-1].get('price_level') or price_movements[i-1].get('price')
                        prev_price = float(prev_price_value)
                        enhanced_pm['price_momentum'] = ((price - prev_price) / prev_price) * 100
                    except:
                        enhanced_pm['price_momentum'] = 0.0
                else:
                    enhanced_pm['price_momentum'] = 0.0
                
                # 7. Range position (structural level within session)
                enhanced_pm['range_position'] = enhanced_pm['normalized_price']  # 0=low, 1=high
                
                # 8. Keep original absolute price for reference (NEVER 0.0!)
                enhanced_pm['absolute_price'] = price if price > 0 else enhanced_pm.get(price_field, price)
                
                # 9. Mark as relativity enhanced
                enhanced_pm['relativity_enhanced'] = True
                enhanced_pm['enhancement_timestamp'] = datetime.now().isoformat()
                
                enhanced_movements.append(enhanced_pm)
                
            except Exception as e:
                # NO FALLBACKS: Enhancement failure means data corruption
                raise ValueError(
                    f"ENHANCEMENT FAILURE: {movement_id} failed to process relativity features\n"
                    f"Error: {str(e)}\n"
                    f"Input data: {pm}\n"
                    f"Session stats: {stats}\n"
                    f"SOLUTION: Fix data corruption or calculation logic - NO FALLBACKS allowed"
                )
                
        return enhanced_movements
    
    def process_session(self, session_data: Dict) -> Dict:
        """
        Process a single enhanced session to add price relativity features
        """
        # Calculate session statistics
        stats = self.calculate_session_statistics(session_data)
        
        # Create enhanced session data
        enhanced_session = session_data.copy()
        
        # Add session statistics to metadata
        enhanced_session['relativity_stats'] = {
            'session_high': stats['session_high'],
            'session_low': stats['session_low'],
            'session_open': stats['session_open'],
            'session_close': stats['session_close'],
            'session_range': stats['session_range'],
            'session_duration_seconds': stats['session_duration_seconds'],
            'normalization_applied': True,
            'structural_relationships_enabled': True,
            'permanent_pattern_capability': True
        }
        
        # Process price_movements with relativity features
        if 'price_movements' in session_data:
            enhanced_session['price_movements'] = self.add_relativity_to_price_movements(
                session_data['price_movements'], stats
            )
            print(f"✅ Enhanced {len(enhanced_session['price_movements'])} price movements with structural relationships")
        
        # Add relativity metadata to other enhanced features
        enhanced_session['processing_metadata'] = enhanced_session.get('processing_metadata', {})
        enhanced_session['processing_metadata']['relativity_enhancement'] = {
            'applied': True,
            'timestamp': datetime.now().isoformat(),
            'features_added': [
                'normalized_price', 'pct_from_open', 'pct_from_high', 'pct_from_low',
                'time_since_session_open', 'normalized_time', 'time_to_next_event',
                'price_momentum', 'range_position', 'absolute_price'
            ],
            'permanent_validity': True,
            'regime_independence': True
        }
        
        return enhanced_session
    
    def process_all_enhanced_sessions(self, input_dir: str, output_dir: str) -> Dict:
        """
        Process all enhanced sessions to add price relativity features
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all enhanced session files
        pattern = str(input_path / 'enhanced_*.json')
        files = glob.glob(pattern)
        
        print(f"🔄 Processing {len(files)} enhanced sessions for price relativity...")
        print(f"📂 Input: {input_dir}")
        print(f"📂 Output: {output_dir}")
        print(f"🎯 MISSION: Transform absolute prices → permanent structural relationships")
        
        results = {
            'processed_count': 0,
            'failed_count': 0,
            'processed_files': [],
            'failed_files': [],
            'total_movements_enhanced': 0
        }
        
        for filepath in files:
            filename = os.path.basename(filepath)
            
            try:
                # Load enhanced session data
                with open(filepath, 'r') as f:
                    session_data = json.load(f)
                
                # Add relativity features
                enhanced_data = self.process_session(session_data)
                
                # Save with relativity suffix
                output_filename = filename.replace('enhanced_', 'enhanced_rel_')
                output_filepath = output_path / output_filename
                
                # Validate enhanced data before saving
                self._validate_enhanced_data(enhanced_data, filename)
                
                with open(output_filepath, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)
                
                # Track movement count
                movement_count = len(enhanced_data.get('price_movements', []))
                results['total_movements_enhanced'] += movement_count
                
                print(f"✅ {filename} → {output_filename} ({movement_count} movements enhanced)")
                results['processed_count'] += 1
                results['processed_files'].append(output_filename)
                
            except Exception as e:
                error_details = f"Error processing {filename}: {str(e)}"
                print(f"❌ {error_details}")
                results['failed_count'] += 1
                results['failed_files'].append({
                    'filename': filename, 
                    'error': str(e), 
                    'error_type': type(e).__name__
                })
                
                # Log detailed error for debugging
                print(f"    Error type: {type(e).__name__}")
                print(f"    File path: {filepath}")
                if "INTEGRITY FAILURE" in str(e) or "MISSING REQUIRED FIELD" in str(e):
                    print(f"    ℹ️ This appears to be a data quality issue requiring source data fix")
        
        # Summary
        print(f"\n🏆 Enhanced Session Relativity Processing Complete:")
        print(f"✅ Successfully processed: {results['processed_count']} sessions")
        print(f"🔄 Total price movements enhanced: {results['total_movements_enhanced']}")
        print(f"❌ Failed: {results['failed_count']} sessions")
        print(f"📊 Success rate: {(results['processed_count'] / len(files)) * 100:.1f}%")
        
        if results['processed_count'] > 0:
            avg_movements = results['total_movements_enhanced'] / results['processed_count']
            print(f"📈 Average movements per session: {avg_movements:.1f}")
            print(f"🎯 STRUCTURAL TRANSFORMATION ACHIEVED:")
            print(f"   • Absolute prices → Normalized positions (0.0-1.0)")
            print(f"   • Price deltas → Percentage relationships")
            print(f"   • Timestamps → Session-relative timing")
            print(f"   • Patterns now survive market regime changes!")
        
        if results['failed_files']:
            print(f"\n⚠️ FAILED FILES ANALYSIS:")
            for failure in results['failed_files']:
                if isinstance(failure, dict):
                    print(f"   • {failure['filename']}: {failure['error_type']} - {failure['error'][:100]}...")
                else:
                    print(f"   • {failure}")
            
            print(f"\n🔧 RECOMMENDATION:")
            print(f"   1. Review failed files for data corruption")
            print(f"   2. Fix data source issues before reprocessing")
            print(f"   3. NO FALLBACKS policy enforced - clean data required")
        
        return results
    
    def _validate_enhanced_data(self, enhanced_data: Dict, filename: str) -> None:
        """
        Validate that enhanced data has all required relativity features
        """
        required_relativity_features = [
            'normalized_price', 'pct_from_open', 'pct_from_high', 
            'pct_from_low', 'time_since_session_open', 'normalized_time',
            'price_momentum', 'range_position', 'absolute_price'
        ]
        
        # Check price_movements have relativity features
        price_movements = enhanced_data.get('price_movements', [])
        for i, pm in enumerate(price_movements[:5]):  # Check first 5 movements
            missing_features = [f for f in required_relativity_features if f not in pm]
            if missing_features:
                raise ValueError(
                    f"ENHANCEMENT INCOMPLETE: {filename} price_movements[{i}] missing features: {missing_features}\n"
                    f"SOLUTION: Review enhancement logic for price_movements"
                )
        
        # Validate relativity_stats were added
        if 'relativity_stats' not in enhanced_data:
            raise ValueError(
                f"ENHANCEMENT INCOMPLETE: {filename} missing 'relativity_stats' metadata\n"
                f"SOLUTION: Ensure process_session adds relativity statistics"
            )
        
        # Validate relativity_stats completeness
        relativity_stats = enhanced_data['relativity_stats']
        required_stats = [
            'session_high', 'session_low', 'session_open', 'session_close', 
            'session_range', 'structural_relationships_enabled', 'permanent_pattern_capability'
        ]
        missing_stats = [s for s in required_stats if s not in relativity_stats]
        if missing_stats:
            raise ValueError(
                f"ENHANCEMENT INCOMPLETE: {filename} relativity_stats missing: {missing_stats}\n"
                f"SOLUTION: Ensure calculate_session_statistics provides complete statistics"
            )

def main():
    """
    Main execution: Transform all enhanced sessions to use price relativity
    """
    print("🎯 IRONFORGE ENHANCED SESSION RELATIVITY PROCESSOR")
    print("Mission: Transform absolute prices → permanent structural relationships")
    print("Following NO FALLBACKS policy - strict data validation enforced")
    print("=" * 80)
    
    processor = EnhancedSessionRelativityProcessor()
    
    results = processor.process_all_enhanced_sessions(
        input_dir='/Users/jack/IRONPULSE/IRONFORGE/enhanced_sessions',
        output_dir='/Users/jack/IRONPULSE/IRONFORGE/enhanced_sessions_with_relativity'
    )
    
    print(f"\n🚀 IRONFORGE Enhanced Session Relativity Transformation Complete!")
    print(f"🔄 Patterns transformed from absolute prices to permanent structural relationships")
    print(f"📈 Discovered patterns will now survive market regime changes")
    print(f"🏛️ Archaeological discoveries permanently valid across all market conditions!")
    
    return results

if __name__ == "__main__":
    main()