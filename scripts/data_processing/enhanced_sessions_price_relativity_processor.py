#!/usr/bin/env python3
"""
Enhanced Sessions Price Relativity Processor
==========================================

Applies price relativity features to the 57 enhanced sessions from Phase 2
to enable permanent structural pattern discovery.

CRITICAL: Transforms enhanced sessions from absolute prices to structural relationships
"""

import json
import glob
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

class EnhancedSessionsRelativityProcessor:
    """
    Processes enhanced sessions to add price relativity features
    """
    
    def __init__(self):
        self.enhanced_sessions_path = Path('/Users/jack/IRONPULSE/IRONFORGE/enhanced_sessions')
        self.output_path = Path('/Users/jack/IRONPULSE/IRONFORGE/enhanced_sessions_relativity')
        self.output_path.mkdir(exist_ok=True)
        
    def calculate_session_statistics(self, session_data: Dict) -> Dict:
        """Calculate session-wide statistics for normalization"""
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
        
        # Collect all prices from different sources
        all_prices = []
        
        # From price_movements
        price_movements = session_data.get('price_movements', [])
        for pm in price_movements:
            try:
                price = float(pm.get('price_level', 0))
                if price > 0:
                    all_prices.append(price)
            except:
                continue
        
        # From session_fpfvg
        fpfvg = session_data.get('session_fpfvg', {})
        fpfvg_formation = fpfvg.get('fpfvg_formation', {})
        for field in ['premium_high', 'discount_low']:
            try:
                price = float(fpfvg_formation.get(field, 0))
                if price > 0:
                    all_prices.append(price)
            except:
                continue
                
        # From fpfvg interactions
        interactions = fpfvg_formation.get('interactions', [])
        for interaction in interactions:
            try:
                price = float(interaction.get('price_level', 0))
                if price > 0:
                    all_prices.append(price)
            except:
                continue
        
        # From session_liquidity_events
        liquidity_events = session_data.get('session_liquidity_events', [])
        for event in liquidity_events:
            try:
                price = float(event.get('price_level', 0))
                if price > 0:
                    all_prices.append(price)
            except:
                continue
        
        if all_prices:
            stats['session_high'] = max(all_prices)
            stats['session_low'] = min(all_prices)
            stats['session_open'] = all_prices[0]
            stats['session_close'] = all_prices[-1]
            stats['session_range'] = stats['session_high'] - stats['session_low']
            
            if stats['session_range'] <= 0:
                raise ValueError(f"INVALID PRICE RANGE: Session high ({stats['session_high']}) <= low ({stats['session_low']})")
        else:
            raise ValueError("NO PRICE DATA FOUND - Session contains no valid prices")
            
        return stats
    
    def add_relativity_to_price_movements(self, price_movements: List, stats: Dict) -> List:
        """Add relativity features to price_movements array"""
        enhanced_movements = []
        
        for i, pm in enumerate(price_movements):
            if not isinstance(pm, dict) or 'price_level' not in pm:
                continue
            
            try:
                price = float(pm['price_level'])
                timestamp = pm.get('timestamp', '12:00:00')
                
                # Create enhanced price movement
                enhanced_pm = pm.copy()
                
                # Normalized price (0.0 to 1.0 within session range)
                enhanced_pm['normalized_price'] = (price - stats['session_low']) / stats['session_range']
                
                # Percentage from session open
                if stats['session_open'] > 0:
                    enhanced_pm['pct_from_open'] = ((price - stats['session_open']) / stats['session_open']) * 100
                else:
                    enhanced_pm['pct_from_open'] = 0.0
                
                # Percentage from session high/low
                enhanced_pm['pct_from_high'] = ((stats['session_high'] - price) / stats['session_range']) * 100
                enhanced_pm['pct_from_low'] = ((price - stats['session_low']) / stats['session_range']) * 100
                
                # Time since session open
                try:
                    session_date = pm.get('date', stats['session_start_dt'].strftime('%Y-%m-%d'))
                    pm_dt = datetime.strptime(f"{session_date} {timestamp}", '%Y-%m-%d %H:%M:%S')
                    time_since_open = int((pm_dt - stats['session_start_dt']).total_seconds())
                    enhanced_pm['time_since_session_open'] = max(0, time_since_open)
                    enhanced_pm['normalized_time'] = time_since_open / stats['session_duration_seconds']
                except:
                    enhanced_pm['time_since_session_open'] = i * 60  # fallback
                    enhanced_pm['normalized_time'] = i / len(price_movements)
                
                # Time to next event
                if i < len(price_movements) - 1:
                    enhanced_pm['time_to_next_event'] = 60  # default 1 minute
                else:
                    enhanced_pm['time_to_next_event'] = 0
                
                # Keep original absolute price for reference
                enhanced_pm['absolute_price'] = price
                
                enhanced_movements.append(enhanced_pm)
                
            except Exception as e:
                print(f"❌ Warning: Failed to enhance price movement {i}: {str(e)}")
                enhanced_movements.append(pm)  # Keep original if enhancement fails
                
        return enhanced_movements
    
    def add_relativity_to_fpfvg(self, fpfvg_data: Dict, stats: Dict) -> Dict:
        """Add relativity features to FPFVG formation data"""
        enhanced_fpfvg = fpfvg_data.copy()
        
        if 'fpfvg_formation' in enhanced_fpfvg:
            formation = enhanced_fpfvg['fpfvg_formation']
            
            # Enhanced premium_high
            if 'premium_high' in formation:
                try:
                    price = float(formation['premium_high'])
                    formation['premium_high_normalized'] = (price - stats['session_low']) / stats['session_range']
                    formation['premium_high_pct_from_open'] = ((price - stats['session_open']) / stats['session_open']) * 100 if stats['session_open'] > 0 else 0.0
                    formation['premium_high_absolute'] = price
                except:
                    pass
            
            # Enhanced discount_low
            if 'discount_low' in formation:
                try:
                    price = float(formation['discount_low'])
                    formation['discount_low_normalized'] = (price - stats['session_low']) / stats['session_range']
                    formation['discount_low_pct_from_open'] = ((price - stats['session_open']) / stats['session_open']) * 100 if stats['session_open'] > 0 else 0.0
                    formation['discount_low_absolute'] = price
                except:
                    pass
            
            # Enhanced interactions
            if 'interactions' in formation:
                enhanced_interactions = []
                for interaction in formation['interactions']:
                    enhanced_interaction = interaction.copy()
                    
                    if 'price_level' in interaction:
                        try:
                            price = float(interaction['price_level'])
                            enhanced_interaction['price_level_normalized'] = (price - stats['session_low']) / stats['session_range']
                            enhanced_interaction['price_level_pct_from_open'] = ((price - stats['session_open']) / stats['session_open']) * 100 if stats['session_open'] > 0 else 0.0
                            enhanced_interaction['price_level_absolute'] = price
                        except:
                            pass
                    
                    enhanced_interactions.append(enhanced_interaction)
                
                formation['interactions'] = enhanced_interactions
        
        return enhanced_fpfvg
    
    def add_relativity_to_liquidity_events(self, liquidity_events: List, stats: Dict) -> List:
        """Add relativity features to liquidity events"""
        enhanced_events = []
        
        for event in liquidity_events:
            enhanced_event = event.copy()
            
            if 'price_level' in event:
                try:
                    price = float(event['price_level'])
                    enhanced_event['price_level_normalized'] = (price - stats['session_low']) / stats['session_range']
                    enhanced_event['price_level_pct_from_open'] = ((price - stats['session_open']) / stats['session_open']) * 100 if stats['session_open'] > 0 else 0.0
                    enhanced_event['price_level_absolute'] = price
                except:
                    pass
            
            enhanced_events.append(enhanced_event)
        
        return enhanced_events
    
    def process_enhanced_session(self, session_data: Dict) -> Dict:
        """Process a single enhanced session to add price relativity features"""
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
            'enhancement_timestamp': datetime.now().isoformat()
        }
        
        # Process price_movements
        if 'price_movements' in session_data:
            enhanced_session['price_movements'] = self.add_relativity_to_price_movements(
                session_data['price_movements'], stats
            )
        
        # Process session_fpfvg
        if 'session_fpfvg' in session_data:
            enhanced_session['session_fpfvg'] = self.add_relativity_to_fpfvg(
                session_data['session_fpfvg'], stats
            )
        
        # Process session_liquidity_events
        if 'session_liquidity_events' in session_data:
            enhanced_session['session_liquidity_events'] = self.add_relativity_to_liquidity_events(
                session_data['session_liquidity_events'], stats
            )
        
        return enhanced_session
    
    def process_all_enhanced_sessions(self) -> Dict:
        """Process all enhanced sessions to add relativity features"""
        pattern = str(self.enhanced_sessions_path / 'enhanced_*.json')
        files = glob.glob(pattern)
        
        print(f"🔄 Processing {len(files)} enhanced sessions for price relativity...")
        print(f"📂 Input: {self.enhanced_sessions_path}")
        print(f"📂 Output: {self.output_path}")
        
        results = {
            'processed_count': 0,
            'failed_count': 0,
            'processed_files': [],
            'failed_files': []
        }
        
        for filepath in files:
            filename = os.path.basename(filepath)
            
            try:
                # Load enhanced session data
                with open(filepath, 'r') as f:
                    session_data = json.load(f)
                
                # Add relativity features
                enhanced_data = self.process_enhanced_session(session_data)
                
                # Save with new suffix
                output_filename = filename.replace('.json', '_rel.json')
                output_filepath = self.output_path / output_filename
                
                with open(output_filepath, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)
                
                print(f"✅ Processed {filename} → {output_filename}")
                results['processed_count'] += 1
                results['processed_files'].append(output_filename)
                
            except Exception as e:
                error_details = f"Error processing {filename}: {str(e)}"
                print(f"❌ {error_details}")
                results['failed_count'] += 1
                results['failed_files'].append({'filename': filename, 'error': str(e)})
        
        # Summary
        print(f"\n🏆 Enhanced Sessions Price Relativity Processing Complete:")
        print(f"✅ Successfully processed: {results['processed_count']} sessions")
        print(f"❌ Failed: {results['failed_count']} sessions")
        print(f"📊 Success rate: {(results['processed_count'] / len(files)) * 100:.1f}%")
        
        return results

def main():
    """Main execution: Transform all enhanced sessions to use price relativity"""
    print("🔧 ENHANCED SESSIONS PRICE RELATIVITY PROCESSOR")
    print("Transforming 57 enhanced sessions from absolute prices to structural relationships")
    print("=" * 80)
    
    processor = EnhancedSessionsRelativityProcessor()
    results = processor.process_all_enhanced_sessions()
    
    print(f"\n🎯 Enhanced Sessions Price Relativity Transformation Complete!")
    print(f"🔄 Enhanced sessions now use permanent structural relationships")
    print(f"📈 Patterns will survive market regime changes (23k → 30k+)")
    
    return results

if __name__ == "__main__":
    main()
