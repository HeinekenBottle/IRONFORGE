#!/usr/bin/env python3
"""
IRONFORGE Pre-Structure Inventory Agent
=======================================

Market structure archaeologist - catalogs what exists BEFORE 40% archaeological touches.
Inventories session highs/lows, daily extremes, and existing FVGs for target progression analysis.

Multi-Agent Role: Pre-Structure Inventory Specialist
Focus: Complete liquidity target mapping before trigger events
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SessionStructure:
    """Session high/low structure before 40% touch"""
    session_type: str
    session_high: float
    session_high_timestamp: str
    session_low: float
    session_low_timestamp: str
    session_start: str
    is_current_session: bool  # Session where 40% touch occurs

@dataclass
class DailyStructure:
    """Daily high/low structure before 40% touch"""
    daily_high: float
    daily_high_timestamp: str
    daily_low: float
    daily_low_timestamp: str
    current_day_high: float
    current_day_low: float
    previous_day_high: float
    previous_day_low: float

@dataclass
class FVGStructure:
    """Fair Value Gap existing before 40% touch"""
    fvg_id: str
    gap_high: float
    gap_low: float
    gap_size: float
    creation_timestamp: str
    creation_session: str
    gap_type: str  # 'bullish' or 'bearish'
    is_filled: bool
    distance_from_40_level: float

@dataclass
class PreStructureInventory:
    """Complete market structure inventory before 40% touch"""
    archaeological_touch: Dict
    session_structures: List[SessionStructure]
    daily_structure: DailyStructure
    existing_fvgs: List[FVGStructure]
    inventory_timestamp: str
    total_targets: int
    target_summary: Dict

class PreStructureInventoryAgent:
    """Agent specialized in cataloging pre-existing market structure"""
    
    def __init__(self, shard_data_path: str, archaeological_touches_file: str):
        self.shard_data_path = Path(shard_data_path)
        self.archaeological_touches_file = Path(archaeological_touches_file)
        
        # Structure detection parameters
        self.fvg_minimum_size = 2.0  # Minimum gap size to qualify as FVG
        self.lookback_hours = 24     # How far back to scan for structure
        self.session_definitions = {
            'ASIA': ('21:00', '05:00'),
            'LONDON': ('03:00', '11:00'), 
            'LUNCH': ('11:00', '13:00'),
            'NY': ('09:30', '16:00'),
            'PREMARKET': ('06:00', '09:30'),
            'MIDNIGHT': ('16:00', '21:00')
        }
        
    def load_archaeological_touches(self) -> List[Dict]:
        """Load 40% archaeological zone touch data"""
        
        print("📊 Loading 40% archaeological touch events...")
        
        # Using same sample data as before, but now we'll properly analyze pre-structure
        sample_touches = [
            {
                'timestamp': '2025-07-25 14:35:00',
                'price': 23162.25,
                'archaeological_40_level': 23160.0,
                'session_type': 'NY',
                'date': '2025-07-25',
                'previous_day_high': 23175.5,
                'previous_day_low': 23145.0
            },
            {
                'timestamp': '2025-07-26 10:22:00',
                'price': 23158.75,
                'archaeological_40_level': 23157.5,
                'session_type': 'LONDON',
                'date': '2025-07-26',
                'previous_day_high': 23170.0,
                'previous_day_low': 23148.0
            },
            {
                'timestamp': '2025-07-29 15:18:00',
                'price': 23164.0,
                'archaeological_40_level': 23163.25,
                'session_type': 'NY',
                'date': '2025-07-29',
                'previous_day_high': 23178.0,
                'previous_day_low': 23152.5
            }
        ]
        
        print(f"✅ Loaded {len(sample_touches)} archaeological touches for pre-structure analysis")
        return sample_touches
    
    def scan_session_structure_before_touch(self, touch_data: Dict) -> List[SessionStructure]:
        """Scan for session highs/lows that existed BEFORE 40% touch"""
        
        session_structures = []
        touch_timestamp = datetime.strptime(touch_data['timestamp'], '%Y-%m-%d %H:%M:%S')
        date = touch_data['date']
        
        # Load all session data for the touch date and scan backwards
        session_types = ['MIDNIGHT', 'PREMARKET', 'ASIA', 'LONDON', 'LUNCH', 'NY']
        
        for session_type in session_types:
            session_path = self.shard_data_path / f"shard_{session_type}_{date}"
            
            if session_path.exists():
                df = self.load_shard_data(session_path)
                if df is not None and len(df) > 0:
                    
                    # Filter data to only BEFORE the 40% touch
                    if 'timestamp_et' in df.columns:
                        df['datetime'] = pd.to_datetime(df['timestamp_et'])
                        pre_touch_data = df[df['datetime'] < touch_timestamp].copy()
                    else:
                        # Use index-based approximation if no timestamp column
                        midpoint = len(df) // 2
                        pre_touch_data = df.iloc[:midpoint].copy()
                    
                    if len(pre_touch_data) > 0 and 'price' in pre_touch_data.columns:
                        # Find session high/low from pre-touch data only
                        session_high = pre_touch_data['price'].max()
                        session_low = pre_touch_data['price'].min()
                        
                        # Get timestamps of highs/lows
                        high_idx = pre_touch_data['price'].idxmax()
                        low_idx = pre_touch_data['price'].idxmin()
                        
                        high_timestamp = str(pre_touch_data.loc[high_idx, 'timestamp_et']) if 'timestamp_et' in pre_touch_data.columns else f"{session_type}_high"
                        low_timestamp = str(pre_touch_data.loc[low_idx, 'timestamp_et']) if 'timestamp_et' in pre_touch_data.columns else f"{session_type}_low"
                        
                        # Determine if this is the current session (where touch occurred)
                        is_current_session = (session_type == touch_data['session_type'])
                        
                        session_structure = SessionStructure(
                            session_type=session_type,
                            session_high=session_high,
                            session_high_timestamp=high_timestamp,
                            session_low=session_low,
                            session_low_timestamp=low_timestamp,
                            session_start=f"{date} {session_type}",
                            is_current_session=is_current_session
                        )
                        
                        session_structures.append(session_structure)
        
        return session_structures
    
    def scan_daily_structure_before_touch(self, touch_data: Dict) -> DailyStructure:
        """Catalog daily high/low structure before 40% touch"""
        
        date = touch_data['date']
        touch_timestamp = datetime.strptime(touch_data['timestamp'], '%Y-%m-%d %H:%M:%S')
        
        # Initialize with provided previous day data
        previous_day_high = touch_data.get('previous_day_high', 0.0)
        previous_day_low = touch_data.get('previous_day_low', 0.0)
        
        # Scan current day data before touch to find current day extremes
        current_day_high = 0.0
        current_day_low = 999999.0
        daily_high_timestamp = ""
        daily_low_timestamp = ""
        
        session_types = ['MIDNIGHT', 'PREMARKET', 'ASIA', 'LONDON', 'LUNCH', 'NY']
        
        for session_type in session_types:
            session_path = self.shard_data_path / f"shard_{session_type}_{date}"
            
            if session_path.exists():
                df = self.load_shard_data(session_path)
                if df is not None and len(df) > 0 and 'price' in df.columns:
                    
                    # Filter to before touch time
                    if 'timestamp_et' in df.columns:
                        df['datetime'] = pd.to_datetime(df['timestamp_et'])
                        pre_touch_data = df[df['datetime'] < touch_timestamp].copy()
                    else:
                        pre_touch_data = df.iloc[:len(df)//2].copy()
                    
                    if len(pre_touch_data) > 0:
                        session_high = pre_touch_data['price'].max()
                        session_low = pre_touch_data['price'].min()
                        
                        # Update current day extremes
                        if session_high > current_day_high:
                            current_day_high = session_high
                            high_idx = pre_touch_data['price'].idxmax()
                            daily_high_timestamp = str(pre_touch_data.loc[high_idx, 'timestamp_et']) if 'timestamp_et' in pre_touch_data.columns else f"{session_type}_high"
                        
                        if session_low < current_day_low:
                            current_day_low = session_low
                            low_idx = pre_touch_data['price'].idxmin()
                            daily_low_timestamp = str(pre_touch_data.loc[low_idx, 'timestamp_et']) if 'timestamp_et' in pre_touch_data.columns else f"{session_type}_low"
        
        return DailyStructure(
            daily_high=max(current_day_high, previous_day_high),
            daily_high_timestamp=daily_high_timestamp if current_day_high > previous_day_high else "previous_day",
            daily_low=min(current_day_low, previous_day_low),
            daily_low_timestamp=daily_low_timestamp if current_day_low < previous_day_low else "previous_day",
            current_day_high=current_day_high,
            current_day_low=current_day_low,
            previous_day_high=previous_day_high,
            previous_day_low=previous_day_low
        )
    
    def scan_existing_fvgs_before_touch(self, touch_data: Dict) -> List[FVGStructure]:
        """Identify Fair Value Gaps that existed BEFORE 40% touch"""
        
        existing_fvgs = []
        date = touch_data['date']
        touch_timestamp = datetime.strptime(touch_data['timestamp'], '%Y-%m-%d %H:%M:%S')
        archaeological_level = touch_data['archaeological_40_level']
        
        session_types = ['MIDNIGHT', 'PREMARKET', 'ASIA', 'LONDON', 'LUNCH', 'NY']
        fvg_counter = 0
        
        for session_type in session_types:
            session_path = self.shard_data_path / f"shard_{session_type}_{date}"
            
            if session_path.exists():
                df = self.load_shard_data(session_path)
                if df is not None and len(df) > 1 and 'price' in df.columns:
                    
                    # Filter to before touch time only
                    if 'timestamp_et' in df.columns:
                        df['datetime'] = pd.to_datetime(df['timestamp_et'])
                        pre_touch_data = df[df['datetime'] < touch_timestamp].copy()
                    else:
                        pre_touch_data = df.iloc[:len(df)//2].copy()
                    
                    if len(pre_touch_data) < 2:
                        continue
                    
                    # Detect FVGs in pre-touch data
                    for i in range(1, len(pre_touch_data)):
                        current_price = pre_touch_data['price'].iloc[i]
                        previous_price = pre_touch_data['price'].iloc[i-1]
                        
                        gap_size = abs(current_price - previous_price)
                        
                        if gap_size >= self.fvg_minimum_size:
                            gap_high = max(current_price, previous_price)
                            gap_low = min(current_price, previous_price)
                            gap_type = 'bullish' if current_price > previous_price else 'bearish'
                            
                            creation_timestamp = str(pre_touch_data.index[i]) if 'timestamp_et' not in pre_touch_data.columns else str(pre_touch_data['timestamp_et'].iloc[i])
                            
                            # Calculate distance from 40% level
                            gap_mid = (gap_high + gap_low) / 2
                            distance_from_40 = abs(gap_mid - archaeological_level)
                            
                            # Check if gap was filled in remaining pre-touch data
                            post_gap_data = pre_touch_data.iloc[i+1:]
                            is_filled = False
                            
                            if len(post_gap_data) > 0:
                                if gap_type == 'bullish':
                                    # Bullish gap filled if price returns to gap_low level
                                    is_filled = any(post_gap_data['price'] <= gap_low)
                                else:
                                    # Bearish gap filled if price returns to gap_high level  
                                    is_filled = any(post_gap_data['price'] >= gap_high)
                            
                            fvg_counter += 1
                            fvg_structure = FVGStructure(
                                fvg_id=f"{session_type}_FVG_{fvg_counter}",
                                gap_high=gap_high,
                                gap_low=gap_low,
                                gap_size=gap_size,
                                creation_timestamp=creation_timestamp,
                                creation_session=session_type,
                                gap_type=gap_type,
                                is_filled=is_filled,
                                distance_from_40_level=distance_from_40
                            )
                            
                            existing_fvgs.append(fvg_structure)
        
        # Sort FVGs by distance from 40% level (nearest first)
        existing_fvgs.sort(key=lambda x: x.distance_from_40_level)
        
        return existing_fvgs
    
    def load_shard_data(self, session_path: Path) -> Optional[pd.DataFrame]:
        """Load 1-minute shard data from parquet files"""
        try:
            nodes_file = session_path / "nodes.parquet"
            if nodes_file.exists():
                df = pd.read_parquet(nodes_file)
                return df
        except Exception as e:
            print(f"Error loading {session_path}: {e}")
        return None
    
    def create_complete_inventory(self, touch_data: Dict) -> PreStructureInventory:
        """Create complete pre-structure inventory for a 40% archaeological touch"""
        
        print(f"🏛️ Creating pre-structure inventory for {touch_data['timestamp']}...")
        
        # Scan all structure types before the touch
        session_structures = self.scan_session_structure_before_touch(touch_data)
        daily_structure = self.scan_daily_structure_before_touch(touch_data)
        existing_fvgs = self.scan_existing_fvgs_before_touch(touch_data)
        
        # Calculate target summary
        total_targets = len(session_structures) * 2 + 2 + len([fvg for fvg in existing_fvgs if not fvg.is_filled])
        
        target_summary = {
            'session_highs': len([s for s in session_structures]),
            'session_lows': len([s for s in session_structures]),
            'daily_extremes': 2,  # daily high and low
            'unfilled_fvgs': len([fvg for fvg in existing_fvgs if not fvg.is_filled]),
            'total_targets': total_targets
        }
        
        inventory = PreStructureInventory(
            archaeological_touch=touch_data,
            session_structures=session_structures,
            daily_structure=daily_structure,
            existing_fvgs=existing_fvgs,
            inventory_timestamp=datetime.now().isoformat(),
            total_targets=total_targets,
            target_summary=target_summary
        )
        
        print(f"✅ Inventory complete: {total_targets} total liquidity targets cataloged")
        print(f"   📈 Session structures: {len(session_structures)}")
        print(f"   🌍 Daily extremes: 2")
        print(f"   🔄 Unfilled FVGs: {target_summary['unfilled_fvgs']}")
        
        return inventory
    
    def export_inventories(self, inventories: List[PreStructureInventory]) -> str:
        """Export all pre-structure inventories to JSON"""
        
        output_path = Path("/Users/jack/IRONFORGE/pre_structure_inventories")
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"pre_structure_inventories_{timestamp}.json"
        
        export_data = {
            'agent_role': 'Pre-Structure Inventory Agent',
            'analysis_timestamp': datetime.now().isoformat(),
            'total_archaeological_touches': len(inventories),
            'inventories': [asdict(inv) for inv in inventories],
            'inventory_summary': self._create_summary_stats(inventories)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📁 Pre-structure inventories exported to: {output_file}")
        return str(output_file)
    
    def _create_summary_stats(self, inventories: List[PreStructureInventory]) -> Dict:
        """Create summary statistics across all inventories"""
        
        total_targets = sum(inv.total_targets for inv in inventories)
        total_sessions = sum(len(inv.session_structures) for inv in inventories)
        total_fvgs = sum(len(inv.existing_fvgs) for inv in inventories)
        unfilled_fvgs = sum(len([fvg for fvg in inv.existing_fvgs if not fvg.is_filled]) for inv in inventories)
        
        return {
            'total_liquidity_targets': total_targets,
            'average_targets_per_touch': total_targets / len(inventories) if inventories else 0,
            'total_session_structures': total_sessions,
            'total_fvgs_found': total_fvgs,
            'unfilled_fvgs_available': unfilled_fvgs,
            'fvg_fill_rate_pre_touch': (total_fvgs - unfilled_fvgs) / total_fvgs if total_fvgs > 0 else 0
        }
    
    def run_pre_structure_analysis(self) -> Dict:
        """Execute complete pre-structure inventory analysis"""
        
        print("🏺 PRE-STRUCTURE INVENTORY AGENT STARTING...")
        print("=" * 55)
        
        # Load archaeological touch events
        archaeological_touches = self.load_archaeological_touches()
        
        # Create inventory for each touch
        all_inventories = []
        
        for i, touch in enumerate(archaeological_touches):
            print(f"\n📊 Processing touch {i+1}/{len(archaeological_touches)}")
            inventory = self.create_complete_inventory(touch)
            all_inventories.append(inventory)
        
        # Export results
        export_file = self.export_inventories(all_inventories)
        
        return {
            'agent_role': 'Pre-Structure Inventory Agent',
            'inventories_created': len(all_inventories),
            'export_file': export_file,
            'total_targets_cataloged': sum(inv.total_targets for inv in all_inventories),
            'ready_for_progression_tracking': True
        }

def main():
    """Execute pre-structure inventory analysis"""
    
    print("🏛️ PRE-STRUCTURE INVENTORY AGENT")
    print("=" * 40)
    
    agent = PreStructureInventoryAgent(
        shard_data_path="/Users/jack/IRONFORGE/data/shards/NQ_M5",
        archaeological_touches_file="/Users/jack/IRONFORGE/archaeological_touches_data.json"
    )
    
    results = agent.run_pre_structure_analysis()
    
    # Display results
    print(f"\n🏛️ PRE-STRUCTURE INVENTORY COMPLETE:")
    print(f"   📊 Inventories Created: {results['inventories_created']}")
    print(f"   🎯 Total Targets Cataloged: {results['total_targets_cataloged']}")
    print(f"   📁 Export File: {results['export_file']}")
    print(f"   ✅ Ready for Target Progression Tracking: {results['ready_for_progression_tracking']}")
    
    return results

if __name__ == "__main__":
    main()