#!/usr/bin/env python3
"""
IRONFORGE Schema Normalization - Technical Debt Surgeon Implementation
====================================================================

Migrates legacy 34D data to current 37D schema while maintaining strict data integrity.
Follows NO FALLBACKS policy - fails cleanly if data cannot be migrated properly.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

class SchemaNormalizer:
    """
    Technical Debt Surgeon implementation for schema normalization
    Migrates 34D legacy data to 37D temporal cycle schema
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Expected feature counts for validation
        self.LEGACY_34D_FEATURES = 34
        self.CURRENT_37D_FEATURES = 37
        self.TEMPORAL_CYCLE_ADDITIONS = 3
        
        # Feature structure validation
        self.REQUIRED_RELATIVITY_FEATURES = [
            'normalized_price', 'pct_from_open', 'pct_from_high', 
            'pct_from_low', 'price_to_HTF_ratio', 'time_since_session_open', 
            'normalized_time'
        ]
        
    def migrate_legacy_to_37d(self, session_data: Dict) -> Dict:
        """
        Migrate legacy 34D session data to current 37D schema
        
        NO FALLBACKS: Fails cleanly if data cannot be properly migrated
        """
        
        # Phase 1: Validate input data structure
        self._validate_legacy_structure(session_data)
        
        # Phase 2: Extract session date for temporal cycle calculation
        session_date = self._extract_session_date(session_data)
        temporal_cycles = self._calculate_temporal_cycles(session_date)
        
        # Phase 3: Enhance price movements with temporal cycle features
        enhanced_session = self._add_temporal_cycle_features(
            session_data, temporal_cycles
        )
        
        # Phase 4: Validate migrated data
        self._validate_migrated_structure(enhanced_session)
        
        return enhanced_session
    
    def _validate_legacy_structure(self, session_data: Dict) -> None:
        """Validate legacy data structure before migration"""
        
        # Check required session metadata
        if 'session_metadata' not in session_data:
            raise ValueError(
                "SCHEMA MIGRATION FAILED: Missing session_metadata field\n"
                "SOLUTION: Ensure session data includes session_metadata with date information"
            )
        
        # Check price movements exist
        if 'price_movements' not in session_data:
            raise ValueError(
                "SCHEMA MIGRATION FAILED: Missing price_movements field\n"
                "SOLUTION: Ensure session data includes price_movements array"
            )
        
        price_movements = session_data['price_movements']
        if not price_movements or len(price_movements) < 2:
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Insufficient price movements ({len(price_movements)})\n"
                "SOLUTION: Need at least 2 price movements for valid session data"
            )
        
        # Validate relativity features are present (indicates 34D legacy data)
        missing_relativity = []
        sample_movement = price_movements[0]
        
        for required_feature in self.REQUIRED_RELATIVITY_FEATURES:
            if required_feature not in sample_movement:
                missing_relativity.append(required_feature)
        
        if missing_relativity:
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Missing relativity features: {missing_relativity}\n"
                "SOLUTION: Run price_relativity_generator.py first to create 34D legacy schema"
            )
        
        # Check for existing temporal cycle features (would indicate already migrated)
        temporal_cycle_features = ['week_of_month', 'month_of_year', 'day_of_week_cycle']
        existing_cycles = [f for f in temporal_cycle_features if f in sample_movement]
        
        if existing_cycles:
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Data already contains temporal cycle features: {existing_cycles}\n"
                "SOLUTION: This data appears to already be 37D schema. No migration needed."
            )
    
    def _extract_session_date(self, session_data: Dict) -> datetime:
        """Extract session date from metadata for temporal cycle calculation"""
        
        session_metadata = session_data['session_metadata']
        
        # Try different date field names
        date_fields = ['session_date', 'date', 'trading_date']
        session_date_str = None
        
        for field in date_fields:
            if field in session_metadata:
                session_date_str = session_metadata[field]
                break
        
        if not session_date_str:
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: No session date found in metadata\n"
                f"Checked fields: {date_fields}\n"
                f"Available fields: {list(session_metadata.keys())}\n"
                "SOLUTION: Ensure session_metadata contains session_date field in YYYY-MM-DD format"
            )
        
        # Parse date string
        try:
            if isinstance(session_date_str, str):
                # Handle multiple date formats
                for date_format in ['%Y-%m-%d', '%Y%m%d', '%m/%d/%Y']:
                    try:
                        return datetime.strptime(session_date_str, date_format)
                    except ValueError:
                        continue
                
                raise ValueError(f"Unrecognized date format: {session_date_str}")
            else:
                raise ValueError(f"Session date must be string, got {type(session_date_str)}")
                
        except Exception as e:
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Cannot parse session date '{session_date_str}'\n"
                f"Error: {e}\n"
                "SOLUTION: Use YYYY-MM-DD format for session_date"
            )
    
    def _calculate_temporal_cycles(self, session_date: datetime) -> Dict[str, int]:
        """Calculate temporal cycle features from session date"""
        
        # Week of month: 1-5 (which week of the month)
        week_of_month = min(5, ((session_date.day - 1) // 7) + 1)
        
        # Month of year: 1-12 (which month)
        month_of_year = session_date.month
        
        # Day of week cycle: 0-6 (Monday=0, Sunday=6)
        day_of_week_cycle = session_date.weekday()
        
        return {
            'week_of_month': week_of_month,
            'month_of_year': month_of_year,
            'day_of_week_cycle': day_of_week_cycle
        }
    
    def _add_temporal_cycle_features(self, session_data: Dict, 
                                   temporal_cycles: Dict[str, int]) -> Dict:
        """Add temporal cycle features to all price movements"""
        
        enhanced_session = session_data.copy()
        enhanced_price_movements = []
        
        for movement in session_data['price_movements']:
            enhanced_movement = movement.copy()
            
            # Add temporal cycle features
            enhanced_movement.update(temporal_cycles)
            
            enhanced_price_movements.append(enhanced_movement)
        
        enhanced_session['price_movements'] = enhanced_price_movements
        
        # Update metadata to reflect 37D schema
        if 'feature_dimensions' in enhanced_session.get('session_metadata', {}):
            enhanced_session['session_metadata']['feature_dimensions'] = self.CURRENT_37D_FEATURES
        
        return enhanced_session
    
    def _validate_migrated_structure(self, enhanced_session: Dict) -> None:
        """Validate the migrated 37D structure"""
        
        price_movements = enhanced_session['price_movements']
        sample_movement = price_movements[0]
        
        # Validate temporal cycle features were added
        required_cycles = ['week_of_month', 'month_of_year', 'day_of_week_cycle']
        missing_cycles = []
        
        for cycle_feature in required_cycles:
            if cycle_feature not in sample_movement:
                missing_cycles.append(cycle_feature)
        
        if missing_cycles:
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Failed to add temporal cycle features: {missing_cycles}\n"
                "SOLUTION: Internal migration error - check _add_temporal_cycle_features method"
            )
        
        # Validate temporal cycle values are reasonable
        week_of_month = sample_movement['week_of_month']
        month_of_year = sample_movement['month_of_year']
        day_of_week_cycle = sample_movement['day_of_week_cycle']
        
        if not (1 <= week_of_month <= 5):
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Invalid week_of_month: {week_of_month} (must be 1-5)\n"
                "SOLUTION: Check date parsing logic"
            )
        
        if not (1 <= month_of_year <= 12):
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Invalid month_of_year: {month_of_year} (must be 1-12)\n"
                "SOLUTION: Check date parsing logic"
            )
        
        if not (0 <= day_of_week_cycle <= 6):
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Invalid day_of_week_cycle: {day_of_week_cycle} (must be 0-6)\n"
                "SOLUTION: Check date parsing logic"
            )
        
        # Validate all original relativity features still present
        missing_relativity = []
        for required_feature in self.REQUIRED_RELATIVITY_FEATURES:
            if required_feature not in sample_movement:
                missing_relativity.append(required_feature)
        
        if missing_relativity:
            raise ValueError(
                f"SCHEMA MIGRATION FAILED: Lost relativity features during migration: {missing_relativity}\n"
                "SOLUTION: Check migration logic preserves all original features"
            )
    
    def migrate_session_file(self, input_path: Path, output_path: Path) -> bool:
        """
        Migrate a single session file from 34D to 37D schema
        
        Returns True if successful, False if skipped due to corruption
        Raises exception if migration fails unexpectedly
        """
        
        try:
            # Load legacy session data
            with open(input_path, 'r') as f:
                legacy_data = json.load(f)
            
            # Migrate to 37D schema
            migrated_data = self.migrate_legacy_to_37d(legacy_data)
            
            # Save migrated data
            with open(output_path, 'w') as f:
                json.dump(migrated_data, f, indent=2)
            
            self.logger.info(f"Successfully migrated {input_path.name} to 37D schema")
            return True
            
        except (ValueError, KeyError) as e:
            # Data quality issues - log and skip (NO FALLBACKS)
            self.logger.warning(f"Skipping {input_path.name} due to data quality issue: {e}")
            return False
        
        except Exception as e:
            # Unexpected error - re-raise for investigation
            self.logger.error(f"Unexpected error migrating {input_path.name}: {e}")
            raise
    
    def batch_migrate_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Migrate all session files in a directory
        
        Returns migration summary with success/failure counts
        """
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files
        json_files = list(input_dir.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {input_dir}")
        
        # Migration summary
        summary = {
            'total_files': len(json_files),
            'successful_migrations': 0,
            'skipped_corrupted': 0,
            'failed_migrations': [],
            'migrated_files': [],
            'skipped_files': []
        }
        
        for json_file in json_files:
            output_file = output_dir / json_file.name
            
            try:
                success = self.migrate_session_file(json_file, output_file)
                
                if success:
                    summary['successful_migrations'] += 1
                    summary['migrated_files'].append(json_file.name)
                else:
                    summary['skipped_corrupted'] += 1
                    summary['skipped_files'].append(json_file.name)
                    
            except Exception as e:
                summary['failed_migrations'].append({
                    'file': json_file.name,
                    'error': str(e)
                })
        
        return summary

def main():
    """Command-line interface for schema migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate IRONFORGE session data from 34D to 37D schema")
    parser.add_argument('input_dir', help="Directory containing legacy 34D session files")
    parser.add_argument('output_dir', help="Directory for migrated 37D session files")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run migration
    normalizer = SchemaNormalizer()
    
    try:
        summary = normalizer.batch_migrate_directory(
            Path(args.input_dir), 
            Path(args.output_dir)
        )
        
        print(f"\n📊 Migration Summary:")
        print(f"   Total files: {summary['total_files']}")
        print(f"   ✅ Successful: {summary['successful_migrations']}")
        print(f"   ⚠️  Skipped (corrupted): {summary['skipped_corrupted']}")
        print(f"   ❌ Failed: {len(summary['failed_migrations'])}")
        
        if summary['failed_migrations']:
            print(f"\n❌ Failed migrations:")
            for failure in summary['failed_migrations']:
                print(f"   - {failure['file']}: {failure['error']}")
        
        if summary['skipped_files']:
            print(f"\n⚠️  Skipped files (data quality issues):")
            for skipped in summary['skipped_files']:
                print(f"   - {skipped}")
        
        success_rate = summary['successful_migrations'] / summary['total_files'] * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())