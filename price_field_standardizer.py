#!/usr/bin/env python3
"""
Price Field Standardizer
========================
Standardizes price movement field names across all enhanced sessions.
Converts 'price' fields to 'price_level' for graph builder compatibility.

Final fix for TGAT Model Quality Recovery Plan Phase 5 validation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


class PriceFieldStandardizer:
    """Standardizes price movement field formats for TGAT validation."""
    
    def __init__(self):
        self.enhanced_sessions_dir = Path(__file__).parent / "enhanced_sessions_with_relativity"
        self.stats = {
            'sessions_processed': 0,
            'movements_processed': 0,
            'fields_standardized': 0,
            'validation_errors': []
        }
    
    def standardize_session_file(self, session_path: Path) -> Dict[str, Any]:
        """Standardize price fields in a single session file."""
        print(f"Processing: {session_path.name}")
        
        try:
            # Load session data
            with open(session_path, 'r') as f:
                session_data = json.load(f)
            
            # Process price movements
            movements_standardized = 0
            price_movements = session_data.get('price_movements', [])
            
            for i, movement in enumerate(price_movements):
                if 'price' in movement and 'price_level' not in movement:
                    # Copy price to price_level
                    movement['price_level'] = movement['price']
                    movements_standardized += 1
                    print(f"  Movement {i}: Standardized price → price_level ({movement['price']})")
                
                # Validate required fields exist
                if 'price_level' not in movement:
                    error_msg = f"Movement {i} missing price_level after standardization"
                    self.stats['validation_errors'].append(f"{session_path.name}: {error_msg}")
                    print(f"  ⚠️ {error_msg}")
            
            # Save standardized session
            with open(session_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Update stats
            self.stats['sessions_processed'] += 1
            self.stats['movements_processed'] += len(price_movements)
            self.stats['fields_standardized'] += movements_standardized
            
            print(f"  ✅ {movements_standardized} fields standardized, {len(price_movements)} total movements")
            
            return {
                'session': session_path.name,
                'movements_total': len(price_movements),
                'fields_standardized': movements_standardized,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Failed to process {session_path.name}: {str(e)}"
            self.stats['validation_errors'].append(error_msg)
            print(f"  ❌ {error_msg}")
            
            return {
                'session': session_path.name,
                'error': str(e),
                'success': False
            }
    
    def validate_standardization(self) -> Dict[str, Any]:
        """Validate all enhanced sessions have consistent price_level fields."""
        print("\n🔍 Validating standardization across all sessions...")
        
        validation_results = {
            'total_sessions': 0,
            'valid_sessions': 0,
            'total_movements': 0,
            'invalid_movements': [],
            'validation_passed': True
        }
        
        # Check all enhanced session files
        for session_path in self.enhanced_sessions_dir.glob("enhanced_rel_*.json"):
            validation_results['total_sessions'] += 1
            
            try:
                with open(session_path, 'r') as f:
                    session_data = json.load(f)
                
                price_movements = session_data.get('price_movements', [])
                session_valid = True
                
                for i, movement in enumerate(price_movements):
                    validation_results['total_movements'] += 1
                    
                    if 'price_level' not in movement:
                        validation_results['invalid_movements'].append(
                            f"{session_path.name}: Movement {i} missing price_level"
                        )
                        session_valid = False
                        validation_results['validation_passed'] = False
                
                if session_valid:
                    validation_results['valid_sessions'] += 1
                    print(f"  ✅ {session_path.name}: {len(price_movements)} movements validated")
                else:
                    print(f"  ❌ {session_path.name}: Validation failed")
                    
            except Exception as e:
                print(f"  ❌ {session_path.name}: Validation error - {str(e)}")
                validation_results['validation_passed'] = False
        
        return validation_results
    
    def run_standardization(self) -> Dict[str, Any]:
        """Execute complete price field standardization process."""
        print("🚀 PRICE FIELD STANDARDIZATION")
        print("=" * 50)
        print(f"Target directory: {self.enhanced_sessions_dir}")
        
        # Find all enhanced session files
        session_files = list(self.enhanced_sessions_dir.glob("enhanced_rel_*.json"))
        print(f"Found {len(session_files)} enhanced session files\n")
        
        if not session_files:
            print("❌ No enhanced session files found!")
            return {'error': 'No session files found'}
        
        # Process each session file
        session_results = []
        for session_path in session_files:
            result = self.standardize_session_file(session_path)
            session_results.append(result)
        
        # Validate standardization
        validation_results = self.validate_standardization()
        
        # Generate summary report
        print(f"\n🎯 STANDARDIZATION COMPLETE")
        print("=" * 50)
        print(f"Sessions processed: {self.stats['sessions_processed']}")
        print(f"Total movements: {self.stats['movements_processed']}")
        print(f"Fields standardized: {self.stats['fields_standardized']}")
        print(f"Validation passed: {validation_results['validation_passed']}")
        
        if self.stats['validation_errors']:
            print(f"\n⚠️ Validation errors: {len(self.stats['validation_errors'])}")
            for error in self.stats['validation_errors']:
                print(f"  - {error}")
        
        if validation_results['invalid_movements']:
            print(f"\n❌ Invalid movements: {len(validation_results['invalid_movements'])}")
            for invalid in validation_results['invalid_movements']:
                print(f"  - {invalid}")
        
        # Final status
        if validation_results['validation_passed'] and not self.stats['validation_errors']:
            print(f"\n✅ SUCCESS: All {len(session_files)} sessions standardized")
            print("🏆 Ready for final TGAT validation!")
        else:
            print(f"\n❌ ISSUES DETECTED: Manual review required")
        
        return {
            'standardization_stats': self.stats,
            'validation_results': validation_results,
            'session_results': session_results
        }


def main():
    """Run price field standardization."""
    standardizer = PriceFieldStandardizer()
    results = standardizer.run_standardization()
    
    # Save results for tracking
    results_path = Path(__file__).parent / "price_standardization_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    return results


if __name__ == "__main__":
    main()