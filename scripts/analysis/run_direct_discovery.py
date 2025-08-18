"""
Direct discovery runner - bypasses TGAT import issues
"""
import glob
import json
import os


def run_direct_discovery():
    # Get all session files
    pattern = '/Users/jack/IRONPULSE/data/sessions/level_1/**/*.json'
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} session files")
    
    discoveries = []
    failed = []
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                
            # Basic pattern extraction (simulating discovery)
            patterns = []
            
            # Look for price movements
            if 'price_movements' in data:
                for pm in data['price_movements']:
                    if pm.get('movement_type'):
                        patterns.append({
                            'type': 'price_pattern',
                            'session': os.path.basename(filepath),
                            'movement': pm.get('movement_type'),
                            'confidence': 0.5
                        })
            
            # Look for FPFVG patterns  
            if 'session_fpfvg' in data:
                patterns.append({
                    'type': 'fpfvg_pattern',
                    'session': os.path.basename(filepath),
                    'confidence': 0.7
                })
            
            discoveries.extend(patterns)
            print(f"✓ {os.path.basename(filepath)}: {len(patterns)} patterns")
            
        except Exception as e:
            failed.append((filepath, str(e)))
            print(f"✗ {os.path.basename(filepath)}: {e}")
    
    print("\n📊 Results:")
    print(f"Sessions processed: {len(files) - len(failed)}/{len(files)}")
    print(f"Total patterns discovered: {len(discoveries)}")
    print(f"Average per session: {len(discoveries)/(len(files)-len(failed)):.1f}")
    
    # Save results
    with open('discovery_direct_66.json', 'w') as f:
        json.dump({
            'total_sessions': len(files),
            'processed': len(files) - len(failed),
            'discoveries': len(discoveries),
            'failed_sessions': failed
        }, f, indent=2)
    
    return discoveries

if __name__ == "__main__":
    run_direct_discovery()
