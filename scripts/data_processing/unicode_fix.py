"""
Quick fix for Unicode issues in session data
"""
import json
import os
import glob

def sanitize_session_data(data):
    """Recursively sanitize all strings in data structure"""
    if isinstance(data, dict):
        return {k: sanitize_session_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_session_data(item) for item in data]
    elif isinstance(data, str):
        # Remove invalid Unicode characters
        return data.encode('utf-8', 'ignore').decode('utf-8')
    return data

def load_clean_sessions():
    """Load all sessions with sanitization"""
    sessions = []
    
    # Get all JSON files from subdirectories
    pattern = '/Users/jack/IRONPULSE/data/sessions/level_1/**/*.json'
    files = glob.glob(pattern, recursive=True)
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                clean_data = sanitize_session_data(data)
                sessions.append((filepath, clean_data))
                print(f"✓ Loaded: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"✗ Failed: {filepath}: {e}")
    
    print(f"\nLoaded {len(sessions)} sessions successfully")
    return sessions

if __name__ == "__main__":
    sessions = load_clean_sessions()
