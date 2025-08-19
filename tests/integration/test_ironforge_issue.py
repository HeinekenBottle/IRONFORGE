# This should trigger IRONFORGE-specific review
session_data = {"events": []}  # Wrong format - should be 'price_movements'
detector.extract_events(session_data)  # Will fail
