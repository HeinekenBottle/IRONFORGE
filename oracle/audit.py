#!/usr/bin/env python3
"""
Oracle Audit - Comprehensive session discovery and validation

Provides detailed audit capabilities for Oracle training pipeline,
including session discovery, validation, and gap analysis.
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .session_mapping import SessionMapper, SessionMappingError

logger = logging.getLogger(__name__)


class AuditError(Exception):
    """Raised when audit operations fail"""
    pass


class OracleAuditor:
    """Comprehensive Oracle training pipeline auditor"""
    
    # Error codes for classification
    ERROR_CODES = {
        'SUCCESS': 'Session fully processable',
        'SHARD_NOT_FOUND': 'Shard directory does not exist',
        'META_MISSING': 'meta.json file not found',
        'META_INVALID': 'meta.json file corrupted or invalid format',
        'NODES_MISSING': 'nodes.parquet file not found',
        'NODES_CORRUPTED': 'nodes.parquet file unreadable or corrupted',
        'EDGES_MISSING': 'edges.parquet file not found', 
        'EDGES_CORRUPTED': 'edges.parquet file unreadable or corrupted',
        'INSUFFICIENT_NODES': 'Too few nodes for meaningful graph construction',
        'INSUFFICIENT_EDGES': 'Too few edges for graph connectivity',
        'DATE_OUT_OF_RANGE': 'Session date outside specified range',
        'TF_MISMATCH': 'Timeframe mismatch between request and shard metadata',
        'SYMBOL_MISMATCH': 'Symbol mismatch between request and shard metadata'
    }
    
    def __init__(self, data_dir: str = "data/shards", verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.session_mapper = SessionMapper(base_shard_dir=data_dir)
        
        # Audit results
        self.audit_results = []
        self.error_counts = {code: 0 for code in self.ERROR_CODES.keys()}
        
    def log(self, message: str) -> None:
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[ORACLE_AUDIT] {message}")
            
    def validate_shard_structure(self, shard_path: Path) -> Tuple[str, str, Dict]:
        """
        Validate shard directory structure and contents
        
        Args:
            shard_path: Path to shard directory
            
        Returns:
            Tuple of (error_code, reason, metadata_dict)
        """
        if not shard_path.exists():
            return 'SHARD_NOT_FOUND', f'Directory does not exist: {shard_path}', {}
        
        # Validate meta.json
        meta_file = shard_path / 'meta.json'
        if not meta_file.exists():
            return 'META_MISSING', f'meta.json not found in {shard_path}', {}
        
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return 'META_INVALID', f'meta.json corrupted: {e}', {}
        
        # Validate nodes.parquet
        nodes_file = shard_path / 'nodes.parquet'
        if not nodes_file.exists():
            return 'NODES_MISSING', f'nodes.parquet not found in {shard_path}', metadata
        
        try:
            nodes_df = pd.read_parquet(nodes_file)
            node_count = len(nodes_df)
        except Exception as e:
            return 'NODES_CORRUPTED', f'nodes.parquet unreadable: {e}', metadata
        
        # Validate edges.parquet
        edges_file = shard_path / 'edges.parquet'
        if not edges_file.exists():
            return 'EDGES_MISSING', f'edges.parquet not found in {shard_path}', metadata
        
        try:
            edges_df = pd.read_parquet(edges_file)
            edge_count = len(edges_df)
        except Exception as e:
            return 'EDGES_CORRUPTED', f'edges.parquet unreadable: {e}', metadata
        
        # Validate minimum content requirements
        if node_count < 5:  # Minimum nodes for meaningful graph
            return 'INSUFFICIENT_NODES', f'Only {node_count} nodes (minimum 5 required)', metadata
        
        if edge_count < 2:  # Minimum edges for connectivity
            return 'INSUFFICIENT_EDGES', f'Only {edge_count} edges (minimum 2 required)', metadata
        
        # Update metadata with actual counts
        metadata['actual_node_count'] = node_count
        metadata['actual_edge_count'] = edge_count
        
        return 'SUCCESS', 'Session fully processable', metadata
    
    def validate_metadata_consistency(self, metadata: Dict, symbol: str, 
                                    timeframe: str, session_date: str) -> Tuple[str, str]:
        """
        Validate metadata consistency with request parameters
        
        Args:
            metadata: Loaded shard metadata
            symbol: Requested symbol
            timeframe: Requested timeframe  
            session_date: Requested session date
            
        Returns:
            Tuple of (error_code, reason)
        """
        # Validate symbol consistency
        meta_symbol = metadata.get('symbol', '')
        if meta_symbol != symbol:
            return 'SYMBOL_MISMATCH', f'Expected symbol {symbol}, found {meta_symbol}'
        
        # Validate timeframe consistency
        meta_tf = metadata.get('timeframe', '')
        normalized_tf = self.session_mapper.normalize_timeframe(timeframe)
        if meta_tf != normalized_tf:
            return 'TF_MISMATCH', f'Expected timeframe {normalized_tf}, found {meta_tf}'
        
        # Validate date consistency (extract from session_id or date field)
        meta_date = metadata.get('date', '')
        if not meta_date:
            # Try to extract from session_id
            session_id = metadata.get('session_id', '')
            try:
                parsed = self.session_mapper.parse_session_id(session_id)
                meta_date = parsed['date']
            except SessionMappingError:
                pass
        
        if meta_date != session_date:
            return 'DATE_OUT_OF_RANGE', f'Expected date {session_date}, found {meta_date}'
        
        return 'SUCCESS', 'Metadata consistent'
    
    def audit_session(self, session_info: Dict, symbol: str, timeframe: str) -> Dict:
        """
        Comprehensive audit of a single session
        
        Args:
            session_info: Session info from SessionMapper.discover_sessions()
            symbol: Trading symbol
            timeframe: Timeframe string
            
        Returns:
            Dict with audit results for the session
        """
        session_id = session_info['session_id']
        shard_path = Path(session_info['shard_path'])
        
        self.log(f"Auditing session: {session_id}")
        
        # Initialize audit result
        audit_result = {
            'session_id': session_id,
            'session_type': session_info['session_type'],
            'date': session_info['date'],
            'shard_path': str(shard_path),
            'shard_exists': session_info['shard_exists'],
            'meta_valid': False,
            'nodes_count': 0,
            'edges_count': 0,
            'processable': False,
            'error_code': 'UNKNOWN',
            'reason': 'Audit not completed'
        }
        
        # Validate shard structure
        error_code, reason, metadata = self.validate_shard_structure(shard_path)
        
        if error_code != 'SUCCESS':
            audit_result['error_code'] = error_code
            audit_result['reason'] = reason
            self.error_counts[error_code] += 1
            return audit_result
        
        # Update counts from metadata
        audit_result['meta_valid'] = True
        audit_result['nodes_count'] = metadata.get('actual_node_count', 0)
        audit_result['edges_count'] = metadata.get('actual_edge_count', 0)
        
        # Validate metadata consistency
        consistency_error, consistency_reason = self.validate_metadata_consistency(
            metadata, symbol, timeframe, session_info['date']
        )
        
        if consistency_error != 'SUCCESS':
            audit_result['error_code'] = consistency_error
            audit_result['reason'] = consistency_reason
            self.error_counts[consistency_error] += 1
            return audit_result
        
        # Session is fully processable
        audit_result['processable'] = True
        audit_result['error_code'] = 'SUCCESS'
        audit_result['reason'] = 'Session fully processable'
        self.error_counts['SUCCESS'] += 1
        
        return audit_result
    
    def run_audit(self, symbol: str, timeframe: str, from_date: str, to_date: str) -> Dict:
        """
        Run comprehensive audit of Oracle training pipeline
        
        Args:
            symbol: Trading symbol (e.g., 'NQ')
            timeframe: Timeframe string (e.g., '5' or 'M5')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            Dict with complete audit results
        """
        self.log(f"Starting Oracle audit: {symbol} {timeframe} from {from_date} to {to_date}")
        
        # Reset audit state
        self.audit_results = []
        self.error_counts = {code: 0 for code in self.ERROR_CODES.keys()}
        
        # Discover all sessions matching criteria
        try:
            discovered_sessions = self.session_mapper.discover_sessions(
                symbol, timeframe, from_date, to_date
            )
        except SessionMappingError as e:
            raise AuditError(f"Session discovery failed: {e}")
        
        self.log(f"Discovered {len(discovered_sessions)} sessions")
        
        # Audit each session
        for session_info in discovered_sessions:
            try:
                audit_result = self.audit_session(session_info, symbol, timeframe)
                self.audit_results.append(audit_result)
            except Exception as e:
                # Handle unexpected audit failures
                audit_result = {
                    'session_id': session_info.get('session_id', 'UNKNOWN'),
                    'session_type': session_info.get('session_type', 'UNKNOWN'),
                    'date': session_info.get('date', 'UNKNOWN'),
                    'shard_path': session_info.get('shard_path', 'UNKNOWN'),
                    'shard_exists': False,
                    'meta_valid': False,
                    'nodes_count': 0,
                    'edges_count': 0,
                    'processable': False,
                    'error_code': 'AUDIT_FAILED',
                    'reason': f'Unexpected audit failure: {e}'
                }
                self.audit_results.append(audit_result)
                # Add AUDIT_FAILED to error counts if not present
                if 'AUDIT_FAILED' not in self.error_counts:
                    self.error_counts['AUDIT_FAILED'] = 0
                self.error_counts['AUDIT_FAILED'] += 1
        
        # Calculate audit totals
        audit_total = sum(1 for r in self.audit_results if r['processable'])
        total_discovered = len(self.audit_results)
        
        # Generate summary
        audit_summary = {
            'symbol': symbol,
            'timeframe': timeframe,
            'date_range': f"{from_date} to {to_date}",
            'audit_timestamp': datetime.now().isoformat(),
            'sessions_discovered': total_discovered,
            'audit_total': audit_total,  # Key metric for train-oracle
            'success_rate': audit_total / total_discovered if total_discovered > 0 else 0.0,
            'error_breakdown': dict(self.error_counts),
            'sessions': self.audit_results
        }
        
        self.log(f"Audit complete: {audit_total}/{total_discovered} sessions processable")
        
        return audit_summary
    
    def save_audit_ledger(self, audit_summary: Dict, output_path: Path) -> None:
        """
        Save audit results as CSV ledger
        
        Args:
            audit_summary: Audit results from run_audit()
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV headers
        headers = [
            'session_id', 'session_type', 'date', 'shard_path', 'shard_exists',
            'meta_valid', 'nodes_count', 'edges_count', 'processable', 
            'error_code', 'reason'
        ]
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for session in audit_summary['sessions']:
                writer.writerow({
                    header: session.get(header, '') for header in headers
                })
        
        self.log(f"Audit ledger saved: {output_path}")
    
    def generate_gap_analysis(self, audit_summary: Dict, target_sessions: int = 57) -> Dict:
        """
        Generate detailed gap analysis and remediation guidance
        
        Args:
            audit_summary: Audit results from run_audit()
            target_sessions: Target number of sessions (default 57)
            
        Returns:
            Dict with gap analysis and remediation guidance
        """
        audit_total = audit_summary['audit_total']
        
        if audit_total >= target_sessions:
            return {
                'gap_exists': False,
                'audit_total': audit_total,
                'target_sessions': target_sessions,
                'message': f"✅ Coverage sufficient: {audit_total}/{target_sessions} sessions"
            }
        
        # Analyze gaps
        missing_count = target_sessions - audit_total
        processable_sessions = [s for s in audit_summary['sessions'] if s['processable']]
        failed_sessions = [s for s in audit_summary['sessions'] if not s['processable']]
        
        # Date analysis
        if processable_sessions:
            dates = sorted(set(s['date'] for s in processable_sessions))
            date_range_start = dates[0]
            date_range_end = dates[-1]
        else:
            date_range_start = audit_summary['date_range'].split(' to ')[0]
            date_range_end = audit_summary['date_range'].split(' to ')[1]
        
        # Get session type coverage
        session_types = set(s['session_type'] for s in processable_sessions)
        available_types = self.session_mapper.SESSION_TYPES
        missing_types = available_types - session_types
        
        # Generate concrete remediation steps
        remediation_steps = []
        
        if failed_sessions:
            # Group failures by error type
            error_groups = {}
            for session in failed_sessions:
                error_code = session['error_code']
                if error_code not in error_groups:
                    error_groups[error_code] = []
                error_groups[error_code].append(session)
            
            remediation_steps.append(f"Fix {len(failed_sessions)} failed sessions:")
            for error_code, sessions in error_groups.items():
                session_ids = [s['session_id'] for s in sessions[:3]]  # Show first 3
                more_text = f" (and {len(sessions)-3} more)" if len(sessions) > 3 else ""
                remediation_steps.append(f"  - {error_code}: {', '.join(session_ids)}{more_text}")
        
        if missing_count > len(failed_sessions):
            additional_needed = missing_count - len(failed_sessions)
            remediation_steps.append(f"Discover {additional_needed} additional sessions:")
            remediation_steps.append(f"  - Extend date range beyond {date_range_end}")
            remediation_steps.append(f"  - Check earlier dates before {date_range_start}")
            
            if missing_types:
                remediation_steps.append(f"  - Look for missing session types: {', '.join(sorted(missing_types))}")
        
        # Expected shard paths for missing sessions
        expected_paths = []
        symbol = audit_summary['symbol']
        timeframe = audit_summary['timeframe']
        
        # Generate example paths for next few days
        from datetime import datetime, timedelta
        end_date = datetime.strptime(date_range_end, '%Y-%m-%d')
        for i in range(1, min(8, missing_count + 1)):  # Show up to 7 example paths
            next_date = end_date + timedelta(days=i)
            date_str = next_date.strftime('%Y-%m-%d')
            # Use most common session type as example
            common_type = 'MIDNIGHT'  # Default
            if processable_sessions:
                type_counts = {}
                for s in processable_sessions:
                    st = s['session_type']
                    type_counts[st] = type_counts.get(st, 0) + 1
                common_type = max(type_counts.keys(), key=lambda k: type_counts[k])
            
            expected_path = self.session_mapper.resolve_shard_path(
                symbol, timeframe, common_type, date_str
            )
            expected_paths.append(str(expected_path))
        
        return {
            'gap_exists': True,
            'audit_total': audit_total,
            'target_sessions': target_sessions,
            'missing_count': missing_count,
            'processable_sessions': len(processable_sessions),
            'failed_sessions': len(failed_sessions),
            'error_breakdown': dict(audit_summary['error_breakdown']),
            'date_range': f"{date_range_start} to {date_range_end}",
            'session_types_found': sorted(session_types),
            'session_types_missing': sorted(missing_types),
            'remediation_steps': remediation_steps,
            'expected_missing_paths': expected_paths[:5]  # Show first 5 paths
        }


def main():
    """CLI entry point for Oracle audit"""
    parser = argparse.ArgumentParser(description="Audit Oracle training pipeline sessions")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g., NQ,ES)")
    parser.add_argument("--tf", required=True, help="Timeframe (e.g., 5 or M5)")
    parser.add_argument("--from", dest="from_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data/shards", help="Parquet shard data directory")
    parser.add_argument("--output", help="Output CSV ledger file")
    parser.add_argument("--min-sessions", type=int, default=57, help="Minimum required sessions")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    # Parse symbols (for now, just use first symbol)
    symbols = args.symbols.split(',')
    symbol = symbols[0].strip()
    
    # Initialize auditor
    auditor = OracleAuditor(data_dir=args.data_dir, verbose=args.verbose)
    
    try:
        # Run audit
        audit_summary = auditor.run_audit(symbol, args.tf, args.from_date, args.to_date)
        
        # Save ledger if requested
        if args.output:
            ledger_path = Path(args.output)
            auditor.save_audit_ledger(audit_summary, ledger_path)
        
        # Generate gap analysis
        gap_analysis = auditor.generate_gap_analysis(audit_summary, args.min_sessions)
        
        # Print results
        print(f"\n📊 Oracle Training Pipeline Audit")
        print(f"{'='*60}")
        print(f"Symbol: {audit_summary['symbol']}")
        print(f"Timeframe: {audit_summary['timeframe']}")
        print(f"Date range: {audit_summary['date_range']}")
        print(f"Sessions discovered: {audit_summary['sessions_discovered']}")
        print(f"Sessions processable (audit_total): {audit_summary['audit_total']}")
        print(f"Success rate: {audit_summary['success_rate']:.1%}")
        
        # Error breakdown
        if any(count > 0 for code, count in audit_summary['error_breakdown'].items() if code != 'SUCCESS'):
            print(f"\n❌ Error Breakdown:")
            for error_code, count in audit_summary['error_breakdown'].items():
                if count > 0 and error_code != 'SUCCESS':
                    print(f"  {error_code}: {count} sessions")
        
        # Gap analysis
        if gap_analysis['gap_exists']:
            print(f"\n⚠️  Coverage Gap Analysis")
            print(f"Target sessions: {gap_analysis['target_sessions']}")
            print(f"Missing: {gap_analysis['missing_count']} sessions")
            
            print(f"\n🔧 Remediation Steps:")
            for step in gap_analysis['remediation_steps']:
                print(f"  {step}")
            
            if gap_analysis['expected_missing_paths']:
                print(f"\n📂 Expected Missing Shard Paths (examples):")
                for path in gap_analysis['expected_missing_paths']:
                    print(f"  {path}")
        else:
            print(f"\n✅ {gap_analysis['message']}")
        
        # Output audit_total for train-oracle integration
        print(f"\naudit_total: {audit_summary['audit_total']}")
        
        # Exit with appropriate code
        return 0 if audit_summary['audit_total'] >= args.min_sessions else 1
        
    except Exception as e:
        print(f"❌ Audit failed: {e}")
        return 2


if __name__ == "__main__":
    exit(main())