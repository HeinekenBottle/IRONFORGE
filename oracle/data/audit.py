#!/usr/bin/env python3
"""
Oracle Audit - Comprehensive session discovery and validation

Provides detailed audit capabilities for Oracle training pipeline,
including session discovery, validation, and gap analysis.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..core import (
    ERROR_CODES, MIN_NODES_THRESHOLD, MIN_EDGES_THRESHOLD,
    AuditError, create_audit_error, handle_oracle_errors
)
from ..models import SessionMetadata, AuditResult
from .session_mapping import SessionMapper, SessionMappingError

logger = logging.getLogger(__name__)


class OracleAuditor:
    """Comprehensive Oracle training pipeline auditor"""
    
    def __init__(self, data_dir: str = "data/shards", verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.session_mapper = SessionMapper(base_shard_dir=data_dir)
        
        # Audit results
        self.audit_results: List[AuditResult] = []
        self.error_counts = {code: 0 for code in ERROR_CODES.keys()}
        
    def log(self, message: str) -> None:
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[ORACLE_AUDIT] {message}")
            
    @handle_oracle_errors(AuditError, "AUDIT_VALIDATION_ERROR")
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
        
        # Check minimum thresholds
        if node_count < MIN_NODES_THRESHOLD:
            return 'INSUFFICIENT_NODES', f'Only {node_count} nodes (minimum: {MIN_NODES_THRESHOLD})', metadata
        
        if edge_count < MIN_EDGES_THRESHOLD:
            return 'INSUFFICIENT_EDGES', f'Only {edge_count} edges (minimum: {MIN_EDGES_THRESHOLD})', metadata
        
        # Update metadata with actual counts
        metadata['actual_node_count'] = node_count
        metadata['actual_edge_count'] = edge_count
        
        return 'SUCCESS', 'Session fully processable', metadata
    
    @handle_oracle_errors(AuditError, "AUDIT_SESSION_ERROR")
    def audit_session(self, session_id: str, symbol: str, timeframe: str, 
                     session_date: str, from_date: str, to_date: str) -> AuditResult:
        """
        Audit a single session for Oracle training compatibility
        
        Args:
            session_id: Session identifier
            symbol: Trading symbol
            timeframe: Timeframe string
            session_date: Session date in YYYY-MM-DD format
            from_date: Start date for range validation
            to_date: End date for range validation
            
        Returns:
            AuditResult with validation status and metadata
        """
        self.log(f"Auditing session: {session_id}")
        
        try:
            # Parse session components
            session_type, parsed_date = self.session_mapper.parse_session_id(session_id)
            
            # Validate date range
            if parsed_date < from_date or parsed_date > to_date:
                return AuditResult(
                    session_id=session_id,
                    status='DATE_OUT_OF_RANGE',
                    error_message=f'Session date {parsed_date} outside range {from_date} to {to_date}'
                )
            
            # Resolve shard path
            shard_path = self.session_mapper.resolve_shard_path(
                symbol, timeframe, session_type, parsed_date
            )
            
            # Validate shard structure
            error_code, reason, metadata = self.validate_shard_structure(shard_path)
            
            if error_code != 'SUCCESS':
                return AuditResult(
                    session_id=session_id,
                    status=error_code,
                    error_message=reason,
                    shard_path=shard_path
                )
            
            # Validate metadata consistency
            if 'symbol' in metadata and metadata['symbol'] != symbol:
                return AuditResult(
                    session_id=session_id,
                    status='SYMBOL_MISMATCH',
                    error_message=f'Expected {symbol}, found {metadata["symbol"]}',
                    shard_path=shard_path
                )
            
            if 'timeframe' in metadata and metadata['timeframe'] != timeframe:
                return AuditResult(
                    session_id=session_id,
                    status='TF_MISMATCH',
                    error_message=f'Expected {timeframe}, found {metadata["timeframe"]}',
                    shard_path=shard_path
                )
            
            # Create session metadata
            session_metadata = SessionMetadata(
                session_id=session_id,
                symbol=symbol,
                timeframe=timeframe,
                session_type=session_type,
                session_date=parsed_date,
                node_count=metadata.get('actual_node_count', 0),
                edge_count=metadata.get('actual_edge_count', 0),
                quality_level=metadata.get('quality', 'fair'),
                data_source='parquet'
            )
            
            return AuditResult(
                session_id=session_id,
                status='SUCCESS',
                metadata=session_metadata,
                shard_path=shard_path
            )
            
        except SessionMappingError as e:
            return AuditResult(
                session_id=session_id,
                status='SESSION_MAPPING_ERROR',
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error auditing session {session_id}: {e}")
            return AuditResult(
                session_id=session_id,
                status='AUDIT_ERROR',
                error_message=f'Unexpected audit error: {e}'
            )
    
    @handle_oracle_errors(AuditError, "AUDIT_DISCOVERY_ERROR")
    def audit_oracle_training_data(self, symbol: str, timeframe: str, 
                                  from_date: str, to_date: str, 
                                  min_sessions: int = 57) -> Dict:
        """
        Comprehensive audit of Oracle training data availability
        
        Args:
            symbol: Trading symbol (e.g., 'NQ')
            timeframe: Timeframe string (e.g., 'M5')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            min_sessions: Minimum required sessions for training
            
        Returns:
            Dict with complete audit results
        """
        self.log(f"Starting Oracle audit: {symbol} {timeframe} from {from_date} to {to_date}")
        
        # Reset audit state
        self.audit_results = []
        self.error_counts = {code: 0 for code in ERROR_CODES.keys()}
        
        # Discover all sessions matching criteria
        try:
            discovered_sessions = self.session_mapper.discover_sessions(
                symbol, timeframe, from_date, to_date
            )
        except SessionMappingError as e:
            raise AuditError(f"Session discovery failed: {e}")
        
        self.log(f"Discovered {len(discovered_sessions)} sessions")
        
        # Audit each discovered session
        for session_id in discovered_sessions:
            try:
                # Parse session to get date
                _, session_date = self.session_mapper.parse_session_id(session_id)
                
                # Audit the session
                audit_result = self.audit_session(
                    session_id, symbol, timeframe, session_date, from_date, to_date
                )
                
                self.audit_results.append(audit_result)
                self.error_counts[audit_result.status] += 1
                
                if audit_result.is_success:
                    self.log(f"✓ {session_id}: SUCCESS")
                else:
                    self.log(f"✗ {session_id}: {audit_result.status} - {audit_result.error_message}")
                    
            except Exception as e:
                logger.error(f"Failed to audit session {session_id}: {e}")
                error_result = AuditResult(
                    session_id=session_id,
                    status='AUDIT_ERROR',
                    error_message=f'Audit failed: {e}'
                )
                self.audit_results.append(error_result)
                self.error_counts['AUDIT_ERROR'] = self.error_counts.get('AUDIT_ERROR', 0) + 1
        
        # Generate summary
        total_sessions = len(self.audit_results)
        successful_sessions = self.error_counts.get('SUCCESS', 0)
        success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0.0
        
        audit_summary = {
            'symbol': symbol,
            'timeframe': timeframe,
            'date_range': f'{from_date} to {to_date}',
            'sessions_discovered': len(discovered_sessions),
            'audit_total': total_sessions,
            'audit_successful': successful_sessions,
            'success_rate': success_rate,
            'min_sessions_required': min_sessions,
            'meets_minimum': successful_sessions >= min_sessions,
            'error_breakdown': dict(self.error_counts),
            'audit_timestamp': datetime.now().isoformat()
        }
        
        self.log(f"Audit complete: {successful_sessions}/{total_sessions} sessions processable")
        
        return audit_summary

    def generate_gap_analysis(self, audit_summary: Dict, min_sessions: int) -> Dict:
        """
        Generate gap analysis for Oracle training requirements

        Args:
            audit_summary: Results from audit_oracle_training_data
            min_sessions: Minimum required sessions

        Returns:
            Dict with gap analysis and recommendations
        """
        successful_sessions = audit_summary['audit_successful']
        gap = max(0, min_sessions - successful_sessions)

        gap_analysis = {
            'required_sessions': min_sessions,
            'available_sessions': successful_sessions,
            'gap': gap,
            'gap_percentage': (gap / min_sessions * 100) if min_sessions > 0 else 0,
            'training_ready': gap == 0,
            'recommendations': []
        }

        if gap > 0:
            gap_analysis['recommendations'].extend([
                f"Need {gap} more processable sessions for Oracle training",
                "Consider expanding date range to find more sessions",
                "Check data quality issues in error breakdown"
            ])

            # Analyze error patterns for specific recommendations
            error_breakdown = audit_summary.get('error_breakdown', {})

            if error_breakdown.get('SHARD_NOT_FOUND', 0) > 0:
                gap_analysis['recommendations'].append(
                    "Some sessions missing from data directory - check data ingestion"
                )

            if error_breakdown.get('INSUFFICIENT_NODES', 0) > 0:
                gap_analysis['recommendations'].append(
                    "Some sessions have too few events - consider lowering node threshold"
                )

            if error_breakdown.get('NODES_CORRUPTED', 0) > 0:
                gap_analysis['recommendations'].append(
                    "Some parquet files are corrupted - re-process affected sessions"
                )
        else:
            gap_analysis['recommendations'].append(
                f"Training data requirements met with {successful_sessions} sessions"
            )

        return gap_analysis

    def export_audit_results(self, output_path: str, format: str = 'csv') -> None:
        """
        Export audit results to file

        Args:
            output_path: Output file path
            format: Export format ('csv' or 'json')
        """
        output_file = Path(output_path)

        if format.lower() == 'csv':
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'session_id', 'status', 'error_message', 'node_count',
                    'edge_count', 'quality_level', 'shard_path'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.audit_results:
                    row = {
                        'session_id': result.session_id,
                        'status': result.status,
                        'error_message': result.error_message or '',
                        'node_count': result.metadata.node_count if result.metadata else 0,
                        'edge_count': result.metadata.edge_count if result.metadata else 0,
                        'quality_level': result.metadata.quality_level if result.metadata else '',
                        'shard_path': str(result.shard_path) if result.shard_path else ''
                    }
                    writer.writerow(row)

        elif format.lower() == 'json':
            export_data = {
                'audit_results': [result.to_dict() for result in self.audit_results],
                'error_counts': self.error_counts,
                'export_timestamp': datetime.now().isoformat()
            }

            with open(output_file, 'w') as jsonfile:
                json.dump(export_data, jsonfile, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.log(f"Audit results exported to {output_file}")

    def get_processable_sessions(self) -> List[SessionMetadata]:
        """
        Get list of all processable sessions from audit results

        Returns:
            List of SessionMetadata for successfully audited sessions
        """
        processable = []
        for result in self.audit_results:
            if result.is_processable:
                processable.append(result.metadata)
        return processable

    def get_error_summary(self) -> Dict[str, List[str]]:
        """
        Get summary of errors grouped by error code

        Returns:
            Dict mapping error codes to lists of affected session IDs
        """
        error_summary = {}
        for result in self.audit_results:
            if not result.is_success:
                if result.status not in error_summary:
                    error_summary[result.status] = []
                error_summary[result.status].append(result.session_id)
        return error_summary

    def print_audit_summary(self, audit_summary: Dict) -> None:
        """Print formatted audit summary to console"""
        print(f"\n📊 Oracle Audit Summary")
        print(f"Symbol: {audit_summary['symbol']}")
        print(f"Timeframe: {audit_summary['timeframe']}")
        print(f"Date Range: {audit_summary['date_range']}")
        print(f"Sessions Discovered: {audit_summary['sessions_discovered']}")
        print(f"Sessions Processable: {audit_summary['audit_successful']}")
        print(f"Success Rate: {audit_summary['success_rate']:.1%}")
        print(f"Minimum Required: {audit_summary['min_sessions_required']}")
        print(f"Training Ready: {'✅' if audit_summary['meets_minimum'] else '❌'}")

        # Error breakdown
        error_breakdown = audit_summary['error_breakdown']
        if any(count > 0 for code, count in error_breakdown.items() if code != 'SUCCESS'):
            print(f"\n❌ Error Breakdown:")
            for error_code, count in error_breakdown.items():
                if count > 0 and error_code != 'SUCCESS':
                    description = ERROR_CODES.get(error_code, 'Unknown error')
                    print(f"  {error_code}: {count} sessions - {description}")

    def validate_training_readiness(self, min_sessions: int = 57,
                                  min_success_rate: float = 0.8) -> Tuple[bool, List[str]]:
        """
        Validate if data is ready for Oracle training

        Args:
            min_sessions: Minimum required processable sessions
            min_success_rate: Minimum required success rate

        Returns:
            Tuple of (is_ready, list_of_issues)
        """
        issues = []

        successful_sessions = self.error_counts.get('SUCCESS', 0)
        total_sessions = len(self.audit_results)
        success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0.0

        if successful_sessions < min_sessions:
            issues.append(f"Insufficient sessions: {successful_sessions} < {min_sessions}")

        if success_rate < min_success_rate:
            issues.append(f"Low success rate: {success_rate:.1%} < {min_success_rate:.1%}")

        if total_sessions == 0:
            issues.append("No sessions discovered")

        return len(issues) == 0, issues
