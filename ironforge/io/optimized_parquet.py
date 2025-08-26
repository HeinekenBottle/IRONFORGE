"""
Optimized Parquet I/O with PyArrow Context7-Guided Improvements
Safe opt-in optimizations for E2R (Execution to Results) layer
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from ..learning.translation_config import get_global_config


logger = logging.getLogger(__name__)


class OptimizedParquetWriter:
    """
    Enhanced Parquet writer with PyArrow-guided optimizations.
    
    Features (opt-in via flags):
    - Content-defined chunking (CDC) for optimal compression
    - Row group size optimization based on query patterns
    - Schema evolution tracking for incremental updates
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = get_global_config()
        if config_override:
            # Allow runtime config override for testing
            for key, value in config_override.items():
                if hasattr(self.config.parquet, key):
                    setattr(self.config.parquet, key, value)
                    
        self._schema_history: Dict[str, pa.Schema] = {}
        self._row_group_stats: Dict[str, List[int]] = {}
        
        logger.info(f"OptimizedParquetWriter initialized, CDC: {self.config.enable_optimized_parquet}")
    
    def write_motif_results(self, results: pd.DataFrame, 
                           output_path: str,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Write motif analysis results with optional optimizations
        
        Args:
            results: DataFrame with motif analysis results
            output_path: Path to write parquet file
            metadata: Optional metadata to include in schema
        """
        if not self.config.enable_optimized_parquet:
            # Use standard parquet write without optimizations
            self._write_standard(results, output_path, metadata)
            return
            
        try:
            self._write_optimized(results, output_path, metadata)
        except Exception as e:
            logger.warning(f"Optimized write failed: {e}, falling back to standard")
            self._write_standard(results, output_path, metadata)
    
    def _write_standard(self, results: pd.DataFrame, 
                       output_path: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Standard parquet write (fallback)"""
        table = pa.Table.from_pandas(results)
        
        if metadata:
            # Add metadata to schema
            schema_metadata = {str(k): str(v) for k, v in metadata.items()}
            table = table.replace_schema_metadata(schema_metadata)
            
        pq.write_table(table, output_path)
        logger.info(f"Written {len(results)} motif results to {output_path} (standard)")
    
    def _write_optimized(self, results: pd.DataFrame,
                        output_path: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Optimized parquet write with Context7 best practices"""
        
        # Convert to Arrow table
        table = pa.Table.from_pandas(results)
        
        # Apply schema optimization
        if self.config.parquet.enable_cdc:
            table = self._apply_schema_evolution(table, output_path, metadata)
            
        # Prepare write options
        write_options = self._prepare_write_options()
        
        # Write with optimizations
        if self.config.parquet.enable_content_defined_chunking:
            # Use PyArrow's content-defined chunking
            cdc_options = {
                'min_chunk_size': self.config.parquet.cdc_min_chunk_size,
                'max_chunk_size': self.config.parquet.cdc_max_chunk_size,
            }
            
            pq.write_table(
                table, 
                output_path,
                use_content_defined_chunking=cdc_options,
                **write_options
            )
        else:
            pq.write_table(table, output_path, **write_options)
            
        # Track row group statistics
        if self.config.parquet.enable_row_group_optimization:
            self._update_row_group_stats(output_path, len(results))
            
        logger.info(f"Written {len(results)} motif results to {output_path} (optimized)")
    
    def _apply_schema_evolution(self, table: pa.Table, 
                               output_path: str,
                               metadata: Optional[Dict[str, Any]] = None) -> pa.Table:
        """Apply schema evolution and CDC tracking"""
        
        # Generate schema fingerprint
        schema_hash = self._compute_schema_hash(table.schema)
        
        # Check for schema changes
        path_key = str(Path(output_path).parent)
        if path_key in self._schema_history:
            previous_hash = self._compute_schema_hash(self._schema_history[path_key])
            if schema_hash != previous_hash:
                logger.info(f"Schema evolution detected for {path_key}")
                
        # Store current schema
        self._schema_history[path_key] = table.schema
        
        # Add CDC metadata
        cdc_metadata = {
            'ironforge.schema_hash': schema_hash,
            'ironforge.write_timestamp': pd.Timestamp.now().isoformat(),
            'ironforge.cdc_enabled': 'true'
        }
        
        if metadata:
            cdc_metadata.update({f'ironforge.{k}': str(v) for k, v in metadata.items()})
            
        # Apply metadata to schema
        enhanced_schema = table.schema.with_metadata(cdc_metadata)
        return table.cast(enhanced_schema)
    
    def _prepare_write_options(self) -> Dict[str, Any]:
        """Prepare PyArrow write options based on configuration"""
        options = {
            'compression': 'snappy',  # Good balance of speed/compression
            'version': '2.6',  # Modern Parquet version
        }
        
        if self.config.parquet.enable_row_group_optimization:
            # Optimize row group size for query patterns
            options['row_group_size'] = self.config.parquet.target_row_group_size
            
        return options
    
    def _compute_schema_hash(self, schema: pa.Schema) -> str:
        """Compute deterministic hash of schema structure"""
        schema_dict = {
            'fields': [
                {
                    'name': field.name,
                    'type': str(field.type),
                    'nullable': field.nullable
                }
                for field in schema
            ]
        }
        
        schema_json = json.dumps(schema_dict, sort_keys=True)
        return hashlib.md5(schema_json.encode()).hexdigest()[:16]
    
    def _update_row_group_stats(self, file_path: str, row_count: int) -> None:
        """Update row group statistics for optimization"""
        if file_path not in self._row_group_stats:
            self._row_group_stats[file_path] = []
            
        self._row_group_stats[file_path].append(row_count)
        
        # Log statistics periodically
        if len(self._row_group_stats[file_path]) % 10 == 0:
            stats = self._row_group_stats[file_path]
            logger.info(f"Row group stats for {file_path}: "
                       f"mean={np.mean(stats):.0f}, std={np.std(stats):.0f}")
    
    def read_motif_results(self, file_path: str, 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read motif results with optional optimizations
        
        Args:
            file_path: Path to parquet file  
            columns: Optional list of columns to read
            
        Returns:
            DataFrame with motif results
        """
        if not self.config.enable_optimized_parquet:
            # Standard read
            return pd.read_parquet(file_path, columns=columns)
            
        try:
            return self._read_optimized(file_path, columns)
        except Exception as e:
            logger.warning(f"Optimized read failed: {e}, falling back to standard")
            return pd.read_parquet(file_path, columns=columns)
    
    def _read_optimized(self, file_path: str, 
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Optimized parquet read with memory mapping"""
        
        # Use memory mapping for potential performance benefits
        table = pq.read_table(
            file_path, 
            columns=columns,
            memory_map=True  # PyArrow best practice for large files
        )
        
        # Check for CDC metadata
        if table.schema.metadata:
            metadata = table.schema.metadata
            if b'ironforge.cdc_enabled' in metadata:
                logger.debug(f"CDC metadata found: {metadata}")
                
        return table.to_pandas()
    
    def get_schema_info(self, file_path: str) -> Dict[str, Any]:
        """Get schema information and CDC metadata"""
        try:
            metadata = pq.read_metadata(file_path)
            schema = metadata.schema.to_arrow_schema()
            
            info = {
                'num_columns': len(schema),
                'num_rows': metadata.num_rows,
                'num_row_groups': metadata.num_row_groups,
                'file_size_bytes': Path(file_path).stat().st_size,
                'schema_hash': self._compute_schema_hash(schema)
            }
            
            # Extract CDC metadata if present
            if schema.metadata:
                cdc_keys = [k for k in schema.metadata.keys() 
                           if k.startswith(b'ironforge.')]
                if cdc_keys:
                    info['cdc_metadata'] = {
                        k.decode(): v.decode() 
                        for k, v in schema.metadata.items()
                        if k in cdc_keys
                    }
                    
            return info
            
        except Exception as e:
            logger.error(f"Failed to read schema info: {e}")
            return {'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get writer statistics"""
        return {
            'optimizations_enabled': self.config.enable_optimized_parquet,
            'cdc_enabled': self.config.parquet.enable_cdc,
            'content_chunking_enabled': self.config.parquet.enable_content_defined_chunking,
            'schemas_tracked': len(self._schema_history),
            'files_written': len(self._row_group_stats),
            'avg_row_group_size': np.mean([
                np.mean(stats) for stats in self._row_group_stats.values()
            ]) if self._row_group_stats else 0
        }