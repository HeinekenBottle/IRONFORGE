"""
Context7-Optimized Parquet I/O Module  
Enhanced Parquet operations with performance optimizations

Key optimizations:
1. ZSTD compression with optimal levels
2. Content-defined chunking  
3. Optimized row group sizes
4. Data type normalization
5. Batch operations
6. Memory-mapped reading
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pyarrow import fs

logger = logging.getLogger(__name__)


@dataclass
class OptimizedParquetConfig:
    """Configuration for Context7 Parquet optimizations"""
    
    # Compression optimization (Context7 recommendations)
    compression: str = 'zstd'
    compression_level: int = 3  # Optimal balance from Context7
    
    # Row group optimization  
    row_group_size: int = 10000  # Context7 recommended size
    
    # Content-defined chunking
    enable_content_chunking: bool = True
    min_chunk_size: int = 256 * 1024  # 256 KiB
    max_chunk_size: int = 1024 * 1024  # 1 MiB
    
    # Data type optimization
    optimize_dtypes: bool = True
    use_dictionary_encoding: List[str] = None  # Columns to dictionary encode
    
    # Performance options
    use_memory_map: bool = True
    enable_parallel_io: bool = True
    max_open_files: int = 900  # Context7 recommendation
    
    # File organization
    partition_by_session: bool = True
    write_metadata: bool = True
    
    def __post_init__(self):
        if self.use_dictionary_encoding is None:
            self.use_dictionary_encoding = ['event_type', 'session_id']


class OptimizedParquetWriter:
    """
    Context7-optimized Parquet writer with advanced performance features
    """
    
    def __init__(self, config: OptimizedParquetConfig):
        self.config = config
        self._metadata_collector = []
        
    def write_table(self, table: pa.Table, file_path: Union[str, Path], 
                   **kwargs) -> Dict[str, Any]:
        """
        Write PyArrow table with Context7 optimizations
        
        Args:
            table: PyArrow table to write
            file_path: Output file path
            **kwargs: Additional write options
            
        Returns:
            Write statistics and metadata
        """
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Optimize table schema and data
        optimized_table = self._optimize_table(table)
        
        # Prepare write options
        write_options = self._get_write_options(**kwargs)
        
        # Write with optimizations
        start_time = pd.Timestamp.now()
        
        if self.config.enable_content_chunking:
            pq.write_table(
                optimized_table, 
                file_path,
                **write_options,
                use_content_defined_chunking={
                    'min_chunk_size': self.config.min_chunk_size,
                    'max_chunk_size': self.config.max_chunk_size
                },
                metadata_collector=self._metadata_collector
            )
        else:
            pq.write_table(
                optimized_table,
                file_path, 
                **write_options,
                metadata_collector=self._metadata_collector
            )
            
        write_time = pd.Timestamp.now() - start_time
        
        # Collect statistics
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        stats = {
            'file_path': str(file_path),
            'file_size_bytes': file_size,
            'write_time_seconds': write_time.total_seconds(),
            'num_rows': len(optimized_table),
            'num_columns': len(optimized_table.schema),
            'compression_ratio': len(table) * len(table.schema) * 8 / max(file_size, 1)  # Rough estimate
        }
        
        logger.info(f"Wrote Parquet file: {file_path.name} "
                   f"({file_size / (1024**2):.2f} MB, "
                   f"{write_time.total_seconds():.3f}s)")
        
        return stats
        
    def write_dataset(self, table: pa.Table, base_dir: Union[str, Path],
                     partition_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Write partitioned dataset with Context7 optimizations
        
        Args:
            table: PyArrow table to write
            base_dir: Base directory for dataset
            partition_cols: Columns to partition by
            
        Returns:
            Dataset write statistics
        """
        
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimize table
        optimized_table = self._optimize_table(table)
        
        # Setup partitioning
        partitioning = None
        if partition_cols:
            partition_fields = [
                pa.field(col, optimized_table.schema.field(col).type)
                for col in partition_cols if col in optimized_table.column_names
            ]
            if partition_fields:
                partitioning = ds.partitioning(pa.schema(partition_fields), flavor="hive")
                
        # Write dataset with optimizations
        start_time = pd.Timestamp.now()
        
        ds.write_dataset(
            optimized_table,
            base_dir,
            format="parquet",
            partitioning=partitioning,
            max_open_files=self.config.max_open_files,
            max_rows_per_file=50000,  # Context7 recommendation for file size control
            min_rows_per_group=self.config.row_group_size,
            max_rows_per_group=self.config.row_group_size * 2,
            file_options=self._get_dataset_file_options()
        )
        
        write_time = pd.Timestamp.now() - start_time
        
        # Write metadata files if enabled
        if self.config.write_metadata:
            self._write_metadata_files(optimized_table, base_dir)
            
        # Collect statistics
        total_size = sum(f.stat().st_size for f in base_dir.rglob('*.parquet'))
        
        stats = {
            'base_dir': str(base_dir),
            'total_size_bytes': total_size,
            'write_time_seconds': write_time.total_seconds(),
            'num_rows': len(optimized_table),
            'num_columns': len(optimized_table.schema),
            'num_partitions': len(list(base_dir.rglob('*.parquet')))
        }
        
        logger.info(f"Wrote partitioned dataset: {base_dir.name} "
                   f"({total_size / (1024**2):.2f} MB, "
                   f"{stats['num_partitions']} files, "
                   f"{write_time.total_seconds():.3f}s)")
        
        return stats
        
    def _optimize_table(self, table: pa.Table) -> pa.Table:
        """Optimize table schema and data types"""
        
        if not self.config.optimize_dtypes:
            return table
            
        optimized_columns = []
        optimized_names = []
        
        for i, column_name in enumerate(table.column_names):
            column = table.column(i)
            optimized_column = self._optimize_column(column, column_name)
            optimized_columns.append(optimized_column)
            optimized_names.append(column_name)
            
        return pa.table(optimized_columns, names=optimized_names)
        
    def _optimize_column(self, column: pa.ChunkedArray, column_name: str) -> pa.ChunkedArray:
        """Optimize individual column data type and encoding"""
        
        # Dictionary encoding for categorical columns
        if column_name in self.config.use_dictionary_encoding:
            if not pa.types.is_dictionary(column.type):
                try:
                    return pa.chunked_array([
                        chunk.dictionary_encode() for chunk in column.chunks
                    ])
                except Exception as e:
                    logger.warning(f"Failed to dictionary encode column {column_name}: {e}")
                    
        # Optimize numeric types
        if pa.types.is_integer(column.type):
            return self._optimize_integer_column(column, column_name)
        elif pa.types.is_floating(column.type):
            return self._optimize_float_column(column, column_name)
            
        return column
        
    def _optimize_integer_column(self, column: pa.ChunkedArray, column_name: str) -> pa.ChunkedArray:
        """Optimize integer column precision"""
        
        # Find min/max values to determine optimal type
        try:
            min_val = pa.compute.min(column).as_py()
            max_val = pa.compute.max(column).as_py()
            
            if min_val is None or max_val is None:
                return column
                
            # Choose optimal integer type
            if min_val >= 0:
                # Unsigned types
                if max_val <= 255:
                    target_type = pa.uint8()
                elif max_val <= 65535:
                    target_type = pa.uint16()
                elif max_val <= 4294967295:
                    target_type = pa.uint32()
                else:
                    target_type = pa.uint64()
            else:
                # Signed types
                if min_val >= -128 and max_val <= 127:
                    target_type = pa.int8()
                elif min_val >= -32768 and max_val <= 32767:
                    target_type = pa.int16()
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    target_type = pa.int32()
                else:
                    target_type = pa.int64()
                    
            # Cast if beneficial (smaller type)
            if target_type != column.type:
                return column.cast(target_type)
                
        except Exception as e:
            logger.debug(f"Could not optimize integer column {column_name}: {e}")
            
        return column
        
    def _optimize_float_column(self, column: pa.ChunkedArray, column_name: str) -> pa.ChunkedArray:
        """Optimize float column precision"""
        
        # For financial data, float32 is often sufficient
        if pa.types.is_float64(column.type):
            # Check if values fit in float32 range
            try:
                min_val = pa.compute.min(column).as_py()
                max_val = pa.compute.max(column).as_py()
                
                if (min_val is not None and max_val is not None and
                    abs(min_val) < 3.4e38 and abs(max_val) < 3.4e38):
                    
                    return column.cast(pa.float32())
            except Exception as e:
                logger.debug(f"Could not optimize float column {column_name}: {e}")
                
        return column
        
    def _get_write_options(self, **kwargs) -> Dict[str, Any]:
        """Get optimized write options"""
        
        options = {
            'compression': self.config.compression,
            'compression_level': self.config.compression_level,
            'row_group_size': self.config.row_group_size,
            'use_dictionary': self.config.use_dictionary_encoding,
            'version': '2.6',  # Use modern Parquet version
            'data_page_version': '2.0',
        }
        
        # Override with any explicit kwargs
        options.update(kwargs)
        
        return options
        
    def _get_dataset_file_options(self) -> ds.ParquetFileFormat:
        """Get file format options for dataset writing"""
        
        parquet_format = ds.ParquetFileFormat()
        write_options = parquet_format.make_write_options(
            compression=self.config.compression,
            compression_level=self.config.compression_level,
            use_dictionary=self.config.use_dictionary_encoding,
            version='2.6'
        )
        
        return write_options
        
    def _write_metadata_files(self, table: pa.Table, base_dir: Path):
        """Write _metadata and _common_metadata files for dataset"""
        
        try:
            # Write common metadata (schema only)
            pq.write_metadata(
                table.schema,
                base_dir / '_common_metadata'
            )
            
            # Write full metadata if we have collected metadata
            if self._metadata_collector:
                pq.write_metadata(
                    table.schema,
                    base_dir / '_metadata',
                    metadata_collector=self._metadata_collector
                )
                
        except Exception as e:
            logger.warning(f"Failed to write metadata files: {e}")


class OptimizedParquetReader:
    """
    Context7-optimized Parquet reader with performance enhancements
    """
    
    def __init__(self, config: OptimizedParquetConfig):
        self.config = config
        
    def read_table(self, file_path: Union[str, Path], 
                  columns: Optional[List[str]] = None,
                  filters: Optional[List[Tuple]] = None,
                  **kwargs) -> Tuple[pa.Table, Dict[str, Any]]:
        """
        Read Parquet table with Context7 optimizations
        
        Args:
            file_path: Path to Parquet file
            columns: Columns to read (None for all)
            filters: Row filters to apply
            **kwargs: Additional read options
            
        Returns:
            (table, read_stats): Table and read statistics
        """
        
        file_path = Path(file_path)
        
        # Prepare read options
        read_options = {
            'memory_map': self.config.use_memory_map,
            'use_threads': True,  # Enable parallel column decoding
            'columns': columns,
            'filters': filters
        }
        read_options.update(kwargs)
        
        start_time = pd.Timestamp.now()
        
        # Read with optimizations
        table = pq.read_table(file_path, **read_options)
        
        read_time = pd.Timestamp.now() - start_time
        
        # Collect statistics
        file_size = file_path.stat().st_size
        
        stats = {
            'file_path': str(file_path),
            'file_size_bytes': file_size,
            'read_time_seconds': read_time.total_seconds(),
            'num_rows': len(table),
            'num_columns': len(table.schema),
            'throughput_mb_per_sec': (file_size / (1024**2)) / max(read_time.total_seconds(), 0.001)
        }
        
        logger.info(f"Read Parquet file: {file_path.name} "
                   f"({file_size / (1024**2):.2f} MB, "
                   f"{read_time.total_seconds():.3f}s, "
                   f"{stats['throughput_mb_per_sec']:.1f} MB/s)")
        
        return table, stats
        
    def read_dataset(self, dataset_path: Union[str, Path],
                    columns: Optional[List[str]] = None,
                    filters: Optional[List[Tuple]] = None,
                    **kwargs) -> Tuple[pa.Table, Dict[str, Any]]:
        """
        Read Parquet dataset with optimizations
        
        Args:
            dataset_path: Path to dataset directory  
            columns: Columns to read
            filters: Row filters to apply
            **kwargs: Additional read options
            
        Returns:
            (table, read_stats): Combined table and statistics
        """
        
        dataset_path = Path(dataset_path)
        
        start_time = pd.Timestamp.now()
        
        # Create dataset with optimizations
        dataset = ds.dataset(
            dataset_path,
            format="parquet",
            partitioning="hive"  # Auto-detect partitioning
        )
        
        # Read with filters and column selection
        table = dataset.to_table(
            columns=columns,
            filter=ds.dataset._filters_to_expression(filters) if filters else None
        )
        
        read_time = pd.Timestamp.now() - start_time
        
        # Calculate total dataset size
        total_size = sum(f.stat().st_size for f in dataset_path.rglob('*.parquet'))
        num_files = len(list(dataset_path.rglob('*.parquet')))
        
        stats = {
            'dataset_path': str(dataset_path),
            'total_size_bytes': total_size,
            'num_files': num_files,
            'read_time_seconds': read_time.total_seconds(),
            'num_rows': len(table),
            'num_columns': len(table.schema),
            'throughput_mb_per_sec': (total_size / (1024**2)) / max(read_time.total_seconds(), 0.001)
        }
        
        logger.info(f"Read Parquet dataset: {dataset_path.name} "
                   f"({num_files} files, {total_size / (1024**2):.2f} MB, "
                   f"{read_time.total_seconds():.3f}s, "
                   f"{stats['throughput_mb_per_sec']:.1f} MB/s)")
        
        return table, stats
        
    def read_batches(self, file_path: Union[str, Path], 
                    batch_size: int = 10000) -> Iterator[pa.RecordBatch]:
        """Read Parquet file in batches for memory efficiency"""
        
        parquet_file = pq.ParquetFile(file_path)
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield batch


class OptimizedParquetManager:
    """
    High-level manager for optimized Parquet operations
    """
    
    def __init__(self, config: Optional[OptimizedParquetConfig] = None):
        self.config = config or OptimizedParquetConfig()
        self.writer = OptimizedParquetWriter(self.config)
        self.reader = OptimizedParquetReader(self.config)
        
    def save_session_data(self, session_data: Dict[str, Any], 
                         output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Save session data with optimal Parquet format
        
        Args:
            session_data: Session data dictionary
            output_dir: Output directory
            
        Returns:
            Save operation statistics
        """
        
        output_dir = Path(output_dir) 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert session data to PyArrow table
        table = self._session_data_to_table(session_data)
        
        # Determine output strategy
        if self.config.partition_by_session and 'session_id' in table.column_names:
            # Write as partitioned dataset
            return self.writer.write_dataset(
                table, 
                output_dir,
                partition_cols=['session_id']
            )
        else:
            # Write as single file
            session_id = session_data.get('session_id', 'unknown')
            output_file = output_dir / f"session_{session_id}.parquet"
            return self.writer.write_table(table, output_file)
            
    def load_session_data(self, input_path: Union[str, Path],
                         session_id: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load session data from optimized Parquet format
        
        Args:
            input_path: Path to Parquet file or dataset
            session_id: Specific session ID to load (for datasets)
            
        Returns:
            (session_data, load_stats): Session data and load statistics
        """
        
        input_path = Path(input_path)
        
        # Determine if it's a file or dataset
        if input_path.is_file():
            table, stats = self.reader.read_table(input_path)
        else:
            # Dataset - apply session filter if specified
            filters = []
            if session_id:
                filters = [('session_id', '=', session_id)]
                
            table, stats = self.reader.read_dataset(input_path, filters=filters)
            
        # Convert table back to session data format
        session_data = self._table_to_session_data(table)
        
        return session_data, stats
        
    def _session_data_to_table(self, session_data: Dict[str, Any]) -> pa.Table:
        """Convert session data dictionary to PyArrow table"""
        
        # Extract events and metadata
        events = session_data.get('events', [])
        session_id = session_data.get('session_id', 'unknown')
        
        if not events:
            # Create minimal table
            return pa.table({
                'session_id': [session_id],
                'num_events': [0]
            })
            
        # Convert events to columnar format
        event_data = {
            'session_id': [session_id] * len(events),
            'event_id': list(range(len(events))),
            'timestamp_et': [event.get('timestamp_et') for event in events],
            'event_type': [event.get('event_type', 'unknown') for event in events],
            'price_level': [event.get('price_level', 0.0) for event in events],
            'volume_profile': [event.get('volume_profile', 0.0) for event in events],
        }
        
        # Add node features if available
        if events and 'node_features' in events[0]:
            features_matrix = np.array([
                event.get('node_features', np.zeros(45)) for event in events
            ])
            event_data['node_features'] = features_matrix.tolist()
            
        return pa.table(event_data)
        
    def _table_to_session_data(self, table: pa.Table) -> Dict[str, Any]:
        """Convert PyArrow table back to session data format"""
        
        # Convert to pandas for easier manipulation
        df = table.to_pandas()
        
        if len(df) == 0:
            return {'events': [], 'session_id': 'unknown'}
            
        # Extract session metadata
        session_id = df['session_id'].iloc[0] if 'session_id' in df.columns else 'unknown'
        
        # Convert rows back to events
        events = []
        for _, row in df.iterrows():
            event = {
                'timestamp_et': row.get('timestamp_et'),
                'event_type': row.get('event_type', 'unknown'),
                'price_level': row.get('price_level', 0.0),
                'volume_profile': row.get('volume_profile', 0.0)
            }
            
            # Add node features if available
            if 'node_features' in row:
                event['node_features'] = np.array(row['node_features'])
                
            events.append(event)
            
        return {
            'session_id': session_id,
            'events': events,
            'num_events': len(events)
        }


# Factory functions
def create_optimized_writer(enable_all_optimizations: bool = True) -> OptimizedParquetWriter:
    """Create optimized Parquet writer with Context7 settings"""
    
    config = OptimizedParquetConfig(
        compression='zstd',
        compression_level=3,
        row_group_size=10000,
        enable_content_chunking=enable_all_optimizations,
        optimize_dtypes=enable_all_optimizations,
        use_memory_map=enable_all_optimizations,
        enable_parallel_io=enable_all_optimizations
    )
    
    return OptimizedParquetWriter(config)


def create_optimized_reader(enable_all_optimizations: bool = True) -> OptimizedParquetReader:
    """Create optimized Parquet reader with Context7 settings"""
    
    config = OptimizedParquetConfig(
        use_memory_map=enable_all_optimizations,
        enable_parallel_io=enable_all_optimizations
    )
    
    return OptimizedParquetReader(config)