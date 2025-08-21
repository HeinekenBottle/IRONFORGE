#!/usr/bin/env python3
"""
Oracle Session Data Models
Centralized data models and structures for Oracle training system
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

from ..core.constants import QUALITY_LEVELS, SESSION_TYPES


@dataclass
class SessionMetadata:
    """Metadata for a trading session"""
    session_id: str
    symbol: str
    timeframe: str
    session_type: str
    session_date: str  # YYYY-MM-DD format
    node_count: int
    edge_count: int
    quality_level: str = "fair"
    data_source: str = "parquet"  # "parquet" or "enhanced"
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate metadata after initialization"""
        if self.session_type.upper() not in SESSION_TYPES:
            raise ValueError(f"Invalid session_type: {self.session_type}")
        
        if self.quality_level.lower() not in QUALITY_LEVELS:
            raise ValueError(f"Invalid quality_level: {self.quality_level}")
        
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def quality_score(self) -> int:
        """Get numeric quality score"""
        return QUALITY_LEVELS[self.quality_level.lower()]
    
    @property
    def date_obj(self) -> date:
        """Get session date as date object"""
        return datetime.strptime(self.session_date, "%Y-%m-%d").date()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'session_type': self.session_type,
            'session_date': self.session_date,
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'quality_level': self.quality_level,
            'data_source': self.data_source,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """Create from dictionary"""
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class TrainingPair:
    """A single training pair for Oracle model"""
    session_id: str
    symbol: str
    timeframe: str
    session_date: str
    early_pct: float
    early_embedding: np.ndarray  # TGAT embedding from early portion
    target_center: float
    target_half_range: float
    htf_mode: bool = False
    n_events: int = 0
    pattern_id: Optional[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate training pair after initialization"""
        if not 0.0 < self.early_pct <= 1.0:
            raise ValueError(f"Invalid early_pct: {self.early_pct}")
        
        if self.early_embedding.shape[0] != 44:  # TGAT embedding dimension
            raise ValueError(f"Invalid embedding dimension: {self.early_embedding.shape[0]}")
        
        if self.target_half_range < 0:
            raise ValueError(f"Invalid target_half_range: {self.target_half_range}")
    
    @property
    def target_low(self) -> float:
        """Calculate target low from center and half_range"""
        return self.target_center - self.target_half_range
    
    @property
    def target_high(self) -> float:
        """Calculate target high from center and half_range"""
        return self.target_center + self.target_half_range
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'session_date': self.session_date,
            'early_pct': self.early_pct,
            'early_embedding': self.early_embedding.tolist(),
            'target_center': self.target_center,
            'target_half_range': self.target_half_range,
            'htf_mode': self.htf_mode,
            'n_events': self.n_events,
            'pattern_id': self.pattern_id,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingPair':
        """Create from dictionary"""
        data['early_embedding'] = np.array(data['early_embedding'])
        return cls(**data)


@dataclass
class AuditResult:
    """Result of session audit operation"""
    session_id: str
    status: str  # SUCCESS or error code
    error_message: Optional[str] = None
    metadata: Optional[SessionMetadata] = None
    shard_path: Optional[Path] = None
    audit_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.audit_timestamp is None:
            self.audit_timestamp = datetime.now()
    
    @property
    def is_success(self) -> bool:
        """Check if audit was successful"""
        return self.status == 'SUCCESS'
    
    @property
    def is_processable(self) -> bool:
        """Check if session is processable for training"""
        return self.is_success and self.metadata is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'status': self.status,
            'error_message': self.error_message,
            'metadata': self.metadata.to_dict() if self.metadata else None,
            'shard_path': str(self.shard_path) if self.shard_path else None,
            'audit_timestamp': self.audit_timestamp.isoformat() if self.audit_timestamp else None
        }


@dataclass
class TrainingManifest:
    """Manifest for Oracle training run"""
    version: str
    training_date: datetime
    model_path: str
    scaler_path: str
    training_sessions: List[str]
    validation_sessions: List[str]
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    early_pct: float
    total_training_pairs: int
    total_validation_pairs: int
    training_duration_seconds: float
    git_commit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'version': self.version,
            'training_date': self.training_date.isoformat(),
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'training_sessions': self.training_sessions,
            'validation_sessions': self.validation_sessions,
            'hyperparameters': self.hyperparameters,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'early_pct': self.early_pct,
            'total_training_pairs': self.total_training_pairs,
            'total_validation_pairs': self.total_validation_pairs,
            'training_duration_seconds': self.training_duration_seconds,
            'git_commit': self.git_commit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingManifest':
        """Create from dictionary"""
        data['training_date'] = datetime.fromisoformat(data['training_date'])
        return cls(**data)


@dataclass
class OraclePrediction:
    """Oracle model prediction result"""
    session_id: str
    session_date: str
    pct_seen: float
    n_events: int
    pred_center: float
    pred_half_range: float
    confidence: float
    pattern_id: Optional[str] = None
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    early_stats: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.early_stats is None:
            self.early_stats = {}
    
    @property
    def pred_low(self) -> float:
        """Calculate predicted low"""
        return self.pred_center - self.pred_half_range
    
    @property
    def pred_high(self) -> float:
        """Calculate predicted high"""
        return self.pred_center + self.pred_half_range
    
    @property
    def pred_range(self) -> float:
        """Calculate predicted range"""
        return 2 * self.pred_half_range
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'session_date': self.session_date,
            'pct_seen': self.pct_seen,
            'n_events': self.n_events,
            'pred_center': self.pred_center,
            'pred_half_range': self.pred_half_range,
            'pred_low': self.pred_low,
            'pred_high': self.pred_high,
            'confidence': self.confidence,
            'pattern_id': self.pattern_id,
            'start_timestamp': self.start_timestamp.isoformat() if self.start_timestamp else None,
            'end_timestamp': self.end_timestamp.isoformat() if self.end_timestamp else None,
            'early_stats': self.early_stats
        }


@dataclass
class TrainingDataset:
    """Container for Oracle training dataset"""
    training_pairs: List[TrainingPair]
    validation_pairs: List[TrainingPair]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_pairs(self) -> int:
        """Total number of training pairs"""
        return len(self.training_pairs) + len(self.validation_pairs)
    
    @property
    def training_sessions(self) -> List[str]:
        """Unique session IDs in training set"""
        return list(set(pair.session_id for pair in self.training_pairs))
    
    @property
    def validation_sessions(self) -> List[str]:
        """Unique session IDs in validation set"""
        return list(set(pair.session_id for pair in self.validation_pairs))
    
    def get_training_dataframe(self) -> pd.DataFrame:
        """Convert training pairs to DataFrame"""
        return self._pairs_to_dataframe(self.training_pairs)
    
    def get_validation_dataframe(self) -> pd.DataFrame:
        """Convert validation pairs to DataFrame"""
        return self._pairs_to_dataframe(self.validation_pairs)
    
    def _pairs_to_dataframe(self, pairs: List[TrainingPair]) -> pd.DataFrame:
        """Convert list of training pairs to DataFrame"""
        if not pairs:
            return pd.DataFrame()
        
        # Extract basic columns
        data = []
        for pair in pairs:
            row = {
                'symbol': pair.symbol,
                'tf': pair.timeframe,
                'session_date': pair.session_date,
                'htf_mode': pair.htf_mode,
                'early_pct': pair.early_pct,
                'target_center': pair.target_center,
                'target_half_range': pair.target_half_range
            }
            
            # Add embedding columns
            for i, val in enumerate(pair.early_embedding):
                row[f'pooled_{i:02d}'] = val
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'training_pairs': [pair.to_dict() for pair in self.training_pairs],
            'validation_pairs': [pair.to_dict() for pair in self.validation_pairs],
            'metadata': self.metadata
        }
