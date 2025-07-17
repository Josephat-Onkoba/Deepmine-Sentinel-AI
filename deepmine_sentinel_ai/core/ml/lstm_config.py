"""
LSTM Model Configuration Classes

Configuration classes for LSTM architecture design and training
Task 8: Design LSTM Architecture configuration
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class LSTMConfig:
    """
    Core LSTM model configuration
    
    Defines architecture parameters for stability prediction models
    """
    # Model Architecture
    input_features: int = 59  # Features from Task 7 feature engineering
    sequence_length: int = 168  # One week of hourly data
    lstm_units: List[int] = None  # LSTM layer sizes
    dense_units: List[int] = None  # Dense layer sizes
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1
    
    # Output Configuration
    num_classes: int = 4  # STABLE, ELEVATED, HIGH_RISK, CRITICAL
    prediction_horizons: List[int] = None  # Hours ahead: [24, 48, 72, 168]
    
    # Attention Configuration
    use_attention: bool = True
    attention_units: int = 64
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Multi-output Configuration
    multi_output: bool = True
    output_names: List[str] = None
    
    def __post_init__(self):
        """Set default values for mutable fields"""
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]
            
        if self.dense_units is None:
            self.dense_units = [64, 32]
            
        if self.prediction_horizons is None:
            self.prediction_horizons = [24, 48, 72, 168]
            
        if self.output_names is None:
            self.output_names = [
                f'risk_{h}h' for h in self.prediction_horizons
            ]


@dataclass
class AttentionConfig:
    """
    Attention mechanism configuration
    
    Defines parameters for attention layers in LSTM models
    """
    attention_type: str = 'multi_head'  # 'basic' or 'multi_head'
    num_heads: int = 8
    key_dim: int = 64
    value_dim: Optional[int] = None
    dropout_rate: float = 0.1
    use_bias: bool = True
    
    # Domain-specific attention for mining events
    use_domain_attention: bool = True
    event_types: List[str] = None
    event_embedding_dim: int = 16
    
    def __post_init__(self):
        """Set default values for mutable fields"""
        if self.value_dim is None:
            self.value_dim = self.key_dim
            
        if self.event_types is None:
            self.event_types = [
                'background', 'seismic', 'blast', 
                'maintenance', 'equipment', 'environmental'
            ]


@dataclass 
class MultiStepConfig:
    """
    Multi-step prediction configuration
    
    Defines parameters for different prediction horizons
    """
    horizons: List[int] = None  # Prediction horizons in hours
    horizon_names: List[str] = None  # Names for each horizon
    
    # Training strategy
    training_strategy: str = 'joint'  # 'joint', 'sequential', 'independent'
    use_teacher_forcing: bool = True
    teacher_forcing_ratio: float = 0.5
    
    # Recursive prediction
    max_recursive_steps: int = 72  # Maximum steps for recursive prediction
    recursive_strategy: str = 'autoregressive'  # 'autoregressive', 'direct'
    
    def __post_init__(self):
        """Set default values for mutable fields"""
        if self.horizons is None:
            self.horizons = [24, 48, 72, 168]
            
        if self.horizon_names is None:
            self.horizon_names = [
                f'horizon_{h}h' for h in self.horizons
            ]


@dataclass
class ModelArchitectureConfig:
    """
    Complete model architecture configuration
    
    Combines all configuration aspects for complete LSTM models
    """
    model_type: str = 'complete'  # Model type identifier
    
    # Sub-configurations
    lstm_config: LSTMConfig = None
    attention_config: AttentionConfig = None
    multi_step_config: MultiStepConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided"""
        if self.lstm_config is None:
            self.lstm_config = LSTMConfig()
            
        if self.attention_config is None:
            self.attention_config = AttentionConfig()
            
        if self.multi_step_config is None:
            self.multi_step_config = MultiStepConfig()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of complete model configuration"""
        return {
            'model_type': self.model_type,
            'input_shape': (self.lstm_config.sequence_length, self.lstm_config.input_features),
            'lstm_layers': len(self.lstm_config.lstm_units),
            'lstm_units': self.lstm_config.lstm_units,
            'dense_units': self.lstm_config.dense_units,
            'dropout_rate': self.lstm_config.dropout_rate,
            'num_classes': self.lstm_config.num_classes,
            'prediction_horizons': self.lstm_config.prediction_horizons,
            'attention': {
                'enabled': self.lstm_config.use_attention,
                'type': self.attention_config.attention_type,
                'num_heads': self.attention_config.num_heads,
                'key_dim': self.attention_config.key_dim
            },
            'multi_step': {
                'horizons': self.multi_step_config.horizons,
                'training_strategy': self.multi_step_config.training_strategy,
                'use_teacher_forcing': self.multi_step_config.use_teacher_forcing
            }
        }
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return {
            'model_type': self.model_type,
            'lstm': {
                'input_features': self.lstm_config.input_features,
                'sequence_length': self.lstm_config.sequence_length,
                'lstm_units': self.lstm_config.lstm_units,
                'dense_units': self.lstm_config.dense_units,
                'dropout_rate': self.lstm_config.dropout_rate,
                'num_classes': self.lstm_config.num_classes
            },
            'attention': {
                'use_attention': self.lstm_config.use_attention,
                'attention_type': self.attention_config.attention_type,
                'num_heads': self.attention_config.num_heads,
                'key_dim': self.attention_config.key_dim
            },
            'multi_step': {
                'horizons': self.multi_step_config.horizons,
                'training_strategy': self.multi_step_config.training_strategy,
                'use_teacher_forcing': self.multi_step_config.use_teacher_forcing
            }
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for validation"""
        return {
            'model_type': self.model_type,
            'lstm': {
                'input_features': self.lstm_config.input_features,
                'sequence_length': self.lstm_config.sequence_length,
                'lstm_units': self.lstm_config.lstm_units,
                'dense_units': self.lstm_config.dense_units,
                'dropout_rate': self.lstm_config.dropout_rate,
                'num_classes': self.lstm_config.num_classes
            },
            'attention': {
                'use_attention': self.lstm_config.use_attention,
                'attention_type': self.attention_config.attention_type,
                'num_heads': self.attention_config.num_heads,
                'key_dim': self.attention_config.key_dim
            },
            'multi_step': {
                'horizons': self.multi_step_config.horizons,
                'training_strategy': self.multi_step_config.training_strategy,
                'use_teacher_forcing': self.multi_step_config.use_teacher_forcing
            }
        }


@dataclass
class LSTMModelConfig:
    """Configuration for LSTM model training and optimization"""
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    loss_function: str = 'sparse_categorical_crossentropy'
    metrics: List[str] = None
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    class_weight: Optional[Dict[int, float]] = None
    sample_weight: Optional[str] = None
    
    def __post_init__(self):
        """Set default values for mutable fields"""
        if self.metrics is None:
            self.metrics = ['accuracy']
