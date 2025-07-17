"""
Training Configuration for LSTM Models

This module implements Task 9 training configuration:
- Hyperparameter tuning configuration
- Cross-validation settings
- Training monitoring parameters
- Model checkpointing configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import os
from datetime import datetime


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter tuning"""
    
    # LSTM Architecture Parameters
    lstm_units_options: List[List[int]] = field(default_factory=lambda: [
        [64, 32], [128, 64], [256, 128], [128, 64, 32]
    ])
    dense_units_options: List[List[int]] = field(default_factory=lambda: [
        [64, 32], [128, 64], [256, 128]
    ])
    dropout_rate_options: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    
    # Attention Parameters
    attention_heads_options: List[int] = field(default_factory=lambda: [4, 8, 16])
    attention_key_dim_options: List[int] = field(default_factory=lambda: [32, 64, 128])
    
    # Training Parameters
    learning_rate_options: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01])
    batch_size_options: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    optimizer_options: List[str] = field(default_factory=lambda: ['adam', 'rmsprop', 'adamw'])
    
    # Regularization Parameters
    l1_reg_options: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01])
    l2_reg_options: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01])
    
    # Tuning Strategy
    tuning_strategy: str = 'random_search'  # 'grid_search', 'random_search', 'bayesian'
    max_trials: int = 50
    executions_per_trial: int = 1
    random_state: int = 42
    
    def get_search_space(self) -> Dict[str, Any]:
        """Get hyperparameter search space for tuning"""
        return {
            'lstm_units': self.lstm_units_options,
            'dense_units': self.dense_units_options,
            'dropout_rate': self.dropout_rate_options,
            'attention_heads': self.attention_heads_options,
            'attention_key_dim': self.attention_key_dim_options,
            'learning_rate': self.learning_rate_options,
            'batch_size': self.batch_size_options,
            'optimizer': self.optimizer_options,
            'l1_reg': self.l1_reg_options,
            'l2_reg': self.l2_reg_options
        }


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation"""
    
    # CV Strategy
    cv_type: str = 'time_series'  # 'k_fold', 'time_series', 'stratified'
    n_splits: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    
    # Time Series Specific
    time_series_split_method: str = 'rolling_window'  # 'rolling_window', 'expanding_window'
    min_train_size: Optional[int] = None
    max_train_size: Optional[int] = None
    
    # Stratification
    stratify_column: Optional[str] = None
    shuffle: bool = True
    random_state: int = 42
    
    # Gap handling for time series
    gap_size: int = 0  # Number of periods to skip between train and test
    
    def get_cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration"""
        return {
            'cv_type': self.cv_type,
            'n_splits': self.n_splits,
            'test_size': self.test_size,
            'validation_size': self.validation_size,
            'time_series_split_method': self.time_series_split_method,
            'min_train_size': self.min_train_size,
            'max_train_size': self.max_train_size,
            'stratify_column': self.stratify_column,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'gap_size': self.gap_size
        }


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing and versioning"""
    
    # Checkpoint Settings
    save_best_only: bool = True
    monitor_metric: str = 'val_loss'
    monitor_mode: str = 'min'  # 'min', 'max', 'auto'
    save_freq: str = 'epoch'  # 'epoch' or integer for batch frequency
    verbose: int = 1
    
    # Versioning
    version_format: str = 'v{major}.{minor}.{patch}'
    auto_increment: bool = True
    include_timestamp: bool = True
    
    # Storage Configuration
    checkpoint_dir: str = 'models/checkpoints'
    model_registry_dir: str = 'models/registry'
    backup_dir: str = 'models/backups'
    
    # Retention Policy
    keep_best_n: int = 5
    keep_latest_n: int = 3
    cleanup_old_checkpoints: bool = True
    max_age_days: int = 30
    
    # Metadata
    save_model_config: bool = True
    save_training_history: bool = True
    save_hyperparameters: bool = True
    save_metrics: bool = True
    
    def get_checkpoint_path(self, model_name: str, version: str = None) -> str:
        """Get checkpoint file path"""
        if version is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = f"auto_{timestamp}"
        
        filename = f"{model_name}_{version}.h5"
        return os.path.join(self.checkpoint_dir, filename)
    
    def get_model_registry_path(self, model_name: str, version: str) -> str:
        """Get model registry path"""
        return os.path.join(self.model_registry_dir, model_name, version)


@dataclass
class MonitoringConfig:
    """Configuration for training monitoring and logging"""
    
    # Logging Configuration
    log_level: str = 'INFO'
    log_dir: str = 'logs/training'
    log_to_file: bool = True
    log_to_console: bool = True
    
    # TensorBoard Configuration
    use_tensorboard: bool = True
    tensorboard_dir: str = 'logs/tensorboard'
    histogram_freq: int = 1
    write_graph: bool = True
    write_images: bool = False
    update_freq: str = 'epoch'  # 'batch' or 'epoch'
    
    # Metrics Tracking
    track_custom_metrics: bool = True
    metrics_to_track: List[str] = field(default_factory=lambda: [
        'loss', 'accuracy', 'precision', 'recall', 'f1_score',
        'val_loss', 'val_accuracy', 'learning_rate'
    ])
    
    # Early Stopping
    use_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    restore_best_weights: bool = True
    
    # Learning Rate Scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', 'exponential'
    lr_reduction_factor: float = 0.5
    lr_patience: int = 5
    lr_min_value: float = 1e-7
    
    # Progress Tracking
    show_progress_bar: bool = True
    log_frequency: int = 10  # Log every N batches
    
    # Performance Monitoring
    monitor_gpu_usage: bool = True
    monitor_memory_usage: bool = True
    profile_training: bool = False
    
    def get_callbacks_config(self) -> Dict[str, Any]:
        """Get training callbacks configuration"""
        return {
            'tensorboard': {
                'enabled': self.use_tensorboard,
                'log_dir': self.tensorboard_dir,
                'histogram_freq': self.histogram_freq,
                'write_graph': self.write_graph,
                'write_images': self.write_images,
                'update_freq': self.update_freq
            },
            'early_stopping': {
                'enabled': self.use_early_stopping,
                'patience': self.patience,
                'min_delta': self.min_delta,
                'restore_best_weights': self.restore_best_weights
            },
            'lr_scheduler': {
                'enabled': self.use_lr_scheduler,
                'type': self.lr_scheduler_type,
                'factor': self.lr_reduction_factor,
                'patience': self.lr_patience,
                'min_lr': self.lr_min_value
            }
        }


@dataclass
class TrainingPipelineConfig:
    """Complete training pipeline configuration"""
    
    # Component Configurations
    hyperparameter_config: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    cv_config: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Training Settings
    epochs: int = 100
    verbose: int = 1
    use_mixed_precision: bool = False
    
    # Data Configuration
    sequence_length: int = 24
    prediction_horizons: List[str] = field(default_factory=lambda: ['24h', '48h', '72h'])
    
    # Resource Configuration
    use_gpu: bool = True
    gpu_memory_limit: Optional[int] = None  # MB
    num_workers: int = 4
    
    # Experiment Configuration
    experiment_name: str = 'lstm_training'
    experiment_description: str = 'LSTM model training for mining stability prediction'
    
    def get_complete_config(self) -> Dict[str, Any]:
        """Get complete training pipeline configuration"""
        return {
            'hyperparameters': self.hyperparameter_config.get_search_space(),
            'cross_validation': self.cv_config.get_cv_config(),
            'checkpointing': {
                'save_best_only': self.checkpoint_config.save_best_only,
                'monitor_metric': self.checkpoint_config.monitor_metric,
                'checkpoint_dir': self.checkpoint_config.checkpoint_dir
            },
            'monitoring': self.monitoring_config.get_callbacks_config(),
            'training': {
                'epochs': self.epochs,
                'verbose': self.verbose,
                'use_mixed_precision': self.use_mixed_precision
            },
            'experiment': {
                'name': self.experiment_name,
                'description': self.experiment_description
            }
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration settings"""
        warnings = []
        
        # Check directory existence
        required_dirs = [
            self.checkpoint_config.checkpoint_dir,
            self.checkpoint_config.model_registry_dir,
            self.monitoring_config.log_dir,
            self.monitoring_config.tensorboard_dir
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                warnings.append(f"Directory does not exist: {dir_path}")
        
        # Check resource settings
        if self.use_gpu and not self.gpu_memory_limit:
            warnings.append("GPU enabled but no memory limit set - may cause OOM errors")
        
        # Check cross-validation settings
        if self.cv_config.n_splits < 2:
            warnings.append("Cross-validation splits should be >= 2")
        
        # Check hyperparameter ranges
        if not self.hyperparameter_config.learning_rate_options:
            warnings.append("No learning rate options specified")
        
        return warnings
