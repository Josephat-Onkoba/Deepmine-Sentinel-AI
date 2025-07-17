"""
Training Monitoring and Logging System

This module implements Task 9 monitoring infrastructure:
- Comprehensive training metrics tracking
- TensorBoard integration for visualization
- Custom callbacks for mining-specific monitoring
- Performance profiling and resource monitoring
"""

import os
import json
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

from .training_config import MonitoringConfig


class MetricsTracker:
    """
    Comprehensive metrics tracking for LSTM training
    
    Tracks:
    - Training and validation metrics
    - Resource usage (CPU, memory, GPU)
    - Training performance statistics
    - Custom mining-specific metrics
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Metrics storage
        self.metrics_history = {
            'training': {},
            'validation': {},
            'system': {},
            'custom': {}
        }
        
        # Timing information
        self.training_start_time = None
        self.epoch_times = []
        self.batch_times = []
        
        # System monitoring
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        self.process = psutil.Process()
        
        # Initialize metrics storage
        self._initialize_metrics()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training monitoring"""
        logger = logging.getLogger('training_monitor')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        if self.config.log_to_file:
            os.makedirs(self.config.log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(self.config.log_dir, f'training_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.log_level))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_metrics(self):
        """Initialize metrics storage dictionaries"""
        for metric_name in self.config.metrics_to_track:
            if metric_name.startswith('val_'):
                self.metrics_history['validation'][metric_name] = []
            else:
                self.metrics_history['training'][metric_name] = []
        
        # System metrics
        if self.config.monitor_memory_usage:
            self.metrics_history['system']['memory_usage'] = []
            self.metrics_history['system']['memory_percent'] = []
        
        if self.config.monitor_gpu_usage and self.gpu_available:
            self.metrics_history['system']['gpu_memory_usage'] = []
            self.metrics_history['system']['gpu_utilization'] = []
    
    def start_training(self):
        """Mark the start of training"""
        self.training_start_time = time.time()
        self.logger.info("Training monitoring started")
    
    def end_training(self):
        """Mark the end of training and calculate final statistics"""
        if self.training_start_time is None:
            return
        
        total_time = time.time() - self.training_start_time
        
        training_stats = {
            'total_training_time': total_time,
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'average_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'total_epochs': len(self.epoch_times),
            'total_batches': len(self.batch_times)
        }
        
        self.metrics_history['training_stats'] = training_stats
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Average epoch time: {training_stats['average_epoch_time']:.2f} seconds")
    
    def log_epoch_metrics(self, epoch: int, logs: Dict[str, float]):
        """Log metrics for a completed epoch"""
        
        # Record epoch timing
        current_time = time.time()
        if len(self.epoch_times) == 0 and self.training_start_time:
            epoch_time = current_time - self.training_start_time
        else:
            epoch_time = current_time - (self.training_start_time + sum(self.epoch_times))
        
        self.epoch_times.append(epoch_time)
        
        # Log metrics
        for metric_name, value in logs.items():
            if metric_name in self.config.metrics_to_track:
                if metric_name.startswith('val_'):
                    self.metrics_history['validation'][metric_name].append(value)
                else:
                    self.metrics_history['training'][metric_name].append(value)
        
        # Log system metrics
        self._log_system_metrics()
        
        # Log to file
        epoch_info = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'epoch_time': epoch_time,
            'metrics': logs
        }
        
        if epoch % self.config.log_frequency == 0:
            self.logger.info(f"Epoch {epoch + 1}: {logs}")
    
    def log_batch_metrics(self, batch: int, logs: Dict[str, float]):
        """Log metrics for a completed batch"""
        
        # Record batch timing
        batch_time = time.time()
        if len(self.batch_times) > 0:
            batch_duration = batch_time - self.last_batch_time
            self.batch_times.append(batch_duration)
        
        self.last_batch_time = batch_time
        
        # Log batch info periodically
        if batch % (self.config.log_frequency * 10) == 0:
            self.logger.debug(f"Batch {batch}: {logs}")
    
    def _log_system_metrics(self):
        """Log system resource usage metrics"""
        
        try:
            # Memory usage
            if self.config.monitor_memory_usage:
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self.process.memory_percent()
                
                self.metrics_history['system']['memory_usage'].append(memory_mb)
                self.metrics_history['system']['memory_percent'].append(memory_percent)
            
            # GPU usage
            if self.config.monitor_gpu_usage and self.gpu_available:
                try:
                    import nvidia_ml_py3 as nvml
                    nvml.nvmlInit()
                    
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpu_memory_mb = gpu_memory.used / 1024 / 1024
                    gpu_utilization = gpu_util.gpu
                    
                    self.metrics_history['system']['gpu_memory_usage'].append(gpu_memory_mb)
                    self.metrics_history['system']['gpu_utilization'].append(gpu_utilization)
                    
                except ImportError:
                    # Fallback to TensorFlow GPU monitoring
                    pass
                except Exception as e:
                    self.logger.debug(f"GPU monitoring error: {e}")
        
        except Exception as e:
            self.logger.debug(f"System monitoring error: {e}")
    
    def add_custom_metric(self, name: str, value: float, epoch: int = None):
        """Add a custom metric value"""
        
        if name not in self.metrics_history['custom']:
            self.metrics_history['custom'][name] = []
        
        self.metrics_history['custom'][name].append({
            'value': value,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics"""
        
        summary = {
            'training_metrics': {},
            'validation_metrics': {},
            'system_metrics': {},
            'custom_metrics': {}
        }
        
        # Training metrics summary
        for metric, values in self.metrics_history['training'].items():
            if values:
                summary['training_metrics'][metric] = {
                    'final_value': values[-1],
                    'best_value': max(values) if 'acc' in metric else min(values),
                    'average': np.mean(values),
                    'std': np.std(values),
                    'trend': 'improving' if self._is_improving(values) else 'declining'
                }
        
        # Validation metrics summary
        for metric, values in self.metrics_history['validation'].items():
            if values:
                summary['validation_metrics'][metric] = {
                    'final_value': values[-1],
                    'best_value': max(values) if 'acc' in metric else min(values),
                    'average': np.mean(values),
                    'std': np.std(values),
                    'trend': 'improving' if self._is_improving(values) else 'declining'
                }
        
        # System metrics summary
        for metric, values in self.metrics_history['system'].items():
            if values:
                summary['system_metrics'][metric] = {
                    'average': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'final': values[-1]
                }
        
        # Training statistics
        if 'training_stats' in self.metrics_history:
            summary['training_stats'] = self.metrics_history['training_stats']
        
        return summary
    
    def _is_improving(self, values: List[float], window: int = 5) -> bool:
        """Check if metric values are improving over time"""
        
        if len(values) < window * 2:
            return True  # Not enough data
        
        early_values = values[:window]
        recent_values = values[-window:]
        
        return np.mean(recent_values) > np.mean(early_values)
    
    def save_metrics(self, filepath: str):
        """Save all metrics to file"""
        
        metrics_data = {
            'metrics_history': self.metrics_history,
            'config': self.config.__dict__,
            'summary': self.get_metrics_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to {filepath}")
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Monitoring Dashboard', fontsize=16)
            
            # 1. Loss curves
            if 'loss' in self.metrics_history['training'] and 'val_loss' in self.metrics_history['validation']:
                epochs = range(1, len(self.metrics_history['training']['loss']) + 1)
                
                axes[0, 0].plot(epochs, self.metrics_history['training']['loss'], label='Training Loss')
                axes[0, 0].plot(epochs, self.metrics_history['validation']['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('Model Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # 2. Accuracy curves
            if 'accuracy' in self.metrics_history['training'] and 'val_accuracy' in self.metrics_history['validation']:
                epochs = range(1, len(self.metrics_history['training']['accuracy']) + 1)
                
                axes[0, 1].plot(epochs, self.metrics_history['training']['accuracy'], label='Training Accuracy')
                axes[0, 1].plot(epochs, self.metrics_history['validation']['val_accuracy'], label='Validation Accuracy')
                axes[0, 1].set_title('Model Accuracy')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # 3. System metrics
            if 'memory_usage' in self.metrics_history['system']:
                memory_data = self.metrics_history['system']['memory_usage']
                axes[1, 0].plot(memory_data, label='Memory Usage (MB)')
                axes[1, 0].set_title('System Resource Usage')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Memory (MB)')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # 4. Training time per epoch
            if self.epoch_times:
                axes[1, 1].plot(range(1, len(self.epoch_times) + 1), self.epoch_times)
                axes[1, 1].set_title('Training Time per Epoch')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Time (seconds)')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Metrics plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")


class TrainingMonitorCallback(keras.callbacks.Callback):
    """
    Custom Keras callback for comprehensive training monitoring
    """
    
    def __init__(self, metrics_tracker: MetricsTracker):
        super().__init__()
        self.metrics_tracker = metrics_tracker
        self.epoch_start_time = None
        self.batch_start_time = None
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.metrics_tracker.start_training()
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        self.metrics_tracker.end_training()
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        logs = logs or {}
        self.metrics_tracker.log_epoch_metrics(epoch, logs)
    
    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of each batch"""
        self.batch_start_time = time.time()
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch"""
        logs = logs or {}
        self.metrics_tracker.log_batch_metrics(batch, logs)


class MiningSpecificMonitor:
    """
    Mining-specific monitoring for stability prediction models
    
    Tracks domain-specific metrics:
    - Prediction confidence for different stability classes
    - Temporal pattern analysis
    - Event-driven metric changes
    """
    
    def __init__(self):
        self.stability_predictions = {
            'stable': [],
            'unstable': [],
            'critical': [],
            'emergency': []
        }
        
        self.confidence_scores = []
        self.temporal_patterns = {}
        
    def log_predictions(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       confidence_scores: np.ndarray = None):
        """Log model predictions for analysis"""
        
        class_names = ['stable', 'unstable', 'critical', 'emergency']
        
        for i, class_name in enumerate(class_names):
            class_mask = (y_true == i)
            if np.any(class_mask):
                class_predictions = y_pred[class_mask]
                self.stability_predictions[class_name].extend(class_predictions.tolist())
        
        if confidence_scores is not None:
            self.confidence_scores.extend(confidence_scores.tolist())
    
    def analyze_temporal_patterns(self, predictions: np.ndarray, timestamps: List[datetime]):
        """Analyze temporal patterns in predictions"""
        
        df = pd.DataFrame({
            'prediction': predictions,
            'timestamp': pd.to_datetime(timestamps)
        })
        
        # Group by hour of day
        df['hour'] = df['timestamp'].dt.hour
        hourly_patterns = df.groupby('hour')['prediction'].mean()
        
        # Group by day of week
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        daily_patterns = df.groupby('dayofweek')['prediction'].mean()
        
        self.temporal_patterns = {
            'hourly': hourly_patterns.to_dict(),
            'daily': daily_patterns.to_dict()
        }
    
    def get_mining_metrics(self) -> Dict[str, Any]:
        """Get mining-specific metrics summary"""
        
        metrics = {
            'stability_class_distribution': {},
            'average_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'temporal_patterns': self.temporal_patterns
        }
        
        # Calculate class distributions
        total_predictions = sum(len(preds) for preds in self.stability_predictions.values())
        
        for class_name, predictions in self.stability_predictions.items():
            if total_predictions > 0:
                metrics['stability_class_distribution'][class_name] = {
                    'count': len(predictions),
                    'percentage': len(predictions) / total_predictions * 100,
                    'average_confidence': np.mean(predictions) if predictions else 0
                }
        
        return metrics


def create_tensorboard_callback(config: MonitoringConfig, 
                               experiment_name: str = None) -> keras.callbacks.TensorBoard:
    """
    Create TensorBoard callback with comprehensive logging
    
    Args:
        config: Monitoring configuration
        experiment_name: Name of the experiment
        
    Returns:
        Configured TensorBoard callback
    """
    
    if not config.use_tensorboard:
        return None
    
    # Create experiment-specific log directory
    if experiment_name:
        log_dir = os.path.join(config.tensorboard_dir, experiment_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(config.tensorboard_dir, f'training_{timestamp}')
    
    os.makedirs(log_dir, exist_ok=True)
    
    return keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=config.histogram_freq,
        write_graph=config.write_graph,
        write_images=config.write_images,
        update_freq=config.update_freq,
        profile_batch=2 if config.profile_training else 0,
        embeddings_freq=0
    )


def create_monitoring_callbacks(config: MonitoringConfig,
                               metrics_tracker: MetricsTracker = None,
                               experiment_name: str = None) -> List[keras.callbacks.Callback]:
    """
    Create a comprehensive set of monitoring callbacks
    
    Args:
        config: Monitoring configuration
        metrics_tracker: MetricsTracker instance
        experiment_name: Name of the experiment
        
    Returns:
        List of configured callbacks
    """
    
    callbacks = []
    
    # TensorBoard callback
    tensorboard_callback = create_tensorboard_callback(config, experiment_name)
    if tensorboard_callback:
        callbacks.append(tensorboard_callback)
    
    # Custom monitoring callback
    if metrics_tracker:
        callbacks.append(TrainingMonitorCallback(metrics_tracker))
    
    # Early stopping
    if config.use_early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.patience,
            min_delta=config.min_delta,
            restore_best_weights=config.restore_best_weights,
            verbose=1
        ))
    
    # Learning rate scheduler
    if config.use_lr_scheduler:
        if config.lr_scheduler_type == 'reduce_on_plateau':
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.lr_reduction_factor,
                patience=config.lr_patience,
                min_lr=config.lr_min_value,
                verbose=1
            ))
        elif config.lr_scheduler_type == 'cosine':
            callbacks.append(keras.callbacks.LearningRateScheduler(
                lambda epoch: config.lr_min_value + 
                             (0.001 - config.lr_min_value) * 
                             (1 + np.cos(np.pi * epoch / 100)) / 2
            ))
    
    # Progress bar (if enabled)
    if config.show_progress_bar:
        callbacks.append(keras.callbacks.ProgbarLogger(count_mode='steps'))
    
    return callbacks
