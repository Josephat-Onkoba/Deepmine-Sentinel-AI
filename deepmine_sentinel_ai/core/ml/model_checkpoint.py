"""
Model Checkpointing and Version Management

This module implements Task 9 model management:
- Automated model checkpointing during training
- Version control and model registry
- Model metadata and lineage tracking
- Backup and restoration capabilities
"""

import os
import json
import shutil
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging

from .training_config import CheckpointConfig


class ModelVersion:
    """
    Model version information and metadata
    """
    
    def __init__(self, 
                 version: str,
                 model_name: str,
                 timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        self.version = version
        self.model_name = model_name
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        
        # Generate unique model ID
        self.model_id = self._generate_model_id()
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID based on name, version, and timestamp"""
        content = f"{self.model_name}_{self.version}_{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create ModelVersion from dictionary"""
        return cls(
            version=data['version'],
            model_name=data['model_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class ModelCheckpoint:
    """
    Enhanced model checkpointing with versioning and metadata
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Create directories
        self._create_directories()
        
        # Version tracking
        self.current_version = None
        self.checkpoint_history = []
        
        # Load existing checkpoint registry
        self.registry_file = os.path.join(config.model_registry_dir, 'registry.json')
        self.checkpoint_registry = self._load_registry()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for checkpointing"""
        logger = logging.getLogger('model_checkpoint')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_dir = 'logs/checkpointing'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'checkpoint_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger
    
    def _create_directories(self):
        """Create necessary directories for checkpointing"""
        directories = [
            self.config.checkpoint_dir,
            self.config.model_registry_dir,
            self.config.backup_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load checkpoint registry from file"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load registry: {e}")
        
        return {
            'models': {},
            'versions': {},
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_registry(self):
        """Save checkpoint registry to file"""
        self.checkpoint_registry['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.checkpoint_registry, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def create_checkpoint(self,
                         model: keras.Model,
                         model_name: str,
                         metrics: Dict[str, float],
                         hyperparameters: Dict[str, Any] = None,
                         training_history: Dict[str, List] = None,
                         version: str = None) -> ModelVersion:
        """
        Create a new model checkpoint with full metadata
        
        Args:
            model: Trained Keras model
            model_name: Name of the model
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            training_history: Training history
            version: Specific version string (optional)
            
        Returns:
            ModelVersion object with checkpoint information
        """
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name)
        
        # Create model version
        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            metadata={
                'metrics': metrics,
                'hyperparameters': hyperparameters or {},
                'model_params': model.count_params(),
                'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else None,
                'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else None
            }
        )
        
        # Create version directory
        version_dir = os.path.join(
            self.config.model_registry_dir,
            model_name,
            version
        )
        os.makedirs(version_dir, exist_ok=True)
        
        try:
            # Save model
            model_path = os.path.join(version_dir, 'model.h5')
            model.save(model_path)
            
            # Save model in SavedModel format for serving
            savedmodel_path = os.path.join(version_dir, 'saved_model')
            model.save(savedmodel_path, save_format='tf')
            
            # Save model weights separately
            weights_path = os.path.join(version_dir, 'model_weights.h5')
            model.save_weights(weights_path)
            
            # Save model configuration
            if self.config.save_model_config:
                config_path = os.path.join(version_dir, 'model_config.json')
                with open(config_path, 'w') as f:
                    json.dump(model.get_config(), f, indent=2, default=str)
            
            # Save hyperparameters
            if self.config.save_hyperparameters and hyperparameters:
                hyperparams_path = os.path.join(version_dir, 'hyperparameters.json')
                with open(hyperparams_path, 'w') as f:
                    json.dump(hyperparameters, f, indent=2, default=str)
            
            # Save training history
            if self.config.save_training_history and training_history:
                history_path = os.path.join(version_dir, 'training_history.json')
                with open(history_path, 'w') as f:
                    json.dump(training_history, f, indent=2, default=str)
            
            # Save metrics
            if self.config.save_metrics:
                metrics_path = os.path.join(version_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
            
            # Save version metadata
            version_metadata_path = os.path.join(version_dir, 'version_info.json')
            with open(version_metadata_path, 'w') as f:
                json.dump(model_version.to_dict(), f, indent=2, default=str)
            
            # Update registry
            self._update_registry(model_version, version_dir, metrics)
            
            # Create backup if configured
            if self.config.backup_dir:
                self._create_backup(version_dir, model_version)
            
            # Cleanup old checkpoints
            if self.config.cleanup_old_checkpoints:
                self._cleanup_old_checkpoints(model_name)
            
            self.logger.info(f"Checkpoint created: {model_name} v{version}")
            self.current_version = model_version
            
            return model_version
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            # Cleanup partial checkpoint
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
            raise
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for model"""
        
        # Get existing versions for this model
        model_versions = self.checkpoint_registry.get('models', {}).get(model_name, {}).get('versions', [])
        
        if not model_versions:
            return "1.0.0"
        
        # Find highest version number
        max_major, max_minor, max_patch = 0, 0, 0
        
        for version_info in model_versions:
            try:
                version_parts = version_info['version'].split('.')
                major = int(version_parts[0])
                minor = int(version_parts[1]) if len(version_parts) > 1 else 0
                patch = int(version_parts[2]) if len(version_parts) > 2 else 0
                
                if major > max_major or (major == max_major and minor > max_minor) or \
                   (major == max_major and minor == max_minor and patch > max_patch):
                    max_major, max_minor, max_patch = major, minor, patch
            except (ValueError, IndexError):
                continue
        
        # Increment patch version
        return f"{max_major}.{max_minor}.{max_patch + 1}"
    
    def _update_registry(self, model_version: ModelVersion, version_dir: str, metrics: Dict[str, float]):
        """Update the checkpoint registry"""
        
        model_name = model_version.model_name
        
        # Initialize model entry if it doesn't exist
        if model_name not in self.checkpoint_registry['models']:
            self.checkpoint_registry['models'][model_name] = {
                'created': datetime.now().isoformat(),
                'versions': [],
                'latest_version': None,
                'best_version': None
            }
        
        # Add version info
        version_info = {
            'version': model_version.version,
            'model_id': model_version.model_id,
            'timestamp': model_version.timestamp.isoformat(),
            'path': version_dir,
            'metrics': metrics,
            'metadata': model_version.metadata
        }
        
        self.checkpoint_registry['models'][model_name]['versions'].append(version_info)
        self.checkpoint_registry['models'][model_name]['latest_version'] = model_version.version
        
        # Update best version based on primary metric
        primary_metric = self.config.monitor_metric.replace('val_', '')
        if primary_metric in metrics:
            current_best = self.checkpoint_registry['models'][model_name]['best_version']
            
            if current_best is None:
                self.checkpoint_registry['models'][model_name]['best_version'] = model_version.version
            else:
                # Find current best metrics
                for v in self.checkpoint_registry['models'][model_name]['versions']:
                    if v['version'] == current_best:
                        best_metric = v['metrics'].get(primary_metric, float('-inf'))
                        current_metric = metrics.get(primary_metric, float('-inf'))
                        
                        # Update if current is better (depends on metric type)
                        if self.config.monitor_mode == 'max' and current_metric > best_metric:
                            self.checkpoint_registry['models'][model_name]['best_version'] = model_version.version
                        elif self.config.monitor_mode == 'min' and current_metric < best_metric:
                            self.checkpoint_registry['models'][model_name]['best_version'] = model_version.version
                        break
        
        # Add to global versions registry
        self.checkpoint_registry['versions'][model_version.model_id] = version_info
        
        # Save registry
        self._save_registry()
    
    def _create_backup(self, version_dir: str, model_version: ModelVersion):
        """Create backup of checkpoint"""
        
        backup_name = f"{model_version.model_name}_{model_version.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(self.config.backup_dir, backup_name)
        
        try:
            shutil.copytree(version_dir, backup_path)
            self.logger.info(f"Backup created: {backup_path}")
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
    
    def _cleanup_old_checkpoints(self, model_name: str):
        """Cleanup old checkpoints based on retention policy"""
        
        model_info = self.checkpoint_registry.get('models', {}).get(model_name, {})
        versions = model_info.get('versions', [])
        
        if len(versions) <= self.config.keep_latest_n:
            return
        
        # Sort versions by timestamp
        versions_sorted = sorted(versions, key=lambda x: x['timestamp'], reverse=True)
        
        # Keep best versions
        best_version = model_info.get('best_version')
        versions_to_keep = set()
        
        # Keep latest N versions
        for i in range(min(self.config.keep_latest_n, len(versions_sorted))):
            versions_to_keep.add(versions_sorted[i]['version'])
        
        # Keep best N versions by metric
        if self.config.keep_best_n > 0:
            primary_metric = self.config.monitor_metric.replace('val_', '')
            versions_by_metric = sorted(
                versions,
                key=lambda x: x['metrics'].get(primary_metric, float('-inf')),
                reverse=(self.config.monitor_mode == 'max')
            )
            
            for i in range(min(self.config.keep_best_n, len(versions_by_metric))):
                versions_to_keep.add(versions_by_metric[i]['version'])
        
        # Always keep the best version
        if best_version:
            versions_to_keep.add(best_version)
        
        # Remove old versions
        versions_to_remove = []
        for version_info in versions:
            if version_info['version'] not in versions_to_keep:
                # Check age
                version_date = datetime.fromisoformat(version_info['timestamp'])
                age_days = (datetime.now() - version_date).days
                
                if age_days > self.config.max_age_days:
                    versions_to_remove.append(version_info)
        
        # Delete old version directories
        for version_info in versions_to_remove:
            version_path = version_info['path']
            if os.path.exists(version_path):
                try:
                    shutil.rmtree(version_path)
                    self.logger.info(f"Removed old checkpoint: {version_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoint {version_path}: {e}")
        
        # Update registry
        self.checkpoint_registry['models'][model_name]['versions'] = [
            v for v in versions if v not in versions_to_remove
        ]
        self._save_registry()
    
    def load_model(self, model_name: str, version: str = None) -> Tuple[keras.Model, ModelVersion]:
        """
        Load a model from checkpoint
        
        Args:
            model_name: Name of the model
            version: Specific version (if None, loads latest)
            
        Returns:
            Tuple of (loaded_model, model_version)
        """
        
        model_info = self.checkpoint_registry.get('models', {}).get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Get version to load
        if version is None:
            version = model_info.get('latest_version')
            if not version:
                raise ValueError(f"No versions found for model '{model_name}'")
        
        # Find version info
        version_info = None
        for v in model_info['versions']:
            if v['version'] == version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        # Load model
        version_dir = version_info['path']
        model_path = os.path.join(version_dir, 'model.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = keras.models.load_model(model_path)
            
            # Load version metadata
            version_metadata_path = os.path.join(version_dir, 'version_info.json')
            if os.path.exists(version_metadata_path):
                with open(version_metadata_path, 'r') as f:
                    version_data = json.load(f)
                model_version = ModelVersion.from_dict(version_data)
            else:
                model_version = ModelVersion(version, model_name)
            
            self.logger.info(f"Loaded model: {model_name} v{version}")
            return model, model_version
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model and its versions"""
        
        model_info = self.checkpoint_registry.get('models', {}).get(model_name)
        if not model_info:
            return {}
        
        return model_info.copy()
    
    def list_models(self) -> List[str]:
        """List all models in the registry"""
        return list(self.checkpoint_registry.get('models', {}).keys())
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model"""
        
        model_info = self.checkpoint_registry.get('models', {}).get(model_name, {})
        return model_info.get('versions', [])
    
    def get_best_model(self, model_name: str) -> Tuple[keras.Model, ModelVersion]:
        """Load the best performing model version"""
        
        model_info = self.checkpoint_registry.get('models', {}).get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        best_version = model_info.get('best_version')
        if not best_version:
            raise ValueError(f"No best version determined for model '{model_name}'")
        
        return self.load_model(model_name, best_version)
    
    def export_model(self, model_name: str, version: str, export_path: str, export_format: str = 'savedmodel'):
        """
        Export model for deployment
        
        Args:
            model_name: Name of the model
            version: Version to export
            export_path: Path to export the model
            export_format: Export format ('savedmodel', 'tflite', 'h5')
        """
        
        model, model_version = self.load_model(model_name, version)
        
        os.makedirs(export_path, exist_ok=True)
        
        if export_format == 'savedmodel':
            model.save(export_path, save_format='tf')
        elif export_format == 'h5':
            model_file = os.path.join(export_path, f"{model_name}_v{version}.h5")
            model.save(model_file)
        elif export_format == 'tflite':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            tflite_file = os.path.join(export_path, f"{model_name}_v{version}.tflite")
            with open(tflite_file, 'wb') as f:
                f.write(tflite_model)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Export metadata
        metadata_file = os.path.join(export_path, 'model_info.json')
        with open(metadata_file, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Model exported: {model_name} v{version} -> {export_path}")


def create_training_checkpoint_callback(checkpoint_manager: ModelCheckpoint,
                                      model_name: str,
                                      save_best_only: bool = True,
                                      monitor: str = 'val_accuracy',
                                      mode: str = 'max') -> keras.callbacks.Callback:
    """
    Create a custom checkpoint callback for training
    
    Args:
        checkpoint_manager: ModelCheckpoint instance
        model_name: Name of the model
        save_best_only: Whether to save only the best model
        monitor: Metric to monitor
        mode: 'min' or 'max' for the monitored metric
        
    Returns:
        Custom checkpoint callback
    """
    
    class TrainingCheckpointCallback(keras.callbacks.Callback):
        
        def __init__(self):
            super().__init__()
            self.best_metric = float('inf') if mode == 'min' else float('-inf')
            self.checkpoint_count = 0
        
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current_metric = logs.get(monitor)
            
            if current_metric is None:
                return
            
            should_save = False
            
            if not save_best_only:
                should_save = True
            else:
                if mode == 'min' and current_metric < self.best_metric:
                    self.best_metric = current_metric
                    should_save = True
                elif mode == 'max' and current_metric > self.best_metric:
                    self.best_metric = current_metric
                    should_save = True
            
            if should_save:
                try:
                    # Create checkpoint
                    checkpoint_manager.create_checkpoint(
                        model=self.model,
                        model_name=model_name,
                        metrics=logs,
                        training_history=None,  # Will be added later
                        version=None  # Auto-generate
                    )
                    self.checkpoint_count += 1
                    
                except Exception as e:
                    print(f"Failed to create checkpoint: {e}")
    
    return TrainingCheckpointCallback()
