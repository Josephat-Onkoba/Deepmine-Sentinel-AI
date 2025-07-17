"""
Hyperparameter Tuning for LSTM Models

This module implements Task 9 hyperparameter optimization:
- Random search and Bayesian optimization
- Grid search for systematic exploration
- Custom hyperparameter spaces for mining data
- Performance tracking and comparison
"""

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
import json
import os
from datetime import datetime
import logging
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .training_config import HyperparameterConfig, TrainingPipelineConfig
from .lstm_config import LSTMConfig, AttentionConfig, MultiStepConfig
from .model_utils import ModelBuilder

# Try to import LSTM models, create basic ones if not available
try:
    from .lstm_models import BasicLSTMModel, AttentionLSTMModel
except ImportError:
    # Create basic model classes for hyperparameter tuning
    class BasicLSTMModel(keras.Model):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.lstm_layers = []
            self.dense_layers = []
            self.dropout_layers = []
            
            # Build LSTM layers
            for i, units in enumerate(config.lstm_units):
                return_sequences = i < len(config.lstm_units) - 1
                self.lstm_layers.append(
                    keras.layers.LSTM(units, return_sequences=return_sequences)
                )
                if i < len(config.lstm_units) - 1:
                    self.dropout_layers.append(keras.layers.Dropout(config.dropout_rate))
            
            # Build dense layers
            for units in config.dense_units:
                self.dense_layers.append(keras.layers.Dense(units, activation='relu'))
                self.dropout_layers.append(keras.layers.Dropout(config.dropout_rate))
            
            # Output layer
            self.output_layer = keras.layers.Dense(config.num_classes, activation='softmax')
        
        def call(self, inputs):
            x = inputs
            
            # LSTM layers
            for i, lstm_layer in enumerate(self.lstm_layers):
                x = lstm_layer(x)
                if i < len(self.lstm_layers) - 1:
                    x = self.dropout_layers[i](x)
            
            # Dense layers
            for i, dense_layer in enumerate(self.dense_layers):
                x = dense_layer(x)
                x = self.dropout_layers[len(self.lstm_layers) - 1 + i](x)
            
            return self.output_layer(x)
    
    class AttentionLSTMModel(BasicLSTMModel):
        def __init__(self, config, attention_config):
            super().__init__(config)
            self.attention_config = attention_config
from .lstm_models import BasicLSTMModel, MultiStepLSTMModel, AttentionLSTMModel
from .model_utils import ModelBuilder, ModelTrainer


class LSTMHyperModel(kt.HyperModel):
    """
    Hyperparameter tuning model for LSTM architectures
    
    Implements searchable hyperparameter space for:
    - LSTM layer configurations
    - Attention mechanisms
    - Dense layer architectures
    - Regularization parameters
    """
    
    def __init__(self, 
                 config: HyperparameterConfig,
                 input_shape: Tuple[int, int],
                 num_classes: int = 4,
                 model_type: str = 'basic'):
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
    
    def build(self, hp):
        """Build model with hyperparameter search space"""
        
        # LSTM Architecture - Use predefined architecture choices
        architecture_choice = hp.Choice('architecture', ['small', 'medium', 'large', 'extra_large'])
        
        # Map architecture choices to layer configurations
        architecture_map = {
            'small': {'lstm_units': [64, 32], 'dense_units': [64, 32]},
            'medium': {'lstm_units': [128, 64], 'dense_units': [128, 64]},
            'large': {'lstm_units': [256, 128], 'dense_units': [256, 128]},
            'extra_large': {'lstm_units': [128, 64, 32], 'dense_units': [128, 64, 32]}
        }
        
        lstm_units = architecture_map[architecture_choice]['lstm_units']
        dense_units = architecture_map[architecture_choice]['dense_units']
        
        # Individual hyperparameters
        dropout_rate = hp.Choice('dropout_rate', self.config.dropout_rate_options)
        
        # Learning Parameters
        learning_rate = hp.Choice('learning_rate', self.config.learning_rate_options)
        optimizer_name = hp.Choice('optimizer', self.config.optimizer_options)
        
        # Regularization
        l1_reg = hp.Choice('l1_regularization', self.config.l1_reg_options)
        l2_reg = hp.Choice('l2_regularization', self.config.l2_reg_options)
        
        # Create LSTM configuration
        lstm_config = LSTMConfig(
            input_features=self.input_shape[1],
            sequence_length=self.input_shape[0],
            lstm_units=lstm_units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            num_classes=self.num_classes,
            use_attention=(self.model_type == 'attention')
        )
        
        # Build model directly using Keras
        model = keras.Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            if i == 0:
                model.add(keras.layers.LSTM(
                    units, 
                    return_sequences=return_sequences,
                    input_shape=self.input_shape
                ))
            else:
                model.add(keras.layers.LSTM(units, return_sequences=return_sequences))
            
            if return_sequences:
                model.add(keras.layers.Dropout(dropout_rate))
        
        # Add dense layers
        for units in dense_units:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
        # Configure optimizer
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
        
        # Add regularization to dense layers
        if l1_reg > 0 or l2_reg > 0:
            regularizer = keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
            for layer in model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = regularizer
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning system
    
    Supports multiple tuning strategies:
    - Random search for efficient exploration
    - Bayesian optimization for intelligent search
    - Grid search for comprehensive evaluation
    """
    
    def __init__(self, 
                 config: HyperparameterConfig,
                 input_shape: Tuple[int, int],
                 num_classes: int = 4,
                 model_type: str = 'basic'):
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Results storage
        self.tuning_results = []
        self.best_params = None
        self.best_score = float('-inf')
        
        # Create tuner based on strategy
        self.tuner = self._create_tuner()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for hyperparameter tuning"""
        logger = logging.getLogger('hyperparameter_tuner')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_dir = 'logs/hyperparameter_tuning'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'tuning_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_tuner(self):
        """Create appropriate tuner based on strategy"""
        
        # Create hypermodel
        hypermodel = LSTMHyperModel(
            config=self.config,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            model_type=self.model_type
        )
        
        # Tuner directory
        tuner_dir = f'tuning_results/{self.model_type}'
        os.makedirs(tuner_dir, exist_ok=True)
        
        if self.config.tuning_strategy == 'random_search':
            return kt.RandomSearch(
                hypermodel,
                objective='val_accuracy',
                max_trials=self.config.max_trials,
                executions_per_trial=self.config.executions_per_trial,
                directory=tuner_dir,
                project_name=f'lstm_{self.model_type}_random_search',
                seed=self.config.random_state
            )
        
        elif self.config.tuning_strategy == 'bayesian':
            return kt.BayesianOptimization(
                hypermodel,
                objective='val_accuracy',
                max_trials=self.config.max_trials,
                num_initial_points=min(10, self.config.max_trials // 5),
                directory=tuner_dir,
                project_name=f'lstm_{self.model_type}_bayesian',
                seed=self.config.random_state
            )
        
        elif self.config.tuning_strategy == 'grid_search':
            # For grid search, we'll use a custom implementation
            return None
        
        else:
            raise ValueError(f"Unknown tuning strategy: {self.config.tuning_strategy}")
    
    def tune(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: np.ndarray, 
             y_val: np.ndarray,
             epochs: int = 50,
             batch_size: int = 32) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Training epochs per trial
            batch_size: Batch size for training
            
        Returns:
            Dictionary with tuning results
        """
        
        self.logger.info(f"Starting hyperparameter tuning with {self.config.tuning_strategy}")
        self.logger.info(f"Search space size: {self._calculate_search_space_size()}")
        
        if self.config.tuning_strategy == 'grid_search':
            return self._grid_search_tune(X_train, y_train, X_val, y_val, epochs, batch_size)
        else:
            return self._keras_tuner_tune(X_train, y_train, X_val, y_val, epochs, batch_size)
    
    def _keras_tuner_tune(self, 
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         X_val: np.ndarray, 
                         y_val: np.ndarray,
                         epochs: int,
                         batch_size: int) -> Dict[str, Any]:
        """Perform tuning using Keras Tuner"""
        
        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Start search
        start_time = datetime.now()
        
        self.tuner.search(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        tuning_duration = end_time - start_time
        
        # Get best results
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = self.tuner.get_best_models(num_models=1)[0]
        
        # Evaluate best model
        val_loss, val_accuracy, val_precision, val_recall = best_model.evaluate(
            X_val, y_val, verbose=0
        )
        
        # Store results
        self.best_params = {param: best_hps.get(param) for param in best_hps.values.keys()}
        self.best_score = val_accuracy
        
        results = {
            'best_hyperparameters': self.best_params,
            'best_score': self.best_score,
            'best_model': best_model,
            'tuning_duration': tuning_duration.total_seconds(),
            'trials_completed': len(self.tuner.oracle.trials),
            'validation_metrics': {
                'loss': val_loss,
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall
            }
        }
        
        self.logger.info(f"Tuning completed in {tuning_duration}")
        self.logger.info(f"Best validation accuracy: {val_accuracy:.4f}")
        self.logger.info(f"Best hyperparameters: {self.best_params}")
        
        return results
    
    def _grid_search_tune(self, 
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         X_val: np.ndarray, 
                         y_val: np.ndarray,
                         epochs: int,
                         batch_size: int) -> Dict[str, Any]:
        """Perform grid search tuning"""
        
        # Create parameter grid
        param_grid = self.config.get_search_space()
        grid = list(ParameterGrid(param_grid))
        
        # Limit grid size for practical purposes
        if len(grid) > self.config.max_trials:
            np.random.seed(self.config.random_state)
            grid = np.random.choice(grid, size=self.config.max_trials, replace=False)
        
        best_score = float('-inf')
        best_params = None
        best_model = None
        
        self.logger.info(f"Starting grid search with {len(grid)} parameter combinations")
        
        start_time = datetime.now()
        
        for i, params in enumerate(grid):
            self.logger.info(f"Trial {i+1}/{len(grid)}: {params}")
            
            try:
                # Create model with current parameters
                model = self._build_model_from_params(params)
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=10,
                            restore_best_weights=True
                        )
                    ]
                )
                
                # Evaluate model
                val_loss, val_accuracy, val_precision, val_recall = model.evaluate(
                    X_val, y_val, verbose=0
                )
                
                # Track results
                trial_result = {
                    'trial': i + 1,
                    'parameters': params,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'epochs_trained': len(history.history['loss'])
                }
                
                self.tuning_results.append(trial_result)
                
                # Update best results
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = params.copy()
                    best_model = model
                
                self.logger.info(f"Trial {i+1} - Val Accuracy: {val_accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Trial {i+1} failed: {str(e)}")
                continue
        
        end_time = datetime.now()
        tuning_duration = end_time - start_time
        
        # Store final results
        self.best_params = best_params
        self.best_score = best_score
        
        results = {
            'best_hyperparameters': self.best_params,
            'best_score': self.best_score,
            'best_model': best_model,
            'tuning_duration': tuning_duration.total_seconds(),
            'trials_completed': len(self.tuning_results),
            'all_trials': self.tuning_results
        }
        
        self.logger.info(f"Grid search completed in {tuning_duration}")
        self.logger.info(f"Best validation accuracy: {best_score:.4f}")
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        return results
    
    def _build_model_from_params(self, params: Dict[str, Any]):
        """Build model from parameter dictionary"""
        
        # Create LSTM configuration
        lstm_config = LSTMConfig(
            input_features=self.input_shape[1],
            sequence_length=self.input_shape[0],
            lstm_units=params['lstm_units'],
            dense_units=params['dense_units'],
            dropout_rate=params['dropout_rate'],
            num_classes=self.num_classes,
            use_attention=(self.model_type == 'attention')
        )
        
        # Build model
        if self.model_type == 'attention':
            attention_config = AttentionConfig(
                attention_type='multi_head',
                num_heads=params.get('attention_heads', 4),
                key_dim=params.get('attention_key_dim', 64),
                dropout_rate=params['dropout_rate']
            )
            model = AttentionLSTMModel(lstm_config, attention_config)
        else:
            model = BasicLSTMModel(lstm_config)
        
        # Build the model
        dummy_input = tf.random.normal((1, *self.input_shape))
        _ = model(dummy_input)
        
        # Configure optimizer
        optimizer_name = params.get('optimizer', 'adam')
        learning_rate = params.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _calculate_search_space_size(self) -> int:
        """Calculate total search space size"""
        space = self.config.get_search_space()
        total_combinations = 1
        for key, values in space.items():
            total_combinations *= len(values)
        return total_combinations
    
    def save_results(self, filepath: str):
        """Save tuning results to file"""
        
        results_data = {
            'config': {
                'tuning_strategy': self.config.tuning_strategy,
                'max_trials': self.config.max_trials,
                'model_type': self.model_type,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes
            },
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'all_trials': self.tuning_results,
            'search_space': self.config.get_search_space(),
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load tuning results from file"""
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.best_params = results_data['best_parameters']
        self.best_score = results_data['best_score']
        self.tuning_results = results_data.get('all_trials', [])
        
        return results_data
    
    def get_top_n_configs(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N parameter configurations"""
        
        if not self.tuning_results:
            return []
        
        # Sort by validation accuracy
        sorted_results = sorted(
            self.tuning_results, 
            key=lambda x: x.get('val_accuracy', 0), 
            reverse=True
        )
        
        return sorted_results[:n]
