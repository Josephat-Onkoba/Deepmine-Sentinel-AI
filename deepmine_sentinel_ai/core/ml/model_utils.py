"""
Model Utilities for LSTM Training and Evaluation

This module provides utilities for:
- Building different LSTM model types
- Training LSTM models with callbacks
- Evaluating model performance
- Model ensemble management
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import numpy as np
import os
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
import json

from .lstm_config import LSTMConfig, LSTMModelConfig, AttentionConfig, MultiStepConfig, ModelArchitectureConfig
from .lstm_models import (
    BasicLSTMModel, MultiStepLSTMModel, AttentionLSTMModel, 
    MultiOutputLSTMModel, CompleteLSTMPredictor
)


class ModelBuilder:
    """
    Builder class for creating different types of LSTM models
    """
    
    @staticmethod
    def build_model(model_type: str, config: ModelArchitectureConfig) -> keras.Model:
        """
        Build LSTM model based on type and configuration
        
        Args:
            model_type: Type of model ('basic', 'multi_step', 'attention', 'multi_output', 'complete')
            config: Model architecture configuration
            
        Returns:
            Built Keras model
        """
        if model_type == 'basic':
            return BasicLSTMModel(config.lstm_config)
        
        elif model_type == 'multi_step':
            return MultiStepLSTMModel(config.lstm_config, config.multi_step_config)
        
        elif model_type == 'attention':
            return AttentionLSTMModel(config.lstm_config, config.attention_config)
        
        elif model_type == 'multi_output':
            return MultiOutputLSTMModel(config.lstm_config, config.multi_step_config)
        
        elif model_type == 'complete':
            return CompleteLSTMPredictor(
                config.lstm_config, 
                config.attention_config, 
                config.multi_step_config
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def build_ensemble(model_type: str, config: ModelArchitectureConfig, num_models: int = 5) -> List[keras.Model]:
        """
        Build ensemble of LSTM models
        
        Args:
            model_type: Type of models in ensemble
            config: Model configuration
            num_models: Number of models in ensemble
            
        Returns:
            List of trained models
        """
        ensemble = []
        for i in range(num_models):
            # Add some variation to each model
            varied_config = ModelBuilder._vary_config(config, variation_factor=0.1)
            model = ModelBuilder.build_model(model_type, varied_config)
            ensemble.append(model)
        
        return ensemble
    
    @staticmethod
    def _vary_config(config: ModelArchitectureConfig, variation_factor: float = 0.1) -> ModelArchitectureConfig:
        """Add small variations to configuration for ensemble diversity"""
        # Create a copy of the configuration
        varied_config = ModelArchitectureConfig(
            model_type=config.model_type,
            lstm_config=LSTMConfig(**config.lstm_config.__dict__.copy()),
            attention_config=AttentionConfig(**config.attention_config.__dict__.copy()),
            multi_step_config=MultiStepConfig(**config.multi_step_config.__dict__.copy())
        )
        
        # Add small variations
        varied_config.lstm_config.dropout_rate = max(0.0, min(0.5, 
            config.lstm_config.dropout_rate + np.random.normal(0, variation_factor)))
        
        if hasattr(varied_config.lstm_config, 'recurrent_dropout'):
            varied_config.lstm_config.recurrent_dropout = max(0.0, min(0.5,
                getattr(config.lstm_config, 'recurrent_dropout', 0.0) + np.random.normal(0, variation_factor)))
        
        return varied_config
    
    @staticmethod
    def load_model(model_path: str, model_type: str = 'complete') -> keras.Model:
        """Load saved model from path"""
        if os.path.exists(model_path):
            return keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    @staticmethod
    def save_model(model: keras.Model, model_path: str, save_format: str = 'tf'):
        """Save model to path"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path, save_format=save_format)


class ModelTrainer:
    """
    Trainer class for LSTM models with comprehensive training features
    """
    
    def __init__(self, config: LSTMModelConfig):
        self.config = config
        self.training_history = {}
    
    def compile_model(self, model: keras.Model) -> keras.Model:
        """
        Compile model with appropriate loss, optimizer, and metrics
        
        Args:
            model: Keras model to compile
            
        Returns:
            Compiled model
        """
        # Handle different optimizer configurations
        if self.config.optimizer == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=self.config.learning_rate)
        else:
            optimizer = self.config.optimizer
        
        # Compile based on model type
        if hasattr(model, 'compile_multi_output'):
            # Multi-output model
            model.compile_multi_output(optimizer=optimizer)
        else:
            # Single output model
            model.compile(
                optimizer=optimizer,
                loss=self.config.loss_function,
                metrics=self.config.metrics
            )
        
        return model
    
    def create_callbacks(self, model_name: str) -> List[callbacks.Callback]:
        """
        Create training callbacks
        
        Args:
            model_name: Name for saving checkpoints and logs
            
        Returns:
            List of Keras callbacks
        """
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpointing
        checkpoint_path = f'models/checkpoints/{model_name}_best.h5'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        model_checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callback_list.append(model_checkpoint)
        
        # CSV logging
        csv_path = f'logs/{model_name}_training.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        csv_logger = callbacks.CSVLogger(csv_path, append=True)
        callback_list.append(csv_logger)
        
        return callback_list
    
    def train_model(self, 
                   model: keras.Model,
                   train_data: Tuple[np.ndarray, np.ndarray],
                   validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   model_name: str = "lstm_model") -> Dict[str, Any]:
        """
        Train LSTM model with full configuration
        
        Args:
            model: Compiled Keras model
            train_data: Training data (X, y)
            validation_data: Validation data (X, y)
            model_name: Name for saving and logging
            
        Returns:
            Training history and metrics
        """
        X_train, y_train = train_data
        
        # Create callbacks
        callback_list = self.create_callbacks(model_name)
        
        # Prepare class weights if specified
        class_weight = self.config.class_weight
        
        # Training arguments
        fit_kwargs = {
            'x': X_train,
            'y': y_train,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'validation_split': self.config.validation_split,
            'callbacks': callback_list,
            'verbose': 1,
            'shuffle': True
        }
        
        # Add validation data if provided
        if validation_data is not None:
            fit_kwargs['validation_data'] = validation_data
            fit_kwargs.pop('validation_split')
        
        # Add class weights if specified
        if class_weight is not None:
            fit_kwargs['class_weight'] = class_weight
        
        # Train model
        history = model.fit(**fit_kwargs)
        
        # Store training history
        self.training_history[model_name] = {
            'history': history.history,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'history': history.history,
            'model': model,
            'best_epoch': len(history.history['loss']),
            'best_val_loss': min(history.history.get('val_loss', [float('inf')]))
        }
    
    def save_training_history(self, filepath: str):
        """Save training history to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)


class ModelEvaluator:
    """
    Evaluator class for comprehensive model performance assessment
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, 
                      model: keras.Model,
                      test_data: Tuple[np.ndarray, np.ndarray],
                      model_name: str = "lstm_model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained Keras model
            test_data: Test data (X, y)
            model_name: Model identifier
            
        Returns:
            Evaluation metrics and results
        """
        X_test, y_test = test_data
        
        # Get predictions
        predictions = model.predict(X_test, verbose=0)
        
        # Handle different output types
        if isinstance(predictions, dict):
            # Multi-output model
            results = {}
            for output_name, pred in predictions.items():
                output_results = self._evaluate_single_output(y_test, pred, f"{model_name}_{output_name}")
                results[output_name] = output_results
            
            # Calculate average metrics
            avg_results = self._calculate_average_metrics(results)
            results['average'] = avg_results
            
        else:
            # Single output model
            results = self._evaluate_single_output(y_test, predictions, model_name)
        
        # Store results
        self.evaluation_results[model_name] = results
        
        return results
    
    def _evaluate_single_output(self, y_true: np.ndarray, y_pred: np.ndarray, output_name: str) -> Dict[str, Any]:
        """Evaluate single model output"""
        # Convert predictions to class labels
        if y_pred.shape[-1] > 1:  # Multi-class
            y_pred_classes = np.argmax(y_pred, axis=-1)
        else:  # Binary
            y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        results = self._calculate_metrics(y_true, y_pred_classes, y_pred, output_name)
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred_classes: np.ndarray, 
                          y_pred_proba: np.ndarray, output_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Classification report
        class_report = classification_report(y_true, y_pred_classes, output_dict=True, zero_division=0)
        
        # ROC AUC (for multi-class, use ovr strategy)
        try:
            if y_pred_proba.shape[-1] > 2:
                auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.shape[-1] > 1 else y_pred_proba)
        except (ValueError, IndexError):
            auc_roc = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'output_name': output_name
        }
    
    def _calculate_average_metrics(self, multi_output_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate average metrics across multiple outputs"""
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        average_metrics = {}
        
        for metric in metric_names:
            values = []
            for output_name, results in multi_output_results.items():
                if metric in results:
                    values.append(results[metric])
            
            if values:
                average_metrics[f'avg_{metric}'] = float(np.mean(values))
                average_metrics[f'std_{metric}'] = float(np.std(values))
        
        return average_metrics
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple evaluated models"""
        if not model_names:
            return {}
        
        comparison = {
            'models': model_names,
            'metrics': {}
        }
        
        # Extract metrics for comparison
        for model_name in model_names:
            if model_name in self.evaluation_results:
                results = self.evaluation_results[model_name]
                
                # Handle multi-output vs single output
                if 'average' in results:
                    comparison['metrics'][model_name] = results['average']
                else:
                    comparison['metrics'][model_name] = {
                        key: results[key] for key in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
                        if key in results
                    }
        
        # Find best model for each metric
        comparison['best_models'] = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        for metric in metric_names:
            best_score = -1
            best_model = None
            
            for model_name in model_names:
                if model_name in comparison['metrics']:
                    score = comparison['metrics'][model_name].get(f'avg_{metric}', 
                                                                comparison['metrics'][model_name].get(metric, 0))
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                comparison['best_models'][metric] = {'model': best_model, 'score': best_score}
        
        return comparison
    
    def save_evaluation_results(self, filepath: str):
        """Save evaluation results to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.evaluation_results.items():
            serializable_results[model_name] = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


class EnsemblePredictor:
    """
    Ensemble predictor for combining multiple LSTM models
    """
    
    def __init__(self, models: List[keras.Model], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Handle different output types
        if isinstance(predictions[0], dict):
            # Multi-output models
            ensemble_pred = {}
            for output_name in predictions[0].keys():
                output_preds = [pred[output_name] for pred in predictions]
                weighted_pred = self._weighted_average(output_preds)
                ensemble_pred[output_name] = weighted_pred
            return ensemble_pred
        else:
            # Single output models
            return self._weighted_average(predictions)
    
    def _weighted_average(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Calculate weighted average of predictions"""
        weighted_sum = np.zeros_like(predictions[0])
        
        for pred, weight in zip(predictions, self.weights):
            weighted_sum += pred * weight
        
        return weighted_sum
