"""
Cross-Validation System for LSTM Models

This module implements Task 9 cross-validation:
- Time series cross-validation for temporal data
- Stratified cross-validation for balanced evaluation
- Rolling window and expanding window strategies
- Performance metrics aggregation and statistical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Generator
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import logging
from datetime import datetime
import json
import os

from .training_config import CrossValidationConfig
from .lstm_config import LSTMConfig, AttentionConfig, MultiStepConfig
from .lstm_models import BasicLSTMModel, MultiStepLSTMModel, AttentionLSTMModel
from .model_utils import ModelBuilder


class TimeSeriesCV:
    """
    Custom time series cross-validation
    
    Implements mining-specific temporal validation:
    - Preserves temporal order
    - Configurable train/test gaps
    - Rolling and expanding window support
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 gap_size: int = 0,
                 window_type: str = 'rolling'):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_size = gap_size
        self.window_type = window_type
    
    def split(self, X: np.ndarray, y: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits for time series data
        
        Args:
            X: Feature array with temporal order
            y: Target array (optional)
            
        Yields:
            train_indices, test_indices for each split
        """
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)
        
        # Calculate split points
        if self.window_type == 'rolling':
            # Rolling window: fixed train size
            train_size = (n_samples - test_size_samples * self.n_splits) // self.n_splits
            
            for i in range(self.n_splits):
                test_start = train_size + i * test_size_samples + self.gap_size
                test_end = test_start + test_size_samples
                
                if test_end > n_samples:
                    break
                
                if self.window_type == 'rolling':
                    train_start = i * test_size_samples
                    train_end = train_start + train_size
                else:  # expanding
                    train_start = 0
                    train_end = test_start - self.gap_size
                
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                
                yield train_indices, test_indices
        
        else:  # expanding window
            # Expanding window: increasing train size
            min_train_size = int(n_samples * 0.3)  # Minimum 30% for training
            
            for i in range(self.n_splits):
                test_start = min_train_size + i * test_size_samples + self.gap_size
                test_end = test_start + test_size_samples
                
                if test_end > n_samples:
                    break
                
                train_start = 0
                train_end = test_start - self.gap_size
                
                if train_end <= train_start:
                    continue
                
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                
                yield train_indices, test_indices


class CrossValidator:
    """
    Comprehensive cross-validation system for LSTM models
    
    Features:
    - Multiple CV strategies (K-fold, stratified, time series)
    - Robust performance metrics calculation
    - Statistical significance testing
    - Detailed result reporting
    """
    
    def __init__(self, config: CrossValidationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # CV results storage
        self.cv_results = []
        self.fold_models = []
        self.performance_stats = {}
        
        # Create CV splitter
        self.cv_splitter = self._create_cv_splitter()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for cross-validation"""
        logger = logging.getLogger('cross_validator')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_dir = 'logs/cross_validation'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'cv_{timestamp}.log')
        
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
    
    def _create_cv_splitter(self):
        """Create appropriate CV splitter based on configuration"""
        
        if self.config.cv_type == 'k_fold':
            return KFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
        
        elif self.config.cv_type == 'stratified':
            return StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
        
        elif self.config.cv_type == 'time_series':
            if self.config.time_series_split_method == 'sklearn':
                return TimeSeriesSplit(
                    n_splits=self.config.n_splits,
                    test_size=int(self.config.test_size * 1000),  # Approximate
                    gap=self.config.gap_size
                )
            else:
                return TimeSeriesCV(
                    n_splits=self.config.n_splits,
                    test_size=self.config.test_size,
                    gap_size=self.config.gap_size,
                    window_type=self.config.time_series_split_method
                )
        
        else:
            raise ValueError(f"Unknown CV type: {self.config.cv_type}")
    
    def validate_model(self, 
                      model_builder: ModelBuilder,
                      X: np.ndarray, 
                      y: np.ndarray,
                      model_params: Dict[str, Any],
                      training_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-validation on LSTM model
        
        Args:
            model_builder: ModelBuilder instance
            X: Feature data
            y: Target data
            model_params: Model configuration parameters
            training_params: Training configuration parameters
            
        Returns:
            Comprehensive CV results dictionary
        """
        
        self.logger.info(f"Starting {self.config.cv_type} cross-validation with {self.config.n_splits} splits")
        
        # Prepare data for stratification if needed
        stratify_data = None
        if self.config.cv_type == 'stratified' and self.config.stratify_column:
            stratify_data = y
        
        # Perform cross-validation splits
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X, stratify_data)):
            self.logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create validation split from training data
            val_size = int(len(X_train) * self.config.validation_size)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            
            # Build and train model
            try:
                fold_result = self._train_fold(
                    model_builder=model_builder,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    model_params=model_params,
                    training_params=training_params,
                    fold_idx=fold_idx
                )
                
                fold_results.append(fold_result)
                
            except Exception as e:
                self.logger.error(f"Fold {fold_idx + 1} failed: {str(e)}")
                continue
        
        # Aggregate results
        cv_summary = self._aggregate_results(fold_results)
        
        # Store results
        self.cv_results = fold_results
        self.performance_stats = cv_summary
        
        self.logger.info("Cross-validation completed")
        self.logger.info(f"Mean accuracy: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        
        return {
            'fold_results': fold_results,
            'summary': cv_summary,
            'config': self.config.get_cv_config()
        }
    
    def _train_fold(self, 
                   model_builder: ModelBuilder,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   model_params: Dict[str, Any],
                   training_params: Dict[str, Any],
                   fold_idx: int) -> Dict[str, Any]:
        """Train and evaluate model for a single fold"""
        
        # Build model
        model = model_builder.build_model(**model_params)
        
        # Setup training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=training_params.get('patience', 10),
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        start_time = datetime.now()
        
        history = model.fit(
            X_train, y_train,
            epochs=training_params.get('epochs', 50),
            batch_size=training_params.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on test set
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Get predictions for detailed metrics
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred_classes, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_test, y_pred_classes, average=None, zero_division=0
        )
        
        # Store model for ensemble
        self.fold_models.append(model)
        
        return {
            'fold': fold_idx + 1,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'metrics': {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'weighted_precision': precision,
                'weighted_recall': recall,
                'weighted_f1': f1
            },
            'per_class_metrics': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1': per_class_f1.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            }
        }
    
    def _aggregate_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all folds"""
        
        if not fold_results:
            return {}
        
        # Extract metrics from all folds
        accuracies = [fold['metrics']['test_accuracy'] for fold in fold_results]
        precisions = [fold['metrics']['weighted_precision'] for fold in fold_results]
        recalls = [fold['metrics']['weighted_recall'] for fold in fold_results]
        f1_scores = [fold['metrics']['weighted_f1'] for fold in fold_results]
        losses = [fold['metrics']['test_loss'] for fold in fold_results]
        training_times = [fold['training_time'] for fold in fold_results]
        
        # Calculate statistics
        summary = {
            'num_folds': len(fold_results),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'total_training_time': np.sum(training_times),
            'mean_training_time': np.mean(training_times)
        }
        
        # Calculate 95% confidence intervals
        from scipy import stats
        
        for metric_name, values in [
            ('accuracy', accuracies),
            ('precision', precisions),
            ('recall', recalls),
            ('f1', f1_scores)
        ]:
            ci_lower, ci_upper = stats.t.interval(
                0.95, len(values)-1, 
                loc=np.mean(values), 
                scale=stats.sem(values)
            )
            summary[f'{metric_name}_ci_lower'] = ci_lower
            summary[f'{metric_name}_ci_upper'] = ci_upper
        
        # Aggregate confusion matrices
        all_cms = [fold['confusion_matrix'] for fold in fold_results]
        if all_cms:
            aggregated_cm = np.sum(all_cms, axis=0)
            summary['aggregated_confusion_matrix'] = aggregated_cm.tolist()
        
        return summary
    
    def get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble predictions from all fold models"""
        
        if not self.fold_models:
            raise ValueError("No trained models available for ensemble prediction")
        
        # Get predictions from all models
        all_predictions = []
        for model in self.fold_models:
            pred = model.predict(X, verbose=0)
            all_predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        return ensemble_pred
    
    def save_results(self, filepath: str):
        """Save cross-validation results"""
        
        results_data = {
            'config': self.config.get_cv_config(),
            'fold_results': self.cv_results,
            'performance_stats': self.performance_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"CV results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load cross-validation results"""
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.cv_results = results_data['fold_results']
        self.performance_stats = results_data['performance_stats']
        
        return results_data
    
    def plot_cv_results(self, save_path: Optional[str] = None):
        """Plot cross-validation results"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.cv_results:
                self.logger.warning("No CV results to plot")
                return
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Cross-Validation Results', fontsize=16)
            
            # 1. Accuracy across folds
            fold_nums = [result['fold'] for result in self.cv_results]
            accuracies = [result['metrics']['test_accuracy'] for result in self.cv_results]
            
            axes[0, 0].bar(fold_nums, accuracies)
            axes[0, 0].set_title('Test Accuracy by Fold')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].axhline(y=np.mean(accuracies), color='r', linestyle='--', label='Mean')
            axes[0, 0].legend()
            
            # 2. Training history (first fold as example)
            if self.cv_results:
                history = self.cv_results[0]['training_history']
                epochs = range(1, len(history['loss']) + 1)
                
                axes[0, 1].plot(epochs, history['loss'], label='Training Loss')
                axes[0, 1].plot(epochs, history['val_loss'], label='Validation Loss')
                axes[0, 1].set_title('Training History (Fold 1)')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
            
            # 3. Metrics distribution
            metrics_data = {
                'Accuracy': [result['metrics']['test_accuracy'] for result in self.cv_results],
                'Precision': [result['metrics']['weighted_precision'] for result in self.cv_results],
                'Recall': [result['metrics']['weighted_recall'] for result in self.cv_results],
                'F1-Score': [result['metrics']['weighted_f1'] for result in self.cv_results]
            }
            
            bp = axes[1, 0].boxplot(metrics_data.values(), labels=metrics_data.keys())
            axes[1, 0].set_title('Metrics Distribution Across Folds')
            axes[1, 0].set_ylabel('Score')
            
            # 4. Confusion matrix (aggregated)
            if 'aggregated_confusion_matrix' in self.performance_stats:
                cm = np.array(self.performance_stats['aggregated_confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
                axes[1, 1].set_title('Aggregated Confusion Matrix')
                axes[1, 1].set_xlabel('Predicted')
                axes[1, 1].set_ylabel('Actual')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"CV plots saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            self.logger.error(f"Error creating plots: {str(e)}")


def perform_nested_cv(outer_cv_config: CrossValidationConfig,
                     inner_cv_config: CrossValidationConfig,
                     model_builder: ModelBuilder,
                     hyperparameter_space: Dict[str, List],
                     X: np.ndarray,
                     y: np.ndarray) -> Dict[str, Any]:
    """
    Perform nested cross-validation for unbiased performance estimation
    
    Args:
        outer_cv_config: Configuration for outer CV loop
        inner_cv_config: Configuration for inner CV loop (hyperparameter tuning)
        model_builder: ModelBuilder instance
        hyperparameter_space: Hyperparameter search space
        X: Feature data
        y: Target data
        
    Returns:
        Nested CV results with unbiased performance estimates
    """
    
    logger = logging.getLogger('nested_cv')
    
    # Create outer CV splitter
    outer_cv = CrossValidator(outer_cv_config)
    
    outer_results = []
    
    # Outer CV loop
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.cv_splitter.split(X, y)):
        logger.info(f"Outer fold {fold_idx + 1}")
        
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Inner CV loop for hyperparameter tuning
        inner_cv = CrossValidator(inner_cv_config)
        
        # Simplified hyperparameter search for nested CV
        best_params = None
        best_score = -np.inf
        
        # Sample hyperparameter combinations
        from sklearn.model_selection import ParameterSampler
        param_sampler = ParameterSampler(
            hyperparameter_space, 
            n_iter=min(20, len(list(ParameterGrid(hyperparameter_space)))),
            random_state=42
        )
        
        for params in param_sampler:
            # Evaluate parameters using inner CV
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.cv_splitter.split(X_train_outer, y_train_outer):
                X_inner_train, X_inner_val = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
                y_inner_train, y_inner_val = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]
                
                # Train model with current parameters
                model = model_builder.build_model(**params)
                model.fit(
                    X_inner_train, y_inner_train,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_inner_val, y_inner_val),
                    verbose=0
                )
                
                # Evaluate
                _, accuracy, _, _ = model.evaluate(X_inner_val, y_inner_val, verbose=0)
                inner_scores.append(accuracy)
            
            mean_score = np.mean(inner_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        # Train final model with best parameters on full outer training set
        final_model = model_builder.build_model(**best_params)
        final_model.fit(
            X_train_outer, y_train_outer,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate on outer test set
        test_loss, test_accuracy, test_precision, test_recall = final_model.evaluate(
            X_test_outer, y_test_outer, verbose=0
        )
        
        outer_results.append({
            'fold': fold_idx + 1,
            'best_params': best_params,
            'inner_cv_score': best_score,
            'outer_test_accuracy': test_accuracy,
            'outer_test_precision': test_precision,
            'outer_test_recall': test_recall
        })
    
    # Aggregate outer results
    outer_accuracies = [result['outer_test_accuracy'] for result in outer_results]
    
    nested_cv_results = {
        'outer_results': outer_results,
        'unbiased_performance': {
            'mean_accuracy': np.mean(outer_accuracies),
            'std_accuracy': np.std(outer_accuracies),
            'accuracy_scores': outer_accuracies
        },
        'config': {
            'outer_cv': outer_cv_config.get_cv_config(),
            'inner_cv': inner_cv_config.get_cv_config()
        }
    }
    
    logger.info(f"Nested CV completed. Unbiased accuracy: {np.mean(outer_accuracies):.4f} ± {np.std(outer_accuracies):.4f}")
    
    return nested_cv_results
