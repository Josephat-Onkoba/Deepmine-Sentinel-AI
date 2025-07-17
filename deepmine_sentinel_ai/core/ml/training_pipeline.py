"""
Complete LSTM Training Pipeline

This module implements Task 9: LSTM Training Pipeline with:
- Comprehensive training orchestration
- Hyperparameter tuning integration
- Cross-validation support
- Model checkpointing and monitoring
- Mining-specific training strategies
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from .training_config import TrainingPipelineConfig, HyperparameterConfig, CrossValidationConfig
from .hyperparameter_tuner import HyperparameterTuner
from .cross_validation import CrossValidator
from .model_checkpoint import ModelCheckpoint
from .training_monitor import MetricsTracker, MiningSpecificMonitor, create_monitoring_callbacks
from .lstm_config import LSTMConfig, AttentionConfig, MultiStepConfig
from .model_utils import ModelBuilder, ModelTrainer


class LSTMTrainingPipeline:
    """
    Complete LSTM training pipeline for mining stability prediction
    
    Orchestrates:
    - Data preparation and validation
    - Hyperparameter optimization
    - Cross-validation
    - Model training with monitoring
    - Checkpointing and version management
    """
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.hyperparameter_tuner = None
        self.cross_validator = None
        self.checkpoint_manager = ModelCheckpoint(config.checkpoint_config)
        self.metrics_tracker = MetricsTracker(config.monitoring_config)
        self.mining_monitor = MiningSpecificMonitor()
        
        # Training state
        self.training_history = {}
        self.best_model = None
        self.best_metrics = None
        self.experiment_id = self._generate_experiment_id()
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.data_prepared = False
        
        # Results storage
        self.results = {
            'experiment_id': self.experiment_id,
            'config': config.get_complete_config(),
            'hyperparameter_tuning': None,
            'cross_validation': None,
            'training_results': None,
            'model_performance': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for the training pipeline"""
        logger = logging.getLogger('lstm_training_pipeline')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create log directory
        log_dir = 'logs/training_pipeline'
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'pipeline_{timestamp}.log')
        
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
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"lstm_mining_{timestamp}"
    
    def prepare_data(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    feature_names: List[str] = None,
                    normalize: bool = True,
                    test_size: float = 0.2,
                    validation_size: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Prepare data for training with proper splits and preprocessing
        
        Args:
            X: Feature data [samples, sequence_length, features]
            y: Target labels [samples]
            feature_names: Names of features
            normalize: Whether to normalize features
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            
        Returns:
            Dictionary with train/val/test splits
        """
        
        self.logger.info("Preparing data for training...")
        
        # Validate input data
        if len(X.shape) != 3:
            raise ValueError("X must be 3D array: [samples, sequence_length, features]")
        
        if len(y.shape) != 1:
            raise ValueError("y must be 1D array: [samples]")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        # Log data statistics
        self.logger.info(f"Data shape: {X.shape}")
        self.logger.info(f"Number of classes: {len(np.unique(y))}")
        self.logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Store data info
        self.data_info = {
            'input_shape': X.shape[1:],
            'num_samples': X.shape[0],
            'num_features': X.shape[2],
            'sequence_length': X.shape[1],
            'num_classes': len(np.unique(y)),
            'class_distribution': np.bincount(y).tolist(),
            'feature_names': feature_names or [f'feature_{i}' for i in range(X.shape[2])]
        }
        
        # Create train/validation/test splits
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=42
        )
        
        # Second split: separate validation from training
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=42
        )
        
        # Normalize features if requested
        if normalize:
            self.logger.info("Normalizing features...")
            
            # Reshape for scaling (combine samples and sequence dimensions)
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            # Fit scaler on training data only
            self.scaler.fit(X_train_reshaped)
            
            # Transform all sets
            X_train_scaled = self.scaler.transform(X_train_reshaped)
            X_val_scaled = self.scaler.transform(X_val_reshaped)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            
            # Reshape back to original format
            X_train = X_train_scaled.reshape(X_train.shape)
            X_val = X_val_scaled.reshape(X_val.shape)
            X_test = X_test_scaled.reshape(X_test.shape)
        
        # Store prepared data
        self.data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        self.data_prepared = True
        
        self.logger.info(f"Data splits - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return self.data_splits
    
    def run_hyperparameter_tuning(self, 
                                 model_type: str = 'basic',
                                 max_trials: int = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Args:
            model_type: Type of LSTM model ('basic', 'attention', 'multi_step')
            max_trials: Maximum number of trials (overrides config)
            
        Returns:
            Hyperparameter tuning results
        """
        
        if not self.data_prepared:
            raise ValueError("Data must be prepared before hyperparameter tuning")
        
        self.logger.info(f"Starting hyperparameter tuning for {model_type} model...")
        
        # Update config if max_trials provided
        if max_trials:
            self.config.hyperparameter_config.max_trials = max_trials
        
        # Initialize hyperparameter tuner
        self.hyperparameter_tuner = HyperparameterTuner(
            config=self.config.hyperparameter_config,
            input_shape=self.data_info['input_shape'],
            num_classes=self.data_info['num_classes'],
            model_type=model_type
        )
        
        # Run tuning
        tuning_results = self.hyperparameter_tuner.tune(
            X_train=self.data_splits['X_train'],
            y_train=self.data_splits['y_train'],
            X_val=self.data_splits['X_val'],
            y_val=self.data_splits['y_val'],
            epochs=50,
            batch_size=32
        )
        
        # Store results
        self.results['hyperparameter_tuning'] = tuning_results
        
        # Save tuning results
        tuning_results_file = f'results/hyperparameter_tuning_{self.experiment_id}.json'
        self.hyperparameter_tuner.save_results(tuning_results_file)
        
        self.logger.info("Hyperparameter tuning completed")
        self.logger.info(f"Best parameters: {tuning_results['best_hyperparameters']}")
        
        return tuning_results
    
    def run_cross_validation(self, 
                            model_params: Dict[str, Any] = None,
                            cv_folds: int = None) -> Dict[str, Any]:
        """
        Run cross-validation evaluation
        
        Args:
            model_params: Model parameters (uses best from tuning if None)
            cv_folds: Number of CV folds (overrides config)
            
        Returns:
            Cross-validation results
        """
        
        if not self.data_prepared:
            raise ValueError("Data must be prepared before cross-validation")
        
        self.logger.info("Starting cross-validation...")
        
        # Use best hyperparameters if not provided
        if model_params is None:
            if self.results['hyperparameter_tuning'] is None:
                raise ValueError("No model parameters provided and no hyperparameter tuning results available")
            model_params = self.results['hyperparameter_tuning']['best_hyperparameters']
        
        # Update CV config if folds provided
        if cv_folds:
            self.config.cv_config.n_splits = cv_folds
        
        # Initialize cross-validator
        self.cross_validator = CrossValidator(self.config.cv_config)
        
        # Prepare data for CV (combine train and validation for CV)
        X_cv = np.concatenate([self.data_splits['X_train'], self.data_splits['X_val']], axis=0)
        y_cv = np.concatenate([self.data_splits['y_train'], self.data_splits['y_val']], axis=0)
        
        # Create model builder
        model_builder = ModelBuilder()
        
        # Convert hyperparameters to model configuration
        lstm_config = LSTMConfig(
            input_features=self.data_info['num_features'],
            sequence_length=self.data_info['sequence_length'],
            lstm_units=model_params.get('lstm_units', [64, 32]),
            dense_units=model_params.get('dense_units', [64, 32]),
            dropout_rate=model_params.get('dropout_rate', 0.2),
            num_classes=self.data_info['num_classes']
        )
        
        model_build_params = {
            'model_type': 'basic',
            'lstm_config': lstm_config
        }
        
        training_params = {
            'epochs': self.config.epochs,
            'batch_size': model_params.get('batch_size', 32),
            'patience': self.config.monitoring_config.patience
        }
        
        # Run cross-validation
        cv_results = self.cross_validator.validate_model(
            model_builder=model_builder,
            X=X_cv,
            y=y_cv,
            model_params=model_build_params,
            training_params=training_params
        )
        
        # Store results
        self.results['cross_validation'] = cv_results
        
        # Save CV results
        cv_results_file = f'results/cross_validation_{self.experiment_id}.json'
        self.cross_validator.save_results(cv_results_file)
        
        self.logger.info("Cross-validation completed")
        self.logger.info(f"Mean CV accuracy: {cv_results['summary']['mean_accuracy']:.4f} ± {cv_results['summary']['std_accuracy']:.4f}")
        
        return cv_results
    
    def train_final_model(self, 
                         model_params: Dict[str, Any] = None,
                         save_checkpoints: bool = True) -> keras.Model:
        """
        Train the final model with best parameters
        
        Args:
            model_params: Model parameters (uses best from tuning if None)
            save_checkpoints: Whether to save model checkpoints
            
        Returns:
            Trained Keras model
        """
        
        if not self.data_prepared:
            raise ValueError("Data must be prepared before training")
        
        self.logger.info("Starting final model training...")
        
        # Use best hyperparameters if not provided
        if model_params is None:
            if self.results['hyperparameter_tuning'] is None:
                raise ValueError("No model parameters provided and no hyperparameter tuning results available")
            model_params = self.results['hyperparameter_tuning']['best_hyperparameters']
        
        # Build model configuration
        lstm_config = LSTMConfig(
            input_features=self.data_info['num_features'],
            sequence_length=self.data_info['sequence_length'],
            lstm_units=model_params.get('lstm_units', [64, 32]),
            dense_units=model_params.get('dense_units', [64, 32]),
            dropout_rate=model_params.get('dropout_rate', 0.2),
            num_classes=self.data_info['num_classes']
        )
        
        # Build model
        model_builder = ModelBuilder()
        model = model_builder.build_model(
            model_type='basic',
            lstm_config=lstm_config
        )
        
        # Setup training callbacks
        callbacks = create_monitoring_callbacks(
            config=self.config.monitoring_config,
            metrics_tracker=self.metrics_tracker,
            experiment_name=self.experiment_id
        )
        
        # Add checkpoint callback if requested
        if save_checkpoints:
            from .model_checkpoint import create_training_checkpoint_callback
            checkpoint_callback = create_training_checkpoint_callback(
                checkpoint_manager=self.checkpoint_manager,
                model_name=f"lstm_mining_{self.experiment_id}",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            callbacks.append(checkpoint_callback)
        
        # Train model
        self.logger.info("Training model...")
        
        history = model.fit(
            self.data_splits['X_train'],
            self.data_splits['y_train'],
            epochs=self.config.epochs,
            batch_size=model_params.get('batch_size', 32),
            validation_data=(self.data_splits['X_val'], self.data_splits['y_val']),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        self.logger.info("Evaluating model on test set...")
        
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
            self.data_splits['X_test'],
            self.data_splits['y_test'],
            verbose=0
        )
        
        # Get detailed predictions for analysis
        test_predictions = model.predict(self.data_splits['X_test'])
        test_pred_classes = np.argmax(test_predictions, axis=1)
        
        # Calculate detailed metrics
        test_report = classification_report(
            self.data_splits['y_test'],
            test_pred_classes,
            output_dict=True
        )
        
        test_cm = confusion_matrix(
            self.data_splits['y_test'],
            test_pred_classes
        )
        
        # Store training results
        self.training_history = history.history
        self.best_model = model
        self.best_metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'classification_report': test_report,
            'confusion_matrix': test_cm.tolist()
        }
        
        # Log mining-specific predictions
        confidence_scores = np.max(test_predictions, axis=1)
        self.mining_monitor.log_predictions(
            y_true=self.data_splits['y_test'],
            y_pred=test_pred_classes,
            confidence_scores=confidence_scores
        )
        
        # Store results
        self.results['training_results'] = {
            'training_history': self.training_history,
            'model_parameters': model_params,
            'final_metrics': self.best_metrics,
            'mining_metrics': self.mining_monitor.get_mining_metrics()
        }
        
        # Create final checkpoint
        if save_checkpoints:
            final_checkpoint = self.checkpoint_manager.create_checkpoint(
                model=model,
                model_name=f"lstm_mining_final_{self.experiment_id}",
                metrics=self.best_metrics,
                hyperparameters=model_params,
                training_history=self.training_history
            )
            
            self.logger.info(f"Final model checkpoint created: {final_checkpoint.version}")
        
        self.logger.info("Model training completed")
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Test precision: {test_precision:.4f}")
        self.logger.info(f"Test recall: {test_recall:.4f}")
        
        return model
    
    def run_complete_pipeline(self, 
                             X: np.ndarray, 
                             y: np.ndarray,
                             model_type: str = 'basic',
                             skip_hyperparameter_tuning: bool = False,
                             skip_cross_validation: bool = False) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            X: Feature data
            y: Target labels
            model_type: Type of LSTM model
            skip_hyperparameter_tuning: Skip hyperparameter optimization
            skip_cross_validation: Skip cross-validation
            
        Returns:
            Complete pipeline results
        """
        
        self.logger.info(f"Starting complete LSTM training pipeline: {self.experiment_id}")
        
        try:
            # Step 1: Prepare data
            self.prepare_data(X, y)
            
            # Step 2: Hyperparameter tuning (optional)
            if not skip_hyperparameter_tuning:
                self.run_hyperparameter_tuning(model_type=model_type)
            else:
                # Use default parameters
                default_params = {
                    'lstm_units': [64, 32],
                    'dense_units': [64, 32],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'optimizer': 'adam'
                }
                self.results['hyperparameter_tuning'] = {
                    'best_hyperparameters': default_params,
                    'best_score': 0.0
                }
            
            # Step 3: Cross-validation (optional)
            if not skip_cross_validation:
                self.run_cross_validation()
            
            # Step 4: Train final model
            final_model = self.train_final_model()
            
            # Step 5: Save complete results
            self.save_pipeline_results()
            
            self.logger.info("Complete pipeline finished successfully")
            
            return {
                'experiment_id': self.experiment_id,
                'final_model': final_model,
                'results': self.results,
                'data_info': self.data_info
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def save_pipeline_results(self):
        """Save complete pipeline results to file"""
        
        results_dir = 'results/training_pipeline'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save main results
        results_file = os.path.join(results_dir, f'pipeline_results_{self.experiment_id}.json')
        
        # Prepare serializable results
        serializable_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.get_complete_config(),
            'data_info': self.data_info,
            'results': {
                'hyperparameter_tuning': self.results.get('hyperparameter_tuning'),
                'cross_validation': self.results.get('cross_validation'),
                'training_results': self.results.get('training_results')
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save metrics
        if self.metrics_tracker:
            metrics_file = os.path.join(results_dir, f'metrics_{self.experiment_id}.json')
            self.metrics_tracker.save_metrics(metrics_file)
        
        self.logger.info(f"Pipeline results saved to {results_dir}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive training report"""
        
        report_lines = [
            f"LSTM Training Pipeline Report",
            f"=" * 50,
            f"Experiment ID: {self.experiment_id}",
            f"Timestamp: {datetime.now().isoformat()}",
            f"",
            f"Data Information:",
            f"- Input Shape: {self.data_info['input_shape']}",
            f"- Number of Samples: {self.data_info['num_samples']}",
            f"- Number of Features: {self.data_info['num_features']}",
            f"- Sequence Length: {self.data_info['sequence_length']}",
            f"- Number of Classes: {self.data_info['num_classes']}",
            f"- Class Distribution: {self.data_info['class_distribution']}",
            f""
        ]
        
        # Hyperparameter tuning results
        if self.results.get('hyperparameter_tuning'):
            hp_results = self.results['hyperparameter_tuning']
            report_lines.extend([
                f"Hyperparameter Tuning Results:",
                f"- Best Score: {hp_results.get('best_score', 'N/A'):.4f}",
                f"- Best Parameters: {hp_results.get('best_hyperparameters', 'N/A')}",
                f"- Trials Completed: {hp_results.get('trials_completed', 'N/A')}",
                f""
            ])
        
        # Cross-validation results
        if self.results.get('cross_validation'):
            cv_results = self.results['cross_validation']['summary']
            report_lines.extend([
                f"Cross-Validation Results:",
                f"- Mean Accuracy: {cv_results.get('mean_accuracy', 'N/A'):.4f} ± {cv_results.get('std_accuracy', 'N/A'):.4f}",
                f"- Mean Precision: {cv_results.get('mean_precision', 'N/A'):.4f} ± {cv_results.get('std_precision', 'N/A'):.4f}",
                f"- Mean Recall: {cv_results.get('mean_recall', 'N/A'):.4f} ± {cv_results.get('std_recall', 'N/A'):.4f}",
                f"- Mean F1-Score: {cv_results.get('mean_f1', 'N/A'):.4f} ± {cv_results.get('std_f1', 'N/A'):.4f}",
                f""
            ])
        
        # Final model results
        if self.best_metrics:
            report_lines.extend([
                f"Final Model Performance:",
                f"- Test Accuracy: {self.best_metrics['test_accuracy']:.4f}",
                f"- Test Precision: {self.best_metrics['test_precision']:.4f}",
                f"- Test Recall: {self.best_metrics['test_recall']:.4f}",
                f"- Test Loss: {self.best_metrics['test_loss']:.4f}",
                f""
            ])
        
        # Mining-specific metrics
        if self.results.get('training_results', {}).get('mining_metrics'):
            mining_metrics = self.results['training_results']['mining_metrics']
            report_lines.extend([
                f"Mining-Specific Metrics:",
                f"- Average Confidence: {mining_metrics.get('average_confidence', 'N/A'):.4f}",
                f"- Stability Class Distribution:",
            ])
            
            for class_name, metrics in mining_metrics.get('stability_class_distribution', {}).items():
                report_lines.append(f"  - {class_name}: {metrics.get('percentage', 0):.1f}% ({metrics.get('count', 0)} samples)")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = f'results/training_pipeline/report_{self.experiment_id}.txt'
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Training report saved to {report_file}")
        
        return report_text
