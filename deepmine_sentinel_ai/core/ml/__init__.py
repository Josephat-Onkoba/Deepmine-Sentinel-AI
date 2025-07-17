"""
Machine Learning Models for Mining Stability Prediction

This module implements:
- Task 8: Design LSTM Architecture with TensorFlow/Keras, multi-step prediction, attention mechanisms
- Task 9: LSTM Training Pipeline with hyperparameter tuning, cross-validation, checkpointing, monitoring
"""

# Import LSTM models and configurations
try:
    from .lstm_config import LSTMConfig, LSTMModelConfig, AttentionConfig, MultiStepConfig, ModelArchitectureConfig
    from .attention_layers import AttentionLayer, MultiHeadAttention
    from .lstm_models import (
        BasicLSTMModel,
        MultiStepLSTMModel, 
        AttentionLSTMModel,
        MultiOutputLSTMModel,
        CompleteLSTMPredictor
    )
    from .model_utils import ModelBuilder, ModelTrainer, ModelEvaluator
    lstm_models_available = True
except ImportError as e:
    print(f"Warning: LSTM models not available: {e}")
    lstm_models_available = False

# Define exports
__all__ = []

# Add LSTM models if available  
if lstm_models_available:
    __all__.extend([
        'LSTMConfig', 'LSTMModelConfig', 'AttentionConfig', 'MultiStepConfig',
        'ModelArchitectureConfig', 'AttentionLayer', 'MultiHeadAttention',
        'BasicLSTMModel', 'MultiStepLSTMModel', 'AttentionLSTMModel', 
        'MultiOutputLSTMModel', 'CompleteLSTMPredictor',
        'ModelBuilder', 'ModelTrainer', 'ModelEvaluator',
        'TrainingPipelineConfig', 'HyperparameterConfig', 'CrossValidationConfig',
        'CheckpointConfig', 'MonitoringConfig', 'HyperparameterTuner',
        'CrossValidator', 'ModelCheckpoint', 'MetricsTracker', 
        'MiningSpecificMonitor', 'LSTMTrainingPipeline'
    ])
