"""
Core Data Processing Package for LSTM Training Data Preparation

This package contains modules for:
- Synthetic dataset generation
- Data preprocessing pipelines
- Feature engineering
- Data augmentation
- Train/validation/test splits
"""

from .synthetic_generator import SyntheticDataGenerator
from .preprocessor import DataPreprocessor, PreprocessingConfig, SequenceData
from .feature_engineer import FeatureEngineer, FeatureEngineeringConfig
from .augmentor import DataAugmentor, AugmentationConfig
from .lstm_pipeline import LSTMDataPipeline, PipelineConfig

__all__ = [
    'SyntheticDataGenerator',
    'DataPreprocessor', 
    'PreprocessingConfig',
    'SequenceData',
    'FeatureEngineer',
    'FeatureEngineeringConfig',
    'DataAugmentor',
    'AugmentationConfig',
    'LSTMDataPipeline',
    'PipelineConfig'
]
