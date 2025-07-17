"""
LSTM Data Pipeline - Coordinated Task 7 Implementation

Main pipeline that coordinates all Task 7 components:
1. Synthetic dataset generation 
2. Data preprocessing for time series sequences
3. Feature engineering for operational event patterns
4. Training/validation/test data splits
5. Data augmentation for rare event scenarios

Based on PROJECT_FINAL_REPORT.md specifications and Task 7 requirements
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import os
import pickle
import json

from django.utils import timezone
from django.conf import settings

from core.models import Stope, OperationalEvent
from core.data.synthetic_generator import SyntheticDataGenerator
from core.data.preprocessor import DataPreprocessor, PreprocessingConfig, SequenceData
from core.data.feature_engineer import FeatureEngineer, FeatureEngineeringConfig
from core.data.augmentor import DataAugmentor, AugmentationConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for complete LSTM data pipeline"""
    # Synthetic data generation
    generate_synthetic_data: bool = True
    synthetic_random_seed: int = 42
    
    # Data preprocessing  
    sequence_length: int = 168  # 7 days in hours
    prediction_horizon: int = 24  # 24 hours ahead
    validation_split: float = 0.2
    test_split: float = 0.2
    
    # Feature engineering
    enable_feature_engineering: bool = True
    rolling_windows: List[int] = None
    enable_spectral_features: bool = True
    enable_interaction_features: bool = True
    pca_components: int = 50  # Reduce to 50 components
    
    # Data augmentation
    enable_augmentation: bool = True
    target_samples_per_class: int = 800
    critical_oversample_factor: float = 2.0
    
    # Output settings
    save_processed_data: bool = True
    output_dir: str = "data/lstm_training"
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [6, 12, 24, 48]


class LSTMDataPipeline:
    """
    Complete LSTM training data preparation pipeline
    
    Implements all Task 7 requirements:
    - Create data preprocessing pipeline for time series sequences
    - Generate feature engineering for operational event patterns  
    - Build training/validation/test data splits
    - Implement data augmentation for rare event scenarios
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize pipeline with configuration"""
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.synthetic_generator = SyntheticDataGenerator(
            random_seed=self.config.synthetic_random_seed
        ) if self.config.generate_synthetic_data else None
        
        self.preprocessor = DataPreprocessor(
            PreprocessingConfig(
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon,
                validation_split=self.config.validation_split,
                test_split=self.config.test_split,
                random_seed=self.config.synthetic_random_seed
            )
        )
        
        self.feature_engineer = FeatureEngineer(
            FeatureEngineeringConfig(
                rolling_windows=self.config.rolling_windows,
                spectral_features=self.config.enable_spectral_features,
                interaction_features=self.config.enable_interaction_features,
                pca_components=self.config.pca_components
            )
        ) if self.config.enable_feature_engineering else None
        
        self.augmentor = DataAugmentor(
            AugmentationConfig(
                target_samples_per_class=self.config.target_samples_per_class,
                critical_oversample_factor=self.config.critical_oversample_factor
            )
        ) if self.config.enable_augmentation else None
        
        # Setup output directory
        self.output_dir = os.path.join(settings.BASE_DIR, self.config.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("LSTM Data Pipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Sequence length: {self.config.sequence_length} hours")
        logger.info(f"Prediction horizon: {self.config.prediction_horizon} hours")
    
    def generate_synthetic_dataset(self) -> Tuple[List[Stope], List[OperationalEvent]]:
        """
        Generate synthetic dataset according to PROJECT_FINAL_REPORT.md specifications
        
        Returns:
        - 156 synthetic stopes with realistic geological parameters
        - 2,847 operational events over 18-month simulation
        """
        if not self.synthetic_generator:
            raise ValueError("Synthetic data generation is disabled")
        
        logger.info("Generating synthetic dataset...")
        
        stopes, events = self.synthetic_generator.generate_complete_dataset()
        
        logger.info(f"Generated synthetic dataset:")
        logger.info(f"  Stopes: {len(stopes)}")
        logger.info(f"  Events: {len(events)}")
        
        return stopes, events
    
    def preprocess_sequences(self, stopes: List[Stope] = None) -> SequenceData:
        """
        Preprocess raw data into LSTM-ready sequences
        
        Creates 168-hour sliding windows with 24-hour prediction horizons
        """
        logger.info("Preprocessing sequences for LSTM training...")
        
        if stopes is None:
            stopes = list(Stope.objects.filter(is_active=True))
        
        sequence_data = self.preprocessor.process_all_stopes(stopes)
        
        # Validate data quality
        validation_results = self.preprocessor.validate_data_quality(sequence_data)
        
        if not validation_results['valid']:
            logger.warning("Data quality issues detected:")
            for issue in validation_results['issues']:
                logger.warning(f"  - {issue}")
        
        if validation_results['recommendations']:
            logger.info("Recommendations:")
            for rec in validation_results['recommendations']:
                logger.info(f"  - {rec}")
        
        logger.info(f"Sequence preprocessing completed:")
        logger.info(f"  Total sequences: {len(sequence_data.risk_labels)}")
        logger.info(f"  Sequence shape: {sequence_data.dynamic_features.shape}")
        logger.info(f"  Static features: {sequence_data.static_features.shape}")
        logger.info(f"  Class distribution: {sequence_data.metadata['class_distribution']}")
        
        return sequence_data
    
    def engineer_features(self, sequence_data: SequenceData) -> SequenceData:
        """
        Apply feature engineering for operational event patterns
        
        Adds temporal patterns, rolling statistics, event patterns, and interactions
        """
        if not self.feature_engineer:
            logger.info("Feature engineering is disabled, skipping...")
            return sequence_data
        
        logger.info("Applying feature engineering...")
        
        # Apply feature engineering
        engineered_data = self.feature_engineer.engineer_features(sequence_data)
        
        # Apply dimensionality reduction if configured
        if self.config.pca_components > 0:
            engineered_data = self.feature_engineer.apply_dimensionality_reduction(engineered_data)
        
        logger.info(f"Feature engineering completed:")
        logger.info(f"  Original features: {sequence_data.dynamic_features.shape[2]}")
        logger.info(f"  Engineered features: {engineered_data.dynamic_features.shape[2]}")
        
        if 'pca_applied' in engineered_data.metadata:
            logger.info(f"  PCA variance explained: {engineered_data.metadata['pca_total_variance_explained']:.3f}")
        
        return engineered_data
    
    def augment_data(self, sequence_data: SequenceData) -> SequenceData:
        """
        Apply data augmentation for rare event scenarios
        
        Balances classes and generates synthetic critical scenarios
        """
        if not self.augmentor:
            logger.info("Data augmentation is disabled, skipping...")
            return sequence_data
        
        logger.info("Applying data augmentation...")
        
        augmented_data = self.augmentor.augment_data(sequence_data)
        
        logger.info(f"Data augmentation completed:")
        logger.info(f"  Original samples: {sequence_data.metadata['total_sequences']}")
        logger.info(f"  Augmented samples: {len(augmented_data.risk_labels)}")
        logger.info(f"  Augmentation ratio: {augmented_data.metadata.get('augmentation_ratio', 0):.2f}")
        
        return augmented_data
    
    def create_data_splits(self, sequence_data: SequenceData) -> Tuple[Dict, Dict, Dict]:
        """
        Create training/validation/test splits with temporal considerations
        
        Uses temporal split to prevent data leakage
        """
        logger.info("Creating train/validation/test splits...")
        
        train_data, val_data, test_data = self.preprocessor.create_train_val_test_splits(sequence_data)
        
        logger.info(f"Data splits created:")
        logger.info(f"  Training: {len(train_data['risk_labels'])} samples")
        logger.info(f"  Validation: {len(val_data['risk_labels'])} samples") 
        logger.info(f"  Test: {len(test_data['risk_labels'])} samples")
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, train_data: Dict, val_data: Dict, test_data: Dict,
                          metadata: Dict) -> Dict[str, str]:
        """
        Save processed data to disk for LSTM training
        
        Saves data in multiple formats for flexibility
        """
        if not self.config.save_processed_data:
            logger.info("Data saving is disabled, skipping...")
            return {}
        
        logger.info("Saving processed data...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_paths = {}
        
        # Save as NumPy arrays (efficient for training)
        train_path = os.path.join(self.output_dir, f"train_data_{timestamp}.npz")
        np.savez_compressed(
            train_path,
            static_features=train_data['static_features'],
            dynamic_features=train_data['dynamic_features'],
            risk_labels=train_data['risk_labels'],
            timestamps=train_data['timestamps'],
            stope_ids=train_data['stope_ids']
        )
        save_paths['train'] = train_path
        
        val_path = os.path.join(self.output_dir, f"val_data_{timestamp}.npz")
        np.savez_compressed(
            val_path,
            static_features=val_data['static_features'],
            dynamic_features=val_data['dynamic_features'],
            risk_labels=val_data['risk_labels'],
            timestamps=val_data['timestamps'],
            stope_ids=val_data['stope_ids']
        )
        save_paths['validation'] = val_path
        
        test_path = os.path.join(self.output_dir, f"test_data_{timestamp}.npz")
        np.savez_compressed(
            test_path,
            static_features=test_data['static_features'],
            dynamic_features=test_data['dynamic_features'],
            risk_labels=test_data['risk_labels'],
            timestamps=test_data['timestamps'],
            stope_ids=test_data['stope_ids']
        )
        save_paths['test'] = test_path
        
        # Save metadata as JSON
        metadata_path = os.path.join(self.output_dir, f"metadata_{timestamp}.json")
        # Convert numpy types to Python types for JSON serialization
        serializable_metadata = self._make_json_serializable(metadata)
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2, default=str)
        save_paths['metadata'] = metadata_path
        
        # Save pipeline configuration
        config_path = os.path.join(self.output_dir, f"pipeline_config_{timestamp}.json")
        config_dict = {
            'sequence_length': self.config.sequence_length,
            'prediction_horizon': self.config.prediction_horizon,
            'validation_split': self.config.validation_split,
            'test_split': self.config.test_split,
            'feature_engineering_enabled': self.config.enable_feature_engineering,
            'augmentation_enabled': self.config.enable_augmentation,
            'rolling_windows': self.config.rolling_windows,
            'pca_components': self.config.pca_components,
            'target_samples_per_class': self.config.target_samples_per_class
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        save_paths['config'] = config_path
        
        # Save preprocessor scalers for inference
        scalers_path = os.path.join(self.output_dir, f"scalers_{timestamp}.pkl")
        scalers_data = {
            'static_scaler': self.preprocessor.static_scaler,
            'dynamic_scaler': self.preprocessor.dynamic_scaler,
            'impact_scaler': self.preprocessor.impact_scaler,
            'rock_type_encoder': self.preprocessor.rock_type_encoder,
            'mining_method_encoder': self.preprocessor.mining_method_encoder
        }
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers_data, f)
        save_paths['scalers'] = scalers_path
        
        logger.info(f"Processed data saved to {self.output_dir}")
        logger.info(f"Files created: {list(save_paths.keys())}")
        
        return save_paths
    
    def validate_data_quality(self, sequence_data: SequenceData) -> Dict:
        """
        Validate data quality for pipeline integration
        
        Returns validation metrics and quality assessment
        """
        validation_report = {
            'num_samples': sequence_data.dynamic_features.shape[0],
            'sequence_length': sequence_data.dynamic_features.shape[1],
            'feature_dimension': sequence_data.dynamic_features.shape[2],
            'static_features': sequence_data.static_features.shape[1],
            'unique_risk_levels': len(np.unique(sequence_data.risk_labels)),
            'data_quality': 'PASS',
            'issues': []
        }
        
        # Check for basic data integrity
        if sequence_data.dynamic_features.shape[0] == 0:
            validation_report['issues'].append("No sequences generated")
            validation_report['data_quality'] = 'FAIL'
        
        if np.any(np.isnan(sequence_data.dynamic_features)):
            validation_report['issues'].append("NaN values in dynamic features")
            validation_report['data_quality'] = 'WARN'
        
        return validation_report
    
    def _make_json_serializable(self, obj):
        """Convert numpy and other types to JSON-serializable formats"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        else:
            return obj
    
    def run_complete_pipeline(self) -> Dict:
        """
        Execute complete Task 7 pipeline
        
        Returns summary of pipeline execution and data paths
        """
        logger.info("Starting complete LSTM data preparation pipeline (Task 7)")
        
        pipeline_start = datetime.now()
        results = {
            'pipeline_start': pipeline_start,
            'config': self.config,
            'stages_completed': [],
            'data_paths': {},
            'statistics': {}
        }
        
        try:
            # Stage 1: Generate synthetic dataset (if enabled)
            if self.config.generate_synthetic_data:
                logger.info("Stage 1: Generating synthetic dataset...")
                stopes, events = self.generate_synthetic_dataset()
                results['stages_completed'].append('synthetic_generation')
                results['statistics']['synthetic_stopes'] = len(stopes)
                results['statistics']['synthetic_events'] = len(events)
            else:
                stopes = None
            
            # Stage 2: Preprocess sequences
            logger.info("Stage 2: Preprocessing sequences...")
            sequence_data = self.preprocess_sequences(stopes)
            results['stages_completed'].append('preprocessing')
            results['statistics']['total_sequences'] = len(sequence_data.risk_labels)
            results['statistics']['original_feature_dim'] = sequence_data.dynamic_features.shape[2]
            results['statistics']['class_distribution'] = sequence_data.metadata['class_distribution']
            
            # Stage 3: Feature engineering (if enabled)
            if self.config.enable_feature_engineering:
                logger.info("Stage 3: Feature engineering...")
                sequence_data = self.engineer_features(sequence_data)
                results['stages_completed'].append('feature_engineering')
                results['statistics']['engineered_feature_dim'] = sequence_data.dynamic_features.shape[2]
            
            # Stage 4: Data augmentation (if enabled)
            if self.config.enable_augmentation:
                logger.info("Stage 4: Data augmentation...")
                sequence_data = self.augment_data(sequence_data)
                results['stages_completed'].append('augmentation')
                results['statistics']['augmented_sequences'] = len(sequence_data.risk_labels)
                results['statistics']['final_class_distribution'] = np.bincount(sequence_data.risk_labels).tolist()
            
            # Stage 5: Create data splits
            logger.info("Stage 5: Creating data splits...")
            train_data, val_data, test_data = self.create_data_splits(sequence_data)
            results['stages_completed'].append('data_splits')
            results['statistics']['train_samples'] = len(train_data['risk_labels'])
            results['statistics']['val_samples'] = len(val_data['risk_labels'])
            results['statistics']['test_samples'] = len(test_data['risk_labels'])
            
            # Stage 6: Save processed data (if enabled)
            if self.config.save_processed_data:
                logger.info("Stage 6: Saving processed data...")
                data_paths = self.save_processed_data(train_data, val_data, test_data, sequence_data.metadata)
                results['stages_completed'].append('data_saving')
                results['data_paths'] = data_paths
            
            # Calculate final statistics
            pipeline_end = datetime.now()
            results['pipeline_end'] = pipeline_end
            results['pipeline_duration'] = (pipeline_end - pipeline_start).total_seconds()
            results['success'] = True
            
            logger.info("LSTM data preparation pipeline completed successfully!")
            logger.info(f"Pipeline duration: {results['pipeline_duration']:.2f} seconds")
            logger.info(f"Stages completed: {results['stages_completed']}")
            
            # Summary statistics
            logger.info("Final Dataset Statistics:")
            for key, value in results['statistics'].items():
                logger.info(f"  {key}: {value}")
            
        except Exception as e:
            logger.error(f"Pipeline failed at stage: {len(results['stages_completed']) + 1}")
            logger.error(f"Error: {e}")
            results['success'] = False
            results['error'] = str(e)
            raise
        
        return results
    
    def generate_pipeline_report(self, results: Dict) -> str:
        """
        Generate comprehensive pipeline report
        
        Returns formatted report string for documentation
        """
        report = []
        report.append("=" * 80)
        report.append("LSTM TRAINING DATA PREPARATION PIPELINE REPORT")
        report.append("Task 7: Prepare LSTM Training Data")
        report.append("=" * 80)
        report.append("")
        
        # Pipeline configuration
        report.append("PIPELINE CONFIGURATION:")
        report.append(f"  Sequence Length: {self.config.sequence_length} hours")
        report.append(f"  Prediction Horizon: {self.config.prediction_horizon} hours")
        report.append(f"  Validation Split: {self.config.validation_split}")
        report.append(f"  Test Split: {self.config.test_split}")
        report.append(f"  Feature Engineering: {self.config.enable_feature_engineering}")
        report.append(f"  Data Augmentation: {self.config.enable_augmentation}")
        report.append(f"  PCA Components: {self.config.pca_components}")
        report.append("")
        
        # Execution summary
        report.append("EXECUTION SUMMARY:")
        if results['success']:
            report.append(f"  Status: SUCCESS")
            report.append(f"  Duration: {results['pipeline_duration']:.2f} seconds")
            report.append(f"  Stages Completed: {', '.join(results['stages_completed'])}")
        else:
            report.append(f"  Status: FAILED")
            report.append(f"  Error: {results.get('error', 'Unknown error')}")
        report.append("")
        
        # Dataset statistics
        if 'statistics' in results:
            report.append("DATASET STATISTICS:")
            stats = results['statistics']
            
            if 'synthetic_stopes' in stats:
                report.append(f"  Synthetic Stopes Generated: {stats['synthetic_stopes']}")
                report.append(f"  Synthetic Events Generated: {stats['synthetic_events']}")
            
            report.append(f"  Total Sequences: {stats.get('total_sequences', 'N/A')}")
            report.append(f"  Original Features: {stats.get('original_feature_dim', 'N/A')}")
            
            if 'engineered_feature_dim' in stats:
                report.append(f"  Engineered Features: {stats['engineered_feature_dim']}")
            
            if 'augmented_sequences' in stats:
                report.append(f"  Augmented Sequences: {stats['augmented_sequences']}")
            
            report.append(f"  Training Samples: {stats.get('train_samples', 'N/A')}")
            report.append(f"  Validation Samples: {stats.get('val_samples', 'N/A')}")
            report.append(f"  Test Samples: {stats.get('test_samples', 'N/A')}")
            
            if 'class_distribution' in stats:
                report.append(f"  Original Class Distribution: {stats['class_distribution']}")
            
            if 'final_class_distribution' in stats:
                report.append(f"  Final Class Distribution: {stats['final_class_distribution']}")
        
        report.append("")
        
        # Data paths
        if 'data_paths' in results and results['data_paths']:
            report.append("OUTPUT FILES:")
            for data_type, path in results['data_paths'].items():
                report.append(f"  {data_type.title()}: {path}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
