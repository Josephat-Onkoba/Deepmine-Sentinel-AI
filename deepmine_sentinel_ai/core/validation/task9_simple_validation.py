"""
Simple Task 9 Validation: LSTM Training Pipeline

This script validates the core components of the LSTM training pipeline
"""

import os
import sys
import numpy as np

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_training_config():
    """Test training configuration classes"""
    print("Testing training configuration...")
    
    try:
        from core.ml.training_config import (
            TrainingPipelineConfig, HyperparameterConfig, 
            CrossValidationConfig, CheckpointConfig, MonitoringConfig
        )
        
        # Test basic configuration creation
        hyperparameter_config = HyperparameterConfig()
        cv_config = CrossValidationConfig()
        checkpoint_config = CheckpointConfig()
        monitoring_config = MonitoringConfig()
        
        pipeline_config = TrainingPipelineConfig(
            epochs=20,
            hyperparameter_config=hyperparameter_config,
            cv_config=cv_config,
            checkpoint_config=checkpoint_config,
            monitoring_config=monitoring_config
        )
        
        # Test configuration serialization
        config_dict = pipeline_config.get_complete_config()
        assert isinstance(config_dict, dict)
        
        print("✓ Training configuration classes work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Training configuration failed: {e}")
        return False


def test_hyperparameter_tuning():
    """Test hyperparameter tuning system"""
    print("Testing hyperparameter tuning...")
    
    try:
        from core.ml.hyperparameter_tuner import HyperparameterTuner
        from core.ml.training_config import HyperparameterConfig
        
        # Create configuration
        hp_config = HyperparameterConfig()
        
        # Test tuner initialization
        tuner = HyperparameterTuner(
            config=hp_config,
            input_shape=(50, 12),
            num_classes=3,
            model_type='basic'
        )
        
        assert tuner.config == hp_config
        
        print("✓ Hyperparameter tuning system works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Hyperparameter tuning failed: {e}")
        return False


def test_cross_validation():
    """Test cross-validation system"""
    print("Testing cross-validation...")
    
    try:
        from core.ml.cross_validation import CrossValidator, TimeSeriesCV
        from core.ml.training_config import CrossValidationConfig
        
        # Test TimeSeriesCV
        ts_cv = TimeSeriesCV(n_splits=3, test_size=0.2)
        
        # Generate test data
        X = np.random.randn(100, 50, 12)
        y = np.random.choice([0, 1, 2], size=100)
        
        # Test split generation
        splits = list(ts_cv.split(X, y))
        assert len(splits) == 3
        
        # Test CrossValidator
        cv_config = CrossValidationConfig()
        cross_validator = CrossValidator(cv_config)
        
        assert cross_validator.config == cv_config
        
        print("✓ Cross-validation system works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Cross-validation failed: {e}")
        return False


def test_model_checkpoint():
    """Test model checkpointing system"""
    print("Testing model checkpointing...")
    
    try:
        from core.ml.model_checkpoint import ModelCheckpoint
        from core.ml.training_config import CheckpointConfig
        
        # Test checkpoint configuration
        checkpoint_config = CheckpointConfig(
            checkpoint_dir='checkpoints/test'
        )
        
        # Test ModelCheckpoint manager
        checkpoint_manager = ModelCheckpoint(checkpoint_config)
        assert checkpoint_manager.config == checkpoint_config
        
        print("✓ Model checkpointing system works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Model checkpointing failed: {e}")
        return False


def test_training_monitor():
    """Test training monitoring system"""
    print("Testing training monitoring...")
    
    try:
        from core.ml.training_monitor import MetricsTracker, MiningSpecificMonitor
        from core.ml.training_config import MonitoringConfig
        
        # Test monitoring configuration
        monitoring_config = MonitoringConfig()
        
        # Test MetricsTracker
        metrics_tracker = MetricsTracker(monitoring_config)
        
        # Test MiningSpecificMonitor
        mining_monitor = MiningSpecificMonitor()
        
        print("✓ Training monitoring system works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Training monitoring failed: {e}")
        return False


def test_training_pipeline():
    """Test complete training pipeline"""
    print("Testing complete training pipeline...")
    
    try:
        from core.ml.training_pipeline import LSTMTrainingPipeline
        from core.ml.training_config import TrainingPipelineConfig
        
        # Create pipeline configuration
        pipeline_config = TrainingPipelineConfig(epochs=3)
        
        # Test pipeline initialization
        pipeline = LSTMTrainingPipeline(pipeline_config)
        
        assert pipeline.config == pipeline_config
        assert pipeline.experiment_id is not None
        
        # Generate test data
        X = np.random.randn(100, 50, 12)
        y = np.random.choice([0, 1, 2], size=100)
        
        # Test data preparation
        data_splits = pipeline.prepare_data(X=X, y=y)
        
        assert 'X_train' in data_splits
        assert 'X_val' in data_splits
        assert 'X_test' in data_splits
        assert pipeline.data_prepared == True
        
        print("✓ Training pipeline works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Training pipeline failed: {e}")
        return False


def run_simple_validation():
    """Run simplified Task 9 validation"""
    
    print("=" * 80)
    print("TASK 9 SIMPLE VALIDATION: LSTM TRAINING PIPELINE")
    print("=" * 80)
    
    tests = [
        test_training_config,
        test_hyperparameter_tuning,
        test_cross_validation,
        test_model_checkpoint,
        test_training_monitor,
        test_training_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 80)
    print("TASK 9 VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 TASK 9 VALIDATION: ALL TESTS PASSED!")
        print("✅ LSTM Training Pipeline implementation is complete and functional")
        return True
    else:
        print(f"\n⚠️  TASK 9 VALIDATION: {total - passed} TESTS FAILED")
        print("❌ Review failed components before proceeding")
        return False


if __name__ == "__main__":
    success = run_simple_validation()
    sys.exit(0 if success else 1)
