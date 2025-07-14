"""
Test script to verify new TimeSeriesData models work correctly.
Creates sample configurations and validates functionality.
"""

import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepmine_sentinel_ai.settings')
django.setup()

from core.models import (
    Stope, TimeSeriesData, FeatureEngineeringConfig, DataQualityMetrics
)
from django.utils import timezone
from datetime import timedelta
import json

def test_time_series_models():
    """Test the new time series models"""
    
    print("🧪 Testing TimeSeriesData Models")
    print("=" * 50)
    
    # Test 1: Create FeatureEngineeringConfig
    print("1. Creating FeatureEngineeringConfig...")
    try:
        config = FeatureEngineeringConfig.objects.create(
            config_name="test_config",
            version="1.0",
            description="Test configuration for LSTM models",
            enabled_sensor_types=['vibration', 'deformation', 'stress'],
            enabled_feature_types=['raw', 'statistical'],
            window_sizes=[1, 4, 12],
            aggregation_functions=['mean', 'std', 'max'],
            include_event_features=True,
            normalization_method='zscore',
            outlier_detection_method='iqr',
            outlier_threshold=3.0
        )
        print(f"   ✅ Created: {config}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Get or create a test stope
    print("2. Getting test stope...")
    try:
        stope, created = Stope.objects.get_or_create(
            stope_name="TEST_STOPE_TS",
            defaults={
                'rqd': 75.0,
                'rock_type': 'granite',
                'depth': 100.0,
                'hr': 5.0,
                'dip': 45.0,
                'direction': 'N',
                'undercut_width': 3.0,
                'mining_method': 'open_stope'
            }
        )
        print(f"   ✅ Stope: {stope} ({'created' if created else 'existing'})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: Create TimeSeriesData
    print("3. Creating TimeSeriesData...")
    try:
        # Sample data for testing
        sample_features = [
            [1.0, 2.0, 3.0],  # timestep 1
            [1.1, 2.1, 3.1],  # timestep 2
            [1.2, 2.2, 3.2],  # timestep 3
        ]
        sample_normalized = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ]
        sample_targets = [0.1, 0.2, 0.3]
        sample_risk_levels = ['stable', 'stable', 'elevated']
        
        ts_data = TimeSeriesData.objects.create(
            stope=stope,
            sequence_id="test_sequence_001",
            sequence_type='training',
            start_timestamp=timezone.now() - timedelta(hours=3),
            end_timestamp=timezone.now(),
            sequence_length=3,
            feature_set='basic',
            feature_count=3,
            raw_features=sample_features,
            normalized_features=sample_normalized,
            feature_names=['vibration_mean', 'deformation_std', 'stress_max'],
            impact_score_sequence=sample_targets,
            risk_level_sequence=sample_risk_levels,
            data_quality_score=0.95,
            missing_data_percentage=5.0,
            anomaly_count=0
        )
        print(f"   ✅ Created: {ts_data}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 4: Validate the sequence
    print("4. Validating sequence...")
    try:
        is_valid = ts_data.validate_sequence()
        print(f"   ✅ Validation result: {is_valid}")
        if not is_valid:
            print(f"   ⚠️  Validation errors: {ts_data.validation_errors}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 5: Create DataQualityMetrics
    print("5. Creating DataQualityMetrics...")
    try:
        quality_metrics = DataQualityMetrics(
            time_series_data=ts_data,
            completeness_score=0.95,
            consistency_score=0.90,
            validity_score=1.0,
            temporal_resolution_score=0.98,
            outlier_count=0,
            outlier_percentage=0.0,
            invalid_readings_count=0,
            timestamp_irregularities=0,
            overall_quality_score=0.0,  # Will be calculated
            quality_grade='F',  # Will be calculated
            analysis_version='1.0'
        )
        quality_metrics.calculate_overall_quality()
        quality_metrics.save()
        print(f"   ✅ Created: {quality_metrics}")
        print(f"   📊 Quality grade: {quality_metrics.quality_grade} ({quality_metrics.overall_quality_score:.3f})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 6: Test model methods
    print("6. Testing model methods...")
    try:
        feature_matrix = ts_data.get_feature_matrix()
        target_vector = ts_data.get_target_vector()
        print(f"   ✅ Feature matrix shape: {feature_matrix.shape}")
        print(f"   ✅ Target vector shape: {target_vector.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n🎉 All tests passed successfully!")
    print("\n📊 Summary:")
    print(f"   - FeatureEngineeringConfig count: {FeatureEngineeringConfig.objects.count()}")
    print(f"   - TimeSeriesData count: {TimeSeriesData.objects.count()}")
    print(f"   - DataQualityMetrics count: {DataQualityMetrics.objects.count()}")
    
    return True

if __name__ == "__main__":
    test_time_series_models()
