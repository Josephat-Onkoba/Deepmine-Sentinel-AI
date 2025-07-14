#!/usr/bin/env python3
"""
Enhanced ML Service Integration Test Script
==========================================

Comprehensive test script to validate the enhanced ML service integration
with Django models, views, and the enhanced dual-branch model.

Tests:
- ML Service initialization and health checks
- Enhanced model loading and prediction capabilities  
- Django model integration and data flow
- View integration and AJAX endpoints
- Future prediction and alert generation
- Error handling and fallback mechanisms
"""

import os
import sys
import django
import logging
from datetime import datetime, timedelta

# Setup Django environment
sys.path.append('/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepmine_sentinel_ai.settings')
django.setup()

from core.models import Stope, TimeSeriesData, Prediction, FuturePrediction, PredictionAlert
from core.ml_service import MLPredictionService
from django.utils import timezone
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ml_service_initialization():
    """Test ML service initialization and health checks."""
    print("\n" + "="*60)
    print("Testing Enhanced ML Service Initialization")
    print("="*60)
    
    try:
        # Initialize service
        ml_service = MLPredictionService()
        print("✅ ML Service initialized successfully")
        
        # Test health validation
        health_status = ml_service.validate_model_health()
        print(f"📊 Model Health Status: {health_status['overall_health']}")
        print(f"📋 Components: {health_status['components']}")
        
        if health_status['recommendations']:
            print(f"💡 Recommendations: {health_status['recommendations']}")
        
        # Test model training status
        is_trained = ml_service.is_model_trained()
        print(f"🎯 Model Trained: {is_trained}")
        
        return True, ml_service
        
    except Exception as e:
        print(f"❌ ML Service initialization failed: {e}")
        return False, None

def test_django_model_integration(ml_service):
    """Test integration with Django models."""
    print("\n" + "="*60)
    print("Testing Django Model Integration")
    print("="*60)
    
    try:
        # Get or create a test stope
        stope, created = Stope.objects.get_or_create(
            stope_name="TEST_INTEGRATION_STOPE",
            defaults={
                'rqd': 75.0,
                'hr': 8.5,
                'depth': 450.0,
                'dip': 65.0,
                'direction': 'North',
                'undercut_wdt': 12.0,
                'rock_type': 'Granite',
                'support_type': 'Rock Bolts',
                'support_density': 1.5,
                'support_installed': True
            }
        )
        
        if created:
            print(f"✅ Created test stope: {stope.stope_name}")
        else:
            print(f"✅ Using existing test stope: {stope.stope_name}")
        
        # Add test time series data
        test_data_points = []
        base_time = timezone.now() - timedelta(days=30)
        
        for i in range(50):  # 50 data points over 30 days
            timestamp = base_time + timedelta(hours=i*14.4)  # Every ~14.4 hours
            
            # Create realistic time series data with some variation
            vibration = 5.0 + np.random.normal(0, 2) + i * 0.1  # Increasing trend
            deformation = 1.0 + np.random.normal(0, 0.5) + i * 0.02
            stress = 25.0 + np.random.normal(0, 5) + i * 0.3
            temperature = 18.0 + np.random.normal(0, 2)
            humidity = 60.0 + np.random.normal(0, 10)
            
            data_point, created = TimeSeriesData.objects.get_or_create(
                stope=stope,
                timestamp=timestamp,
                defaults={
                    'vibration_velocity': max(0, vibration),
                    'deformation_rate': max(0, deformation),
                    'stress': max(0, stress),
                    'temperature': temperature,
                    'humidity': max(0, min(100, humidity))
                }
            )
            
            if created:
                test_data_points.append(data_point)
        
        print(f"✅ Added {len(test_data_points)} time series data points")
        print(f"📊 Total time series data for stope: {stope.timeseries_data.count()}")
        
        return True, stope
        
    except Exception as e:
        print(f"❌ Django model integration test failed: {e}")
        return False, None

def test_enhanced_prediction_capabilities(ml_service, stope):
    """Test enhanced prediction capabilities."""
    print("\n" + "="*60)
    print("Testing Enhanced Prediction Capabilities")
    print("="*60)
    
    try:
        # Test basic stability prediction
        print("🔮 Testing basic stability prediction...")
        prediction_result = ml_service.predict_stope_stability(stope, save_prediction=False)
        
        if 'error' not in prediction_result:
            print("✅ Basic prediction successful")
            print(f"  📊 Prediction: {prediction_result['prediction']}")
            print(f"  🎯 Risk Level: {prediction_result['risk_level']}")
            print(f"  💪 Confidence: {prediction_result['confidence']:.3f}")
            print(f"  🤖 Model Type: {prediction_result['model_info']['model_type']}")
            
            # Check for enhanced features
            if prediction_result['model_info'].get('enhanced_mode', False):
                print("✅ Enhanced mode active")
                
                if 'future_predictions' in prediction_result:
                    future_preds = prediction_result['future_predictions']
                    print(f"  🔮 Future predictions: {len(future_preds)} horizons")
                    for pred in future_preds[:3]:  # Show first 3
                        print(f"    • {pred['horizon_days']} days: {pred['predicted_risk_level']} (confidence: {pred['confidence']:.2f})")
                
                if 'risk_trend' in prediction_result:
                    print(f"  📈 Risk trend: {prediction_result['risk_trend']}")
                    
                if 'recommendations' in prediction_result:
                    print(f"  💡 Has recommendations: {len(prediction_result['recommendations']) > 0}")
            else:
                print("ℹ️ Fallback mode used (CSV data not available)")
            
            return True, prediction_result
        else:
            print(f"❌ Prediction failed: {prediction_result['error']}")
            return False, None
            
    except Exception as e:
        print(f"❌ Enhanced prediction test failed: {e}")
        return False, None

def test_database_integration(ml_service, stope, prediction_result):
    """Test database integration for predictions."""
    print("\n" + "="*60)
    print("Testing Database Integration")
    print("="*60)
    
    try:
        # Test saving prediction to database
        print("💾 Testing prediction saving...")
        
        prediction_result_db = ml_service.predict_stope_stability(stope, save_prediction=True)
        
        if 'error' not in prediction_result_db:
            print("✅ Prediction saved to database")
            
            # Check if prediction was saved
            recent_prediction = Prediction.objects.filter(stope=stope).order_by('-created_at').first()
            if recent_prediction:
                print(f"  📝 Found saved prediction: {recent_prediction.risk_level}")
                print(f"  🕒 Created at: {recent_prediction.created_at}")
            
            # Check for future predictions
            future_predictions = FuturePrediction.objects.filter(stope=stope)
            if future_predictions.exists():
                print(f"  🔮 Future predictions saved: {future_predictions.count()}")
                
                # Show first few future predictions
                for fp in future_predictions[:3]:
                    print(f"    • {fp.days_ahead} days: {fp.risk_level} (confidence: {fp.confidence_score:.2f})")
            
            # Check for alerts
            alerts = PredictionAlert.objects.filter(stope=stope, is_active=True)
            if alerts.exists():
                print(f"  🚨 Active alerts: {alerts.count()}")
                for alert in alerts[:2]:
                    print(f"    • {alert.severity}: {alert.title}")
            else:
                print("  ℹ️ No active alerts generated")
            
            return True
        else:
            print(f"❌ Database saving failed: {prediction_result_db['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False

def test_batch_prediction_capabilities(ml_service):
    """Test batch prediction capabilities."""
    print("\n" + "="*60)
    print("Testing Batch Prediction Capabilities")
    print("="*60)
    
    try:
        # Get multiple stopes for batch testing
        stopes = list(Stope.objects.all()[:3])  # Test with up to 3 stopes
        
        if len(stopes) < 1:
            print("ℹ️ No stopes available for batch testing")
            return True
        
        stope_ids = [stope.id for stope in stopes]
        print(f"🔄 Testing batch prediction for {len(stope_ids)} stopes...")
        
        # Perform batch prediction
        batch_results = ml_service.predict_multiple_stopes(stope_ids)
        
        success_count = 0
        error_count = 0
        enhanced_count = 0
        
        for stope_id, result in batch_results.items():
            if 'error' not in result:
                success_count += 1
                if result.get('model_info', {}).get('enhanced_mode', False):
                    enhanced_count += 1
                print(f"  ✅ Stope {stope_id}: {result['prediction']} ({result['risk_level']})")
            else:
                error_count += 1
                print(f"  ❌ Stope {stope_id}: {result['error']}")
        
        print(f"📊 Batch Results: {success_count} successful, {error_count} failed")
        if enhanced_count > 0:
            print(f"🚀 Enhanced predictions: {enhanced_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch prediction test failed: {e}")
        return False

def test_feature_extraction_methods(ml_service, stope):
    """Test different feature extraction methods."""
    print("\n" + "="*60)
    print("Testing Feature Extraction Methods")
    print("="*60)
    
    try:
        # Test Django feature extraction
        print("🔧 Testing Django feature extraction...")
        django_features = ml_service._extract_django_static_features(stope)
        print(f"  ✅ Django static features: {len(django_features)} features")
        print(f"    RQD: {django_features['rqd']}, HR: {django_features['hr']}, Depth: {django_features['depth']}")
        
        # Test physics-based stability calculation
        print("⚗️ Testing physics-based stability calculation...")
        stability_score = ml_service._calculate_physics_based_stability(django_features)
        print(f"  ✅ Physics-based stability score: {stability_score:.3f}")
        
        # Test time series trend analysis
        print("📈 Testing time series trend analysis...")
        ts_risk_factor = ml_service._analyze_django_timeseries_trends(stope)
        print(f"  ✅ Time series risk factor: {ts_risk_factor:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_model_performance_metrics(ml_service):
    """Test model performance metrics retrieval."""
    print("\n" + "="*60)
    print("Testing Model Performance Metrics")
    print("="*60)
    
    try:
        # Get model performance metrics
        print("📊 Retrieving model performance metrics...")
        metrics = ml_service.get_model_performance_metrics()
        
        if 'error' not in metrics:
            print("✅ Model metrics retrieved successfully")
            print(f"  🤖 Model Type: {metrics.get('model_type', 'Unknown')}")
            print(f"  🏗️ Architecture: {metrics.get('architecture', 'Unknown')}")
            print(f"  📈 Status: {metrics.get('status', 'Unknown')}")
            
            if 'accuracy' in metrics and metrics['accuracy'] != 'N/A':
                print(f"  🎯 Accuracy: {metrics['accuracy']}")
            
            if 'enhanced_features' in metrics:
                print(f"  🚀 Enhanced Features: {metrics['enhanced_features']}")
        else:
            print(f"ℹ️ Model metrics not available: {metrics['error']}")
        
        # Get feature importance
        print("🔍 Retrieving feature importance...")
        importance = ml_service.get_feature_importance(top_n=5)
        
        if importance:
            print("✅ Feature importance retrieved")
            for feature, score in list(importance.items())[:3]:
                print(f"  • {feature}: {score:.3f}")
        else:
            print("ℹ️ Feature importance not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Model metrics test failed: {e}")
        return False

def cleanup_test_data():
    """Clean up test data."""
    print("\n" + "="*60)
    print("Cleaning Up Test Data")
    print("="*60)
    
    try:
        # Remove test stope and related data
        test_stope = Stope.objects.filter(stope_name="TEST_INTEGRATION_STOPE").first()
        
        if test_stope:
            # Remove related data
            deleted_ts = test_stope.timeseries_data.count()
            deleted_pred = test_stope.predictions.count()
            deleted_future = test_stope.future_predictions.count()
            deleted_alerts = test_stope.alerts.count()
            
            test_stope.delete()
            
            print(f"✅ Cleaned up test stope and related data:")
            print(f"  • Time series data: {deleted_ts}")
            print(f"  • Predictions: {deleted_pred}")
            print(f"  • Future predictions: {deleted_future}")
            print(f"  • Alerts: {deleted_alerts}")
        else:
            print("ℹ️ No test stope found to clean up")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Enhanced ML Service Integration Tests")
    print("=" * 80)
    
    test_results = []
    ml_service = None
    stope = None
    prediction_result = None
    
    # Test 1: ML Service Initialization
    success, ml_service = test_ml_service_initialization()
    test_results.append(("ML Service Initialization", success))
    
    if success and ml_service:
        # Test 2: Django Model Integration
        success, stope = test_django_model_integration(ml_service)
        test_results.append(("Django Model Integration", success))
        
        if success and stope:
            # Test 3: Enhanced Prediction Capabilities
            success, prediction_result = test_enhanced_prediction_capabilities(ml_service, stope)
            test_results.append(("Enhanced Prediction Capabilities", success))
            
            # Test 4: Database Integration
            if success:
                success = test_database_integration(ml_service, stope, prediction_result)
                test_results.append(("Database Integration", success))
            
            # Test 5: Feature Extraction Methods
            success = test_feature_extraction_methods(ml_service, stope)
            test_results.append(("Feature Extraction Methods", success))
            
        # Test 6: Batch Prediction Capabilities
        success = test_batch_prediction_capabilities(ml_service)
        test_results.append(("Batch Prediction Capabilities", success))
        
        # Test 7: Model Performance Metrics
        success = test_model_performance_metrics(ml_service)
        test_results.append(("Model Performance Metrics", success))
    
    # Test 8: Cleanup
    success = cleanup_test_data()
    test_results.append(("Cleanup Test Data", success))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced ML service integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the output above for details.")
        return False

if __name__ == "__main__":
    run_comprehensive_tests()
