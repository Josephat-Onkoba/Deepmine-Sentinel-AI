#!/usr/bin/env python3
"""
Simple ML Service Test
======================

Basic test to validate the enhanced ML service integration.
"""

import os
import sys
import django

# Setup Django environment
sys.path.append('/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepmine_sentinel_ai.settings')
django.setup()

def test_basic_ml_service():
    """Test basic ML service functionality."""
    print("🚀 Testing Enhanced ML Service")
    print("=" * 50)
    
    try:
        from core.ml_service import MLPredictionService
        print("✅ Import successful")
        
        # Initialize service
        service = MLPredictionService()
        print("✅ Service initialization successful")
        
        # Test health check
        health = service.validate_model_health()
        print(f"📊 Health Status: {health['overall_health']}")
        
        # Show components
        for component, status in health['components'].items():
            print(f"  • {component}: {status}")
        
        # Show data stats if available
        if 'data_stats' in health:
            stats = health['data_stats']
            print(f"📈 Data Statistics:")
            print(f"  • Stopes: {stats['total_stopes']}")
            print(f"  • Time series points: {stats['total_timeseries_points']}")
            print(f"  • Stopes with data: {stats['stopes_with_timeseries']}")
        
        # Show recommendations
        if health.get('recommendations'):
            print("💡 Recommendations:")
            for rec in health['recommendations']:
                print(f"  • {rec}")
        
        # Test model readiness
        ready = service.is_model_trained()
        print(f"🎯 Model ready: {ready}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_django_models():
    """Test Django models access."""
    print("\n🗃️ Testing Django Models")
    print("=" * 50)
    
    try:
        from core.models import Stope, TimeSeriesData
        
        stope_count = Stope.objects.count()
        ts_count = TimeSeriesData.objects.count()
        
        print(f"✅ Database access successful")
        print(f"📊 Found {stope_count} stopes")
        print(f"📈 Found {ts_count} time series data points")
        
        if stope_count > 0:
            sample_stope = Stope.objects.first()
            print(f"📝 Sample stope: {sample_stope.stope_name}")
            ts_for_stope = sample_stope.timeseries_data.count()
            print(f"📈 Time series for sample: {ts_for_stope}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Simple Enhanced ML Service Test")
    print("=" * 60)
    
    test1_result = test_django_models()
    test2_result = test_basic_ml_service()
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Django Models: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"ML Service: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    if test1_result and test2_result:
        print("🎉 All basic tests passed!")
    else:
        print("⚠️ Some tests failed.")
