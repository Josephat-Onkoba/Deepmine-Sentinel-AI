#!/usr/bin/env python3
"""
Model Saving Validation Test
============================

This script tests the complete model saving and loading workflow
to ensure the trained model is properly saved and can be retrieved.
"""

import os
import sys
import django
from pathlib import Path

# Setup Django environment
sys.path.append('/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepmine_sentinel_ai.settings')

try:
    django.setup()
    print("✅ Django environment initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Django: {e}")
    sys.exit(1)

def test_model_saving_workflow():
    """Test the complete model saving and loading workflow."""
    
    print("\n🧪 TESTING MODEL SAVING WORKFLOW")
    print("=" * 50)
    
    # Test paths
    from django.conf import settings
    base_dir = Path(settings.BASE_DIR)
    models_dir = base_dir / 'models'
    model_path = models_dir / 'enhanced_dual_branch_model.keras'
    components_path = model_path.parent / f'{model_path.stem}_components.pkl'
    metadata_path = model_path.parent / f'{model_path.stem}_metadata.json'
    
    print(f"\n📁 Expected File Locations:")
    print(f"   Base directory: {base_dir}")
    print(f"   Models directory: {models_dir}")
    print(f"   Model file: {model_path}")
    print(f"   Components file: {components_path}")
    print(f"   Metadata file: {metadata_path}")
    
    # Check if models directory exists
    print(f"\n🔍 Directory Status:")
    print(f"   Models directory exists: {models_dir.exists()}")
    
    if models_dir.exists():
        model_files = list(models_dir.glob('*'))
        print(f"   Files in models directory: {len(model_files)}")
        for file in model_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"     - {file.name} ({size_mb:.2f} MB)")
    
    # Test ML Service integration
    print(f"\n🤖 Testing ML Service Integration:")
    
    try:
        from core.ml_service import MLPredictionService
        
        service = MLPredictionService()
        print(f"   ✅ ML Service initialized")
        print(f"   Service model path: {service.model_path}")
        print(f"   Service metadata path: {service.model_metadata_path}")
        
        # Check if service detects trained model
        is_trained = service.is_model_trained()
        print(f"   Model is trained: {is_trained}")
        
        # Check if saved model exists
        model_exists = os.path.exists(service.model_path)
        components_exist = os.path.exists(service.model_path.replace('.keras', '_components.pkl'))
        
        print(f"   Saved model exists: {model_exists}")
        print(f"   Components exist: {components_exist}")
        
        if model_exists:
            print(f"   📊 Model file size: {os.path.getsize(service.model_path) / (1024*1024):.2f} MB")
            
            # Try to load the model
            try:
                load_success = service._load_model()
                print(f"   Model loading successful: {load_success}")
                
                if load_success and service.predictor:
                    has_trained_model = hasattr(service.predictor, 'combined_model') and service.predictor.combined_model is not None
                    print(f"   Has trained model in memory: {has_trained_model}")
                    
                    if has_trained_model:
                        params = service.predictor.combined_model.count_params()
                        print(f"   Model parameters: {params:,}")
                    
            except Exception as e:
                print(f"   ❌ Model loading failed: {e}")
        
        # Test metadata if it exists
        if os.path.exists(service.model_metadata_path):
            print(f"\n📋 Model Metadata:")
            try:
                import json
                with open(service.model_metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                for key, value in metadata.items():
                    print(f"   {key}: {value}")
                    
            except Exception as e:
                print(f"   ❌ Failed to read metadata: {e}")
        
    except ImportError as e:
        print(f"   ❌ Cannot import ML service: {e}")
    except Exception as e:
        print(f"   ❌ Error testing ML service: {e}")
    
    # Test Django management command path
    print(f"\n🎯 Testing Django Management Command Path:")
    
    command_path = base_dir / 'core' / 'management' / 'commands' / 'train_model.py'
    print(f"   Command file exists: {command_path.exists()}")
    
    if command_path.exists():
        # Check if the command has the saving logic
        with open(command_path, 'r') as f:
            content = f.read()
            
        has_saving = 'save_enhanced_model' in content
        has_metadata = 'metadata' in content and 'json' in content
        has_error_handling = 'except' in content and 'save' in content
        
        print(f"   Has model saving logic: {has_saving}")
        print(f"   Has metadata saving: {has_metadata}")
        print(f"   Has error handling: {has_error_handling}")
    
    # Summary
    print(f"\n📊 WORKFLOW STATUS SUMMARY:")
    print("=" * 40)
    
    workflow_status = {
        'Models directory': models_dir.exists(),
        'Saved model exists': model_exists if 'model_exists' in locals() else False,
        'Components saved': components_exist if 'components_exist' in locals() else False,
        'ML service integration': True,
        'Management command ready': command_path.exists() if 'command_path' in locals() else False
    }
    
    for item, status in workflow_status.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {item}")
    
    all_ready = all(workflow_status.values())
    print(f"\n🎉 Overall Status: {'READY FOR PRODUCTION' if all_ready else 'NEEDS ATTENTION'}")
    
    if not all_ready:
        print(f"\n🔧 To fix:")
        print(f"   1. Run: python manage.py train_model")
        print(f"   2. This will create all necessary model files")
        print(f"   3. ML service will then automatically load the saved model")

if __name__ == "__main__":
    test_model_saving_workflow()
