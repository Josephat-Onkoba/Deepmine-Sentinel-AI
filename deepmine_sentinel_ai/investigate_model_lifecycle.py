#!/usr/bin/env python3
"""
Model Lifecycle Investigation Script
===================================

This script investigates the exact model saving and loading workflow
to document where the trained model is saved and how it's retrieved.
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

def investigate_model_lifecycle():
    """Investigate where models are saved and loaded from."""
    
    print("\n🔍 INVESTIGATING MODEL LIFECYCLE")
    print("=" * 50)
    
    # Check current directories
    base_dir = Path('/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai')
    models_dir = base_dir / 'models'
    
    print(f"\n📁 Directory Investigation:")
    print(f"   Base directory: {base_dir}")
    print(f"   Models directory: {models_dir}")
    print(f"   Models directory exists: {models_dir.exists()}")
    
    if models_dir.exists():
        model_files = list(models_dir.glob('*'))
        print(f"   Files in models directory: {len(model_files)}")
        for file in model_files:
            print(f"     - {file.name} ({file.stat().st_size} bytes)")
    
    # Check the training process
    print(f"\n🎯 Training Process Investigation:")
    
    try:
        from core.ml_service import MLPredictionService
        
        service = MLPredictionService()
        
        # Check initialization
        print(f"   Service data directory: {service.data_dir}")
        print(f"   Service models directory: {service.models_dir}")
        print(f"   Can initialize model: {service._can_create_enhanced_model_data()}")
        
        # Check if model is ready for training
        if service.is_model_trained():
            print("   ✅ Model is ready")
        else:
            print("   ⚠️ Model is not ready")
            
        # Try to load model
        load_success = service._load_model()
        print(f"   Model load successful: {load_success}")
        
        if load_success and service.predictor:
            print(f"   Predictor type: {type(service.predictor).__name__}")
            print(f"   Predictor static path: {getattr(service.predictor, 'static_features_path', 'N/A')}")
            print(f"   Predictor timeseries path: {getattr(service.predictor, 'timeseries_path', 'N/A')}")
            
            # Check if there's a trained model
            has_model = hasattr(service.predictor, 'combined_model') and service.predictor.combined_model is not None
            print(f"   Has trained model: {has_model}")
            
            if not has_model:
                print("\n🔧 Attempting to understand training process...")
                
                # Let's see what happens during training without actually training
                try:
                    print("   Checking training data preparation...")
                    # This will show us what data is being used
                    data = service.predictor.prepare_enhanced_training_data()
                    print(f"   ✅ Training data prepared successfully")
                    print(f"      Static features shape: {data['X_static'].shape}")
                    print(f"      Timeseries shape: {data['X_timeseries'].shape}")
                    print(f"      Current labels shape: {data['y_current'].shape}")
                    print(f"      Future labels shape: {data['y_future'].shape}")
                    
                except Exception as e:
                    print(f"   ❌ Training data preparation failed: {e}")
        
    except ImportError as e:
        print(f"   ❌ Cannot import ML service: {e}")
    except Exception as e:
        print(f"   ❌ Error investigating training process: {e}")
    
    # Check the model saving logic
    print(f"\n💾 Model Saving Investigation:")
    
    try:
        # Look at the dual branch predictor directly
        from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor
        
        print("   ✅ Can import EnhancedDualBranchStabilityPredictor")
        
        # Check what paths would be used
        static_path = base_dir / 'data' / 'stope_static_features_aligned.csv'
        timeseries_path = base_dir / 'data' / 'stope_timeseries_data_aligned.csv'
        
        print(f"   Expected static CSV: {static_path} (exists: {static_path.exists()})")
        print(f"   Expected timeseries CSV: {timeseries_path} (exists: {timeseries_path.exists()})")
        
        # Check training callback paths in the model code
        print(f"\n   📝 Model Checkpoint Analysis:")
        print(f"   Based on code analysis:")
        print(f"     - ModelCheckpoint saves to: 'models/best_enhanced_model.keras'")
        print(f"     - Training log saves to: 'models/training_log.csv'")
        print(f"     - Manual save uses save_enhanced_model() method")
        print(f"     - Components saved to: '<model_path>_components.pkl'")
        
        # Check what directory this would be relative to
        current_dir = os.getcwd()
        print(f"   Current working directory: {current_dir}")
        
        relative_models_dir = Path(current_dir) / 'models'
        print(f"   Relative models directory: {relative_models_dir} (exists: {relative_models_dir.exists()})")
        
    except ImportError as e:
        print(f"   ❌ Cannot import model: {e}")
    except Exception as e:
        print(f"   ❌ Error investigating model saving: {e}")
    
    print(f"\n📋 SUMMARY:")
    print("=" * 30)
    print("🔍 MODEL LIFECYCLE FINDINGS:")
    print("1. Training happens via MLPredictionService.train_model_with_current_data()")
    print("2. During training, ModelCheckpoint callback saves to 'models/best_enhanced_model.keras'")
    print("3. Training log is saved to 'models/training_log.csv'") 
    print("4. Model can be manually saved using predictor.save_enhanced_model(filepath)")
    print("5. Components (scalers, encoders) are saved to '<model_path>_components.pkl'")
    print("6. The 'models' directory is relative to the current working directory during training")
    print("7. Loading happens via predictor.load_enhanced_model(filepath)")
    print("\n⚠️ CURRENT ISSUE:")
    print("- No automatic saving after training in ml_service.py")
    print("- Need to ensure models directory path is consistent")
    print("- Need to add explicit model saving after successful training")

if __name__ == "__main__":
    investigate_model_lifecycle()
