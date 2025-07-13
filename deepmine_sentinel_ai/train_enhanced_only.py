#!/usr/bin/env python3
"""
Train only the Enhanced Dual-Branch Stability Predictor
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import models and utilities
from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor

def train_enhanced_model():
    """Train only the Enhanced Dual-Branch model."""
    print("🎯 TRAINING Enhanced Dual-Branch Stability Predictor")
    print("="*60)
    
    # Dataset paths
    static_features_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
    timeseries_path = os.path.join(project_root, 'data', 'stope_timeseries_data_aligned.csv')
    
    # Check if datasets exist
    print(f"📁 Checking for datasets:")
    print(f"   Static features: {static_features_path}")
    print(f"   Exists: {os.path.exists(static_features_path)}")
    print(f"   Timeseries data: {timeseries_path}")
    print(f"   Exists: {os.path.exists(timeseries_path)}")
    
    if not os.path.exists(static_features_path) or not os.path.exists(timeseries_path):
        print(f"❌ Required datasets not found!")
        return None, None
    
    try:
        # Initialize model
        print("🧠 Initializing enhanced dual-branch model...")
        model = EnhancedDualBranchStabilityPredictor(
            static_features_path=static_features_path,
            timeseries_path=timeseries_path
        )
        
        # Train model
        print("🏋️ Training enhanced dual-branch model...")
        history = model.train_enhanced_model(
            epochs=5,  # Reduced for testing
            batch_size=16,
            validation_split=0.2
        )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f'enhanced_dual_branch_{timestamp}.keras')
        model.save_model(model_path)
        print(f"💾 Model saved: {model_path}")
        
        print("✅ Enhanced Dual-Branch model training completed!")
        return model, history
        
    except Exception as e:
        print(f"❌ Error training Enhanced model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    train_enhanced_model()
