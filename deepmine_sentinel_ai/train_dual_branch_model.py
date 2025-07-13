#!/usr/bin/env python3
"""
Training Script for Dual-Branch Stope Stability Predictor
=========================================================

This script trains the dual-branch neural network model using:
- Static stope features + profile summary (Dense branch)
- Timeseries monitoring data (LSTM branch)

Author: Deepmine Sentinel AI Team
Date: July 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from core.ml.models.dual_branch_stability_predictor import DualBranchStopeStabilityPredictor

def main():
    print("🚀 DEEPMINE SENTINEL AI - DUAL BRANCH MODEL TRAINING")
    print("=" * 60)
    
    # Dataset paths
    static_features_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
    timeseries_path = os.path.join(project_root, 'data', 'stope_timeseries_data_aligned.csv')
    
    # Check if datasets exist
    if not os.path.exists(static_features_path):
        print(f"❌ Static features dataset not found: {static_features_path}")
        return
        
    if not os.path.exists(timeseries_path):
        print(f"❌ Timeseries dataset not found: {timeseries_path}")
        return
    
    print(f"📁 Loading datasets...")
    print(f"   Static features: {static_features_path}")
    print(f"   Timeseries data: {timeseries_path}")
    
    # Initialize model
    predictor = DualBranchStopeStabilityPredictor(
        static_features_path=static_features_path,
        timeseries_path=timeseries_path
    )
    
    # Train model
    print(f"\n🏋️ Training dual-branch model...")
    try:
        history = predictor.train(
            epochs=50,
            batch_size=16,
            validation_split=0.2
        )
        
        print(f"\n✅ Training completed successfully!")
        
        # Save model
        model_dir = os.path.join(project_root, 'core', 'ml', 'models', 'saved')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'dual_branch_stability_model.h5')
        
        predictor.save_model(model_path)
        
        # Test predictions on a few stopes
        print(f"\n🧪 Testing predictions...")
        test_stopes = predictor.static_df['stope_name'].head(5).tolist()
        
        for stope in test_stopes:
            try:
                result = predictor.predict_stability(stope)
                if result:
                    print(f"   {stope}: {result['risk_level']} risk "
                          f"(prob: {result['instability_probability']:.3f})")
                else:
                    print(f"   {stope}: Prediction failed")
            except Exception as e:
                print(f"   {stope}: Error - {e}")
        
        # Plot training history if available
        if history:
            plot_training_history(history)
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def plot_training_history(history):
    """Plot training history."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # Precision
        ax3.plot(history.history['precision'], label='Training Precision')
        ax3.plot(history.history['val_precision'], label='Validation Precision')
        ax3.set_title('Model Precision')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        
        # Recall
        ax4.plot(history.history['recall'], label='Training Recall')
        ax4.plot(history.history['val_recall'], label='Validation Recall')
        ax4.set_title('Model Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(__file__), 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ Could not create training plot: {e}")

if __name__ == "__main__":
    main()
