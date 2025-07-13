#!/usr/bin/env python3
"""
Comprehensive Training Script for All Deepmine Sentinel AI Models
================================================================

This script trains all stability prediction models in the system:
1. EnhancedDualBranchStabilityPredictor (Main Model)
2. TemporalStabilityPredictor (LSTM-based)
3. Simple Current Stability Model (Fallback)

Features:
- Complete evaluation metrics for each model
- Training plots and visualizations
- Model comparison and performance analysis
- Automated model saving and versioning

Author: Deepmine Sentinel AI Team
Date: July 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import joblib

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import models and utilities
from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor
from core.utils import get_stope_profile_summary

class ModelTrainingOrchestrator:
    """Orchestrates training of all stability prediction models."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(project_root, 'training_results', self.timestamp)
        self.models_dir = os.path.join(project_root, 'models')
        self.plots_dir = os.path.join(project_root, 'plots')
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.training_results = {}
        
    def setup_gpu(self):
        """Configure GPU if available."""
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"✅ GPU available: {physical_devices[0].name}")
            return True
        else:
            print("⚠️  No GPU available, using CPU")
            return False
            
    def train_enhanced_dual_branch_model(self):
        """Train the main enhanced dual-branch stability predictor."""
        print("\n" + "="*80)
        print("🎯 TRAINING MODEL 1: Enhanced Dual-Branch Stability Predictor")
        print("="*80)
        
        # Dataset paths
        static_features_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
        timeseries_path = os.path.join(project_root, 'data', 'stope_timeseries_data_aligned.csv')
        
        # Check if datasets exist
        if not os.path.exists(static_features_path) or not os.path.exists(timeseries_path):
            print(f"❌ Required datasets not found:")
            print(f"   Static features: {static_features_path}")
            print(f"   Timeseries data: {timeseries_path}")
            return None, None
        
        print(f"📁 Using real datasets:")
        print(f"   Static features: {static_features_path}")
        print(f"   Timeseries data: {timeseries_path}")
        
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
                epochs=100,
                batch_size=16,
                validation_split=0.2,
                learning_rate=0.001
            )
            
            # Save model
            model_path = os.path.join(self.models_dir, f'enhanced_dual_branch_{self.timestamp}.keras')
            model.save_model(model_path)
            print(f"💾 Model saved: {model_path}")
            
            # Generate evaluation metrics
            metrics = self._evaluate_enhanced_model(model, history)
            
            # Save training plots
            self._plot_enhanced_training_history(history, "enhanced_dual_branch")
            
            # Test predictions
            test_results = self._test_enhanced_predictions(model)
            
            self.training_results['enhanced_dual_branch'] = {
                'model_path': model_path,
                'metrics': metrics,
                'test_results': test_results,
                'training_time': datetime.now().isoformat()
            }
            
            print("✅ Enhanced dual-branch model training completed!")
            return model, history
            
        except Exception as e:
            print(f"❌ Enhanced dual-branch model training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_temporal_stability_model(self):
        """Train the temporal stability predictor (LSTM-based)."""
        print("\n" + "="*80)
        print("🕒 TRAINING MODEL 2: Temporal Stability Predictor")
        print("="*80)
        
        try:
            # Import the temporal predictor from utils
            from core.utils import TemporalStabilityPredictor
            
            # Initialize temporal predictor
            temporal_predictor = TemporalStabilityPredictor(
                model_path=self.models_dir, 
                model_version=f'temporal_{self.timestamp}'
            )
            
            # Create temporal training data from real timeseries data
            print("📊 Preparing temporal training data from real timeseries...")
            temporal_data = self._create_temporal_training_data_from_real_data()
            
            # Train temporal model
            print("🏋️ Training temporal LSTM model...")
            model, history = self._train_temporal_model_direct(temporal_data, temporal_predictor)
            
            if model and history:
                # Save model
                model_path = os.path.join(self.models_dir, f'temporal_stability_{self.timestamp}.keras')
                model.save(model_path)
                print(f"💾 Temporal model saved: {model_path}")
                
                # Generate evaluation metrics
                metrics = self._evaluate_temporal_model(model, temporal_data, history)
                
                # Save training plots
                self._plot_temporal_training_history(history, "temporal_stability")
                
                self.training_results['temporal_stability'] = {
                    'model_path': model_path,
                    'metrics': metrics,
                    'training_time': datetime.now().isoformat()
                }
                
                print("✅ Temporal stability model training completed!")
                return model, history
            else:
                print("❌ Temporal model training returned None")
                return None, None
                
        except Exception as e:
            print(f"❌ Temporal stability model training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_simple_stability_model(self):
        """Train the simple current stability model."""
        print("\n" + "="*80)
        print("🔷 TRAINING MODEL 3: Simple Current Stability Model")
        print("="*80)
        
        try:
            # Create simple model training data from real static features
            print("📊 Preparing simple model training data from real static features...")
            simple_data = self._create_simple_training_data_from_real_data()
            
            # Train simple model
            print("🏋️ Training simple stability model...")
            model, history = self._train_simple_model_direct(simple_data)
            
            if model and history:
                # Save model
                model_path = os.path.join(self.models_dir, f'simple_stability_{self.timestamp}.keras')
                model.save(model_path)
                print(f"💾 Simple model saved: {model_path}")
                
                # Save scaler
                scaler_path = os.path.join(self.models_dir, f'simple_scaler_{self.timestamp}.pkl')
                joblib.dump(simple_data['scaler'], scaler_path)
                print(f"💾 Scaler saved: {scaler_path}")
                
                # Generate evaluation metrics
                metrics = self._evaluate_simple_model(model, simple_data, history)
                
                # Save training plots
                self._plot_simple_training_history(history, "simple_stability")
                
                self.training_results['simple_stability'] = {
                    'model_path': model_path,
                    'scaler_path': scaler_path,
                    'metrics': metrics,
                    'training_time': datetime.now().isoformat()
                }
                
                print("✅ Simple stability model training completed!")
                return model, history
            else:
                print("❌ Simple model training returned None")
                return None, None
                
        except Exception as e:
            print(f"❌ Simple stability model training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _create_temporal_training_data_from_real_data(self):
        """Create temporal training data from real timeseries data."""
        try:
            # Load real timeseries data
            timeseries_path = os.path.join(project_root, 'data', 'stope_timeseries_data_aligned.csv')
            df = pd.read_csv(timeseries_path)
            
            print(f"📊 Loaded {len(df)} timeseries records from {len(df['stope_name'].unique())} stopes")
            
            # Prepare sequences for each stope
            sequence_length = 30  # Use 30 time steps for prediction
            n_features = 5  # vibration_velocity, deformation_rate, stress, temperature, humidity
            
            sequences = []
            labels = []
            
            for stope_name in df['stope_name'].unique():
                stope_data = df[df['stope_name'] == stope_name].sort_values('timestamp')
                
                if len(stope_data) >= sequence_length:
                    # Extract features
                    features = stope_data[['vibration_velocity', 'deformation_rate', 'stress', 'temperature', 'humidity']].values
                    
                    # Create sliding windows
                    for i in range(len(features) - sequence_length + 1):
                        sequence = features[i:i+sequence_length]
                        sequences.append(sequence)
                        
                        # Create label based on trend in deformation_rate and stress
                        future_window = features[i+sequence_length-10:i+sequence_length]  # Last 10 points
                        deformation_trend = np.polyfit(range(10), future_window[:, 1], 1)[0]  # deformation_rate trend
                        stress_level = np.mean(future_window[:, 2])  # average stress
                        
                        # Label as unstable (1) if increasing deformation trend or high stress
                        label = 1 if (deformation_trend > 0.01 or stress_level > np.percentile(features[:, 2], 70)) else 0
                        labels.append(label)
            
            X = np.array(sequences)
            y = np.array(labels)
            
            print(f"📊 Created {len(X)} temporal sequences with shape {X.shape}")
            print(f"📊 Label distribution: {np.bincount(y)}")
            
            return {'X': X, 'y': y}
            
        except Exception as e:
            print(f"Error creating temporal training data: {e}")
            return self._create_temporal_training_data()  # Fallback to sample data
    
    def _create_simple_training_data_from_real_data(self):
        """Create simple training data from real static features."""
        try:
            # Load real static features data
            static_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
            df = pd.read_csv(static_path)
            
            print(f"📊 Loaded {len(df)} static feature records")
            
            # Prepare features for simple model
            feature_columns = ['rqd', 'hr', 'depth', 'dip', 'undercut_wdt', 'support_density', 'support_installed']
            
            # Handle categorical columns
            # Convert direction to numeric
            direction_map = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 
                           'Northeast': 4, 'Northwest': 5, 'Southeast': 6, 'Southwest': 7}
            df['direction_num'] = df['direction'].map(direction_map).fillna(0)
            
            # Convert rock_type to numeric
            rock_type_map = {'Basalt': 0, 'Schist': 1, 'Granite': 2, 'Quartzite': 3}
            df['rock_type_num'] = df['rock_type'].map(rock_type_map).fillna(0)
            
            # Convert support_type to numeric
            support_type_map = {'Mesh': 0, 'Shotcrete': 1, 'Rock Bolts': 2, 'Cable Bolts': 3}
            df['support_type_num'] = df['support_type'].map(support_type_map).fillna(0)
            
            # Select features for training
            feature_columns_final = feature_columns + ['direction_num', 'rock_type_num', 'support_type_num']
            X = df[feature_columns_final].values
            
            # Create stability labels based on engineering rules
            # Consider a stope unstable if:
            # - Low RQD (rock quality designation) 
            # - High depth 
            # - High stress concentration (inferred from depth and rock type)
            # - Poor support conditions
            
            labels = []
            for i in range(len(df)):
                risk_score = 0
                
                # RQD contribution (lower RQD = higher risk)
                if df.iloc[i]['rqd'] < 30:
                    risk_score += 0.3
                elif df.iloc[i]['rqd'] < 50:
                    risk_score += 0.2
                
                # Depth contribution (higher depth = higher risk)
                if df.iloc[i]['depth'] > 400:
                    risk_score += 0.25
                elif df.iloc[i]['depth'] > 300:
                    risk_score += 0.15
                
                # Support density contribution (lower density = higher risk)
                if df.iloc[i]['support_density'] < 0.3:
                    risk_score += 0.2
                
                # Support installation contribution (not installed = higher risk)
                if df.iloc[i]['support_installed'] == 0:
                    risk_score += 0.25
                
                # Add some randomness to avoid perfect deterministic labels
                risk_score += np.random.normal(0, 0.1)
                
                # Threshold for instability
                labels.append(1 if risk_score > 0.5 else 0)
            
            y = np.array(labels)
            
            print(f"📊 Created simple training data with shape {X.shape}")
            print(f"📊 Label distribution: {np.bincount(y)}")
            
            return {'X': X, 'y': y}
            
        except Exception as e:
            print(f"Error creating simple training data: {e}")
            return self._create_simple_training_data()  # Fallback to sample data
    
    def _create_temporal_training_data(self):
        """Fallback: Create sample temporal training data for LSTM model."""
        print("⚠️  Using fallback sample temporal data")
        n_samples = 1000
        sequence_length = 30
        n_features = 5  # Match real data features
        
        # Generate sequences
        X = np.random.randn(n_samples, sequence_length, n_features)
        
        # Generate labels based on trend in last few time steps
        y = []
        for i in range(n_samples):
            # Check if there's an increasing trend in deformation
            last_values = X[i, -5:, 1]  # Last 5 deformation values
            trend = np.polyfit(range(5), last_values, 1)[0]
            y.append(1 if trend > 0.1 else 0)
        
        y = np.array(y)
        
        return {'X': X, 'y': y}
    
    def _create_simple_training_data(self):
        """Fallback: Create sample simple training data for basic model."""
        print("⚠️  Using fallback sample simple data")
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels based on simple linear combination
        weights = np.random.randn(n_features)
        y_continuous = X @ weights
        y = (y_continuous > np.median(y_continuous)).astype(int)
        
        return {'X': X, 'y': y}
    
    def _evaluate_enhanced_model(self, model, history):
        """Generate comprehensive evaluation metrics for enhanced model."""
        metrics = {}
        
        if history and history.history:
            # Training metrics
            final_epoch = len(history.history['loss']) - 1
            metrics['final_train_loss'] = float(history.history['loss'][final_epoch])
            metrics['final_val_loss'] = float(history.history['val_loss'][final_epoch])
            metrics['final_train_accuracy'] = float(history.history['accuracy'][final_epoch])
            metrics['final_val_accuracy'] = float(history.history['val_accuracy'][final_epoch])
            
            if 'precision' in history.history:
                metrics['final_train_precision'] = float(history.history['precision'][final_epoch])
                metrics['final_val_precision'] = float(history.history['val_precision'][final_epoch])
            
            if 'recall' in history.history:
                metrics['final_train_recall'] = float(history.history['recall'][final_epoch])
                metrics['final_val_recall'] = float(history.history['val_recall'][final_epoch])
            
            # Best metrics during training
            metrics['best_val_accuracy'] = float(max(history.history['val_accuracy']))
            metrics['best_val_loss'] = float(min(history.history['val_loss']))
            
        return metrics
    
    def _evaluate_temporal_model(self, model, data, history):
        """Generate evaluation metrics for temporal model."""
        metrics = {}
        
        if history and hasattr(history, 'history'):
            final_epoch = len(history.history['loss']) - 1
            metrics['final_train_loss'] = float(history.history['loss'][final_epoch])
            metrics['final_val_loss'] = float(history.history['val_loss'][final_epoch])
            metrics['final_train_accuracy'] = float(history.history['accuracy'][final_epoch])
            metrics['final_val_accuracy'] = float(history.history['val_accuracy'][final_epoch])
        
        return metrics
    
    def _evaluate_simple_model(self, model, data, history):
        """Generate evaluation metrics for simple model."""
        metrics = {}
        
        if history and hasattr(history, 'history'):
            final_epoch = len(history.history['loss']) - 1
            metrics['final_train_loss'] = float(history.history['loss'][final_epoch])
            metrics['final_val_loss'] = float(history.history['val_loss'][final_epoch])
            metrics['final_train_accuracy'] = float(history.history['accuracy'][final_epoch])
            metrics['final_val_accuracy'] = float(history.history['val_accuracy'][final_epoch])
        
        return metrics
    
    def _plot_enhanced_training_history(self, history, model_name):
        """Create comprehensive training plots for enhanced model."""
        if not history or not history.history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name.replace("_", " ").title()} - Training History', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision (if available)
        if 'precision' in history.history:
            axes[0, 2].plot(history.history['precision'], label='Training Precision', linewidth=2)
            axes[0, 2].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
            axes[0, 2].set_title('Model Precision', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'Precision not available', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Precision (N/A)')
        
        # Plot 4: Recall (if available)
        if 'recall' in history.history:
            axes[1, 0].plot(history.history['recall'], label='Training Recall', linewidth=2)
            axes[1, 0].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
            axes[1, 0].set_title('Model Recall', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Recall not available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Recall (N/A)')
        
        # Plot 5: Learning Rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1, 1].set_title('Learning Rate', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate not tracked', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate (N/A)')
        
        # Plot 6: Training Summary
        axes[1, 2].axis('off')
        summary_text = f"""
Training Summary:
• Total Epochs: {len(history.history['loss'])}
• Final Train Loss: {history.history['loss'][-1]:.4f}
• Final Val Loss: {history.history['val_loss'][-1]:.4f}
• Final Train Acc: {history.history['accuracy'][-1]:.4f}
• Final Val Acc: {history.history['val_accuracy'][-1]:.4f}
• Best Val Acc: {max(history.history['val_accuracy']):.4f}
"""
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f'{model_name}_training_history_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plot saved: {plot_path}")
        plt.close()
    
    def _plot_temporal_training_history(self, history, model_name):
        """Create training plots for temporal model."""
        self._plot_standard_training_history(history, model_name)
    
    def _plot_simple_training_history(self, history, model_name):
        """Create training plots for simple model."""
        self._plot_standard_training_history(history, model_name)
    
    def _plot_standard_training_history(self, history, model_name):
        """Create standard training plots."""
        if not history or not hasattr(history, 'history'):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'{model_name.replace("_", " ").title()} - Training History', fontsize=14, fontweight='bold')
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f'{model_name}_training_history_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plot saved: {plot_path}")
        plt.close()
    
    def _train_temporal_model_direct(self, temporal_data, temporal_predictor):
        """Train temporal model directly using the TemporalStabilityPredictor."""
        try:
            X, y = temporal_data['X'], temporal_data['y']
            
            # Prepare data for temporal model
            n_samples, sequence_length, n_features = X.shape
            
            # Create one-hot encoded targets for multi-class
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=2)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_categorical, test_size=0.2, random_state=42
            )
            
            # Build the temporal model architecture
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')  # Binary classification
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=1
            )
            
            return model, history
            
        except Exception as e:
            print(f"Error in temporal model training: {e}")
            return None, None
    
    def _train_simple_model_direct(self, simple_data):
        """Train simple model directly."""
        try:
            X, y = simple_data['X'], simple_data['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler in data for later use
            simple_data['scaler'] = scaler
            
            # Build simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=50,
                batch_size=32,
                verbose=1
            )
            
            return model, history
            
        except Exception as e:
            print(f"Error in simple model training: {e}")
            return None, None
    
    def _test_enhanced_predictions(self, model):
        """Test predictions on enhanced model."""
        try:
            # Get sample stopes
            if hasattr(model, 'static_df') and model.static_df is not None:
                test_stopes = model.static_df['stope_name'].head(5).tolist()
                results = {}
                
                for stope in test_stopes:
                    try:
                        result = model.predict_stability(stope)
                        if result:
                            results[stope] = {
                                'risk_level': result.get('risk_level', 'unknown'),
                                'instability_probability': result.get('instability_probability', 0.0),
                                'future_risk_7_days': result.get('future_risk_7_days', 0.0),
                                'future_risk_30_days': result.get('future_risk_30_days', 0.0)
                            }
                        else:
                            results[stope] = {'error': 'Prediction failed'}
                    except Exception as e:
                        results[stope] = {'error': str(e)}
                
                return results
            else:
                return {'error': 'No test data available'}
                
        except Exception as e:
            return {'error': f'Testing failed: {e}'}
    
    def generate_final_report(self):
        """Generate a comprehensive training report."""
        print("\n" + "="*80)
        print("📊 GENERATING COMPREHENSIVE TRAINING REPORT")
        print("="*80)
        
        # Save results to JSON
        results_path = os.path.join(self.results_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        # Generate summary report
        report_path = os.path.join(self.results_dir, 'training_report.md')
        with open(report_path, 'w') as f:
            f.write(f"# Deepmine Sentinel AI - Training Report\n\n")
            f.write(f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Models Trained\n\n")
            
            for model_name, results in self.training_results.items():
                f.write(f"### {model_name.replace('_', ' ').title()}\n\n")
                f.write(f"- **Model Path:** `{results['model_path']}`\n")
                f.write(f"- **Training Time:** {results['training_time']}\n")
                
                if 'metrics' in results:
                    f.write(f"- **Metrics:**\n")
                    for metric, value in results['metrics'].items():
                        f.write(f"  - {metric}: {value:.4f}\n")
                
                f.write(f"\n")
            
            f.write(f"## Training Artifacts\n\n")
            f.write(f"- Results Directory: `{self.results_dir}`\n")
            f.write(f"- Models Directory: `{self.models_dir}`\n")
            f.write(f"- Plots Directory: `{self.plots_dir}`\n")
        
        print(f"📄 Training report saved: {report_path}")
        print(f"📁 Results directory: {self.results_dir}")
        
        return report_path
    
    def run_complete_training(self):
        """Run complete training for all models."""
        print("🚀 DEEPMINE SENTINEL AI - COMPREHENSIVE MODEL TRAINING")
        print("="*80)
        print(f"Training Session: {self.timestamp}")
        print(f"Results Directory: {self.results_dir}")
        print("="*80)
        
        # Setup GPU
        gpu_available = self.setup_gpu()
        
        # Train all models
        print("\n🎯 Starting comprehensive model training pipeline...")
        
        # 1. Enhanced Dual-Branch Model (Main Model)
        enhanced_model, enhanced_history = self.train_enhanced_dual_branch_model()
        
        # 2. Temporal Stability Model
        temporal_model, temporal_history = self.train_temporal_stability_model()
        
        # 3. Simple Stability Model
        simple_model, simple_history = self.train_simple_stability_model()
        
        # Generate final report
        report_path = self.generate_final_report()
        
        print("\n" + "="*80)
        print("🎉 TRAINING PIPELINE COMPLETED!")
        print("="*80)
        print(f"📄 Report: {report_path}")
        print(f"📁 Results: {self.results_dir}")
        print("="*80)

def main():
    """Main training function."""
    trainer = ModelTrainingOrchestrator()
    trainer.run_complete_training()

if __name__ == "__main__":
    main()
