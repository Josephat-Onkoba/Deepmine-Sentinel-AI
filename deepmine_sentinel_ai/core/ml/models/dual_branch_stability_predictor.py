"""
Enhanced Dual-Branch Neural Network for Stope Stability Prediction
=================================================================

This module implements an advanced dual-branch neural network architecture for:
1. Current stope stability prediction (binary classification)
2. Future risk forecasting (multi-horizon temporal prediction)
3. Risk progression modeling (time series forecasting)

Architecture:
- Dense Feedforward branch: static stope features + geological profile
- LSTM branch: temporal sensor data (vibration, deformation, stress, etc.)
- Temporal projection layers: future risk prediction for multiple horizons
- Attention mechanism: focus on critical temporal patterns

Features:
- Multi-horizon predictions (1, 3, 7, 14, 30 days ahead)
- Risk level classification (stable, slight_elevated, elevated, high, critical, unstable)
- Confidence intervals and uncertainty quantification
- Feature importance analysis and explanations

Author: Deepmine Sentinel AI Team
Date: July 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Concatenate, Dropout, BatchNormalization, 
    MultiHeadAttention, LayerNormalization, RepeatVector,
    TimeDistributed, Reshape, Lambda
)
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime as dt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.utils import get_stope_profile_summary, get_stope_features_for_ml

class EnhancedDualBranchStabilityPredictor:
    """
    Enhanced dual-branch neural network for current and future stope stability prediction.
    
    Features:
    - Dual-branch architecture (static + temporal features)
    - Multi-horizon future predictions (1, 3, 7, 14, 30 days)
    - Risk level classification with confidence intervals
    - Attention mechanisms for temporal pattern recognition
    - Uncertainty quantification and model interpretability
    """
    
    def __init__(self, static_features_path, timeseries_path):
        self.static_features_path = static_features_path
        self.timeseries_path = timeseries_path
        self.static_df = pd.read_csv(static_features_path)
        self.timeseries_df = pd.read_csv(timeseries_path)
        
        # Models for different tasks
        self.current_model = None  # Current stability prediction
        self.temporal_model = None  # Future risk prediction
        self.combined_model = None  # Unified multi-task model
        
        # Data preprocessing
        self.static_scaler = StandardScaler()
        self.timeseries_scaler = StandardScaler()
        self.risk_label_encoder = LabelEncoder()
        
        # Prediction horizons (days)
        self.prediction_horizons = [1, 3, 7, 14, 30]
        
        # Risk levels for classification
        self.risk_levels = ['stable', 'slight_elevated', 'elevated', 'high', 'critical', 'unstable']
        self.risk_to_numeric = {level: idx for idx, level in enumerate(self.risk_levels)}
        self.numeric_to_risk = {idx: level for idx, level in enumerate(self.risk_levels)}
        
        # Feature mappings for categorical variables
        self.direction_map = {
            'North': 0, 'Northeast': 1, 'East': 2, 'Southeast': 3,
            'South': 4, 'Southwest': 5, 'West': 6, 'Northwest': 7
        }
        
        self.rock_type_map = {
            'Granite': 0, 'Basalt': 1, 'Quartzite': 2, 'Schist': 3,
            'Gneiss': 4, 'Marble': 5, 'Slate': 6, 'Shale': 7,
            'Limestone': 8, 'Sandstone': 9, 'Obsidian': 10
        }
        
        self.support_type_map = {
            'None': 0, 'Rock Bolts': 1, 'Mesh': 2, 'Steel Sets': 3,
            'Shotcrete': 4, 'Timber': 5, 'Cable Bolts': 6
        }
        
        # Model configuration
        self.sequence_length = 30  # Days of historical data for LSTM
        self.n_features_temporal = 5  # vibration, deformation, stress, temp, humidity

    def prepare_enhanced_training_data(self):
        """
        Prepare enhanced training data for both current and future stability prediction.
        
        Returns:
            dict containing:
            - X_static: Static features
            - X_timeseries: Temporal sequences 
            - y_current: Current stability labels
            - y_future: Future risk labels for multiple horizons
            - timestamps: Associated timestamps
        """
        # Get unique stopes from both datasets
        static_stopes = set(self.static_df['stope_name'].unique())
        timeseries_stopes = set(self.timeseries_df['stope_name'].unique())
        common_stopes = list(static_stopes.intersection(timeseries_stopes))
        
        print(f"Found {len(common_stopes)} stopes with both static and timeseries data")
        
        # Prepare training sequences
        X_static = []
        X_timeseries = []
        y_current = []
        y_future = []
        timestamps = []
        
        for stope in common_stopes:
            # Get static features
            row = self.static_df[self.static_df['stope_name'] == stope].iloc[0]
            static_features = self._extract_static_features(row)
            
            # Get timeseries data sorted by timestamp
            stope_ts = self.timeseries_df[self.timeseries_df['stope_name'] == stope].copy()
            stope_ts['timestamp'] = pd.to_datetime(stope_ts['timestamp'])
            stope_ts = stope_ts.sort_values('timestamp')
            
            # Create sliding windows for temporal sequences
            ts_features = ['vibration_velocity', 'deformation_rate', 'stress', 'temperature', 'humidity']
            
            # Adjust window logic to ensure we have valid samples
            min_sequence_start = self.sequence_length  # Need at least sequence_length historical points
            max_sequence_end = len(stope_ts) - 1  # At least one point for current prediction
            
            # For future predictions, we'll use the longest available horizon, not max
            max_future_horizon = min(max(self.prediction_horizons), len(stope_ts) - min_sequence_start - 1)
            
            if min_sequence_start >= max_sequence_end:
                print(f"Warning: Stope {stope} has insufficient data ({len(stope_ts)} points), skipping...")
                continue
            
            for i in range(min_sequence_start, max_sequence_end):
                # Historical sequence (input)
                sequence = stope_ts.iloc[i-self.sequence_length:i][ts_features].values
                
                # Current timestamp
                current_timestamp = stope_ts.iloc[i]['timestamp']
                
                # Current stability (target for current prediction)
                current_stability = self._calculate_current_stability(
                    static_features, sequence[-1]  # Use latest sensor reading
                )
                
                # Future stability for different horizons
                future_risks = []
                for horizon in self.prediction_horizons:
                    if i + horizon < len(stope_ts):
                        future_sensor_data = stope_ts.iloc[i + horizon][ts_features].values
                        future_risk = self._calculate_future_risk(
                            static_features, sequence, future_sensor_data, horizon
                        )
                        future_risks.append(future_risk)
                    else:
                        # Use extrapolation for missing future data
                        future_risks.append(self._extrapolate_future_risk(
                            static_features, sequence, horizon
                        ))
                
                # Store training sample
                X_static.append(static_features)
                X_timeseries.append(sequence)
                y_current.append(current_stability)
                y_future.append(future_risks)
                timestamps.append(current_timestamp)
        
        # Convert to numpy arrays
        X_static = np.array(X_static)
        X_timeseries = np.array(X_timeseries)
        y_current = np.array(y_current)
        y_future = np.array(y_future)
        
        print(f"Raw data shapes:")
        print(f"  X_static: {X_static.shape}")
        print(f"  X_timeseries: {X_timeseries.shape}")
        print(f"  y_current: {y_current.shape}")
        print(f"  y_future: {y_future.shape}")
        
        # Check if we have any training samples
        if len(X_static) == 0:
            raise ValueError(
                f"No training samples generated! Check data quality and sequence parameters.\n"
                f"Sequence length: {self.sequence_length}, Max prediction horizon: {max(self.prediction_horizons)}\n"
                f"Available stopes: {common_stopes}\n"
                f"Timeseries data length per stope: {[len(self.timeseries_df[self.timeseries_df['stope_name']==s]) for s in common_stopes[:3]]}"
            )
        
        # Normalize features
        X_static = self.static_scaler.fit_transform(X_static)
        
        # Normalize timeseries (reshape for batch normalization)
        n_samples, n_timesteps, n_features = X_timeseries.shape
        X_timeseries_reshaped = X_timeseries.reshape(-1, n_features)
        X_timeseries_scaled = self.timeseries_scaler.fit_transform(X_timeseries_reshaped)
        X_timeseries = X_timeseries_scaled.reshape(n_samples, n_timesteps, n_features)
        
        print(f"Prepared training data:")
        print(f"  Samples: {len(X_static)}")
        print(f"  Static features shape: {X_static.shape}")
        print(f"  Timeseries shape: {X_timeseries.shape}")
        print(f"  Current stability distribution: {np.bincount(y_current)}")
        print(f"  Future predictions shape: {y_future.shape}")
        
        return {
            'X_static': X_static,
            'X_timeseries': X_timeseries,
            'y_current': y_current,
            'y_future': y_future,
            'timestamps': timestamps
        }
    
    def _extract_static_features(self, row):
        """Extract and encode static features from dataframe row."""
        static_features = [
            float(row['rqd']),
            float(row['hr']),
            float(row['depth']),
            float(row['dip']),
            self.direction_map.get(row['direction'], 0),
            float(row['undercut_wdt']),
            self.rock_type_map.get(row['rock_type'], 0),
            self.support_type_map.get(row['support_type'], 0),
            float(row['support_density']),
            int(row['support_installed'])
        ]
        
        # Add profile summary features
        try:
            profile_summary = get_stope_profile_summary(row['stope_name'])
            static_features.extend(profile_summary)
        except:
            # Add default profile summary if calculation fails
            static_features.extend([0.0] * 10)  # Assuming 10 profile features
            
        return static_features
    
    def _calculate_current_stability(self, static_features, current_sensor_data):
        """
        Calculate current stability label based on static and sensor data.
        Returns binary classification: 0=stable, 1=unstable
        """
        rqd = static_features[0]
        hr = static_features[1]
        depth = static_features[2]
        support_installed = static_features[9]
        support_density = static_features[8]
        
        # Current sensor readings
        vibration = current_sensor_data[0] if len(current_sensor_data) > 0 else 0
        deformation = current_sensor_data[1] if len(current_sensor_data) > 1 else 0
        stress = current_sensor_data[2] if len(current_sensor_data) > 2 else 0
        
        # Enhanced stability scoring
        instability_score = 0
        
        # Geological factors
        if rqd < 25:
            instability_score += 0.4
        elif rqd < 50:
            instability_score += 0.2
        elif rqd < 70:
            instability_score += 0.1
            
        # Span effects
        if hr > 15:
            instability_score += 0.4
        elif hr > 10:
            instability_score += 0.2
        elif hr > 8:
            instability_score += 0.1
            
        # Depth effects  
        if depth > 800:
            instability_score += 0.3
        elif depth > 600:
            instability_score += 0.2
        elif depth > 400:
            instability_score += 0.1
            
        # Support adequacy
        if not support_installed or support_density < 0.3:
            instability_score += 0.3
        elif support_density < 0.6:
            instability_score += 0.1
            
        # Real-time sensor indicators
        if vibration > 5.0:  # High vibration threshold
            instability_score += 0.2
        elif vibration > 3.0:
            instability_score += 0.1
            
        if deformation > 2.0:  # High deformation rate
            instability_score += 0.3
        elif deformation > 1.0:
            instability_score += 0.1
            
        if stress > 100:  # High stress levels
            instability_score += 0.2
        elif stress > 50:
            instability_score += 0.1
        
        # Add controlled randomness for model diversity
        noise = np.random.normal(0, 0.05)
        instability_score += noise
        
        return 1 if instability_score > 0.5 else 0
    
    def _calculate_future_risk(self, static_features, historical_sequence, future_sensor_data, horizon_days):
        """
        Calculate future risk level (0-5) based on projected conditions.
        """
        current_risk = self._calculate_current_stability(static_features, historical_sequence[-1])
        
        # Trend analysis from historical sequence
        if len(historical_sequence) >= 7:
            recent_vibration_trend = np.polyfit(range(7), historical_sequence[-7:, 0], 1)[0]
            recent_deformation_trend = np.polyfit(range(7), historical_sequence[-7:, 1], 1)[0]
            recent_stress_trend = np.polyfit(range(7), historical_sequence[-7:, 2], 1)[0]
        else:
            recent_vibration_trend = 0
            recent_deformation_trend = 0  
            recent_stress_trend = 0
        
        # Project future risk based on trends and horizon
        base_risk = current_risk * 2  # Convert binary to risk scale start
        
        # Trend-based risk escalation
        trend_factor = horizon_days / 30.0  # Normalize by month
        
        if recent_vibration_trend > 0.1:
            base_risk += trend_factor * 1.0
        if recent_deformation_trend > 0.05:
            base_risk += trend_factor * 1.5
        if recent_stress_trend > 1.0:
            base_risk += trend_factor * 1.0
            
        # Future sensor data influence
        future_vibration = future_sensor_data[0] if len(future_sensor_data) > 0 else 0
        future_deformation = future_sensor_data[1] if len(future_sensor_data) > 1 else 0
        future_stress = future_sensor_data[2] if len(future_sensor_data) > 2 else 0
        
        if future_vibration > 4.0:
            base_risk += 1.0
        if future_deformation > 1.5:
            base_risk += 1.5
        if future_stress > 80:
            base_risk += 1.0
            
        # Geological factors affecting long-term stability
        rqd = static_features[0]
        depth = static_features[2]
        
        if horizon_days > 7:
            if rqd < 50:
                base_risk += 0.5
            if depth > 600:
                base_risk += 0.3
        
        # Add some noise and constrain to valid range
        risk_level = base_risk + np.random.normal(0, 0.3)
        risk_level = max(0, min(5, int(round(risk_level))))
        
        return risk_level
    
    def _extrapolate_future_risk(self, static_features, historical_sequence, horizon_days):
        """
        Extrapolate future risk when no future sensor data is available.
        """
        # Use trend analysis to project future conditions
        if len(historical_sequence) >= 7:
            # Calculate trends
            vibration_trend = np.polyfit(range(len(historical_sequence)), 
                                       historical_sequence[:, 0], 1)[0]
            deformation_trend = np.polyfit(range(len(historical_sequence)), 
                                         historical_sequence[:, 1], 1)[0]
            stress_trend = np.polyfit(range(len(historical_sequence)), 
                                    historical_sequence[:, 2], 1)[0]
            
            # Project future values
            future_vibration = historical_sequence[-1, 0] + vibration_trend * horizon_days
            future_deformation = historical_sequence[-1, 1] + deformation_trend * horizon_days
            future_stress = historical_sequence[-1, 2] + stress_trend * horizon_days
            
            future_sensor_data = [future_vibration, future_deformation, future_stress, 
                                historical_sequence[-1, 3], historical_sequence[-1, 4]]
        else:
            # Use last known values with slight degradation
            future_sensor_data = historical_sequence[-1] * (1 + 0.1 * horizon_days / 30)
            
        return self._calculate_future_risk(static_features, historical_sequence, 
                                         future_sensor_data, horizon_days)
    
    def _generate_stability_target(self, row):
        """
        Generate synthetic stability target based on geological rules.
        Returns 1 for unstable, 0 for stable.
        """
        rqd = float(row['rqd'])
        hr = float(row['hr'])
        depth = float(row['depth'])
        support_installed = row['support_installed']
        support_density = float(row['support_density'])
        
        # Rule-based stability assessment
        instability_score = 0
        
        # Poor rock quality
        if rqd < 50:
            instability_score += 0.4
        elif rqd < 70:
            instability_score += 0.2
            
        # Large hydraulic radius (span)
        if hr > 10:
            instability_score += 0.3
        elif hr > 8:
            instability_score += 0.15
            
        # Deep stopes
        if depth > 600:
            instability_score += 0.2
        elif depth > 400:
            instability_score += 0.1
            
        # Inadequate support
        if not support_installed or support_density < 0.5:
            instability_score += 0.3
            
        # Add some randomness to make it more realistic
        noise = np.random.normal(0, 0.1)
        instability_score += noise
        
        return 1 if instability_score > 0.5 else 0

    def build_enhanced_model(self, static_input_dim, timeseries_timesteps, timeseries_feature_dim,
                           dense_units=128, lstm_units=128, attention_heads=8, dropout_rate=0.3):
        """
        Build enhanced dual-branch neural network with temporal prediction capabilities.
        
        Architecture:
        1. Static branch: Dense layers for geological/structural features
        2. Temporal branch: LSTM + Attention for time series patterns
        3. Current prediction head: Binary classification (stable/unstable)
        4. Future prediction heads: Multi-class classification for each horizon
        """
        
        # ===== STATIC FEATURES BRANCH =====
        static_input = Input(shape=(static_input_dim,), name='static_input')
        
        # Static feature processing
        x_static = Dense(dense_units, activation='relu', name='static_dense_1')(static_input)
        x_static = BatchNormalization(name='static_bn_1')(x_static)
        x_static = Dropout(dropout_rate, name='static_dropout_1')(x_static)
        
        x_static = Dense(dense_units, activation='relu', name='static_dense_2')(x_static)
        x_static = BatchNormalization(name='static_bn_2')(x_static)
        x_static = Dropout(dropout_rate, name='static_dropout_2')(x_static)
        
        x_static = Dense(dense_units // 2, activation='relu', name='static_dense_3')(x_static)
        x_static = BatchNormalization(name='static_bn_3')(x_static)
        
        # ===== TEMPORAL FEATURES BRANCH =====
        timeseries_input = Input(shape=(timeseries_timesteps, timeseries_feature_dim), name='timeseries_input')
        
        # Multi-layer LSTM with residual connections
        x_lstm = LSTM(lstm_units, return_sequences=True, name='lstm_1')(timeseries_input)
        x_lstm = Dropout(dropout_rate, name='lstm_dropout_1')(x_lstm)
        
        x_lstm_2 = LSTM(lstm_units, return_sequences=True, name='lstm_2')(x_lstm)
        x_lstm_2 = Dropout(dropout_rate, name='lstm_dropout_2')(x_lstm_2)
        
        # Add residual connection if dimensions match
        if lstm_units == lstm_units:
            x_lstm = tf.keras.layers.Add(name='lstm_residual')([x_lstm, x_lstm_2])
        else:
            x_lstm = x_lstm_2
        
        # Multi-head attention for temporal pattern recognition
        attention_output = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=lstm_units // attention_heads,
            name='temporal_attention'
        )(x_lstm, x_lstm)
        
        # Layer normalization after attention
        x_lstm = LayerNormalization(name='attention_norm')(attention_output)
        
        # Global temporal features (final LSTM output)
        x_lstm_global = LSTM(lstm_units // 2, return_sequences=False, name='lstm_final')(x_lstm)
        x_lstm_global = BatchNormalization(name='lstm_bn_final')(x_lstm_global)
        x_lstm_global = Dropout(dropout_rate, name='lstm_dropout_final')(x_lstm_global)
        
        # ===== FEATURE FUSION =====
        # Combine static and temporal features
        merged_features = Concatenate(name='feature_fusion')([x_static, x_lstm_global])
        
        # Shared representation learning
        x_shared = Dense(dense_units, activation='relu', name='shared_dense_1')(merged_features)
        x_shared = BatchNormalization(name='shared_bn_1')(x_shared)
        x_shared = Dropout(dropout_rate, name='shared_dropout_1')(x_shared)
        
        x_shared = Dense(dense_units // 2, activation='relu', name='shared_dense_2')(x_shared)
        x_shared = BatchNormalization(name='shared_bn_2')(x_shared)
        
        # ===== PREDICTION HEADS =====
        
        # Current stability prediction (binary classification)
        current_pred = Dense(dense_units // 4, activation='relu', name='current_dense')(x_shared)
        current_pred = Dropout(dropout_rate, name='current_dropout')(current_pred)
        current_stability = Dense(1, activation='sigmoid', name='current_stability')(current_pred)
        
        # Future risk predictions for multiple horizons
        future_predictions = []
        
        for i, horizon in enumerate(self.prediction_horizons):
            # Dedicated branch for each horizon
            future_branch = Dense(dense_units // 4, activation='relu', 
                                name=f'future_{horizon}d_dense')(x_shared)
            future_branch = Dropout(dropout_rate, name=f'future_{horizon}d_dropout')(future_branch)
            
            # Risk level prediction (6 classes: stable to unstable)
            future_risk = Dense(len(self.risk_levels), activation='softmax', 
                              name=f'future_risk_{horizon}d')(future_branch)
            future_predictions.append(future_risk)
        
        # ===== MODEL COMPILATION =====
        
        # Create the model
        self.combined_model = Model(
            inputs=[static_input, timeseries_input],
            outputs=[current_stability] + future_predictions,
            name='EnhancedDualBranchStabilityPredictor'
        )
        
        # Compile with multiple loss functions
        loss_dict = {'current_stability': 'binary_crossentropy'}
        loss_weights = {'current_stability': 1.0}
        
        for i, horizon in enumerate(self.prediction_horizons):
            loss_dict[f'future_risk_{horizon}d'] = 'sparse_categorical_crossentropy'
            loss_weights[f'future_risk_{horizon}d'] = 0.5  # Lower weight for future predictions
        
        metrics_dict = {
            'current_stability': ['accuracy', 'precision', 'recall'],
            **{f'future_risk_{horizon}d': ['accuracy'] for horizon in self.prediction_horizons}
        }
        
        self.combined_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics=metrics_dict
        )
        
        return self.combined_model
    
    def train_enhanced_model(self, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the enhanced model with current and future prediction capabilities.
        """
        print("🚀 Preparing enhanced training data...")
        data = self.prepare_enhanced_training_data()
        
        X_static = data['X_static']
        X_timeseries = data['X_timeseries']
        y_current = data['y_current']
        y_future = data['y_future']
        
        print(f"📊 Training data summary:")
        print(f"   Samples: {len(X_static)}")
        print(f"   Static features: {X_static.shape[1]}")
        print(f"   Temporal sequence: {X_timeseries.shape[1]} timesteps × {X_timeseries.shape[2]} features")
        print(f"   Current stability distribution: {np.bincount(y_current)}")
        print(f"   Future predictions: {len(self.prediction_horizons)} horizons")
        
        # Split data
        indices = np.arange(len(X_static))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, 
                                              stratify=y_current, random_state=42)
        
        # Training data
        X_static_train = X_static[train_idx]
        X_ts_train = X_timeseries[train_idx]
        y_current_train = y_current[train_idx]
        y_future_train = y_future[train_idx]
        
        # Validation data
        X_static_val = X_static[val_idx]
        X_ts_val = X_timeseries[val_idx]
        y_current_val = y_current[val_idx]
        y_future_val = y_future[val_idx]
        
        # Build model
        static_input_dim = X_static.shape[1]
        timeseries_timesteps = X_timeseries.shape[1]
        timeseries_feature_dim = X_timeseries.shape[2]
        
        print(f"🧠 Building enhanced dual-branch model...")
        print(f"   Static input dimension: {static_input_dim}")
        print(f"   Temporal sequence: {timeseries_timesteps} × {timeseries_feature_dim}")
        print(f"   Prediction horizons: {self.prediction_horizons} days")
        
        self.build_enhanced_model(static_input_dim, timeseries_timesteps, timeseries_feature_dim)
        
        # Print model summary
        print(f"\n🏗️ Model Architecture:")
        self.combined_model.summary()
        
        # Prepare target dictionaries for multiple outputs
        y_train_dict = {'current_stability': y_current_train}
        y_val_dict = {'current_stability': y_current_val}
        
        for i, horizon in enumerate(self.prediction_horizons):
            y_train_dict[f'future_risk_{horizon}d'] = y_future_train[:, i]
            y_val_dict[f'future_risk_{horizon}d'] = y_future_val[:, i]
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_enhanced_model.keras', 
                monitor='val_loss', save_best_only=True
            )
        ]
        
        # Train the model
        print(f"\n🎯 Training enhanced model for {epochs} epochs...")
        history = self.combined_model.fit(
            [X_static_train, X_ts_train], 
            y_train_dict,
            validation_data=([X_static_val, X_ts_val], y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate final model
        print(f"\n📈 Evaluating model performance...")
        val_results = self.combined_model.evaluate(
            [X_static_val, X_ts_val], y_val_dict, verbose=0
        )
        
        # Print results
        print(f"\n✅ Training completed!")
        print(f"📊 Validation Results:")
        
        for i, metric_name in enumerate(self.combined_model.metrics_names):
            print(f"   {metric_name}: {val_results[i]:.4f}")
        
        return history

    def train(self, epochs=100, batch_size=16, validation_split=0.2):
        """
        Train the model using the prepared data.
        """
        print("Preparing training data...")
        X_static, X_timeseries, y = self.prepare_data()
        
        print(f"Data shapes:")
        print(f"  Static features: {X_static.shape}")
        print(f"  Timeseries features: {X_timeseries.shape}")
        print(f"  Targets: {y.shape}")
        print(f"  Positive samples: {np.sum(y)}/{len(y)} ({np.mean(y)*100:.1f}%)")
        
        # Split data
        indices = np.arange(len(X_static))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, 
                                              stratify=y, random_state=42)
        
        X_static_train, X_static_val = X_static[train_idx], X_static[val_idx]
        X_ts_train, X_ts_val = X_timeseries[train_idx], X_timeseries[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build model
        static_input_dim = X_static.shape[1]
        timeseries_timesteps = X_timeseries.shape[1]
        timeseries_feature_dim = X_timeseries.shape[2]
        
        print(f"Building model with:")
        print(f"  Static input dim: {static_input_dim}")
        print(f"  Timeseries timesteps: {timeseries_timesteps}")
        print(f"  Timeseries features: {timeseries_feature_dim}")
        
        self.build_model(static_input_dim, timeseries_timesteps, timeseries_feature_dim)
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Train model
        print(f"\nTraining for {epochs} epochs...")
        history = self.model.fit(
            [X_static_train, X_ts_train], y_train,
            validation_data=([X_static_val, X_ts_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True
        )
        
        # Evaluate on validation set
        val_loss, val_acc, val_prec, val_rec = self.model.evaluate(
            [X_static_val, X_ts_val], y_val, verbose=0
        )
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  Precision: {val_prec:.4f}")
        print(f"  Recall: {val_rec:.4f}")
        print(f"  F1-Score: {2 * val_prec * val_rec / (val_prec + val_rec):.4f}")
        
        return history

    def predict_comprehensive_stability(self, stope_name):
        """
        Comprehensive stability prediction including current state and future projections.
        
        Returns:
            dict: Complete prediction results with current and future assessments
        """
        if self.combined_model is None:
            raise ValueError("Enhanced model not trained yet. Call train_enhanced_model() first.")
        
        try:
            # Prepare input features
            static_features, timeseries_features = self._prepare_prediction_features(stope_name)
            
            if static_features is None or timeseries_features is None:
                return None
            
            # Make predictions
            predictions = self.combined_model.predict(
                [static_features, timeseries_features], verbose=0
            )
            
            # Parse predictions
            current_prob = float(predictions[0][0][0])
            future_probs = [pred[0] for pred in predictions[1:]]
            
            # Determine current risk level
            current_risk = "High" if current_prob > 0.7 else "Medium" if current_prob > 0.3 else "Low"
            current_stable = current_prob < 0.5
            
            # Process future predictions
            future_predictions = []
            for i, (horizon, prob_dist) in enumerate(zip(self.prediction_horizons, future_probs)):
                predicted_class = np.argmax(prob_dist)
                confidence = float(np.max(prob_dist))
                risk_level = self.numeric_to_risk[predicted_class]
                
                future_predictions.append({
                    'horizon_days': horizon,
                    'predicted_risk_level': risk_level,
                    'confidence': confidence,
                    'risk_probabilities': {
                        level: float(prob_dist[idx]) 
                        for idx, level in enumerate(self.risk_levels)
                    }
                })
            
            # Generate explanations
            explanations = self._generate_prediction_explanations(
                stope_name, static_features, timeseries_features, 
                current_prob, future_predictions
            )
            
            # Risk trend analysis
            risk_trend = self._analyze_risk_trend(future_predictions)
            
            return {
                'stope_name': stope_name,
                'timestamp': dt.datetime.now().isoformat(),
                'model_type': 'Dual-Branch Neural Network',
                'model_version': 'Enhanced v2.0',
                
                # Current stability
                'current_stability': {
                    'stable': current_stable,
                    'instability_probability': float(current_prob),
                    'risk_level': current_risk,
                    'confidence': float(abs(current_prob - 0.5) * 2)  # Distance from decision boundary
                },
                
                # Future predictions
                'future_predictions': future_predictions,
                
                # Risk analysis
                'risk_trend': risk_trend,
                'max_risk_horizon': max(future_predictions, key=lambda x: x['confidence'])['horizon_days'],
                'alert_recommended': any(
                    pred['predicted_risk_level'] in ['high', 'critical', 'unstable'] 
                    for pred in future_predictions
                ),
                
                # Model explanations
                'explanations': explanations,
                
                # Recommendations
                'recommendations': self._generate_recommendations(
                    current_prob, future_predictions, explanations
                )
            }
            
        except Exception as e:
            print(f"Error predicting stability for stope {stope_name}: {e}")
            return None
    
    def _prepare_prediction_features(self, stope_name):
        """Prepare features for prediction from stope name."""
        try:
            # Get static features
            if stope_name not in self.static_df['stope_name'].values:
                print(f"Static data not found for stope {stope_name}")
                return None, None
                
            row = self.static_df[self.static_df['stope_name'] == stope_name].iloc[0]
            static_features = self._extract_static_features(row)
            static_vec = np.array(static_features).reshape(1, -1)
            static_vec = self.static_scaler.transform(static_vec)
            
            # Get timeseries features
            if stope_name not in self.timeseries_df['stope_name'].values:
                print(f"Timeseries data not found for stope {stope_name}")
                return None, None
                
            stope_ts = self.timeseries_df[self.timeseries_df['stope_name'] == stope_name].copy()
            stope_ts['timestamp'] = pd.to_datetime(stope_ts['timestamp'])
            stope_ts = stope_ts.sort_values('timestamp')
            
            # Get recent sequence
            ts_features = ['vibration_velocity', 'deformation_rate', 'stress', 'temperature', 'humidity']
            recent_data = stope_ts.tail(self.sequence_length)[ts_features].values
            
            # Pad if necessary
            if len(recent_data) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(recent_data), len(ts_features)))
                recent_data = np.vstack([padding, recent_data])
            
            # Reshape and normalize
            ts_matrix = recent_data.reshape(1, self.sequence_length, len(ts_features))
            
            # Normalize using fitted scaler
            n_samples, n_timesteps, n_features = ts_matrix.shape
            ts_reshaped = ts_matrix.reshape(-1, n_features)
            ts_scaled = self.timeseries_scaler.transform(ts_reshaped)
            ts_matrix = ts_scaled.reshape(n_samples, n_timesteps, n_features)
            
            return static_vec, ts_matrix
            
        except Exception as e:
            print(f"Error preparing features for {stope_name}: {e}")
            return None, None
    
    def _generate_prediction_explanations(self, stope_name, static_features, timeseries_features, 
                                        current_prob, future_predictions):
        """Generate human-readable explanations for predictions."""
        explanations = []
        
        # Static feature analysis
        static_raw = self.static_scaler.inverse_transform(static_features)[0]
        rqd = static_raw[0]
        hr = static_raw[1]
        depth = static_raw[2]
        support_density = static_raw[8]
        
        if rqd < 50:
            explanations.append(f"Poor rock quality (RQD: {rqd:.1f}%) increases instability risk")
        if hr > 10:
            explanations.append(f"Large span (HR: {hr:.1f}m) creates high stress concentrations")
        if depth > 600:
            explanations.append(f"Deep stope (Depth: {depth:.0f}m) increases geological pressure")
        if support_density < 0.5:
            explanations.append(f"Inadequate support density ({support_density:.2f}) compromises stability")
        
        # Temporal pattern analysis
        recent_vibration = timeseries_features[0, -7:, 0].mean()  # Last week average
        recent_deformation = timeseries_features[0, -7:, 1].mean()
        recent_stress = timeseries_features[0, -7:, 2].mean()
        
        if recent_vibration > 3.0:
            explanations.append(f"Elevated vibration levels ({recent_vibration:.2f}) indicate active movement")
        if recent_deformation > 1.0:
            explanations.append(f"High deformation rate ({recent_deformation:.2f}) suggests ongoing instability")
        if recent_stress > 50:
            explanations.append(f"Excessive stress levels ({recent_stress:.1f}) approach failure thresholds")
        
        # Future risk factors
        high_risk_horizons = [
            pred for pred in future_predictions 
            if pred['predicted_risk_level'] in ['high', 'critical', 'unstable']
        ]
        
        if high_risk_horizons:
            shortest_horizon = min(pred['horizon_days'] for pred in high_risk_horizons)
            explanations.append(
                f"Risk escalation expected within {shortest_horizon} days based on current trends"
            )
        
        return explanations
    
    def _analyze_risk_trend(self, future_predictions):
        """Analyze the trend in future risk predictions."""
        risk_numeric = [
            self.risk_to_numeric[pred['predicted_risk_level']] 
            for pred in future_predictions
        ]
        
        if len(risk_numeric) < 2:
            return "insufficient_data"
        
        # Calculate trend slope
        x = np.array(range(len(risk_numeric)))
        slope = np.polyfit(x, risk_numeric, 1)[0]
        
        if slope > 0.5:
            return "escalating"
        elif slope > 0.1:
            return "increasing"
        elif slope > -0.1:
            return "stable"
        elif slope > -0.5:
            return "improving"
        else:
            return "rapidly_improving"
    
    def _generate_recommendations(self, current_prob, future_predictions, explanations):
        """Generate actionable recommendations based on predictions."""
        recommendations = []
        
        # Current risk recommendations
        if current_prob > 0.7:
            recommendations.append("IMMEDIATE: Evacuate personnel and suspend operations")
            recommendations.append("URGENT: Conduct detailed geological assessment")
        elif current_prob > 0.5:
            recommendations.append("HIGH PRIORITY: Increase monitoring frequency")
            recommendations.append("RECOMMENDED: Review and enhance support systems")
        
        # Future risk recommendations
        high_risk_future = any(
            pred['predicted_risk_level'] in ['high', 'critical', 'unstable']
            for pred in future_predictions
        )
        
        if high_risk_future:
            recommendations.append("PREVENTIVE: Schedule preemptive support installation")
            recommendations.append("MONITORING: Deploy additional sensors for early warning")
        
        # Specific recommendations based on explanations
        if any("rock quality" in exp.lower() for exp in explanations):
            recommendations.append("GEOLOGICAL: Consider alternative mining methods for poor rock")
        if any("span" in exp.lower() for exp in explanations):
            recommendations.append("STRUCTURAL: Implement span reduction measures")
        if any("support" in exp.lower() for exp in explanations):
            recommendations.append("SUPPORT: Increase support density and coverage")
        if any("vibration" in exp.lower() for exp in explanations):
            recommendations.append("OPERATIONAL: Review blasting procedures and timing")
        
        return recommendations
    
    def save_enhanced_model(self, filepath):
        """Save the enhanced model and all preprocessing components."""
        if self.combined_model is None:
            raise ValueError("No enhanced model to save. Train the model first.")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the main model
        self.combined_model.save(filepath)
        
        # Save preprocessing components
        components_path = filepath.replace('.keras', '_components.pkl').replace('.h5', '_components.pkl')
        joblib.dump({
            'static_scaler': self.static_scaler,
            'timeseries_scaler': self.timeseries_scaler,
            'risk_label_encoder': self.risk_label_encoder,
            'direction_map': self.direction_map,
            'rock_type_map': self.rock_type_map,
            'support_type_map': self.support_type_map,
            'risk_levels': self.risk_levels,
            'risk_to_numeric': self.risk_to_numeric,
            'numeric_to_risk': self.numeric_to_risk,
            'prediction_horizons': self.prediction_horizons,
            'sequence_length': self.sequence_length,
            'n_features_temporal': self.n_features_temporal
        }, components_path)
        
        print(f"✅ Enhanced model saved to {filepath}")
        print(f"✅ Components saved to {components_path}")
    
    def load_enhanced_model(self, filepath):
        """Load the enhanced model and preprocessing components."""
        # Load the main model
        self.combined_model = tf.keras.models.load_model(filepath)
        
        # Load preprocessing components
        components_path = filepath.replace('.keras', '_components.pkl').replace('.h5', '_components.pkl')
        components = joblib.load(components_path)
        
        self.static_scaler = components['static_scaler']
        self.timeseries_scaler = components['timeseries_scaler']
        self.risk_label_encoder = components['risk_label_encoder']
        self.direction_map = components['direction_map']
        self.rock_type_map = components['rock_type_map']
        self.support_type_map = components['support_type_map']
        self.risk_levels = components['risk_levels']
        self.risk_to_numeric = components['risk_to_numeric']
        self.numeric_to_risk = components['numeric_to_risk']
        self.prediction_horizons = components['prediction_horizons']
        self.sequence_length = components['sequence_length']
        self.n_features_temporal = components['n_features_temporal']
        
        print(f"✅ Enhanced model loaded from {filepath}")
        print(f"✅ Components loaded from {components_path}")
    
    def plot_enhanced_training_history(self, history, save_path='plots/enhanced_training_history.png'):
        """Plot comprehensive training history for the enhanced model."""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Dual-Branch Model Training History', fontsize=16)
        
        # Overall loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Overall Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Current stability metrics
        if 'current_stability_accuracy' in history.history:
            axes[0, 1].plot(history.history['current_stability_accuracy'], label='Training Accuracy')
            axes[0, 1].plot(history.history['val_current_stability_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Current Stability Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Future prediction accuracy (average across horizons)
        future_acc_keys = [k for k in history.history.keys() if 'future_risk' in k and 'accuracy' in k and 'val' not in k]
        if future_acc_keys:
            future_acc_train = np.mean([history.history[k] for k in future_acc_keys], axis=0)
            future_acc_val = np.mean([history.history[f'val_{k}'] for k in future_acc_keys], axis=0)
            
            axes[0, 2].plot(future_acc_train, label='Training Accuracy')
            axes[0, 2].plot(future_acc_val, label='Validation Accuracy')
            axes[0, 2].set_title('Future Predictions Accuracy (Average)')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Learning rate
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Individual horizon accuracies
        horizon_colors = plt.cm.viridis(np.linspace(0, 1, len(self.prediction_horizons)))
        
        for i, (horizon, color) in enumerate(zip(self.prediction_horizons, horizon_colors)):
            key = f'future_risk_{horizon}d_accuracy'
            if key in history.history:
                axes[1, 1].plot(history.history[key], color=color, label=f'{horizon}d', alpha=0.7)
        
        axes[1, 1].set_title('Future Prediction Accuracy by Horizon')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Loss breakdown
        loss_keys = [k for k in history.history.keys() if 'loss' in k and 'val' not in k and k != 'loss']
        if loss_keys:
            for key in loss_keys:
                axes[1, 2].plot(history.history[key], label=key.replace('_', ' ').title())
            
            axes[1, 2].set_title('Individual Loss Components')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training history plots saved to: {save_path}")
        return save_path

    # ===== LEGACY SUPPORT METHODS =====
    # These methods provide backward compatibility with existing code

    def prepare_data(self):
        """Legacy method - redirects to enhanced data preparation."""
        data = self.prepare_enhanced_training_data()
        return data['X_static'], data['X_timeseries'], data['y_current']

    def build_model(self, static_input_dim, timeseries_timesteps, timeseries_feature_dim, 
                   dense_units=128, lstm_units=128, dropout_rate=0.3):
        """Legacy method - redirects to enhanced model building."""
        return self.build_enhanced_model(
            static_input_dim, timeseries_timesteps, timeseries_feature_dim,
            dense_units, lstm_units, 8, dropout_rate  # 8 attention heads
        )

    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """Legacy method - redirects to enhanced training."""
        return self.train_enhanced_model(epochs, batch_size, validation_split)

    def predict_stability(self, stope_name):
        """
        Legacy prediction method - returns simplified results for backward compatibility.
        For full functionality, use predict_comprehensive_stability().
        """
        comprehensive_result = self.predict_comprehensive_stability(stope_name)
        
        if comprehensive_result is None:
            return None
        
        # Extract simplified results for backward compatibility
        current = comprehensive_result['current_stability']
        
        return {
            'stope_name': stope_name,
            'instability_probability': current['instability_probability'],
            'risk_level': current['risk_level'],
            'risk_color': 'red' if current['risk_level'] == 'High' else 'orange' if current['risk_level'] == 'Medium' else 'green',
            'stable': current['stable']
        }

    def save_model(self, filepath):
        """Legacy method - redirects to enhanced model saving."""
        self.save_enhanced_model(filepath)

    def load_model(self, filepath):
        """Legacy method - redirects to enhanced model loading."""
        self.load_enhanced_model(filepath)

    def plot_training_history(self, history, save_path='plots/training_history.png'):
        """Legacy method - redirects to enhanced plotting."""
        return self.plot_enhanced_training_history(history, save_path)


# Backward compatibility alias
DualBranchStopeStabilityPredictor = EnhancedDualBranchStabilityPredictor
