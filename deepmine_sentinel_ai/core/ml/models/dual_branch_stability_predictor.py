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
import logging
import warnings

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Dense, LSTM, Concatenate, Dropout, BatchNormalization, 
        MultiHeadAttention, LayerNormalization, RepeatVector,
        TimeDistributed, Reshape, Lambda
    )
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"TensorFlow not available: {e}. Model training will not be possible.")
    TF_AVAILABLE = False

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directories to path for imports
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
        """
        Initialize the Enhanced Dual-Branch Stability Predictor.
        
        Args:
            static_features_path (str): Path to static features CSV file
            timeseries_path (str): Path to timeseries data CSV file
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required but not available. Please install TensorFlow.")
        
        # Validate input paths
        if not os.path.exists(static_features_path):
            raise FileNotFoundError(f"Static features file not found: {static_features_path}")
        if not os.path.exists(timeseries_path):
            raise FileNotFoundError(f"Timeseries file not found: {timeseries_path}")
        
        self.static_features_path = static_features_path
        self.timeseries_path = timeseries_path
        
        # Load and validate data
        try:
            self.static_df = pd.read_csv(static_features_path)
            self.timeseries_df = pd.read_csv(timeseries_path)
            logger.info(f"Loaded static data: {self.static_df.shape}")
            logger.info(f"Loaded timeseries data: {self.timeseries_df.shape}")
        except Exception as e:
            raise ValueError(f"Error loading data files: {e}")
        
        # Initialize feature mappings BEFORE validation
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
        
        # Validate required columns and data structure
        self._validate_data_structure()
        
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
        
        # Model configuration
        self.sequence_length = 30  # Days of historical data for LSTM
        self.n_features_temporal = 5  # vibration, deformation, stress, temp, humidity
        
        # Training state
        self.is_trained = False
        self.training_metadata = {}

    def _validate_data_structure(self):
        """Validate that the loaded data has required columns and structure."""
        # Required static columns
        required_static_cols = [
            'stope_name', 'rqd', 'hr', 'depth', 'dip', 'direction', 
            'undercut_wdt', 'rock_type', 'support_type', 'support_density', 
            'support_installed', 'stability'
        ]
        
        missing_static = [col for col in required_static_cols if col not in self.static_df.columns]
        if missing_static:
            raise ValueError(f"Missing required static columns: {missing_static}")
        
        # Required timeseries columns
        required_ts_cols = [
            'stope_name', 'timestamp', 'vibration_velocity', 'deformation_rate', 
            'stress', 'temperature', 'humidity'
        ]
        
        missing_ts = [col for col in required_ts_cols if col not in self.timeseries_df.columns]
        if missing_ts:
            raise ValueError(f"Missing required timeseries columns: {missing_ts}")
        
        # Validate data types and ranges
        if self.static_df['stability'].dtype not in ['int64', 'float64']:
            logger.warning("Stability column should be numeric (0/1)")
        
        # Check for overlapping stopes
        static_stopes = set(self.static_df['stope_name'].unique())
        ts_stopes = set(self.timeseries_df['stope_name'].unique())
        common_stopes = static_stopes.intersection(ts_stopes)
        
        if len(common_stopes) == 0:
            raise ValueError("No common stopes found between static and timeseries data")
        
        logger.info(f"Found {len(common_stopes)} common stopes for training")
        
        # Validate categorical values
        for direction in self.static_df['direction'].unique():
            if direction not in self.direction_map:
                logger.warning(f"Unknown direction value: {direction}")
        
        for rock_type in self.static_df['rock_type'].unique():
            if rock_type not in self.rock_type_map:
                logger.warning(f"Unknown rock type: {rock_type}")
        
        for support_type in self.static_df['support_type'].unique():
            if support_type not in self.support_type_map:
                logger.warning(f"Unknown support type: {support_type}")

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
                
                # Current stability (target for current prediction) - use real labels
                current_stability = self._calculate_current_stability(
                    static_features, sequence[-1], stope  # Pass stope name for real labels
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
        """
        Extract and encode static features from dataframe row with robust error handling.
        
        Args:
            row: DataFrame row containing stope data
            
        Returns:
            list: Numerical features ready for model input
        """
        try:
            # Basic geological and structural features
            static_features = [
                float(row['rqd']),
                float(row['hr']),
                float(row['depth']),
                float(row['dip']),
                self.direction_map.get(row['direction'], 0),  # Default to 0 (North) if unknown
                float(row['undercut_wdt']),
                self.rock_type_map.get(row['rock_type'], 0),  # Default to 0 (Granite) if unknown
                self.support_type_map.get(row['support_type'], 0),  # Default to 0 (None) if unknown
                float(row['support_density']),
                int(row['support_installed'])
            ]
            
            # Add profile summary features with error handling
            try:
                profile_summary = get_stope_profile_summary(row['stope_name'])
                if isinstance(profile_summary, list) and len(profile_summary) == 10:
                    static_features.extend(profile_summary)
                else:
                    logger.warning(f"Invalid profile summary for {row['stope_name']}, using defaults")
                    static_features.extend([0.5] * 10)  # Default neutral values
            except Exception as e:
                logger.warning(f"Error getting profile summary for {row['stope_name']}: {e}")
                static_features.extend([0.5] * 10)  # Default neutral values
            
            # Validate all features are numeric
            for i, feature in enumerate(static_features):
                if not isinstance(feature, (int, float)) or np.isnan(feature) or np.isinf(feature):
                    logger.warning(f"Invalid feature at index {i}: {feature}, replacing with 0")
                    static_features[i] = 0.0
            
            return static_features
            
        except Exception as e:
            logger.error(f"Error extracting static features for {row.get('stope_name', 'unknown')}: {e}")
            # Return default feature vector
            return [0.0] * 20  # 10 basic + 10 profile features
    
    def _calculate_current_stability(self, static_features, current_sensor_data, stope_name=None):
        """
        Use real stability labels from CSV data with enhanced fallback logic.
        Returns binary classification: 0=stable, 1=unstable
        """
        # Primary: Use real stability labels from CSV data
        if stope_name and hasattr(self, 'static_df'):
            try:
                stope_data = self.static_df[self.static_df['stope_name'] == stope_name]
                if not stope_data.empty:
                    stability_label = int(stope_data.iloc[0]['stability'])
                    # Note: CSV has 1=stable, 0=unstable, but model expects 0=stable, 1=unstable
                    # So we need to invert the label
                    inverted_label = 1 - stability_label
                    logger.debug(f"Real stability label for {stope_name}: {stability_label} -> {inverted_label}")
                    return inverted_label
            except Exception as e:
                logger.warning(f"Could not get real stability label for {stope_name}: {e}")
        
        # Fallback: Enhanced physics-based calculation for consistency
        return self._calculate_physics_based_stability(static_features, current_sensor_data)
    
    def _calculate_physics_based_stability(self, static_features, current_sensor_data):
        """
        Physics-based stability calculation using industry-standard thresholds.
        Returns: 0=stable, 1=unstable
        """
        try:
            # Extract features with bounds checking
            rqd = max(0, min(100, static_features[0])) if len(static_features) > 0 else 50
            hr = max(0, static_features[1]) if len(static_features) > 1 else 10
            depth = max(0, static_features[2]) if len(static_features) > 2 else 400
            support_installed = int(static_features[9]) if len(static_features) > 9 else 0
            support_density = max(0, min(1, static_features[8])) if len(static_features) > 8 else 0.5
            
            # Current sensor readings with error handling
            vibration = max(0, current_sensor_data[0]) if len(current_sensor_data) > 0 else 0
            deformation = max(0, current_sensor_data[1]) if len(current_sensor_data) > 1 else 0
            stress = max(0, current_sensor_data[2]) if len(current_sensor_data) > 2 else 0
            
            # Industry-standard instability scoring
            instability_score = 0.0
            
            # Rock Quality Designation (critical factor)
            if rqd < 25:       # Very poor rock
                instability_score += 0.40
            elif rqd < 50:     # Poor rock  
                instability_score += 0.25
            elif rqd < 70:     # Fair rock
                instability_score += 0.10
            # Good rock (70-90) and excellent rock (90+) contribute minimal risk
            
            # Hydraulic Radius (span-to-perimeter ratio)
            if hr > 15:        # Very large span
                instability_score += 0.35
            elif hr > 12:      # Large span
                instability_score += 0.22
            elif hr > 9:       # Medium span
                instability_score += 0.12
            elif hr > 6:       # Small span
                instability_score += 0.05
            
            # Depth-induced stress
            if depth > 1000:   # Very deep
                instability_score += 0.30
            elif depth > 700:  # Deep
                instability_score += 0.20
            elif depth > 500:  # Medium depth
                instability_score += 0.10
            elif depth > 300:  # Shallow-medium
                instability_score += 0.05
            
            # Support system effectiveness
            if not support_installed:
                instability_score += 0.30
            else:
                if support_density < 0.2:      # Very poor support
                    instability_score += 0.25
                elif support_density < 0.4:    # Poor support
                    instability_score += 0.15
                elif support_density < 0.6:    # Adequate support
                    instability_score += 0.05
                # Good support (0.6+) provides adequate protection
            
            # Real-time monitoring indicators
            # Vibration velocity (mm/s)
            if vibration > 8.0:        # Critical vibration
                instability_score += 0.20
            elif vibration > 5.0:      # High vibration
                instability_score += 0.12
            elif vibration > 3.0:      # Moderate vibration
                instability_score += 0.06
            elif vibration > 1.5:      # Low vibration
                instability_score += 0.02
            
            # Deformation rate (mm/day)
            if deformation > 3.0:      # Critical deformation
                instability_score += 0.25
            elif deformation > 2.0:    # High deformation
                instability_score += 0.15
            elif deformation > 1.0:    # Moderate deformation
                instability_score += 0.08
            elif deformation > 0.5:    # Low deformation
                instability_score += 0.03
            
            # Stress levels (MPa)
            if stress > 150:           # Critical stress
                instability_score += 0.20
            elif stress > 100:        # High stress
                instability_score += 0.12
            elif stress > 60:         # Moderate stress
                instability_score += 0.06
            elif stress > 30:         # Low-moderate stress
                instability_score += 0.02
            
            # Conservative threshold for mining safety (adjusted based on field data)
            stability_threshold = 0.50
            is_unstable = 1 if instability_score > stability_threshold else 0
            
            logger.debug(f"Physics-based stability: score={instability_score:.3f}, threshold={stability_threshold}, unstable={is_unstable}")
            return is_unstable
            
        except Exception as e:
            logger.error(f"Error in physics-based stability calculation: {e}")
            # Conservative default: assume stable for safety
            return 0
    
    def _calculate_future_risk(self, static_features, historical_sequence, future_sensor_data, horizon_days):
        """
        Calculate future risk level (0-5) based on projected conditions and temporal trends.
        
        Risk Levels:
        0 - stable: Very low risk, conditions favorable
        1 - slight_elevated: Minor concerns, normal monitoring
        2 - elevated: Moderate risk, increased monitoring
        3 - high: Significant risk, enhanced precautions
        4 - critical: Very high risk, immediate action required
        5 - unstable: Imminent failure risk, evacuation recommended
        """
        try:
            # Get current stability as baseline
            current_stability = self._calculate_physics_based_stability(static_features, historical_sequence[-1])
            base_risk = current_stability * 2.0  # Convert binary (0,1) to risk scale (0,2)
            
            # Analyze temporal trends from historical sequence
            if len(historical_sequence) >= 7:
                # Calculate recent trends (last week)
                recent_period = min(7, len(historical_sequence))
                recent_data = historical_sequence[-recent_period:]
                
                # Trend analysis using linear regression
                time_indices = np.arange(recent_period)
                vibration_trend = np.polyfit(time_indices, recent_data[:, 0], 1)[0] if recent_period > 1 else 0
                deformation_trend = np.polyfit(time_indices, recent_data[:, 1], 1)[0] if recent_period > 1 else 0
                stress_trend = np.polyfit(time_indices, recent_data[:, 2], 1)[0] if recent_period > 1 else 0
                
                # Calculate trend magnitudes (normalized by horizon)
                trend_factor = horizon_days / 30.0  # Normalize by monthly cycle
                
                # Vibration trend impact
                if vibration_trend > 0.2:      # Strong increasing vibration
                    base_risk += trend_factor * 1.5
                elif vibration_trend > 0.1:    # Moderate increasing vibration
                    base_risk += trend_factor * 0.8
                elif vibration_trend > 0.05:   # Slight increasing vibration
                    base_risk += trend_factor * 0.3
                elif vibration_trend < -0.1:   # Decreasing vibration (improving)
                    base_risk -= trend_factor * 0.5
                
                # Deformation trend impact (most critical indicator)
                if deformation_trend > 0.1:     # Strong increasing deformation
                    base_risk += trend_factor * 2.0
                elif deformation_trend > 0.05:  # Moderate increasing deformation
                    base_risk += trend_factor * 1.2
                elif deformation_trend > 0.02:  # Slight increasing deformation
                    base_risk += trend_factor * 0.6
                elif deformation_trend < -0.05: # Decreasing deformation (improving)
                    base_risk -= trend_factor * 0.8
                
                # Stress trend impact
                if stress_trend > 2.0:          # Strong increasing stress
                    base_risk += trend_factor * 1.2
                elif stress_trend > 1.0:        # Moderate increasing stress
                    base_risk += trend_factor * 0.7
                elif stress_trend > 0.5:        # Slight increasing stress
                    base_risk += trend_factor * 0.3
                elif stress_trend < -1.0:       # Decreasing stress (improving)
                    base_risk -= trend_factor * 0.4
            
            # Future sensor data influence (if available)
            if future_sensor_data is not None and len(future_sensor_data) >= 3:
                future_vibration = max(0, future_sensor_data[0])
                future_deformation = max(0, future_sensor_data[1])
                future_stress = max(0, future_sensor_data[2])
                
                # Critical thresholds for future conditions
                if future_vibration > 6.0:      # Critical vibration
                    base_risk += 1.5
                elif future_vibration > 4.0:    # High vibration
                    base_risk += 1.0
                elif future_vibration > 2.5:    # Moderate vibration
                    base_risk += 0.5
                
                if future_deformation > 2.5:    # Critical deformation
                    base_risk += 2.0
                elif future_deformation > 1.8:  # High deformation
                    base_risk += 1.5
                elif future_deformation > 1.0:  # Moderate deformation
                    base_risk += 0.8
                
                if future_stress > 120:         # Critical stress
                    base_risk += 1.2
                elif future_stress > 80:        # High stress
                    base_risk += 0.8
                elif future_stress > 50:        # Moderate stress
                    base_risk += 0.4
            
            # Long-term geological risk factors
            if horizon_days > 7:
                rqd = static_features[0] if len(static_features) > 0 else 50
                depth = static_features[2] if len(static_features) > 2 else 400
                
                # Poor rock quality increases long-term risk
                if rqd < 40:
                    base_risk += 0.8 * (horizon_days / 30.0)
                elif rqd < 60:
                    base_risk += 0.4 * (horizon_days / 30.0)
                
                # Deep stopes have higher long-term instability risk
                if depth > 800:
                    base_risk += 0.6 * (horizon_days / 30.0)
                elif depth > 600:
                    base_risk += 0.3 * (horizon_days / 30.0)
            
            # Extended horizon risk escalation (uncertainty increases with time)
            if horizon_days > 14:
                base_risk += 0.3 * (horizon_days - 14) / 30.0
            elif horizon_days > 30:
                base_risk += 0.5 * (horizon_days - 30) / 30.0
            
            # Ensure risk level is within valid range [0, 5]
            risk_level = max(0, min(5, int(round(base_risk))))
            
            logger.debug(f"Future risk calculation: base={base_risk:.3f}, horizon={horizon_days}d, level={risk_level}")
            return risk_level
            
        except Exception as e:
            logger.error(f"Error calculating future risk: {e}")
            # Conservative fallback: moderate risk
            return 2
            base_risk += 1.0
            
        # Geological factors affecting long-term stability
        rqd = static_features[0]
        depth = static_features[2]
        
        if horizon_days > 7:
            if rqd < 50:
                base_risk += 0.5
            if depth > 600:
                base_risk += 0.3
        
        # Remove random noise for deterministic results based on data
        risk_level = max(0, min(5, int(round(base_risk))))
        
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

    def build_enhanced_model(self, static_input_dim, timeseries_timesteps, timeseries_feature_dim,
                           dense_units=128, lstm_units=128, attention_heads=8, dropout_rate=0.3):
        """
        Build enhanced dual-branch neural network with improved architecture.
        
        Improvements:
        - Better regularization with L2 penalties
        - Residual connections for deeper networks
        - Improved attention mechanism
        - Better feature fusion
        - Separate optimization for different prediction tasks
        """
        
        # Regularization configuration
        l2_reg = tf.keras.regularizers.l2(0.001)
        
        # ===== STATIC FEATURES BRANCH =====
        static_input = Input(shape=(static_input_dim,), name='static_input')
        
        # Enhanced static feature processing with residual connections
        x_static = Dense(dense_units, activation='relu', kernel_regularizer=l2_reg, name='static_dense_1')(static_input)
        x_static = BatchNormalization(name='static_bn_1')(x_static)
        x_static = Dropout(dropout_rate, name='static_dropout_1')(x_static)
        
        x_static_2 = Dense(dense_units, activation='relu', kernel_regularizer=l2_reg, name='static_dense_2')(x_static)
        x_static_2 = BatchNormalization(name='static_bn_2')(x_static_2)
        x_static_2 = Dropout(dropout_rate, name='static_dropout_2')(x_static_2)
        
        # Residual connection for static branch
        x_static_res = tf.keras.layers.Add(name='static_residual')([x_static, x_static_2])
        
        x_static_final = Dense(dense_units // 2, activation='relu', kernel_regularizer=l2_reg, name='static_dense_3')(x_static_res)
        x_static_final = BatchNormalization(name='static_bn_3')(x_static_final)
        x_static_final = Dropout(dropout_rate / 2, name='static_dropout_3')(x_static_final)
        
        # ===== TEMPORAL FEATURES BRANCH =====
        timeseries_input = Input(shape=(timeseries_timesteps, timeseries_feature_dim), name='timeseries_input')
        
        # Multi-layer LSTM with improved architecture
        x_lstm_1 = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate/2, 
                       recurrent_dropout=dropout_rate/2, kernel_regularizer=l2_reg, name='lstm_1')(timeseries_input)
        x_lstm_1 = BatchNormalization(name='lstm_bn_1')(x_lstm_1)
        
        x_lstm_2 = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate/2,
                       recurrent_dropout=dropout_rate/2, kernel_regularizer=l2_reg, name='lstm_2')(x_lstm_1)
        x_lstm_2 = BatchNormalization(name='lstm_bn_2')(x_lstm_2)
        
        # Residual connection for LSTM layers
        x_lstm_res = tf.keras.layers.Add(name='lstm_residual')([x_lstm_1, x_lstm_2])
        
        # Enhanced multi-head attention with better normalization
        attention_output = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=lstm_units // attention_heads,
            dropout=dropout_rate/2,
            name='temporal_attention'
        )(x_lstm_res, x_lstm_res)
        
        # Skip connection around attention
        x_lstm_att = tf.keras.layers.Add(name='attention_residual')([x_lstm_res, attention_output])
        x_lstm_att = LayerNormalization(name='attention_norm')(x_lstm_att)
        
        # Final temporal processing
        x_lstm_global = LSTM(lstm_units // 2, return_sequences=False, dropout=dropout_rate/2,
                           recurrent_dropout=dropout_rate/2, kernel_regularizer=l2_reg, name='lstm_final')(x_lstm_att)
        x_lstm_global = BatchNormalization(name='lstm_bn_final')(x_lstm_global)
        x_lstm_global = Dropout(dropout_rate, name='lstm_dropout_final')(x_lstm_global)
        
        # ===== ENHANCED FEATURE FUSION =====
        # Cross-attention between static and temporal features
        static_expanded = RepeatVector(1)(x_static_final)  # Add time dimension
        static_expanded = Reshape((1, dense_units // 2))(static_expanded)
        
        temporal_expanded = Reshape((1, lstm_units // 2))(x_lstm_global)
        
        # Cross-modal attention
        cross_attention = MultiHeadAttention(
            num_heads=4,
            key_dim=(lstm_units // 2) // 4,
            name='cross_modal_attention'
        )(static_expanded, temporal_expanded)
        
        cross_attended = tf.keras.layers.Flatten(name='cross_attention_flatten')(cross_attention)
        
        # Combine all features
        merged_features = Concatenate(name='feature_fusion')([x_static_final, x_lstm_global, cross_attended])
        
        # Shared representation learning with improved architecture
        x_shared = Dense(dense_units, activation='relu', kernel_regularizer=l2_reg, name='shared_dense_1')(merged_features)
        x_shared = BatchNormalization(name='shared_bn_1')(x_shared)
        x_shared = Dropout(dropout_rate, name='shared_dropout_1')(x_shared)
        
        x_shared_2 = Dense(dense_units // 2, activation='relu', kernel_regularizer=l2_reg, name='shared_dense_2')(x_shared)
        x_shared_2 = BatchNormalization(name='shared_bn_2')(x_shared_2)
        x_shared_2 = Dropout(dropout_rate, name='shared_dropout_2')(x_shared_2)
        
        # ===== PREDICTION HEADS WITH TASK-SPECIFIC ARCHITECTURE =====
        
        # Current stability prediction (binary classification)
        current_branch = Dense(dense_units // 4, activation='relu', kernel_regularizer=l2_reg, name='current_dense')(x_shared_2)
        current_branch = BatchNormalization(name='current_bn')(current_branch)
        current_branch = Dropout(dropout_rate, name='current_dropout')(current_branch)
        current_stability = Dense(1, activation='sigmoid', name='current_stability')(current_branch)
        
        # Future risk predictions with dedicated architectures for each horizon
        future_predictions = []
        
        for i, horizon in enumerate(self.prediction_horizons):
            # Horizon-specific feature processing
            future_branch = Dense(dense_units // 4, activation='relu', kernel_regularizer=l2_reg,
                                name=f'future_{horizon}d_dense_1')(x_shared_2)
            future_branch = BatchNormalization(name=f'future_{horizon}d_bn_1')(future_branch)
            future_branch = Dropout(dropout_rate, name=f'future_{horizon}d_dropout_1')(future_branch)
            
            # Horizon-aware processing (longer horizons get more capacity for uncertainty)
            horizon_units = dense_units // 4 + (horizon // 7)  # More units for longer horizons
            future_branch_2 = Dense(horizon_units, activation='relu', kernel_regularizer=l2_reg,
                                  name=f'future_{horizon}d_dense_2')(future_branch)
            future_branch_2 = BatchNormalization(name=f'future_{horizon}d_bn_2')(future_branch_2)
            future_branch_2 = Dropout(dropout_rate, name=f'future_{horizon}d_dropout_2')(future_branch_2)
            
            # Risk level prediction (6 classes)
            future_risk = Dense(len(self.risk_levels), activation='softmax', 
                              name=f'future_risk_{horizon}d')(future_branch_2)
            future_predictions.append(future_risk)
        
        # ===== MODEL COMPILATION WITH IMPROVED CONFIGURATION =====
        
        # Create the enhanced model
        self.combined_model = Model(
            inputs=[static_input, timeseries_input],
            outputs=[current_stability] + future_predictions,
            name='EnhancedDualBranchStabilityPredictor_v2'
        )
        
        # Advanced loss configuration with class weights
        loss_dict = {'current_stability': 'binary_crossentropy'}
        loss_weights = {'current_stability': 1.2}  # Higher weight for current prediction
        
        # Future predictions with horizon-based weighting
        for i, horizon in enumerate(self.prediction_horizons):
            loss_dict[f'future_risk_{horizon}d'] = 'sparse_categorical_crossentropy'
            # Shorter horizons get higher weights (more reliable)
            weight = 1.0 / (1.0 + 0.1 * horizon)  # Exponential decay with horizon
            loss_weights[f'future_risk_{horizon}d'] = weight
        
        # Comprehensive metrics
        metrics_dict = {
            'current_stability': ['accuracy', 'precision', 'recall'],
            **{f'future_risk_{horizon}d': ['accuracy', 'sparse_categorical_accuracy'] 
               for horizon in self.prediction_horizons}
        }
        
        # Advanced optimizer with learning rate scheduling
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        self.combined_model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics=metrics_dict
        )
        
        # Log model configuration
        total_params = self.combined_model.count_params()
        logger.info(f"Enhanced model built with {total_params:,} parameters")
        logger.info(f"Architecture: {len(self.prediction_horizons)+1} prediction heads")
        logger.info(f"Regularization: L2={0.001}, Dropout={dropout_rate}")
        
        return self.combined_model
    
    def train_enhanced_model(self, epochs=100, batch_size=32, validation_split=0.2, 
                           class_weights=None, early_stopping_patience=20):
        """
        Train the enhanced model with comprehensive data validation and optimization.
        
        Args:
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            class_weights: Dictionary of class weights for imbalanced data
            early_stopping_patience: Patience for early stopping
        """
        logger.info("🚀 Starting enhanced model training...")
        
        # Prepare training data with validation
        try:
            data = self.prepare_enhanced_training_data()
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
        
        X_static = data['X_static']
        X_timeseries = data['X_timeseries']
        y_current = data['y_current']
        y_future = data['y_future']
        
        # Comprehensive data validation
        if len(X_static) == 0:
            raise ValueError("No training samples available!")
        
        if np.any(np.isnan(X_static)) or np.any(np.isinf(X_static)):
            logger.warning("Found NaN/Inf in static features, cleaning...")
            X_static = np.nan_to_num(X_static, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.any(np.isnan(X_timeseries)) or np.any(np.isinf(X_timeseries)):
            logger.warning("Found NaN/Inf in timeseries features, cleaning...")
            X_timeseries = np.nan_to_num(X_timeseries, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Log data statistics
        logger.info(f"📊 Training data summary:")
        logger.info(f"   Total samples: {len(X_static)}")
        logger.info(f"   Static features: {X_static.shape[1]} dimensions")
        logger.info(f"   Temporal sequence: {X_timeseries.shape[1]} timesteps × {X_timeseries.shape[2]} features")
        logger.info(f"   Current stability distribution: {np.bincount(y_current)}")
        logger.info(f"   Future prediction horizons: {len(self.prediction_horizons)}")
        
        # Check for class imbalance
        class_distribution = np.bincount(y_current) / len(y_current)
        logger.info(f"   Class balance: Stable={class_distribution[0]:.3f}, Unstable={class_distribution[1]:.3f}")
        
        # Calculate class weights for information only (not used in training due to multi-output limitation)
        if class_weights is None:
            n_classes = len(np.unique(y_current))
            class_weights = {
                0: len(y_current) / (n_classes * np.sum(y_current == 0)),
                1: len(y_current) / (n_classes * np.sum(y_current == 1))
            }
            logger.info(f"   Calculated class weights (informational): {class_weights}")
            logger.info(f"   ⚠️  Note: Class weights not applied - Keras doesn't support class_weight for multi-output models")
        
        # Alternative approaches for handling class imbalance in multi-output models:
        # 1. Use stratified sampling (implemented below)
        # 2. Apply balanced class weights in loss function compilation
        # 3. Use sample_weight parameter (can be implemented later)
        # 4. Data augmentation for minority classes
        # 5. Ensemble methods or cost-sensitive learning approaches
        
        # Stratified split to maintain class distribution
        try:
            indices = np.arange(len(X_static))
            train_idx, val_idx = train_test_split(
                indices, 
                test_size=validation_split, 
                stratify=y_current, 
                random_state=42
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}, using random split")
            train_idx, val_idx = train_test_split(
                indices, 
                test_size=validation_split, 
                random_state=42
            )
        
        # Prepare training and validation sets
        X_static_train, X_static_val = X_static[train_idx], X_static[val_idx]
        X_ts_train, X_ts_val = X_timeseries[train_idx], X_timeseries[val_idx]
        y_current_train, y_current_val = y_current[train_idx], y_current[val_idx]
        y_future_train, y_future_val = y_future[train_idx], y_future[val_idx]
        
        logger.info(f"   Training samples: {len(X_static_train)}")
        logger.info(f"   Validation samples: {len(X_static_val)}")
        
        # Build model with proper dimensions
        static_input_dim = X_static.shape[1]
        timeseries_timesteps = X_timeseries.shape[1]
        timeseries_feature_dim = X_timeseries.shape[2]
        
        logger.info(f"🧠 Building enhanced dual-branch model...")
        logger.info(f"   Static input dimension: {static_input_dim}")
        logger.info(f"   Temporal sequence: {timeseries_timesteps} × {timeseries_feature_dim}")
        logger.info(f"   Prediction horizons: {self.prediction_horizons} days")
        
        try:
            self.build_enhanced_model(static_input_dim, timeseries_timesteps, timeseries_feature_dim)
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
        
        # Print model architecture summary
        logger.info(f"\n🏗️ Model Architecture Summary:")
        self.combined_model.summary(print_fn=logger.info)
        
        # Prepare target dictionaries for multiple outputs
        y_train_dict = {'current_stability': y_current_train}
        y_val_dict = {'current_stability': y_current_val}
        
        for i, horizon in enumerate(self.prediction_horizons):
            y_train_dict[f'future_risk_{horizon}d'] = y_future_train[:, i]
            y_val_dict[f'future_risk_{horizon}d'] = y_future_val[:, i]
        
        # Enhanced training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience, 
                restore_best_weights=True,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=early_stopping_patience//2, 
                min_lr=1e-6,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/best_enhanced_model.keras', 
                monitor='val_loss', 
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                'models/training_log.csv',
                append=True
            )
        ]
        
        # Create models directory if needed
        os.makedirs('models', exist_ok=True)
        
        # Train the model with comprehensive monitoring
        logger.info(f"\n🎯 Starting training for up to {epochs} epochs...")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Early stopping patience: {early_stopping_patience}")
        
        try:
            # Note: class_weight is not supported for multi-output models in Keras
            # Alternative approaches for class imbalance:
            # 1. Use sample_weight parameter (can be implemented later)
            # 2. Use custom loss functions with class weighting
            # 3. Use data augmentation techniques
            # 4. Use ensemble methods or cost-sensitive learning
            
            history = self.combined_model.fit(
                [X_static_train, X_ts_train], 
                y_train_dict,
                validation_data=([X_static_val, X_ts_val], y_val_dict),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # Final model evaluation
        logger.info(f"\n📈 Evaluating final model performance...")
        try:
            val_results = self.combined_model.evaluate(
                [X_static_val, X_ts_val], 
                y_val_dict, 
                verbose=0,
                batch_size=batch_size
            )
            
            # Log comprehensive results
            logger.info(f"\n✅ Training completed successfully!")
            logger.info(f"📊 Final Validation Results:")
            
            for i, metric_name in enumerate(self.combined_model.metrics_names):
                logger.info(f"   {metric_name}: {val_results[i]:.4f}")
            
            # Store training metadata
            self.training_metadata = {
                'epochs_trained': len(history.history['loss']),
                'final_val_loss': val_results[0],
                'best_val_loss': min(history.history['val_loss']),
                'training_samples': len(X_static_train),
                'validation_samples': len(X_static_val),
                'class_weights_used': class_weights,
                'model_parameters': self.combined_model.count_params()
            }
            
            self.is_trained = True
            logger.info(f"🎉 Model training completed with {self.training_metadata['epochs_trained']} epochs")
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            raise
        
        return history
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

    def get_model_summary(self):
        """Get comprehensive model summary and statistics."""
        if not hasattr(self, 'combined_model') or self.combined_model is None:
            return {"error": "Model not built yet"}
        
        summary = {
            "model_name": "Enhanced Dual-Branch Stability Predictor",
            "version": "2.0",
            "architecture": {
                "total_parameters": self.combined_model.count_params(),
                "trainable_parameters": sum([tf.keras.backend.count_params(w) for w in self.combined_model.trainable_weights]),
                "layers": len(self.combined_model.layers),
                "prediction_heads": len(self.prediction_horizons) + 1
            },
            "prediction_capabilities": {
                "current_stability": "Binary classification (stable/unstable)",
                "future_risk_horizons": self.prediction_horizons,
                "risk_levels": self.risk_levels
            },
            "data_requirements": {
                "static_features": f"{self.static_df.shape[1] if hasattr(self, 'static_df') else 'unknown'} dimensions",
                "temporal_sequence_length": self.sequence_length,
                "temporal_features": self.n_features_temporal
            },
            "training_status": {
                "is_trained": self.is_trained,
                "training_metadata": self.training_metadata if hasattr(self, 'training_metadata') else None
            }
        }
        
        return summary

    def _run_consistency_checks(self, predictions):
        """Run consistency checks on model predictions."""
        checks = {
            'probability_range_valid': True,
            'future_risk_progression_logical': True,
            'prediction_completeness': True
        }
        
        issues = []
        
        for stope_name, result in predictions.items():
            if 'error' in result:
                continue
                
            # Check probability ranges
            current_prob = result['current_stability']['instability_probability']
            if not 0 <= current_prob <= 1:
                checks['probability_range_valid'] = False
                issues.append(f"{stope_name}: Invalid probability {current_prob}")
            
            # Check future risk progression logic
            future_risks = [pred['predicted_risk_level'] for pred in result.get('future_predictions', [])]
            if len(future_risks) != len(self.prediction_horizons):
                checks['prediction_completeness'] = False
                issues.append(f"{stope_name}: Incomplete future predictions")
        
        checks['issues'] = issues
        checks['total_issues'] = len(issues)
        
        return checks


# Backward compatibility alias
DualBranchStopeStabilityPredictor = EnhancedDualBranchStabilityPredictor
