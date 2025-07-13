import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def profile_stope(features):
    """
    Generate a detailed profile of an underground mining stope
    based on static geological and design features.
    
    :param features: dict with keys matching stope feature names
    :return: multi-line string with stope profile
    """
    profile = []

    # --- Rock Quality Assessment ---
    rqd = features['rqd']
    if rqd >= 85:
        rqd_desc = "excellent rock quality"
    elif rqd >= 70:
        rqd_desc = "good rock quality"
    elif rqd >= 50:
        rqd_desc = "moderate rock quality"
    else:
        rqd_desc = "poor and fractured rock"

    profile.append(f"Rock Quality: The stope has {rqd_desc} (RQD = {rqd}%).")

    # --- Depth Assessment ---
    depth = features['depth']
    if depth < 300:
        depth_desc = "shallow"
    elif depth < 700:
        depth_desc = "moderately deep"
    else:
        depth_desc = "deep"

    profile.append(f"Depth: This is a {depth_desc} stope located at {depth} m below surface.")

    # --- Dip and Direction ---
    dip = features['dip']
    direction = features['direction']
    if dip < 40:
        dip_desc = "gently dipping"
    elif dip < 70:
        dip_desc = "moderately inclined"
    else:
        dip_desc = "steeply dipping"
    
    profile.append(f"Geometry: The stope is {dip_desc} at {dip}°, facing {direction}.")

    # --- Hydraulic Radius ---
    hr = features['hr']
    if hr < 5:
        hr_desc = "narrow span opening"
    elif hr < 9:
        hr_desc = "moderate hydraulic radius"
    else:
        hr_desc = "large span opening"

    profile.append(f"Shape: The hydraulic radius (HR) is {hr}, indicating a {hr_desc}.")

    # --- Undercut Width ---
    uw = features['Undercut_wdt']
    if uw <= 3:
        uw_desc = "narrow undercut width"
    elif uw <= 7:
        uw_desc = "moderately wide undercut"
    else:
        uw_desc = "wide undercut"

    profile.append(f"Undercut: An {uw_desc} of {uw} m.")

    # --- Rock Type ---
    rock_type = features['rock_type']
    profile.append(f"Lithology: Dominant rock type is {rock_type}.")

    # --- Support System Analysis ---
    support_type = features['support_type']
    support_density = features['support_density']
    support_installed = features['support_installed']

    if not support_installed or support_type.lower() == "none":
        support_desc = "No engineered support has been installed."
        support_level = "none"
    else:
        if support_density < 0.4:
            density_desc = "light support"
            support_level = "light"
        elif support_density < 1.0:
            density_desc = "moderate support"
            support_level = "moderate"
        else:
            density_desc = "dense support"
            support_level = "heavy"

        support_desc = (
            f"{density_desc.capitalize()} has been installed using {support_type} "
            f"(density = {support_density})."
        )
    
    profile.append(f"Support: {support_desc}")

    # --- Stability Inference (Rule-Based) ---
    # Combine RQD, HR, depth, and support to infer a risk label
    if rqd < 50 and hr > 9 and not support_installed:
        risk = "Poor rock and large opening with no support."
    elif rqd < 65 and hr > 8 and support_level in ['none', 'light']:
        risk = "Weak to moderate rock and medium-to-large span, limited support."
    elif rqd >= 70 and support_level in ['moderate', 'heavy']:
        risk = "Good ground with sufficient support."
    else:
        risk = "Conditions acceptable but site-specific review recommended."

    profile.append(f"Stability Insight: {risk}")

    # --- Final Structured Output ---
    return "\n".join(profile)


def get_stope_profile_summary(stope_name):
    """
    Generate a numerical summary vector from stope profile features.
    This function extracts key numerical features that can be used
    as additional inputs to machine learning models.
    
    :param stope_name: string, name of the stope
    :return: list of numerical features derived from profile analysis
    """
    try:
        # In a production system, this would fetch actual stope data
        # For now, we'll use synthetic features based on the stope name
        
        # Generate pseudo-random but consistent features based on stope name
        import hashlib
        hash_val = int(hashlib.md5(stope_name.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val % 1000)  # Ensure reproducible "random" values
        
        # Generate realistic profile summary features
        profile_summary = [
            np.random.uniform(0.4, 0.9),   # stability_score (0-1)
            np.random.uniform(0.1, 0.8),   # risk_level (0-1)
            np.random.uniform(0.3, 0.9),   # complexity_index (0-1)
            np.random.uniform(0.5, 1.0),   # support_adequacy (0-1)
            np.random.uniform(0.2, 0.7),   # geological_factor (0-1)
            np.random.uniform(0.3, 0.8),   # structural_factor (0-1)
            np.random.uniform(0.1, 0.6),   # environmental_factor (0-1)
            np.random.uniform(0.4, 0.9),   # maintenance_factor (0-1)
            np.random.uniform(0.2, 0.8),   # historical_performance (0-1)
            np.random.uniform(0.3, 0.7),   # monitoring_coverage (0-1)
        ]
        
        return profile_summary
        
    except Exception as e:
        # Return default profile summary if there's an error
        return [0.5, 0.3, 0.6, 0.7, 0.4, 0.5, 0.3, 0.6, 0.5, 0.4]


def get_stope_features_for_ml(stope_name, static_df):
    """
    Extract and preprocess stope features for machine learning model input.
    
    :param stope_name: string, name of the stope
    :param static_df: pandas DataFrame with static stope features
    :return: dict with processed features
    """
    try:
        row = static_df[static_df['stope_name'] == stope_name].iloc[0]
        
        # Map categorical features to numerical
        direction_map = {
            'North': 0, 'Northeast': 1, 'East': 2, 'Southeast': 3,
            'South': 4, 'Southwest': 5, 'West': 6, 'Northwest': 7
        }
        
        rock_type_map = {
            'Granite': 0, 'Basalt': 1, 'Quartzite': 2, 'Schist': 3,
            'Gneiss': 4, 'Marble': 5, 'Slate': 6, 'Shale': 7,
            'Limestone': 8, 'Sandstone': 9, 'Obsidian': 10
        }
        
        support_type_map = {
            'None': 0, 'Rock Bolts': 1, 'Mesh': 2, 'Steel Sets': 3,
            'Shotcrete': 4, 'Timber': 5, 'Cable Bolts': 6
        }
        
        features = {
            'rqd': float(row['rqd']),
            'hr': float(row['hr']),
            'depth': float(row['depth']),
            'dip': float(row['dip']),
            'direction': direction_map.get(row['direction'], 0),
            'Undercut_wdt': float(row['undercut_wdt']),  # Note: key matches utils function
            'rock_type': rock_type_map.get(row['rock_type'], 0),
            'support_type': support_type_map.get(row['support_type'], 0),
            'support_density': float(row['support_density']),
            'support_installed': int(row['support_installed'])
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting features for stope {stope_name}: {e}")
        return None

# ============================================================================
# Enhanced Temporal Prediction System
# ============================================================================

class TemporalStabilityPredictor:
    """
    Advanced temporal stability prediction system that uses time series data
    to predict future stope stability risks.
    """
    
    def __init__(self, model_path='models/', model_version='v2.0'):
        self.model_path = model_path
        self.model_version = model_version
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.time_window = 30  # Days of historical data to consider
        self.prediction_horizons = [1, 3, 7, 14, 30]  # Days ahead to predict
        
        # Risk level mappings
        self.risk_levels = ['stable', 'slight_elevated', 'elevated', 'high', 'critical', 'unstable']
        self.risk_to_numeric = {level: idx for idx, level in enumerate(self.risk_levels)}
        self.numeric_to_risk = {idx: level for idx, level in enumerate(self.risk_levels)}
        
        # Load or initialize model
        self._load_or_initialize_model()
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize new one"""
        try:
            model_file = os.path.join(self.model_path, f'temporal_model_{self.model_version}.h5')
            scaler_file = os.path.join(self.model_path, f'temporal_scaler_{self.model_version}.pkl')
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.model = tf.keras.models.load_model(model_file)
                self.scaler = joblib.load(scaler_file)
                logger.info(f"Loaded temporal model {self.model_version}")
            else:
                logger.info("Creating new temporal model")
                self._create_temporal_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_temporal_model()
    
    def _create_temporal_model(self):
        """Create new temporal prediction model"""
        # LSTM-based model for time series prediction
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(self.time_window, 15)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.prediction_horizons) * len(self.risk_levels), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
    
    def prepare_temporal_features(self, stope_data: Dict, timeseries_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare temporal features combining static stope data with time series
        """
        try:
            # Static features (repeated for each time step)
            static_features = [
                stope_data.get('rqd', 0),
                stope_data.get('hr', 0),
                stope_data.get('depth', 0),
                stope_data.get('dip', 0),
                stope_data.get('direction', 0),
                stope_data.get('Undercut_wdt', 0),
                stope_data.get('rock_type', 0),
                stope_data.get('support_type', 0),
                stope_data.get('support_density', 0),
                stope_data.get('support_installed', 0)
            ]
            
            # Prepare time series features
            if timeseries_data is not None and not timeseries_data.empty:
                # Sort by timestamp
                timeseries_data = timeseries_data.sort_values('timestamp')
                
                # Take last time_window days
                recent_data = timeseries_data.tail(self.time_window)
                
                # Temporal features for each time step
                temporal_features = []
                for _, row in recent_data.iterrows():
                    step_features = static_features + [
                        row.get('vibration_velocity', 0) or 0,
                        row.get('deformation_rate', 0) or 0,
                        row.get('stress', 0) or 0,
                        row.get('temperature', 0) or 0,
                        row.get('humidity', 0) or 0,
                    ]
                    temporal_features.append(step_features)
                
                # Pad or truncate to time_window
                while len(temporal_features) < self.time_window:
                    temporal_features.insert(0, static_features + [0, 0, 0, 0, 0])
                temporal_features = temporal_features[-self.time_window:]
                
            else:
                # No time series data, use static features only
                temporal_features = [static_features + [0, 0, 0, 0, 0]] * self.time_window
            
            return np.array(temporal_features)
            
        except Exception as e:
            logger.error(f"Error preparing temporal features: {e}")
            # Return default features
            default_features = [0] * 15
            return np.array([default_features] * self.time_window)
    
    def predict_future_stability(self, stope_data: Dict, timeseries_data: pd.DataFrame) -> Dict:
        """
        Predict future stability for multiple time horizons
        """
        try:
            # Prepare features
            features = self.prepare_temporal_features(stope_data, timeseries_data)
            features = features.reshape(1, self.time_window, -1)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                features_2d = features.reshape(-1, features.shape[-1])
                features_scaled = self.scaler.transform(features_2d)
                features = features_scaled.reshape(1, self.time_window, -1)
            
            # Make prediction
            prediction = self.model.predict(features, verbose=0)
            
            # Reshape prediction to separate horizons
            prediction = prediction.reshape(len(self.prediction_horizons), len(self.risk_levels))
            
            # Process predictions for each horizon
            predictions = {}
            current_date = datetime.now()
            
            for i, days_ahead in enumerate(self.prediction_horizons):
                horizon_pred = prediction[i]
                
                # Get most likely risk level
                risk_idx = np.argmax(horizon_pred)
                risk_level = self.numeric_to_risk[risk_idx]
                confidence = float(horizon_pred[risk_idx])
                
                # Calculate prediction date
                prediction_date = current_date + timedelta(days=days_ahead)
                
                # Determine prediction type
                if days_ahead <= 3:
                    pred_type = 'short_term'
                elif days_ahead <= 14:
                    pred_type = 'medium_term'
                else:
                    pred_type = 'long_term'
                
                # Generate explanation based on contributing factors
                explanation = self._generate_temporal_explanation(
                    risk_level, days_ahead, stope_data, timeseries_data
                )
                
                # Generate recommendations
                recommendations = self._generate_recommendations(risk_level, days_ahead)
                
                predictions[f"{days_ahead}_days"] = {
                    'prediction_type': pred_type,
                    'days_ahead': days_ahead,
                    'prediction_for_date': prediction_date,
                    'risk_level': risk_level,
                    'confidence_score': confidence,
                    'risk_probability': float(np.sum(horizon_pred[3:])),  # High risk and above
                    'all_probabilities': {
                        level: float(horizon_pred[idx]) 
                        for idx, level in enumerate(self.risk_levels)
                    },
                    'explanation': explanation,
                    'recommended_actions': recommendations,
                    'contributing_factors': self._identify_contributing_factors(
                        stope_data, timeseries_data, risk_level
                    )
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in future stability prediction: {e}")
            return self._get_default_predictions()
    
    def _generate_temporal_explanation(self, risk_level: str, days_ahead: int, 
                                     stope_data: Dict, timeseries_data: pd.DataFrame) -> str:
        """Generate human-readable explanation for temporal prediction"""
        
        explanations = {
            'stable': f"Based on current conditions and trend analysis, the stope is projected to remain stable over the next {days_ahead} days. Monitoring parameters show normal patterns.",
            
            'slight_elevated': f"Analysis indicates a slight increase in stability risk over the next {days_ahead} days. While conditions remain generally acceptable, enhanced monitoring is recommended.",
            
            'elevated': f"Predictive models suggest elevated stability concerns developing over the next {days_ahead} days. Multiple risk factors are showing trending patterns that warrant attention.",
            
            'high': f"High stability risk is projected for {days_ahead} days ahead. Current trends in monitoring data suggest deteriorating conditions that require immediate intervention planning.",
            
            'critical': f"Critical stability conditions are forecasted within {days_ahead} days. Multiple indicators show alarming trends that demand immediate action and enhanced safety measures.",
            
            'unstable': f"Unstable conditions are predicted within {days_ahead} days. The combination of geological factors and monitoring trends indicates imminent stability failure risk."
        }
        
        base_explanation = explanations.get(risk_level, "Risk assessment based on current data trends.")
        
        # Add specific factors if available
        if timeseries_data is not None and not timeseries_data.empty:
            recent_data = timeseries_data.tail(7)  # Last week
            
            trend_info = []
            if 'vibration_velocity' in recent_data.columns:
                vibration_trend = recent_data['vibration_velocity'].diff().mean()
                if vibration_trend > 0.1:
                    trend_info.append("increasing vibration levels")
                elif vibration_trend < -0.1:
                    trend_info.append("decreasing vibration levels")
            
            if 'deformation_rate' in recent_data.columns:
                deformation_trend = recent_data['deformation_rate'].diff().mean()
                if deformation_trend > 0.05:
                    trend_info.append("accelerating deformation")
                elif deformation_trend < -0.05:
                    trend_info.append("stabilizing deformation")
            
            if trend_info:
                base_explanation += f" Key trends observed: {', '.join(trend_info)}."
        
        return base_explanation
    
    def _generate_recommendations(self, risk_level: str, days_ahead: int) -> str:
        """Generate specific recommendations based on risk level and timeframe"""
        
        recommendations = {
            'stable': [
                "Continue routine monitoring schedule",
                "Maintain current safety protocols",
                "Schedule regular visual inspections"
            ],
            'slight_elevated': [
                "Increase monitoring frequency by 25%",
                "Review recent operational changes",
                "Prepare contingency plans",
                "Brief operational teams on status"
            ],
            'elevated': [
                "Increase monitoring frequency by 50%",
                "Conduct detailed geological assessment",
                "Review and update safety procedures",
                "Consider additional support installation",
                "Restrict non-essential access"
            ],
            'high': [
                "Implement daily monitoring protocols",
                "Evacuate non-essential personnel",
                "Install additional monitoring equipment",
                "Prepare immediate support reinforcement",
                "Activate emergency response protocols"
            ],
            'critical': [
                "Implement continuous monitoring",
                "Evacuate all non-critical personnel immediately",
                "Deploy emergency support systems",
                "Activate full emergency response",
                "Suspend all non-emergency operations"
            ],
            'unstable': [
                "IMMEDIATE EVACUATION of all personnel",
                "Deploy emergency stabilization measures",
                "Activate full emergency response",
                "Suspend ALL operations in affected areas",
                "Implement continuous remote monitoring"
            ]
        }
        
        base_recs = recommendations.get(risk_level, ["Monitor conditions closely"])
        
        # Add time-specific recommendations
        if days_ahead <= 1:
            base_recs.insert(0, "URGENT: Implement recommendations within 24 hours")
        elif days_ahead <= 3:
            base_recs.insert(0, "Implement recommendations within 72 hours")
        elif days_ahead <= 7:
            base_recs.insert(0, "Plan implementation within one week")
        
        return "; ".join(base_recs)
    
    def _identify_contributing_factors(self, stope_data: Dict, timeseries_data: pd.DataFrame, 
                                     risk_level: str) -> Dict:
        """Identify factors contributing to the risk prediction"""
        
        factors = {
            'geological': [],
            'operational': [],
            'environmental': [],
            'temporal': []
        }
        
        # Geological factors
        rqd = stope_data.get('rqd', 0)
        if rqd < 50:
            factors['geological'].append(f"Poor rock quality (RQD: {rqd}%)")
        elif rqd < 70:
            factors['geological'].append(f"Moderate rock quality (RQD: {rqd}%)")
        
        hr = stope_data.get('hr', 0)
        if hr > 8:
            factors['geological'].append(f"Large span opening (HR: {hr})")
        
        depth = stope_data.get('depth', 0)
        if depth > 500:
            factors['geological'].append(f"Deep location ({depth}m)")
        
        # Support factors
        support_installed = stope_data.get('support_installed', 0)
        if not support_installed:
            factors['operational'].append("No engineered support installed")
        else:
            support_density = stope_data.get('support_density', 0)
            if support_density < 0.5:
                factors['operational'].append("Insufficient support density")
        
        # Time series factors
        if timeseries_data is not None and not timeseries_data.empty:
            recent_data = timeseries_data.tail(7)
            
            if 'vibration_velocity' in recent_data.columns:
                avg_vibration = recent_data['vibration_velocity'].mean()
                if avg_vibration > 10:
                    factors['temporal'].append(f"High vibration levels ({avg_vibration:.1f} mm/s)")
            
            if 'deformation_rate' in recent_data.columns:
                avg_deformation = recent_data['deformation_rate'].mean()
                if avg_deformation > 5:
                    factors['temporal'].append(f"High deformation rate ({avg_deformation:.1f} mm/day)")
            
            if 'stress' in recent_data.columns:
                avg_stress = recent_data['stress'].mean()
                if avg_stress > 50:
                    factors['temporal'].append(f"Elevated stress levels ({avg_stress:.1f} MPa)")
        
        return factors
    
    def _get_default_predictions(self) -> Dict:
        """Return default predictions when model fails"""
        current_date = datetime.now()
        
        predictions = {}
        for days_ahead in self.prediction_horizons:
            prediction_date = current_date + timedelta(days=days_ahead)
            
            if days_ahead <= 3:
                pred_type = 'short_term'
            elif days_ahead <= 14:
                pred_type = 'medium_term'
            else:
                pred_type = 'long_term'
            
            predictions[f"{days_ahead}_days"] = {
                'prediction_type': pred_type,
                'days_ahead': days_ahead,
                'prediction_for_date': prediction_date,
                'risk_level': 'stable',
                'confidence_score': 0.5,
                'risk_probability': 0.1,
                'explanation': "Default prediction due to insufficient data or model error.",
                'recommended_actions': "Collect more monitoring data and review model performance.",
                'contributing_factors': {'system': ['Insufficient data for reliable prediction']}
            }
        
        return predictions

# Backward compatibility function
def predict_stability_with_neural_network(stope_data: Dict, timeseries_data: pd.DataFrame = None) -> Dict:
    """
    Enhanced stability prediction using the trained dual-branch neural network
    """
    try:
        # Import the enhanced model
        from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor
        
        # Initialize model with default paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        static_features_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
        timeseries_path = os.path.join(project_root, 'data', 'stope_timeseries_data_aligned.csv')
        
        if not os.path.exists(static_features_path) or not os.path.exists(timeseries_path):
            # Fallback to temporal predictor
            predictor = TemporalStabilityPredictor()
            future_predictions = predictor.predict_future_stability(stope_data, timeseries_data)
            current_prediction = get_stability_prediction_simple(stope_data)
            
            return {
                'current': current_prediction,
                'future': future_predictions,
                'model_type': 'Temporal LSTM (Fallback)',
                'prediction_timestamp': datetime.now().isoformat()
            }
        
        # Try to load trained enhanced model
        model = EnhancedDualBranchStabilityPredictor(static_features_path, timeseries_path)
        
        # Load the trained model if available
        enhanced_model_path = 'models/enhanced_dual_branch_model.h5'
        if os.path.exists(enhanced_model_path):
            model.load_enhanced_model(enhanced_model_path)
            
            # Get stope name from data (assuming it's provided)
            stope_name = stope_data.get('stope_name', 'Unknown')
            
            # Make comprehensive prediction
            prediction_result = model.predict_comprehensive_stability(stope_name)
            
            if prediction_result:
                return {
                    'current': prediction_result['current_stability'],
                    'future': prediction_result['future_predictions'],
                    'risk_trend': prediction_result['risk_trend'],
                    'explanations': prediction_result['explanations'],
                    'recommendations': prediction_result['recommendations'],
                    'model_type': 'Dual-Branch Neural Network',
                    'model_version': 'Enhanced v2.0',
                    'prediction_timestamp': prediction_result['timestamp']
                }
        
        # If no trained model available, fallback to simpler methods
        predictor = TemporalStabilityPredictor()
        future_predictions = predictor.predict_future_stability(stope_data, timeseries_data)
        current_prediction = get_stability_prediction_simple(stope_data)
        
        return {
            'current': current_prediction,
            'future': future_predictions,
            'model_type': 'Temporal LSTM + Rule-based (Model not trained)',
            'prediction_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced neural network prediction: {e}")
        
        # Ultimate fallback
        current_prediction = get_stability_prediction_simple(stope_data)
        return {
            'current': current_prediction,
            'future': {'error': str(e)},
            'model_type': 'Rule-based (Fallback)',
            'prediction_timestamp': datetime.now().isoformat()
        }

def get_stability_prediction_simple(stope_data: Dict) -> Dict:
    """
    Simple current stability prediction for backward compatibility
    """
    try:
        # Try to use simple trained model first
        simple_model_path = 'models/stope_stability_model.h5'
        if os.path.exists(simple_model_path):
            model = tf.keras.models.load_model(simple_model_path)
            scaler_path = 'models/scaler.pkl'
            
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                
                # Prepare features
                feature_order = ['rqd', 'hr', 'depth', 'dip', 'direction', 'undercut_wdt', 
                               'rock_type', 'support_type', 'support_density', 'support_installed']
                
                features = np.array([[stope_data.get(key, 0) for key in feature_order]])
                features_scaled = scaler.transform(features)
                
                # Make prediction
                prediction = model.predict(features_scaled, verbose=0)
                
                if len(prediction[0]) > 1:  # Multi-class
                    risk_idx = np.argmax(prediction[0])
                    risk_levels = ['stable', 'slight_elevated', 'elevated', 'high', 'critical', 'unstable']
                    risk_level = risk_levels[min(risk_idx, len(risk_levels)-1)]
                    confidence = float(prediction[0][risk_idx])
                else:  # Binary
                    prob = float(prediction[0][0])
                    risk_level = 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
                    confidence = abs(prob - 0.5) * 2  # Distance from decision boundary
                
                return {
                    'risk_level': risk_level,
                    'confidence_score': confidence,
                    'stable': risk_level in ['stable', 'Low'],
                    'instability_probability': prob if len(prediction[0]) == 1 else 1 - prediction[0][0],
                    'explanation': f"Current stability assessment using trained neural network.",
                    'model_type': 'Neural Network'
                }
        
        # Fallback to rule-based prediction
            return get_rule_based_prediction(stope_data)
            
    except Exception as e:
        logger.error(f"Error in neural network prediction: {e}")
        return get_rule_based_prediction(stope_data)

def get_rule_based_prediction(stope_data: Dict) -> Dict:
    """
    Rule-based prediction fallback
    """
    rqd = stope_data.get('rqd', 0)
    hr = stope_data.get('hr', 0)
    support_installed = stope_data.get('support_installed', 0)
    support_density = stope_data.get('support_density', 0)
    
    # Simple rule-based logic
    if rqd < 50 and hr > 8 and not support_installed:
        risk_level = 'high'
        confidence = 0.8
    elif rqd < 65 and hr > 6:
        risk_level = 'elevated'
        confidence = 0.7
    elif support_installed and support_density > 0.8:
        risk_level = 'stable'
        confidence = 0.75
    else:
        risk_level = 'slight_elevated'
        confidence = 0.6
    
    return {
        'risk_level': risk_level,
        'confidence_score': confidence,
        'explanation': f"Rule-based assessment: RQD={rqd}%, HR={hr}, Support={'Yes' if support_installed else 'No'}",
        'model_type': 'Rule-Based Fallback'
    }