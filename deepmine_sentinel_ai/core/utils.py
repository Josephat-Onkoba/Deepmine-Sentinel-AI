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
    This function extracts key numerical features from real stope data
    that can be used as additional inputs to machine learning models.
    
    :param stope_name: string, name of the stope
    :return: list of numerical features derived from profile analysis
    """
    try:
        # Load the static features data
        project_root = os.path.dirname(os.path.dirname(__file__))  # Go up to deepmine_sentinel_ai directory
        static_features_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
        
        if not os.path.exists(static_features_path):
            logger.warning(f"Static features file not found: {static_features_path}")
            return [0.5, 0.3, 0.6, 0.7, 0.4, 0.5, 0.3, 0.6, 0.5, 0.4]
        
        # Read the CSV file
        static_df = pd.read_csv(static_features_path)
        
        # Find the stope data
        stope_data = static_df[static_df['stope_name'] == stope_name]
        if stope_data.empty:
            logger.warning(f"Stope {stope_name} not found in static features data")
            return [0.5, 0.3, 0.6, 0.7, 0.4, 0.5, 0.3, 0.6, 0.5, 0.4]
        
        row = stope_data.iloc[0]
        
        # Calculate meaningful numerical features based on real data
        
        # 1. Stability Score (0-1): Based on RQD and actual stability label
        rqd = float(row['rqd'])
        actual_stability = int(row['stability']) if 'stability' in row else 0.5
        stability_score = min(1.0, max(0.0, (rqd / 100.0) * 0.7 + actual_stability * 0.3))
        
        # 2. Risk Level (0-1): Inverse of stability, considering multiple factors
        hr = float(row['hr'])
        depth = float(row['depth'])
        support_installed = int(row['support_installed'])
        
        # Higher HR and depth increase risk, support reduces it
        risk_components = [
            1.0 - (rqd / 100.0),  # Poor RQD increases risk
            min(1.0, hr / 15.0),  # Large HR increases risk (normalized to max 15)
            min(1.0, depth / 1000.0),  # Deep stopes increase risk (normalized to max 1000m)
            0.0 if support_installed else 0.3  # No support adds significant risk
        ]
        risk_level = min(1.0, np.mean(risk_components))
        
        # 3. Complexity Index (0-1): Based on geometric and geological complexity
        dip = float(row['dip'])
        undercut_wdt = float(row['undercut_wdt'])
        
        # Steep dips and wide undercuts increase complexity
        complexity_index = min(1.0, (dip / 90.0) * 0.5 + min(1.0, undercut_wdt / 10.0) * 0.3 + (hr / 15.0) * 0.2)
        
        # 4. Support Adequacy (0-1): Based on support type, density, and installation
        support_density = float(row['support_density'])
        support_type = str(row['support_type'])
        
        # Support type strength mapping
        support_strength = {
            'None': 0.0, 'Mesh': 0.3, 'Rock Bolts': 0.6, 'Shotcrete': 0.7,
            'Steel Sets': 0.8, 'Cable Bolts': 0.9, 'Timber': 0.4
        }
        
        type_factor = support_strength.get(support_type, 0.5)
        support_adequacy = min(1.0, (type_factor * 0.6 + support_density * 0.4) if support_installed else 0.1)
        
        # 5. Geological Factor (0-1): Based on rock type and RQD
        rock_type = str(row['rock_type'])
        
        # Rock strength mapping (higher values = stronger rock)
        rock_strength = {
            'Granite': 0.9, 'Basalt': 0.8, 'Quartzite': 0.85, 'Schist': 0.6,
            'Gneiss': 0.7, 'Marble': 0.75, 'Slate': 0.65, 'Shale': 0.4,
            'Limestone': 0.7, 'Sandstone': 0.6, 'Obsidian': 0.8
        }
        
        rock_factor = rock_strength.get(rock_type, 0.6)
        geological_factor = min(1.0, (rock_factor * 0.6 + (rqd / 100.0) * 0.4))
        
        # 6. Structural Factor (0-1): Based on dip, direction, and hydraulic radius
        direction = str(row['direction'])
        
        # Direction stability factor (some orientations are more stable)
        direction_stability = {
            'North': 0.8, 'Northeast': 0.7, 'East': 0.75, 'Southeast': 0.7,
            'South': 0.8, 'Southwest': 0.6, 'West': 0.75, 'Northwest': 0.65
        }
        
        dir_factor = direction_stability.get(direction, 0.7)
        # Moderate dips (45-75°) are often less stable
        dip_factor = 1.0 - abs(dip - 45) / 90.0 if 30 <= dip <= 80 else 0.8
        structural_factor = min(1.0, (dir_factor * 0.4 + dip_factor * 0.3 + (1.0 - hr/15.0) * 0.3))
        
        # 7. Environmental Factor (0-1): Depth-based (deeper = more challenging environment)
        environmental_factor = max(0.1, 1.0 - (depth / 1000.0))
        
        # 8. Maintenance Factor (0-1): Based on support adequacy and accessibility
        # Shallower stopes with good support are easier to maintain
        maintenance_factor = min(1.0, (environmental_factor * 0.4 + support_adequacy * 0.6))
        
        # 9. Historical Performance (0-1): Based on current stability and geological factors
        historical_performance = min(1.0, (stability_score * 0.6 + geological_factor * 0.4))
        
        # 10. Monitoring Coverage (0-1): Estimated based on depth and complexity
        # Assume better monitoring for less complex, shallower stopes
        monitoring_coverage = max(0.3, min(1.0, environmental_factor * 0.6 + (1.0 - complexity_index) * 0.4))
        
        profile_summary = [
            float(stability_score),        # 0: stability_score (0-1)
            float(risk_level),             # 1: risk_level (0-1)
            float(complexity_index),       # 2: complexity_index (0-1)
            float(support_adequacy),       # 3: support_adequacy (0-1)
            float(geological_factor),      # 4: geological_factor (0-1)
            float(structural_factor),      # 5: structural_factor (0-1)
            float(environmental_factor),   # 6: environmental_factor (0-1)
            float(maintenance_factor),     # 7: maintenance_factor (0-1)
            float(historical_performance), # 8: historical_performance (0-1)
            float(monitoring_coverage)     # 9: monitoring_coverage (0-1)
        ]
        
        return profile_summary
        
    except Exception as e:
        logger.error(f"Error calculating profile summary for {stope_name}: {e}")
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

# Backward compatibility function
def predict_stability_with_neural_network(stope_data: Dict, timeseries_data: pd.DataFrame = None) -> Dict:
    """
    Enhanced stability prediction using the trained dual-branch neural network
    """
    try:
        # Import the enhanced model
        from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor
        
        # Initialize model with default paths
        project_root = os.path.dirname(os.path.dirname(__file__))  # Go up to deepmine_sentinel_ai directory
        static_features_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
        timeseries_path = os.path.join(project_root, 'data', 'stope_timeseries_data_aligned.csv')
        
        if not os.path.exists(static_features_path) or not os.path.exists(timeseries_path):
            # Return error if data files are missing
            return {
                'current': {
                    'error': 'Training data files not found. Neural network model requires aligned CSV data files.',
                    'model_type': 'Neural Network (Data Missing)'
                },
                'future': {
                    'error': 'Training data files not found. Neural network model requires aligned CSV data files.'
                },
                'model_type': 'Neural Network (Data Missing)',
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
        
        # If no trained model available, return error
        return {
            'current': {
                'error': 'Neural network model not available. Please ensure the dual-branch stability model is trained.',
                'model_type': 'Neural Network (Not Available)'
            },
            'future': {
                'error': 'Neural network model not available. Please ensure the dual-branch stability model is trained.'
            },
            'model_type': 'Neural Network (Model not trained)',
            'prediction_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced neural network prediction: {e}")
        
        # Return error for neural network failures
        return {
            'current': {
                'error': f'Neural network prediction failed: {str(e)}',
                'model_type': 'Neural Network (Error)'
            },
            'future': {
                'error': f'Neural network prediction failed: {str(e)}'
            },
            'model_type': 'Neural Network (Fallback Error)',
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
        
        # If model is not available, return error
        return {
            'error': 'Neural network model not available. Please ensure the dual-branch stability model is trained and accessible.',
            'model_type': 'Neural Network (Not Available)'
        }
            
    except Exception as e:
        logger.error(f"Error in neural network prediction: {e}")
        return {
            'error': f'Neural network prediction failed: {str(e)}',
            'model_type': 'Neural Network (Error)'
        }

