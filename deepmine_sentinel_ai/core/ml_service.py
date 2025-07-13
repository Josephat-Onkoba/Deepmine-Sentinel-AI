"""
Simple ML Prediction Service for Dual-Branch Model Integration
=============================================================

This service provides a simplified interface to the dual-branch stability predictor
for use in Django views and web interface.
"""

import os
import logging
import numpy as np
from django.conf import settings
from .models import Stope, Prediction
from .ml.models.dual_branch_stability_predictor import DualBranchStopeStabilityPredictor

logger = logging.getLogger(__name__)


class MLPredictionService:
    """
    Service class to handle ML predictions using the dual-branch model.
    """
    
    def __init__(self):
        self.model_path = os.path.join(
            settings.BASE_DIR, 'core', 'ml', 'models', 'saved', 
            'dual_branch_stability_model.h5'
        )
        self.predictor = None
        self._model_loaded = False
    
    def is_model_trained(self):
        """Check if the trained model exists and is loadable."""
        if self._model_loaded:
            return True
            
        if not os.path.exists(self.model_path):
            return False
            
        # Try to load the model
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _load_model(self):
        """Load the trained model."""
        if self._model_loaded:
            return
            
        try:
            # For model loading, we don't need the CSV files, so we can pass dummy paths
            # The actual prediction will need to be handled differently
            import tensorflow as tf
            
            # Load the model directly using TensorFlow
            if os.path.exists(self.model_path):
                model = tf.keras.models.load_model(self.model_path)
                
                # Load scalers
                import joblib
                scaler_path = self.model_path.replace('.h5', '_scalers.pkl')
                if os.path.exists(scaler_path):
                    scalers = joblib.load(scaler_path)
                    
                    # Store model and scalers for later use
                    self.model = model
                    self.scalers = scalers
                    self._model_loaded = True
                    logger.info("Dual-branch model loaded successfully")
                else:
                    raise FileNotFoundError(f"Scalers file not found: {scaler_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_stope_stability(self, stope, save_prediction=True):
        """
        Predict stability for a single stope using static features + time series data + profile summary.
        
        Args:
            stope: Stope model instance
            save_prediction: Whether to save prediction to database
            
        Returns:
            dict: Prediction result with risk level, confidence, etc.
        """
        if not self.is_model_trained():
            return {'error': 'Model is not trained or not available'}
        
        try:
            # Check if stope has time series data (required for prediction)
            timeseries_data = stope.timeseries_data.all()
            
            if not timeseries_data.exists():
                return {
                    'error': 'No time series data available for this stope. Please add time series measurements to enable ML prediction.',
                    'prediction_type': 'none'
                }
            
            # Try to use the actual neural network model first
            try:
                prediction_result = self._predict_with_neural_network(stope)
                if prediction_result and 'error' not in prediction_result:
                    # Save prediction to database if requested
                    if save_prediction:
                        self._save_prediction_to_db(stope, prediction_result)
                    return prediction_result
            except Exception as e:
                logger.warning(f"Neural network prediction failed, falling back to rule-based: {e}")
            
            # Fallback to enhanced rule-based prediction
            prediction_result = self._enhanced_rule_based_prediction(stope)
            
            # Save prediction to database if requested
            if save_prediction:
                self._save_prediction_to_db(stope, prediction_result)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error predicting stability for stope {stope.id}: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _predict_with_neural_network(self, stope):
        """
        Use the actual trained dual-branch neural network for prediction.
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Neural network model not loaded")
        
        # Prepare static features
        static_features = self._extract_static_features(stope)
        profile_features = self._extract_profile_features(stope)
        
        # Combine static and profile features
        combined_static = static_features + profile_features
        static_input = np.array([combined_static])  # Shape: (1, n_features)
        
        # Prepare time series features
        timeseries_input = self._prepare_timeseries_for_model(stope)
        
        # Apply scaling if scalers are available
        if hasattr(self, 'scalers') and self.scalers:
            static_input = self.scalers['static_scaler'].transform(static_input)
            
            # Scale time series data
            if timeseries_input is not None:
                original_shape = timeseries_input.shape
                timeseries_reshaped = timeseries_input.reshape(-1, original_shape[-1])
                timeseries_scaled = self.scalers['timeseries_scaler'].transform(timeseries_reshaped)
                timeseries_input = timeseries_scaled.reshape(original_shape)
        
        # Make prediction
        if timeseries_input is not None:
            prediction = self.model.predict([static_input, timeseries_input], verbose=0)[0][0]
        else:
            # If no time series data can be formatted, fall back
            raise ValueError("Cannot format time series data for neural network")
        
        # Convert prediction to risk level
        instability_probability = float(prediction)
        stable = instability_probability < 0.5
        
        if instability_probability > 0.7:
            risk_level = "High"
        elif instability_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'prediction': 'Stable' if stable else 'Unstable',
            'risk_level': risk_level,
            'confidence': instability_probability,
            'probabilities': {
                'unstable': instability_probability,
                'stable': 1 - instability_probability
            },
            'explanation': self._generate_neural_network_explanation(
                stope, stable, risk_level, instability_probability
            ),
            'model_info': {
                'model_type': 'Dual-Branch Neural Network',
                'version': '2.0',
                'features_used': 'Static + Time Series + Profile (Neural Network)',
                'time_series_points': stope.timeseries_data.count()
            }
        }
    
    def _prepare_timeseries_for_model(self, stope):
        """
        Prepare time series data in the format expected by the neural network.
        """
        timeseries_data = stope.timeseries_data.all().order_by('timestamp')
        
        if not timeseries_data:
            return None
        
        # Extract the 5 expected features in the same order as training
        ts_features = ['vibration_velocity', 'deformation_rate', 'stress', 'temperature', 'humidity']
        
        data_matrix = []
        for data_point in timeseries_data:
            row = []
            for feature in ts_features:
                value = getattr(data_point, feature, None)
                # Use 0 for missing values (could be improved with better imputation)
                row.append(float(value) if value is not None else 0.0)
            data_matrix.append(row)
        
        # Convert to numpy array
        ts_array = np.array(data_matrix)
        
        # The model expects a fixed sequence length, so we need to pad or truncate
        # For now, use the last 50 time steps (or pad if less)
        max_sequence_length = 50
        
        if len(ts_array) > max_sequence_length:
            # Take the most recent data
            ts_array = ts_array[-max_sequence_length:]
        elif len(ts_array) < max_sequence_length:
            # Pad with zeros at the beginning
            padding = np.zeros((max_sequence_length - len(ts_array), len(ts_features)))
            ts_array = np.vstack([padding, ts_array])
        
        # Reshape for model input: (1, timesteps, features)
        return ts_array.reshape(1, max_sequence_length, len(ts_features))
    
    def _save_prediction_to_db(self, stope, prediction_result):
        """Save prediction result to database."""
        try:
            Prediction.objects.create(
                stope=stope,
                risk_level=prediction_result['risk_level'],
                impact_score=prediction_result['confidence'],
                explanation=prediction_result['explanation']
            )
        except Exception as e:
            logger.error(f"Error saving prediction to database: {e}")
    
    def _enhanced_rule_based_prediction(self, stope):
        """Enhanced rule-based prediction as fallback."""
        # Get static features from stope
        static_features = self._extract_static_features(stope)
        
        # Get time series features
        timeseries_features = self._extract_timeseries_features(stope)
        
        # Get stope profile summary
        profile_features = self._extract_profile_features(stope)
        
        # Enhanced prediction with time series analysis
        return self._enhanced_prediction_with_timeseries(
            stope, static_features, timeseries_features, profile_features
        )
    
    def _generate_neural_network_explanation(self, stope, stable, risk_level, probability):
        """Generate explanation for neural network predictions."""
        explanation_parts = []
        
        # Basic prediction explanation
        if stable:
            explanation_parts.append(f"🤖 NEURAL NETWORK PREDICTION: The stope '{stope.stope_name}' is predicted to be STABLE.")
        else:
            explanation_parts.append(f"🤖 NEURAL NETWORK PREDICTION: The stope '{stope.stope_name}' is predicted to be UNSTABLE.")
        
        explanation_parts.append(f"🎯 Risk Level: {risk_level}")
        explanation_parts.append(f"📊 Instability Probability: {probability:.3f}")
        
        # Data used
        ts_count = stope.timeseries_data.count()
        explanation_parts.append(f"📈 Time Series Data Points Analyzed: {ts_count}")
        
        # Model details
        explanation_parts.append(f"\n🧠 Analysis Method:")
        explanation_parts.append(f"  • Dual-Branch Neural Network with LSTM and Dense layers")
        explanation_parts.append(f"  • Static geological features + Dynamic time series data")
        explanation_parts.append(f"  • Trained on historical stope stability patterns")
        
        # Risk interpretation
        explanation_parts.append(f"\n⚠️ Risk Interpretation:")
        if probability > 0.7:
            explanation_parts.append(f"  • HIGH RISK: Immediate attention required")
            explanation_parts.append(f"  • Recommend increased monitoring and safety measures")
        elif probability > 0.4:
            explanation_parts.append(f"  • MEDIUM RISK: Enhanced monitoring recommended")
            explanation_parts.append(f"  • Continue regular assessments")
        else:
            explanation_parts.append(f"  • LOW RISK: Current conditions appear stable")
            explanation_parts.append(f"  • Maintain standard monitoring procedures")
        
        explanation_parts.append(f"\n🔬 Model Confidence: This prediction is based on advanced machine learning analysis")
        
        return "\n".join(explanation_parts)
        """Extract static features from stope model."""
        # Direction mapping
        direction_map = {
            'North': 0, 'Northeast': 1, 'East': 2, 'Southeast': 3,
            'South': 4, 'Southwest': 5, 'West': 6, 'Northwest': 7
        }
        
        # Rock type mapping
        rock_type_map = {
            'Granite': 0, 'Basalt': 1, 'Quartzite': 2, 'Schist': 3,
            'Gneiss': 4, 'Marble': 5, 'Slate': 6, 'Shale': 7,
            'Limestone': 8, 'Sandstone': 9, 'Obsidian': 10
        }
        
        # Support type mapping
        support_type_map = {
            'None': 0, 'Rock Bolts': 1, 'Mesh': 2, 'Steel Sets': 3,
            'Shotcrete': 4
        }
        
        static_features = [
            float(stope.rqd),
            float(stope.hr),
            float(stope.depth),
            float(stope.dip),
            direction_map.get(stope.direction, 0),
            float(stope.undercut_wdt),
            rock_type_map.get(stope.rock_type, 0),
            support_type_map.get(stope.support_type, 0),
            float(stope.support_density),
            int(stope.support_installed)
        ]
        
        return static_features
    
    def _extract_timeseries_features(self, stope):
        """Extract and analyze time series features from stope."""
        timeseries_data = stope.timeseries_data.all().order_by('timestamp')
        
        if not timeseries_data:
            return {}
        
        # Extract numeric time series values
        vibration_values = []
        deformation_values = []
        stress_values = []
        temperature_values = []
        humidity_values = []
        
        for data_point in timeseries_data:
            if data_point.vibration_velocity is not None:
                vibration_values.append(data_point.vibration_velocity)
            if data_point.deformation_rate is not None:
                deformation_values.append(data_point.deformation_rate)
            if data_point.stress is not None:
                stress_values.append(data_point.stress)
            if data_point.temperature is not None:
                temperature_values.append(data_point.temperature)
            if data_point.humidity is not None:
                humidity_values.append(data_point.humidity)
        
        # Calculate statistical features
        features = {
            'data_points_count': len(timeseries_data),
            'time_span_days': (timeseries_data.last().timestamp - timeseries_data.first().timestamp).days if len(timeseries_data) > 1 else 0,
        }
        
        # Vibration analysis
        if vibration_values:
            features.update({
                'vibration_mean': np.mean(vibration_values),
                'vibration_max': np.max(vibration_values),
                'vibration_std': np.std(vibration_values),
                'vibration_trend': self._calculate_trend(vibration_values)
            })
        
        # Deformation analysis
        if deformation_values:
            features.update({
                'deformation_mean': np.mean(deformation_values),
                'deformation_max': np.max(deformation_values),
                'deformation_std': np.std(deformation_values),
                'deformation_trend': self._calculate_trend(deformation_values)
            })
        
        # Stress analysis
        if stress_values:
            features.update({
                'stress_mean': np.mean(stress_values),
                'stress_max': np.max(stress_values),
                'stress_std': np.std(stress_values),
                'stress_trend': self._calculate_trend(stress_values)
            })
        
        return features
    
    def _extract_profile_features(self, stope):
        """Extract features from stope profile summary."""
        try:
            # Get the stope profile using the related name from models
            from .models import StopeProfile
            profile = StopeProfile.objects.filter(stope=stope).first()
            
            if profile and profile.summary:
                # Simple numerical features from profile text
                # This could be enhanced with NLP analysis
                summary_length = len(profile.summary)
                word_count = len(profile.summary.split())
                
                # Look for key risk indicators in the summary
                risk_keywords = ['high risk', 'unstable', 'failure', 'collapse', 'dangerous']
                safety_keywords = ['stable', 'safe', 'secure', 'good', 'excellent']
                
                risk_score = sum(1 for keyword in risk_keywords if keyword.lower() in profile.summary.lower())
                safety_score = sum(1 for keyword in safety_keywords if keyword.lower() in profile.summary.lower())
                
                return [summary_length, word_count, risk_score, safety_score]
            else:
                return [0, 0, 0, 0]  # Default values when no profile
        except Exception as e:
            logger.error(f"Error extracting profile features: {e}")
            return [0, 0, 0, 0]
    
    def _calculate_trend(self, values):
        """Calculate trend in time series values (simplified linear trend)."""
        if len(values) < 2:
            return 0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        try:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except:
            return 0
    
    def _enhanced_prediction_with_timeseries(self, stope, static_features, timeseries_features, profile_features):
        """
        Enhanced rule-based prediction that considers time series data patterns.
        """
        instability_score = 0.0
        explanation_factors = []
        
        # Static features analysis (base score)
        rqd, hr, depth, dip = static_features[0], static_features[1], static_features[2], static_features[3]
        support_installed = bool(static_features[9])
        support_density = static_features[8]
        
        # Poor rock quality
        if rqd < 25:
            instability_score += 0.4
            explanation_factors.append(f"Very poor rock quality (RQD: {rqd}%)")
        elif rqd < 50:
            instability_score += 0.2
            explanation_factors.append(f"Poor rock quality (RQD: {rqd}%)")
            
        # Large hydraulic radius
        if hr > 10:
            instability_score += 0.3
            explanation_factors.append(f"Large hydraulic radius ({hr}m)")
        elif hr > 8:
            instability_score += 0.15
            explanation_factors.append(f"Moderate hydraulic radius ({hr}m)")
            
        # Deep stopes
        if depth > 600:
            instability_score += 0.2
            explanation_factors.append(f"Very deep stope ({depth}m)")
        elif depth > 400:
            instability_score += 0.1
            explanation_factors.append(f"Deep stope ({depth}m)")
            
        # Inadequate support
        if not support_installed or support_density < 0.5:
            instability_score += 0.3
            explanation_factors.append("Inadequate support system")
        
        # Time series analysis enhancement
        if timeseries_features:
            # High vibration levels
            if 'vibration_mean' in timeseries_features:
                if timeseries_features['vibration_mean'] > 50:  # mm/s threshold
                    instability_score += 0.25
                    explanation_factors.append(f"High vibration levels (avg: {timeseries_features['vibration_mean']:.1f} mm/s)")
                
                # Increasing vibration trend
                if timeseries_features.get('vibration_trend', 0) > 0.5:
                    instability_score += 0.15
                    explanation_factors.append("Increasing vibration trend detected")
            
            # High deformation rates
            if 'deformation_mean' in timeseries_features:
                if timeseries_features['deformation_mean'] > 10:  # mm/day threshold
                    instability_score += 0.25
                    explanation_factors.append(f"High deformation rates (avg: {timeseries_features['deformation_mean']:.1f} mm/day)")
                
                # Increasing deformation trend
                if timeseries_features.get('deformation_trend', 0) > 0.5:
                    instability_score += 0.2
                    explanation_factors.append("Accelerating deformation detected")
            
            # High stress levels
            if 'stress_mean' in timeseries_features:
                if timeseries_features['stress_mean'] > 100:  # MPa threshold
                    instability_score += 0.2
                    explanation_factors.append(f"High stress levels (avg: {timeseries_features['stress_mean']:.1f} MPa)")
            
            # Data recency and frequency
            data_points = timeseries_features.get('data_points_count', 0)
            if data_points >= 10:
                instability_score -= 0.05  # Bonus for good monitoring
                explanation_factors.append("Good monitoring data coverage")
        
        # Profile analysis enhancement
        if len(profile_features) >= 4:
            risk_score = profile_features[2]
            safety_score = profile_features[3]
            
            if risk_score > safety_score:
                instability_score += 0.1
                explanation_factors.append("Profile analysis indicates risk factors")
            elif safety_score > risk_score:
                instability_score -= 0.05
                explanation_factors.append("Profile analysis indicates positive factors")
        
        # Ensure score is within bounds
        instability_score = max(0, min(1, instability_score))
        
        # Determine risk level
        if instability_score > 0.7:
            risk_level = "High"
            risk_color = "red"
        elif instability_score > 0.4:
            risk_level = "Medium"
            risk_color = "orange"
        else:
            risk_level = "Low"
            risk_color = "green"
        
        stable = instability_score < 0.5
        
        return {
            'prediction': 'Stable' if stable else 'Unstable',
            'risk_level': risk_level,
            'confidence': instability_score,
            'probabilities': {
                'unstable': instability_score,
                'stable': 1 - instability_score
            },
            'explanation': self._generate_comprehensive_explanation(
                stope, stable, risk_level, instability_score, explanation_factors, timeseries_features
            ),
            'model_info': {
                'model_type': 'Enhanced Rule-Based Predictor',
                'version': '2.0',
                'features_used': 'Static + Time Series + Profile Analysis',
                'time_series_points': timeseries_features.get('data_points_count', 0) if timeseries_features else 0
            }
        }
    
    def _generate_explanation_simple(self, stable, risk_level, instability_score, stope):
        """Generate a human-readable explanation for rule-based predictions."""
        explanation_parts = []
        
        # Basic prediction explanation
        if stable:
            explanation_parts.append(f"The stope '{stope.stope_name}' is predicted to be STABLE.")
        else:
            explanation_parts.append(f"The stope '{stope.stope_name}' is predicted to be UNSTABLE.")
        
        explanation_parts.append(f"Risk Level: {risk_level}")
        explanation_parts.append(f"Instability Score: {instability_score:.2f}")
        
        # Add key factors based on stope characteristics
        factors = []
        
        if stope.rqd < 50:
            factors.append(f"Poor rock quality (RQD: {stope.rqd}%)")
        elif stope.rqd > 80:
            factors.append(f"Good rock quality (RQD: {stope.rqd}%)")
            
        if stope.hr > 9:
            factors.append(f"Large hydraulic radius ({stope.hr}m)")
        elif stope.hr < 6:
            factors.append(f"Moderate hydraulic radius ({stope.hr}m)")
            
        if stope.depth > 500:
            factors.append(f"Deep stope (depth: {stope.depth}m)")
            
        if not stope.support_installed:
            factors.append("No support system installed")
        elif stope.support_density < 0.5:
            factors.append(f"Low support density ({stope.support_density})")
        elif stope.support_density > 1.0:
            factors.append(f"High support density ({stope.support_density})")
        
        if stope.dip > 70:
            factors.append(f"Steep dip angle ({stope.dip}°)")
        
        if factors:
            explanation_parts.append("\nKey contributing factors:")
            for factor in factors:
                explanation_parts.append(f"• {factor}")
        
        explanation_parts.append(f"\nNote: This prediction uses a rule-based system. Train the neural network model for more advanced predictions.")
        
        return "\n".join(explanation_parts)

    def _generate_comprehensive_explanation(self, stope, stable, risk_level, instability_score, explanation_factors, timeseries_features):
        """Generate a comprehensive explanation including time series analysis."""
        explanation_parts = []
        
        # Basic prediction explanation
        if stable:
            explanation_parts.append(f"✅ The stope '{stope.stope_name}' is predicted to be STABLE.")
        else:
            explanation_parts.append(f"⚠️  The stope '{stope.stope_name}' is predicted to be UNSTABLE.")
        
        explanation_parts.append(f"🎯 Risk Level: {risk_level}")
        explanation_parts.append(f"📊 Instability Score: {instability_score:.2f}/1.0")
        
        # Time series monitoring status
        if timeseries_features:
            data_points = timeseries_features.get('data_points_count', 0)
            time_span = timeseries_features.get('time_span_days', 0)
            explanation_parts.append(f"📈 Monitoring Data: {data_points} data points over {time_span} days")
        
        # Key contributing factors
        if explanation_factors:
            explanation_parts.append("\n🔍 Key Analysis Factors:")
            for factor in explanation_factors:
                explanation_parts.append(f"  • {factor}")
        
        # Time series insights
        if timeseries_features:
            explanation_parts.append("\n📊 Time Series Analysis:")
            
            if 'vibration_mean' in timeseries_features:
                vibration_status = "Normal" if timeseries_features['vibration_mean'] <= 50 else "Elevated"
                explanation_parts.append(f"  • Vibration: {vibration_status} (avg: {timeseries_features['vibration_mean']:.1f} mm/s)")
            
            if 'deformation_mean' in timeseries_features:
                deformation_status = "Normal" if timeseries_features['deformation_mean'] <= 10 else "Elevated"
                explanation_parts.append(f"  • Deformation: {deformation_status} (avg: {timeseries_features['deformation_mean']:.1f} mm/day)")
            
            if 'stress_mean' in timeseries_features:
                stress_status = "Normal" if timeseries_features['stress_mean'] <= 100 else "High"
                explanation_parts.append(f"  • Stress: {stress_status} (avg: {timeseries_features['stress_mean']:.1f} MPa)")
        
        # Recommendations
        explanation_parts.append("\n💡 Recommendations:")
        if not stable:
            explanation_parts.append("  • Increase monitoring frequency")
            explanation_parts.append("  • Consider additional support measures")
            explanation_parts.append("  • Review excavation procedures")
        else:
            explanation_parts.append("  • Continue regular monitoring")
            explanation_parts.append("  • Maintain current support systems")
        
        explanation_parts.append(f"\n🤖 Analysis Method: Enhanced rule-based prediction with time series integration")
        
        return "\n".join(explanation_parts)

    def predict_multiple_stopes(self, stope_ids):
        """
        Predict stability for multiple stopes.
        
        Args:
            stope_ids: List of stope IDs
            
        Returns:
            dict: Dictionary with stope_id as key and prediction result as value
        """
        results = {}
        
        for stope_id in stope_ids:
            try:
                stope = Stope.objects.get(id=stope_id)
                result = self.predict_stope_stability(stope, save_prediction=True)
                results[stope_id] = result
            except Stope.DoesNotExist:
                results[stope_id] = {'error': f'Stope with ID {stope_id} not found'}
            except Exception as e:
                results[stope_id] = {'error': f'Prediction failed: {str(e)}'}
                
        return results
    
    def get_model_performance_metrics(self):
        """
        Get basic model performance information.
        This is a simplified version for the web interface.
        """
        if not self.is_model_trained():
            return {'error': 'Model is not trained'}
        
        # Return basic model info
        return {
            'model_type': 'Dual-Branch Neural Network',
            'architecture': 'Dense + LSTM branches',
            'status': 'Trained and Ready',
            'input_features': [
                'RQD', 'Hydraulic Radius', 'Depth', 'Dip', 'Direction',
                'Undercut Width', 'Rock Type', 'Support Type', 'Support Density'
            ],
            'output': 'Binary Classification (Stable/Unstable)',
            'last_updated': 'Model file timestamp'  # Could be enhanced to read actual file timestamp
        }
    
    def get_feature_importance(self, top_n=10):
        """
        Get simplified feature importance information.
        Note: Neural networks don't have direct feature importance like tree models,
        but we can provide general importance based on domain knowledge.
        """
        if not self.is_model_trained():
            return {}
        
        # Simplified feature importance based on domain knowledge
        # In practice, this could be enhanced with SHAP values or other explanation methods
        importance = {
            'RQD (Rock Quality Designation)': 0.25,
            'Hydraulic Radius': 0.20,
            'Support Density': 0.15,
            'Depth': 0.12,
            'Rock Type': 0.10,
            'Support Installation': 0.08,
            'Dip Angle': 0.05,
            'Direction': 0.03,
            'Undercut Width': 0.02
        }
        
        # Return top N features
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])
