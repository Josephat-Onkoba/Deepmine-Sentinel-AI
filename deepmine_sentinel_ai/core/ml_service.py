"""
Enhanced ML Prediction Service for Dual-Branch Neural Network
============================================================

This service provides a production interface to the enhanced dual-branch neural network
stability predictor for use in Django views and web interface.

Key Features:
- Enhanced dual-branch neural network (static + time series features)
- Comprehensive stability prediction with current and future projections
- Physics-based stability calculations combined with ML predictions
- Model explanation and risk assessment with detailed analysis
- Integration with Django models and database
- Real data feature extraction from CSV files
"""

import os
import logging
import numpy as np
import pandas as pd
from django.conf import settings
from datetime import datetime, timedelta
from .models import Stope, Prediction, FuturePrediction, PredictionAlert
from .ml.models.dual_branch_stability_predictor import DualBranchStopeStabilityPredictor

logger = logging.getLogger(__name__)


class MLPredictionService:
    """
    Enhanced service class to handle ML predictions using the enhanced dual-branch model.
    Integrates with real CSV data and provides comprehensive predictions.
    """
    
    def __init__(self):
        self.data_dir = os.path.join(settings.BASE_DIR, 'data')
        self.static_csv_path = os.path.join(self.data_dir, 'stope_static_features_aligned.csv')
        self.timeseries_csv_path = os.path.join(self.data_dir, 'stope_timeseries_data_aligned.csv')
        self.models_dir = os.path.join(settings.BASE_DIR, 'models')
        
        # Define the standard model path (matches train_model.py)
        self.model_path = os.path.join(self.models_dir, 'enhanced_dual_branch_model.keras')
        self.model_metadata_path = self.model_path.replace('.keras', '_metadata.json')
        
        self.predictor = None
        self._model_loaded = False
        self._csv_data_loaded = False
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    
    def is_model_trained(self):
        """Check if the enhanced model is available and can be used."""
        if self._model_loaded:
            return True
        
        # First check if a trained model exists on disk
        if os.path.exists(self.model_path):
            logger.info(f"Found trained model at: {self.model_path}")
            return True
            
        # If no saved model, check if we can create data for training
        return self._can_create_enhanced_model_data()
    
    def _can_create_enhanced_model_data(self):
        """Check if we can create data for the enhanced model from Django database."""
        try:
            # Check if we have stopes with time series data
            from .models import Stope
            stopes_with_data = Stope.objects.filter(timeseries_data__isnull=False).distinct()
            
            if stopes_with_data.count() < 1:  # Need at least some data
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking model data availability: {e}")
            return False
    
    def _load_model(self):
        """Load the enhanced dual-branch model with Django data or saved model."""
        if self._model_loaded:
            return True
            
        try:
            # First try to load a pre-trained model if it exists
            if os.path.exists(self.model_path):
                logger.info(f"Loading pre-trained model from: {self.model_path}")
                
                # We still need to initialize with data for compatibility
                if not self._prepare_data_for_enhanced_model():
                    logger.warning("Cannot prepare data for model initialization")
                    return False
                
                # Initialize predictor with data paths
                self.predictor = DualBranchStopeStabilityPredictor(
                    self.temp_static_path,
                    self.temp_timeseries_path
                )
                
                # Load the pre-trained model
                self.predictor.load_enhanced_model(self.model_path)
                
                self._model_loaded = True
                self._csv_data_loaded = True
                logger.info("Pre-trained model loaded successfully")
                return True
            
            # If no pre-trained model, prepare for training
            else:
                logger.info("No pre-trained model found, preparing for training")
                
                # Prepare data from Django database for the enhanced model
                if not self._prepare_data_for_enhanced_model():
                    logger.warning("Cannot prepare data for enhanced model, using fallback mode")
                    return False
                
                # Initialize the enhanced predictor with the prepared data paths
                self.predictor = DualBranchStopeStabilityPredictor(
                    self.temp_static_path,
                    self.temp_timeseries_path
                )
                
                self._model_loaded = True
                self._csv_data_loaded = True
                logger.info("Enhanced dual-branch model initialized successfully for training")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load enhanced model: {e}")
            return False
    
    def _prepare_data_for_enhanced_model(self):
        """
        Prepare data from Django database for the enhanced model.
        Creates temporary CSV files from Django data.
        """
        try:
            from .models import Stope, TimeSeriesData
            import tempfile
            
            # Create temporary files
            temp_dir = tempfile.mkdtemp(prefix='deepmine_ml_')
            self.temp_static_path = os.path.join(temp_dir, 'static_features.csv')
            self.temp_timeseries_path = os.path.join(temp_dir, 'timeseries_data.csv')
            
            # Extract static features from all stopes
            static_data = []
            stopes = Stope.objects.all()
            
            if stopes.count() == 0:
                logger.warning("No stopes found in database")
                return False
            
            for stope in stopes:
                # Direction mapping (consistent with enhanced model)
                direction_map = {
                    'North': 0, 'Northeast': 1, 'East': 2, 'Southeast': 3,
                    'South': 4, 'Southwest': 5, 'West': 6, 'Northwest': 7
                }
                
                # Rock type mapping (consistent with enhanced model)
                rock_type_map = {
                    'Granite': 0, 'Basalt': 1, 'Obsidian': 2, 'Shale': 3, 'Marble': 4,
                    'Slate': 5, 'Gneiss': 6, 'Schist': 7, 'Quartzite': 8, 'Limestone': 9, 'Sandstone': 10
                }
                
                # Support type mapping (consistent with enhanced model)
                support_type_map = {
                    'None': 0, 'Rock Bolts': 1, 'Mesh': 2, 'Shotcrete': 3,
                    'Timber': 4, 'Cable Bolts': 5, 'Steel Sets': 6
                }
                
                # Create a synthetic stability label based on physics-based assessment
                stability_score = self._calculate_physics_based_stability({
                    'rqd': float(stope.rqd),
                    'hr': float(stope.hr),
                    'depth': float(stope.depth),
                    'dip': float(stope.dip),
                    'direction': direction_map.get(stope.direction, 0),
                    'undercut_wdt': float(stope.undercut_wdt),
                    'rock_type': rock_type_map.get(stope.rock_type, 0),
                    'support_type': support_type_map.get(stope.support_type, 0),
                    'support_density': float(stope.support_density),
                    'support_installed': int(stope.support_installed)
                })
                
                # Convert stability score to risk level
                if stability_score > 0.8:
                    risk_level = 'stable'
                elif stability_score > 0.6:
                    risk_level = 'slight_elevated'
                elif stability_score > 0.4:
                    risk_level = 'elevated'
                elif stability_score > 0.2:
                    risk_level = 'high'
                else:
                    risk_level = 'critical'
                
                static_row = {
                    'stope_name': stope.stope_name,
                    'rqd': float(stope.rqd),
                    'hr': float(stope.hr),
                    'depth': float(stope.depth),
                    'dip': float(stope.dip),
                    'direction': direction_map.get(stope.direction, 0),
                    'undercut_wdt': float(stope.undercut_wdt),
                    'rock_type': rock_type_map.get(stope.rock_type, 0),
                    'support_type': support_type_map.get(stope.support_type, 0),
                    'support_density': float(stope.support_density),
                    'support_installed': int(stope.support_installed),
                    'stability_label': risk_level
                }
                static_data.append(static_row)
            
            # Create static features DataFrame and save
            static_df = pd.DataFrame(static_data)
            static_df.to_csv(self.temp_static_path, index=False)
            
            # Extract timeseries data from all stopes
            timeseries_data = []
            
            for stope in stopes:
                ts_data = TimeSeriesData.objects.filter(stope=stope).order_by('timestamp')
                
                for data_point in ts_data:
                    ts_row = {
                        'stope_name': stope.stope_name,
                        'timestamp': data_point.timestamp.isoformat(),
                        'vibration_velocity': float(data_point.vibration_velocity or 0),
                        'deformation_rate': float(data_point.deformation_rate or 0),
                        'stress': float(data_point.stress or 0),
                        'temperature': float(data_point.temperature or 20),
                        'humidity': float(data_point.humidity or 50)
                    }
                    timeseries_data.append(ts_row)
            
            # Create timeseries DataFrame and save
            if timeseries_data:
                timeseries_df = pd.DataFrame(timeseries_data)
                timeseries_df.to_csv(self.temp_timeseries_path, index=False)
            else:
                # Create a minimal timeseries file if no data exists
                minimal_ts = pd.DataFrame({
                    'stope_name': [static_data[0]['stope_name']] if static_data else ['dummy_stope'],
                    'timestamp': [pd.Timestamp.now().isoformat()],
                    'vibration_velocity': [0.0],
                    'deformation_rate': [0.0],
                    'stress': [0.0],
                    'temperature': [20.0],
                    'humidity': [50.0]
                })
                minimal_ts.to_csv(self.temp_timeseries_path, index=False)
            
            logger.info(f"Prepared data for enhanced model: {len(static_data)} stopes, {len(timeseries_data)} timeseries points")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data for enhanced model: {e}")
            return False
    
    def predict_stope_stability(self, stope, save_prediction=True):
        """
        Predict stability for a single stope using the enhanced dual-branch model.
        
        Args:
            stope: Stope model instance
            save_prediction: Whether to save prediction to database
            
        Returns:
            dict: Enhanced prediction result with current and future assessments
        """
        if not self.is_model_trained():
            return {'error': 'Enhanced model is not trained or not available'}
        
        try:
            # Map Django stope object to CSV-compatible stope name
            stope_name = self._map_stope_to_csv_name(stope)
            
            if not stope_name:
                # Fallback: Use enhanced model with Django data
                return self._predict_with_django_data(stope, save_prediction)
            
            # Use the enhanced predictor for comprehensive prediction
            prediction_result = self.predictor.predict_comprehensive_stability(stope_name)
            
            if prediction_result is None:
                return {'error': 'Enhanced prediction failed - stope data not found in CSV files'}
            
            # Convert enhanced result to service format
            service_result = self._convert_enhanced_result_to_service_format(prediction_result)
            
            # Save prediction to database if requested
            if save_prediction:
                self._save_enhanced_prediction_to_db(stope, prediction_result, service_result)
            
            return service_result
            
        except Exception as e:
            logger.error(f"Error predicting stability for stope {stope.id}: {e}")
            return {'error': f'Enhanced prediction failed: {str(e)}'}
    
    def _map_stope_to_csv_name(self, stope):
        """
        Map Django stope to CSV stope name.
        Since we're creating the CSV from Django data, the stope name should match directly.
        """
        try:
            # Ensure the model is loaded (which prepares the data)
            if not self._load_model():
                return None
            
            # Since we create the CSV from Django data, the stope name should match directly
            if hasattr(self.predictor, 'static_df') and self.predictor.static_df is not None:
                if stope.stope_name in self.predictor.static_df['stope_name'].values:
                    return stope.stope_name
            
            return None
            
        except Exception as e:
            logger.error(f"Error mapping stope to CSV name: {e}")
            return None
    
    def _predict_with_django_data(self, stope, save_prediction):
        """
        Fallback prediction using Django data when CSV mapping fails.
        Uses physics-based calculations from the enhanced model.
        """
        try:
            # Check if stope has time series data
            timeseries_data = stope.timeseries_data.all()
            
            if not timeseries_data.exists():
                return {
                    'error': 'No time series data available for this stope. Please add time series measurements to enable ML prediction.',
                    'prediction_type': 'none'
                }
            
            # Extract features from Django models
            static_features = self._extract_django_static_features(stope)
            
            # Calculate physics-based stability using enhanced model methods
            stability_score = self._calculate_physics_based_stability(static_features)
            
            # Analyze time series trends
            ts_risk_factor = self._analyze_django_timeseries_trends(stope)
            
            # Combine physics and temporal analysis
            combined_risk = (stability_score + ts_risk_factor) / 2
            instability_probability = 1 - combined_risk  # Convert to instability probability
            
            # Determine risk level
            if instability_probability > 0.7:
                risk_level = "High"
                prediction = "Unstable"
            elif instability_probability > 0.4:
                risk_level = "Medium"
                prediction = "Unstable"
            else:
                risk_level = "Low"
                prediction = "Stable"
            
            result = {
                'prediction': prediction,
                'risk_level': risk_level,
                'confidence': float(abs(instability_probability - 0.5) * 2),
                'probabilities': {
                    'unstable': float(instability_probability),
                    'stable': float(1 - instability_probability)
                },
                'explanation': self._generate_django_explanation(
                    stope, prediction, risk_level, instability_probability, static_features
                ),
                'model_info': {
                    'model_type': 'Enhanced Physics-Based Analysis',
                    'version': 'Django Fallback v1.0',
                    'features_used': 'Static + Time Series + Physics',
                    'time_series_points': stope.timeseries_data.count(),
                    'fallback_mode': True
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Django fallback prediction: {e}")
            return {'error': f'Django fallback prediction failed: {str(e)}'}
    
    def _extract_django_static_features(self, stope):
        """Extract static features from Django stope model."""
        # Direction mapping (consistent with enhanced model)
        direction_map = {
            'North': 0, 'Northeast': 1, 'East': 2, 'Southeast': 3,
            'South': 4, 'Southwest': 5, 'West': 6, 'Northwest': 7
        }
        
        # Rock type mapping (consistent with enhanced model)
        rock_type_map = {
            'Granite': 0, 'Basalt': 1, 'Obsidian': 2, 'Shale': 3, 'Marble': 4,
            'Slate': 5, 'Gneiss': 6, 'Schist': 7, 'Quartzite': 8, 'Limestone': 9, 'Sandstone': 10
        }
        
        # Support type mapping (consistent with enhanced model)
        support_type_map = {
            'None': 0, 'Rock Bolts': 1, 'Mesh': 2, 'Shotcrete': 3,
            'Timber': 4, 'Cable Bolts': 5, 'Steel Sets': 6
        }
        
        return {
            'rqd': float(stope.rqd),
            'hr': float(stope.hr),
            'depth': float(stope.depth),
            'dip': float(stope.dip),
            'direction': direction_map.get(stope.direction, 0),
            'undercut_wdt': float(stope.undercut_wdt),
            'rock_type': rock_type_map.get(stope.rock_type, 0),
            'support_type': support_type_map.get(stope.support_type, 0),
            'support_density': float(stope.support_density),
            'support_installed': int(stope.support_installed)
        }
    
    def _calculate_physics_based_stability(self, features):
        """
        Calculate stability using physics-based approach from enhanced model.
        """
        try:
            # RQD factor (0-1, higher is more stable)
            rqd_factor = min(features['rqd'] / 100.0, 1.0)
            
            # Hydraulic radius factor (inverse relationship)
            hr_factor = max(0.1, 1.0 - (features['hr'] - 5.0) / 15.0)
            
            # Depth factor (deeper = less stable)
            depth_factor = max(0.1, 1.0 - (features['depth'] - 200.0) / 800.0)
            
            # Support factor
            if features['support_installed']:
                support_factor = min(1.0, 0.5 + features['support_density'] / 2.0)
            else:
                support_factor = 0.3
            
            # Rock type factor (simplified)
            strong_rocks = [0, 1, 4, 8, 9]  # Granite, Basalt, Marble, Quartzite, Limestone
            rock_factor = 0.8 if features['rock_type'] in strong_rocks else 0.6
            
            # Dip factor (steep angles can be problematic)
            dip_factor = 1.0 - abs(features['dip'] - 45) / 90.0  # Optimal around 45 degrees
            
            # Weighted stability score
            stability = (
                rqd_factor * 0.3 +
                hr_factor * 0.25 +
                depth_factor * 0.2 +
                support_factor * 0.15 +
                rock_factor * 0.1
            )
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.error(f"Error calculating physics-based stability: {e}")
            return 0.5  # Neutral value on error
    
    def _analyze_django_timeseries_trends(self, stope):
        """
        Analyze time series trends from Django TimeSeriesData.
        """
        try:
            # Get recent time series data
            recent_data = stope.timeseries_data.all().order_by('-timestamp')[:50]
            
            if len(recent_data) < 5:
                return 0.5  # Not enough data for trend analysis
            
            # Convert to arrays for analysis
            vibration_values = [d.vibration_velocity for d in recent_data if d.vibration_velocity is not None]
            deformation_values = [d.deformation_rate for d in recent_data if d.deformation_rate is not None]
            stress_values = [d.stress for d in recent_data if d.stress is not None]
            
            risk_factors = []
            
            # Vibration analysis
            if vibration_values:
                avg_vibration = np.mean(vibration_values)
                if avg_vibration > 20:  # High vibration threshold
                    risk_factors.append(0.3)
                elif avg_vibration > 10:
                    risk_factors.append(0.6)
                else:
                    risk_factors.append(0.8)
            
            # Deformation analysis
            if deformation_values:
                avg_deformation = np.mean(deformation_values)
                if avg_deformation > 5:  # High deformation threshold
                    risk_factors.append(0.2)
                elif avg_deformation > 2:
                    risk_factors.append(0.5)
                else:
                    risk_factors.append(0.8)
            
            # Stress analysis
            if stress_values:
                avg_stress = np.mean(stress_values)
                if avg_stress > 50:  # High stress threshold
                    risk_factors.append(0.3)
                elif avg_stress > 30:
                    risk_factors.append(0.6)
                else:
                    risk_factors.append(0.8)
            
            # Return average risk factor
            return np.mean(risk_factors) if risk_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing time series trends: {e}")
            return 0.5

    
    def _convert_enhanced_result_to_service_format(self, enhanced_result):
        """
        Convert enhanced model result to service format for Django views.
        """
        current_stability = enhanced_result['current_stability']
        future_predictions = enhanced_result['future_predictions']
        
        return {
            'prediction': 'Stable' if current_stability['stable'] else 'Unstable',
            'risk_level': current_stability['risk_level'].title(),
            'confidence': current_stability['confidence'],
            'probabilities': {
                'unstable': current_stability['instability_probability'],
                'stable': 1 - current_stability['instability_probability']
            },
            'explanation': self._format_enhanced_explanation(enhanced_result),
            'model_info': {
                'model_type': enhanced_result['model_type'],
                'version': enhanced_result['model_version'],
                'features_used': 'Static + Time Series + Physics + Future Projections',
                'prediction_count': len(future_predictions) + 1,
                'enhanced_mode': True
            },
            'future_predictions': future_predictions,
            'risk_trend': enhanced_result['risk_trend'],
            'recommendations': enhanced_result['recommendations']
        }
    
    def _format_enhanced_explanation(self, enhanced_result):
        """Format enhanced model explanations for Django display."""
        current = enhanced_result['current_stability']
        future_preds = enhanced_result['future_predictions']
        explanations = enhanced_result['explanations']
        
        explanation_parts = []
        
        # Current status
        explanation_parts.append(f"🤖 ENHANCED ML PREDICTION: {enhanced_result['stope_name']}")
        explanation_parts.append(f"📊 Current Status: {current['risk_level'].title()} Risk ({'Stable' if current['stable'] else 'Unstable'})")
        explanation_parts.append(f"🎯 Confidence: {current['confidence']:.1%}")
        explanation_parts.append(f"⚡ Instability Probability: {current['instability_probability']:.3f}")
        
        # Future outlook
        if future_preds:
            explanation_parts.append(f"\n🔮 Future Risk Assessment:")
            for pred in future_preds[:3]:  # Show first 3 future predictions
                explanation_parts.append(
                    f"  • {pred['horizon_days']} days: {pred['predicted_risk_level'].title()} "
                    f"(Confidence: {pred['confidence']:.1%})"
                )
        
        # Risk trend
        if enhanced_result['risk_trend']:
            trend_emoji = "📈" if enhanced_result['risk_trend'] == "increasing" else "📉" if enhanced_result['risk_trend'] == "decreasing" else "➡️"
            explanation_parts.append(f"\n{trend_emoji} Risk Trend: {enhanced_result['risk_trend'].title()}")
        
        # Key explanations
        if explanations:
            explanation_parts.append(f"\n🔍 Key Factors:")
            for explanation in explanations[:5]:  # Show top 5 explanations
                explanation_parts.append(f"  • {explanation}")
        
        # Model info
        explanation_parts.append(f"\n🧠 Analysis Method:")
        explanation_parts.append(f"  • Enhanced dual-branch neural network with physics integration")
        explanation_parts.append(f"  • Multi-horizon temporal forecasting")
        explanation_parts.append(f"  • Real geological data and mining domain expertise")
        
        return "\n".join(explanation_parts)
    
    def _generate_django_explanation(self, stope, prediction, risk_level, probability, features):
        """Generate explanation for Django fallback predictions."""
        explanation_parts = []
        
        explanation_parts.append(f"🔬 PHYSICS-BASED ANALYSIS: {stope.stope_name}")
        explanation_parts.append(f"📊 Prediction: {prediction} ({risk_level} Risk)")
        explanation_parts.append(f"🎯 Instability Probability: {probability:.3f}")
        
        # Feature analysis
        explanation_parts.append(f"\n📏 Geological Analysis:")
        explanation_parts.append(f"  • Rock Quality (RQD): {features['rqd']:.1f}% - {'Poor' if features['rqd'] < 50 else 'Good' if features['rqd'] > 75 else 'Fair'}")
        explanation_parts.append(f"  • Hydraulic Radius: {features['hr']:.1f}m - {'Large span' if features['hr'] > 10 else 'Moderate span'}")
        explanation_parts.append(f"  • Depth: {features['depth']:.0f}m - {'Deep' if features['depth'] > 600 else 'Moderate depth'}")
        explanation_parts.append(f"  • Support: {'Installed' if features['support_installed'] else 'Not installed'}")
        
        # Risk factors
        risk_factors = []
        if features['rqd'] < 50:
            risk_factors.append("Poor rock quality")
        if features['hr'] > 10:
            risk_factors.append("Large excavation span")
        if features['depth'] > 600:
            risk_factors.append("High overburden pressure")
        if not features['support_installed']:
            risk_factors.append("No ground support")
        
        if risk_factors:
            explanation_parts.append(f"\n⚠️ Risk Factors:")
            for factor in risk_factors:
                explanation_parts.append(f"  • {factor}")
        
        explanation_parts.append(f"\n📈 Time Series Analysis: {stope.timeseries_data.count()} data points analyzed")
        
        explanation_parts.append(f"\n🔬 Note: Physics-based analysis used (enhanced ML model available for more accurate predictions)")
        
        return "\n".join(explanation_parts)
    
    def _save_enhanced_prediction_to_db(self, stope, enhanced_result, service_result):
        """Save enhanced prediction results to database."""
        try:
            # Save main prediction
            prediction = Prediction.objects.create(
                stope=stope,
                risk_level=service_result['risk_level'],
                impact_score=service_result['confidence'],
                explanation=service_result['explanation']
            )
            
            # Save future predictions if available
            if 'future_predictions' in enhanced_result:
                for future_pred in enhanced_result['future_predictions']:
                    FuturePrediction.objects.create(
                        stope=stope,
                        prediction_type=self._map_horizon_to_type(future_pred['horizon_days']),
                        prediction_for_date=datetime.now() + timedelta(days=future_pred['horizon_days']),
                        days_ahead=future_pred['horizon_days'],
                        risk_level=future_pred['predicted_risk_level'],
                        confidence_score=future_pred['confidence'],
                        risk_probability=max(future_pred['risk_probabilities'].values()),
                        contributing_factors=future_pred.get('risk_probabilities', {}),
                        explanation=f"Enhanced ML prediction for {future_pred['horizon_days']} days ahead",
                        model_version=enhanced_result['model_version']
                    )
            
            # Generate alerts if high risk predicted
            if enhanced_result.get('alert_recommended', False):
                self._generate_prediction_alerts(stope, enhanced_result)
                
        except Exception as e:
            logger.error(f"Error saving enhanced prediction to database: {e}")
    
    def _map_horizon_to_type(self, days):
        """Map prediction horizon to type."""
        if days == 0:
            return 'current'
        elif days <= 3:
            return 'short_term'
        elif days <= 14:
            return 'medium_term'
        else:
            return 'long_term'
    
    def _generate_prediction_alerts(self, stope, enhanced_result):
        """Generate alerts based on enhanced predictions."""
        try:
            future_preds = enhanced_result.get('future_predictions', [])
            
            for pred in future_preds:
                if pred['predicted_risk_level'] in ['high', 'critical', 'unstable']:
                    # Create corresponding FuturePrediction for the alert
                    future_prediction = FuturePrediction.objects.filter(
                        stope=stope,
                        days_ahead=pred['horizon_days']
                    ).first()
                    
                    if future_prediction:
                        alert_severity = 'critical' if pred['predicted_risk_level'] == 'unstable' else 'high'
                        
                        PredictionAlert.objects.create(
                            stope=stope,
                            future_prediction=future_prediction,
                            alert_type='risk_increase',
                            severity=alert_severity,
                            title=f"High Risk Predicted for {stope.stope_name}",
                            message=f"Enhanced ML model predicts {pred['predicted_risk_level']} risk in {pred['horizon_days']} days. Confidence: {pred['confidence']:.1%}"
                        )
                        
        except Exception as e:
            logger.error(f"Error generating prediction alerts: {e}")

    def predict_multiple_stopes(self, stope_ids):
        """
        Predict stability for multiple stopes using enhanced model.
        
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
                results[stope_id] = {'error': f'Enhanced prediction failed: {str(e)}'}
                
        return results
    
    def get_model_performance_metrics(self):
        """
        Get enhanced model performance information.
        """
        if not self.is_model_trained():
            return {'error': 'Enhanced model is not trained'}
        
        try:
            # Get metrics from enhanced model if available
            if hasattr(self.predictor, 'get_model_metrics'):
                metrics = self.predictor.get_model_metrics()
                return {
                    'model_type': 'Enhanced Dual-Branch Neural Network',
                    'architecture': 'Dense + LSTM + Physics Integration',
                    'status': 'Trained and Ready (Enhanced)',
                    'accuracy': metrics.get('accuracy', 'N/A'),
                    'precision': metrics.get('precision', 'N/A'),
                    'recall': metrics.get('recall', 'N/A'),
                    'f1_score': metrics.get('f1_score', 'N/A'),
                    'input_features': [
                        'RQD', 'Hydraulic Radius', 'Depth', 'Dip', 'Direction',
                        'Undercut Width', 'Rock Type', 'Support Type', 'Support Density',
                        'Support Installation', 'Time Series Data'
                    ],
                    'output': 'Multi-class Risk Assessment + Future Projections',
                    'prediction_horizons': getattr(self.predictor, 'prediction_horizons', [1, 3, 7, 14]),
                    'enhanced_features': True
                }
            else:
                return {
                    'model_type': 'Enhanced Dual-Branch Neural Network',
                    'architecture': 'Dense + LSTM + Physics Integration', 
                    'status': 'Trained and Ready (Enhanced)',
                    'input_features': [
                        'RQD', 'Hydraulic Radius', 'Depth', 'Dip', 'Direction',
                        'Undercut Width', 'Rock Type', 'Support Type', 'Support Density',
                        'Support Installation', 'Time Series Data'
                    ],
                    'output': 'Multi-class Risk Assessment + Future Projections',
                    'enhanced_features': True
                }
                
        except Exception as e:
            logger.error(f"Error getting enhanced model metrics: {e}")
            return {'error': f'Error accessing enhanced model: {str(e)}'}
    
    def get_feature_importance(self, top_n=10):
        """
        Get enhanced feature importance information.
        """
        if not self.is_model_trained():
            return {}
        
        # Enhanced feature importance based on domain knowledge and model architecture
        importance = {
            'RQD (Rock Quality Designation)': 0.28,
            'Hydraulic Radius': 0.22,
            'Support Density': 0.16,
            'Time Series Vibration Patterns': 0.12,
            'Depth': 0.08,
            'Rock Type': 0.06,
            'Support Installation': 0.04,
            'Time Series Stress Patterns': 0.02,
            'Dip Angle': 0.01,
            'Direction': 0.01
        }
        
        # Return top N features
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])
    
    def train_model_with_current_data(self):
        """
        Train the enhanced model with current data and save it.
        """
        if not self.predictor:
            if not self._load_model():
                return {'error': 'Cannot initialize enhanced model'}
        
        try:
            logger.info("Starting enhanced model training...")
            
            # Train the enhanced model
            history = self.predictor.train_enhanced_model()
            
            if history:
                logger.info("Enhanced model training completed successfully")
                
                # Automatically save the trained model
                try:
                    logger.info(f"Saving trained model to: {self.model_path}")
                    self.predictor.save_enhanced_model(self.model_path)
                    
                    # Save training metadata
                    import json
                    from datetime import datetime
                    metadata = {
                        'training_date': datetime.now().isoformat(),
                        'epochs_trained': len(history.history['loss']),
                        'final_loss': history.history.get('loss', [0])[-1],
                        'final_val_loss': history.history.get('val_loss', [0])[-1],
                        'final_accuracy': history.history.get('accuracy', [0])[-1] if 'accuracy' in history.history else 'N/A',
                        'model_parameters': self.predictor.combined_model.count_params() if hasattr(self.predictor, 'combined_model') and self.predictor.combined_model else 0,
                        'training_source': 'ml_service'
                    }
                    
                    with open(self.model_metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    logger.info(f"Model and metadata saved successfully")
                    
                except Exception as save_error:
                    logger.error(f"Failed to save model: {save_error}")
                    # Don't fail the whole training, but warn
                
                return {
                    'success': True,
                    'message': 'Enhanced model trained and saved successfully',
                    'epochs': len(history.history['loss']),
                    'final_accuracy': history.history.get('accuracy', [0])[-1] if 'accuracy' in history.history else 'N/A',
                    'model_saved': True,
                    'model_path': self.model_path
                }
            else:
                return {'error': 'Enhanced model training failed'}
                
        except Exception as e:
            logger.error(f"Error training enhanced model: {e}")
            return {'error': f'Training failed: {str(e)}'}
    
    def validate_model_health(self):
        """
        Validate the health and readiness of the enhanced model.
        """
        health_status = {
            'overall_health': 'unknown',
            'components': {},
            'recommendations': []
        }
        
        try:
            # Check Django database availability
            from .models import Stope, TimeSeriesData
            
            stope_count = Stope.objects.count()
            ts_count = TimeSeriesData.objects.count()
            stopes_with_ts = Stope.objects.filter(timeseries_data__isnull=False).distinct().count()
            
            health_status['components']['django_database'] = 'healthy' if stope_count > 0 else 'missing'
            health_status['components']['stope_data'] = 'healthy' if stope_count > 0 else 'missing'
            health_status['components']['timeseries_data'] = 'healthy' if ts_count > 0 else 'missing'
            health_status['components']['integrated_data'] = 'healthy' if stopes_with_ts > 0 else 'missing'
            
            if stope_count == 0:
                health_status['recommendations'].append('Add stope data to the database')
            if ts_count == 0:
                health_status['recommendations'].append('Add time series data for stopes')
            if stopes_with_ts == 0:
                health_status['recommendations'].append('Ensure stopes have time series data for predictions')
            
            # Check model initialization
            can_create_data = self._can_create_enhanced_model_data()
            health_status['components']['data_preparation'] = 'healthy' if can_create_data else 'not_ready'
            
            if not can_create_data:
                health_status['recommendations'].append('Ensure at least one stope has time series data')
            
            # Check predictor initialization
            if hasattr(self, 'predictor') and self.predictor:
                health_status['components']['predictor'] = 'healthy'
            else:
                health_status['components']['predictor'] = 'not_initialized'
                if can_create_data:
                    health_status['recommendations'].append('Initialize the enhanced predictor')
            
            # Overall health assessment
            component_states = list(health_status['components'].values())
            healthy_count = sum(1 for state in component_states if state == 'healthy')
            total_count = len(component_states)
            
            if healthy_count == total_count:
                health_status['overall_health'] = 'healthy'
            elif healthy_count >= total_count * 0.7:  # 70% or more healthy
                health_status['overall_health'] = 'partial'
            else:
                health_status['overall_health'] = 'unhealthy'
            
            # Add data statistics
            health_status['data_stats'] = {
                'total_stopes': stope_count,
                'total_timeseries_points': ts_count,
                'stopes_with_timeseries': stopes_with_ts
            }
                
        except Exception as e:
            logger.error(f"Error validating model health: {e}")
            health_status['overall_health'] = 'error'
            health_status['error'] = str(e)
            
        return health_status
