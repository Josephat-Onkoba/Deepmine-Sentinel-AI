"""
LSTM Inference Engine - Task 10 Implementation

This module implements real-time prediction capabilities for trained LSTM models:
- Real-time prediction API for trained models
- Batch prediction capabilities for multiple stopes
- Prediction confidence scoring and uncertainty estimation
- Model performance monitoring in production
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import psutil
from dataclasses import dataclass, asdict
from pathlib import Path
import joblib
from scipy import stats
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Data structure for prediction requests"""
    stope_id: str
    sensor_data: np.ndarray  # Shape: (sequence_length, n_features)
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class PredictionResult:
    """Data structure for prediction results"""
    stope_id: str
    prediction: int  # Predicted stability class (0-3)
    confidence_score: float  # Confidence in prediction (0-1)
    class_probabilities: np.ndarray  # Probabilities for each class
    uncertainty_score: float  # Uncertainty estimation (0-1)
    prediction_timestamp: datetime
    model_version: str
    processing_time_ms: float
    metadata: Dict[str, Any] = None


@dataclass
class BatchPredictionResult:
    """Data structure for batch prediction results"""
    batch_id: str
    predictions: List[PredictionResult]
    batch_timestamp: datetime
    total_processing_time_ms: float
    success_count: int
    failure_count: int
    metadata: Dict[str, Any] = None


@dataclass
class ModelPerformanceMetrics:
    """Model performance monitoring metrics"""
    model_version: str
    timestamp: datetime
    predictions_count: int
    average_confidence: float
    average_uncertainty: float
    average_processing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    accuracy_score: Optional[float] = None
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None


class ModelLoader:
    """Handles loading and caching of trained LSTM models"""
    
    def __init__(self, model_directory: str = "models/trained_models"):
        self.model_directory = Path(model_directory)
        self.loaded_models = {}  # Cache for loaded models
        self.model_metadata = {}  # Metadata for loaded models
        
    def load_model(self, model_path: str) -> keras.Model:
        """Load a trained LSTM model"""
        try:
            model_path = Path(model_path)
            
            # Check if model is already cached
            cache_key = str(model_path)
            if cache_key in self.loaded_models:
                logger.info(f"Using cached model: {model_path.name}")
                return self.loaded_models[cache_key]
            
            # Load model from disk
            start_time = time.time()
            model = keras.models.load_model(model_path)
            load_time = (time.time() - start_time) * 1000
            
            # Cache the model
            self.loaded_models[cache_key] = model
            
            # Store metadata
            self.model_metadata[cache_key] = {
                'path': str(model_path),
                'loaded_at': datetime.now(),
                'load_time_ms': load_time,
                'model_size_mb': model_path.stat().st_size / (1024 * 1024),
                'parameters': model.count_params()
            }
            
            logger.info(f"Loaded model: {model_path.name} ({load_time:.2f}ms)")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise
    
    def get_latest_model(self) -> keras.Model:
        """Get the most recently trained model"""
        try:
            model_files = list(self.model_directory.glob("*.keras"))
            if not model_files:
                raise FileNotFoundError("No trained models found")
            
            # Sort by modification time (newest first)
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            return self.load_model(latest_model)
            
        except Exception as e:
            logger.error(f"Failed to get latest model: {str(e)}")
            raise
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available trained models"""
        models = []
        for model_file in self.model_directory.glob("*.keras"):
            stat = model_file.stat()
            models.append({
                'name': model_file.name,
                'path': str(model_file),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'loaded': str(model_file) in self.loaded_models
            })
        
        # Sort by modification time (newest first)
        return sorted(models, key=lambda x: x['modified'], reverse=True)


class ConfidenceEstimator:
    """Estimates prediction confidence and uncertainty"""
    
    def __init__(self):
        self.prediction_history = []
        self.confidence_threshold = 0.7  # Threshold for high confidence
        
    def calculate_confidence(self, class_probabilities: np.ndarray) -> float:
        """Calculate confidence score based on class probabilities"""
        # Use maximum probability as base confidence
        max_prob = np.max(class_probabilities)
        
        # Adjust based on probability distribution entropy
        entropy = -np.sum(class_probabilities * np.log(class_probabilities + 1e-8))
        max_entropy = np.log(len(class_probabilities))  # Maximum possible entropy
        
        # Normalize entropy (0 = confident, 1 = uncertain)
        normalized_entropy = entropy / max_entropy
        
        # Combine max probability with inverse entropy
        confidence = max_prob * (1 - normalized_entropy)
        
        return float(confidence)
    
    def calculate_uncertainty(self, class_probabilities: np.ndarray) -> float:
        """Calculate uncertainty score using entropy"""
        # Calculate Shannon entropy
        entropy = -np.sum(class_probabilities * np.log(class_probabilities + 1e-8))
        max_entropy = np.log(len(class_probabilities))
        
        # Normalize to 0-1 range
        uncertainty = entropy / max_entropy
        
        return float(uncertainty)
    
    def assess_prediction_reliability(self, 
                                   class_probabilities: np.ndarray,
                                   sensor_data: np.ndarray) -> Dict[str, float]:
        """Comprehensive prediction reliability assessment"""
        confidence = self.calculate_confidence(class_probabilities)
        uncertainty = self.calculate_uncertainty(class_probabilities)
        
        # Data quality assessment
        data_variance = np.var(sensor_data)
        data_mean = np.mean(sensor_data)
        data_std = np.std(sensor_data)
        
        # Check for data anomalies
        z_scores = np.abs(stats.zscore(sensor_data.flatten()))
        anomaly_ratio = np.mean(z_scores > 3)  # Proportion of outliers
        
        # Overall reliability score
        reliability = confidence * (1 - uncertainty) * (1 - anomaly_ratio)
        
        return {
            'confidence': confidence,
            'uncertainty': uncertainty,
            'data_quality': 1 - anomaly_ratio,
            'reliability': reliability,
            'anomaly_ratio': anomaly_ratio
        }


class LSTMInferenceEngine:
    """
    Main inference engine for LSTM stability predictions
    
    Provides real-time and batch prediction capabilities with
    confidence scoring and performance monitoring.
    """
    
    def __init__(self, 
                 model_directory: str = "models/trained_models",
                 performance_log_dir: str = "logs/inference_performance"):
        
        self.model_loader = ModelLoader(model_directory)
        self.confidence_estimator = ConfidenceEstimator()
        
        # Performance monitoring
        self.performance_log_dir = Path(performance_log_dir)
        self.performance_log_dir.mkdir(parents=True, exist_ok=True)
        self.performance_metrics = []
        
        # Current model
        self.current_model = None
        self.current_model_version = None
        
        # Prediction cache for performance
        self.prediction_cache = {}
        self.cache_expiry_minutes = 5
        
        # Performance counters
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        
        logger.info("LSTM Inference Engine initialized")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load a specific model or the latest available model"""
        try:
            if model_path:
                self.current_model = self.model_loader.load_model(model_path)
                self.current_model_version = Path(model_path).stem
            else:
                self.current_model = self.model_loader.get_latest_model()
                # Extract version from filename
                latest_models = self.model_loader.list_available_models()
                if latest_models:
                    self.current_model_version = latest_models[0]['name'].replace('.keras', '')
            
            logger.info(f"Active model: {self.current_model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict_single(self, request: PredictionRequest) -> PredictionResult:
        """Make a single prediction for one stope"""
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if self.current_model is None:
                self.load_model()
            
            # Validate input data
            if request.sensor_data.ndim != 2:
                raise ValueError("Sensor data must be 2D (sequence_length, n_features)")
            
            # Prepare input for model (add batch dimension)
            input_data = np.expand_dims(request.sensor_data, axis=0)
            
            # Make prediction
            prediction_probs = self.current_model.predict(input_data, verbose=0)[0]
            predicted_class = int(np.argmax(prediction_probs))
            
            # Calculate confidence and uncertainty
            reliability_metrics = self.confidence_estimator.assess_prediction_reliability(
                prediction_probs, request.sensor_data
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create result
            result = PredictionResult(
                stope_id=request.stope_id,
                prediction=predicted_class,
                confidence_score=reliability_metrics['confidence'],
                class_probabilities=prediction_probs,
                uncertainty_score=reliability_metrics['uncertainty'],
                prediction_timestamp=datetime.now(),
                model_version=self.current_model_version,
                processing_time_ms=processing_time,
                metadata={
                    'reliability_metrics': reliability_metrics,
                    'input_shape': request.sensor_data.shape,
                    'request_timestamp': request.timestamp.isoformat()
                }
            )
            
            # Update counters
            self.total_predictions += 1
            self.successful_predictions += 1
            
            # Store prediction metrics for performance tracking
            prediction_metric = ModelPerformanceMetrics(
                model_version=self.current_model_version or "unknown",
                timestamp=datetime.now(),
                predictions_count=1,
                average_confidence=reliability_metrics['confidence'],
                average_uncertainty=reliability_metrics['uncertainty'],
                average_processing_time_ms=processing_time,
                memory_usage_mb=0.0,  # Will be updated in get_performance_metrics
                cpu_usage_percent=0.0  # Will be updated in get_performance_metrics
            )
            self.performance_metrics.append(prediction_metric)
            
            logger.debug(f"Prediction for {request.stope_id}: class={predicted_class}, "
                        f"confidence={reliability_metrics['confidence']:.3f}, "
                        f"time={processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.failed_predictions += 1
            logger.error(f"Prediction failed for {request.stope_id}: {str(e)}")
            raise
    
    def predict_batch(self, 
                     requests: List[PredictionRequest],
                     batch_id: Optional[str] = None) -> BatchPredictionResult:
        """Make predictions for multiple stopes"""
        start_time = time.time()
        
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        predictions = []
        success_count = 0
        failure_count = 0
        
        try:
            # Ensure model is loaded
            if self.current_model is None:
                self.load_model()
            
            logger.info(f"Starting batch prediction {batch_id} with {len(requests)} requests")
            
            # Process all requests
            for i, request in enumerate(requests):
                try:
                    result = self.predict_single(request)
                    predictions.append(result)
                    success_count += 1
                    
                except Exception as e:
                    failure_count += 1
                    logger.warning(f"Failed prediction {i+1}/{len(requests)} in batch {batch_id}: {str(e)}")
                    continue
            
            # Calculate total processing time
            total_processing_time = (time.time() - start_time) * 1000
            
            # Create batch result
            batch_result = BatchPredictionResult(
                batch_id=batch_id,
                predictions=predictions,
                batch_timestamp=datetime.now(),
                total_processing_time_ms=total_processing_time,
                success_count=success_count,
                failure_count=failure_count,
                metadata={
                    'average_processing_time_ms': total_processing_time / len(requests) if requests else 0,
                    'success_rate': success_count / len(requests) if requests else 0,
                    'model_version': self.current_model_version
                }
            )
            
            logger.info(f"Batch prediction {batch_id} completed: "
                       f"{success_count}/{len(requests)} successful, "
                       f"total_time={total_processing_time:.2f}ms")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch prediction {batch_id} failed: {str(e)}")
            raise
    
    def get_performance_metrics(self) -> ModelPerformanceMetrics:
        """Get current model performance metrics"""
        
        # Get system resource usage
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        cpu_usage = psutil.cpu_percent()
        
        # Calculate recent performance metrics from actual predictions
        if self.performance_metrics:
            recent_metrics = self.performance_metrics[-100:]  # Last 100 predictions
            avg_confidence = np.mean([m.average_confidence for m in recent_metrics if m.average_confidence > 0])
            avg_uncertainty = np.mean([m.average_uncertainty for m in recent_metrics if m.average_uncertainty > 0])
            avg_processing_time = np.mean([m.average_processing_time_ms for m in recent_metrics if m.average_processing_time_ms > 0])
        else:
            avg_confidence = 0.0
            avg_uncertainty = 0.0
            avg_processing_time = 0.0
        
        # Handle NaN values
        if np.isnan(avg_confidence):
            avg_confidence = 0.0
        if np.isnan(avg_uncertainty):
            avg_uncertainty = 0.0
        if np.isnan(avg_processing_time):
            avg_processing_time = 0.0
        
        metrics = ModelPerformanceMetrics(
            model_version=self.current_model_version or "unknown",
            timestamp=datetime.now(),
            predictions_count=self.total_predictions,
            average_confidence=avg_confidence,
            average_uncertainty=avg_uncertainty,
            average_processing_time_ms=avg_processing_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )
        
        return metrics
    
    def monitor_performance(self, save_to_file: bool = True):
        """Monitor and log model performance metrics"""
        
        metrics = self.get_performance_metrics()
        self.performance_metrics.append(metrics)
        
        if save_to_file:
            # Save metrics to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = self.performance_log_dir / f"performance_metrics_{timestamp}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
        
        logger.info(f"Performance metrics: predictions={metrics.predictions_count}, "
                   f"avg_confidence={metrics.average_confidence:.3f}, "
                   f"avg_processing_time={metrics.average_processing_time_ms:.2f}ms")
        
        return metrics
    
    def validate_model_accuracy(self, 
                               test_data: List[Tuple[np.ndarray, int]],
                               save_results: bool = True) -> Dict[str, float]:
        """Validate model accuracy on test data"""
        
        if self.current_model is None:
            self.load_model()
        
        predictions = []
        true_labels = []
        processing_times = []
        
        logger.info(f"Validating model accuracy on {len(test_data)} samples")
        
        for sensor_data, true_label in test_data:
            start_time = time.time()
            
            # Create prediction request
            request = PredictionRequest(
                stope_id="validation",
                sensor_data=sensor_data,
                timestamp=datetime.now()
            )
            
            try:
                result = self.predict_single(request)
                predictions.append(result.prediction)
                true_labels.append(true_label)
                processing_times.append(result.processing_time_ms)
                
            except Exception as e:
                logger.warning(f"Validation prediction failed: {str(e)}")
                continue
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(true_labels, predictions)
        avg_processing_time = np.mean(processing_times)
        
        # Calculate per-class metrics
        from sklearn.metrics import classification_report
        report = classification_report(true_labels, predictions, output_dict=True)
        
        validation_results = {
            'accuracy': accuracy,
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'average_processing_time_ms': avg_processing_time,
            'total_samples': len(predictions),
            'model_version': self.current_model_version,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        if save_results:
            # Save validation results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.performance_log_dir / f"validation_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Model validation completed: accuracy={accuracy:.3f}, "
                   f"avg_time={avg_processing_time:.2f}ms")
        
        return validation_results
    
    def get_prediction_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get prediction summary for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.performance_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        # Always include basic summary info even if no recent metrics
        summary = {
            'period_hours': hours,
            'predictions_count': self.total_predictions,
            'success_rate': self.successful_predictions / self.total_predictions if self.total_predictions > 0 else 1.0,
            'current_model_version': self.current_model_version or "unknown"
        }
        
        if not recent_metrics:
            summary.update({
                'average_confidence': 0.0,
                'average_uncertainty': 0.0,
                'average_processing_time_ms': 0.0,
                'peak_memory_usage_mb': 0.0,
                'peak_cpu_usage_percent': 0.0,
                'summary': 'No performance metrics recorded in the specified period'
            })
        else:
            summary.update({
                'average_confidence': np.mean([m.average_confidence for m in recent_metrics]),
                'average_uncertainty': np.mean([m.average_uncertainty for m in recent_metrics]),
                'average_processing_time_ms': np.mean([m.average_processing_time_ms for m in recent_metrics]),
                'peak_memory_usage_mb': max(m.memory_usage_mb for m in recent_metrics),
                'peak_cpu_usage_percent': max(m.cpu_usage_percent for m in recent_metrics)
            })
        
        return summary


# Example usage and testing functions
def create_sample_prediction_request(stope_id: str = "STOPE_001") -> PredictionRequest:
    """Create a sample prediction request for testing"""
    
    # Generate sample sensor data (20 timesteps, 8 features)
    np.random.seed(42)
    sensor_data = np.random.random((20, 8))
    
    # Add some realistic patterns
    for i in range(20):
        # Add trend
        trend = i * 0.01
        sensor_data[i] += trend
        
        # Add correlations between features
        sensor_data[i, 6] = sensor_data[i, 0] * 0.5 + sensor_data[i, 1] * 0.3  # vibration
        sensor_data[i, 7] = np.cumsum(sensor_data[:i+1, 6])[-1] * 0.1  # displacement
    
    return PredictionRequest(
        stope_id=stope_id,
        sensor_data=sensor_data,
        timestamp=datetime.now()
    )


def test_inference_engine():
    """Test the inference engine functionality"""
    
    print("🔬 Testing LSTM Inference Engine...")
    
    # Initialize inference engine
    engine = LSTMInferenceEngine()
    
    # Load the trained model
    try:
        engine.load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test single prediction
    print("\n📊 Testing single prediction...")
    request = create_sample_prediction_request("TEST_STOPE_001")
    
    try:
        result = engine.predict_single(request)
        print(f"✅ Single prediction successful:")
        print(f"   Stope: {result.stope_id}")
        print(f"   Prediction: {result.prediction}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Uncertainty: {result.uncertainty_score:.3f}")
        print(f"   Processing time: {result.processing_time_ms:.2f}ms")
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
    
    # Test batch prediction
    print("\n📦 Testing batch prediction...")
    batch_requests = [
        create_sample_prediction_request(f"STOPE_{i:03d}") 
        for i in range(1, 6)
    ]
    
    try:
        batch_result = engine.predict_batch(batch_requests)
        print(f"✅ Batch prediction successful:")
        print(f"   Batch ID: {batch_result.batch_id}")
        print(f"   Successful: {batch_result.success_count}/{len(batch_requests)}")
        print(f"   Total time: {batch_result.total_processing_time_ms:.2f}ms")
        print(f"   Average time per prediction: {batch_result.metadata['average_processing_time_ms']:.2f}ms")
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
    
    # Test performance monitoring
    print("\n📈 Testing performance monitoring...")
    try:
        metrics = engine.monitor_performance()
        print(f"✅ Performance monitoring successful:")
        print(f"   Total predictions: {metrics.predictions_count}")
        print(f"   Memory usage: {metrics.memory_usage_mb:.1f} MB")
        print(f"   CPU usage: {metrics.cpu_usage_percent:.1f}%")
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
    
    # Test prediction summary
    print("\n📋 Testing prediction summary...")
    try:
        summary = engine.get_prediction_summary(hours=1)
        print(f"✅ Prediction summary:")
        print(f"   Predictions in last hour: {summary['predictions_count']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        print(f"   Model version: {summary['current_model_version']}")
    except Exception as e:
        print(f"❌ Prediction summary failed: {e}")
    
    print(f"\n🎯 TASK 10 INFERENCE ENGINE TEST COMPLETED")
    print(f"✅ Real-time prediction API: IMPLEMENTED")
    print(f"✅ Batch prediction capabilities: IMPLEMENTED") 
    print(f"✅ Confidence scoring: IMPLEMENTED")
    print(f"✅ Performance monitoring: IMPLEMENTED")


if __name__ == "__main__":
    test_inference_engine()
