"""
Django REST API Views for LSTM Inference Engine

Provides HTTP endpoints for:
- Single stope predictions
- Batch predictions
- Model performance monitoring
- Prediction history and analytics
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.core.serializers.json import DjangoJSONEncoder
from django.conf import settings
import logging

# Import our inference engine
from .ml.inference_engine import (
    LSTMInferenceEngine, 
    PredictionRequest, 
    create_sample_prediction_request
)

logger = logging.getLogger(__name__)

# Global inference engine instance (initialized on first use)
_inference_engine = None


def get_inference_engine():
    """Get or create the global inference engine instance"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = LSTMInferenceEngine()
        try:
            _inference_engine.load_model()
            logger.info("Inference engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise
    return _inference_engine


class InferenceJSONEncoder(DjangoJSONEncoder):
    """Custom JSON encoder for inference results"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def json_response(data: Dict[str, Any], status: int = 200) -> JsonResponse:
    """Create a standardized JSON response"""
    return JsonResponse(
        data, 
        status=status, 
        encoder=InferenceJSONEncoder,
        json_dumps_params={'indent': 2}
    )


def error_response(message: str, status: int = 400, details: Dict[str, Any] = None) -> JsonResponse:
    """Create a standardized error response"""
    response_data = {
        'success': False,
        'error': message,
        'timestamp': datetime.now().isoformat()
    }
    if details:
        response_data['details'] = details
    
    return json_response(response_data, status=status)


def success_response(data: Dict[str, Any], message: str = "Success") -> JsonResponse:
    """Create a standardized success response"""
    response_data = {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    return json_response(response_data)


@csrf_exempt
@require_http_methods(["POST"])
def predict_single(request):
    """
    Single stope stability prediction endpoint
    
    Expected JSON payload:
    {
        "stope_id": "STOPE_001",
        "sensor_data": [[...], [...], ...],  // 2D array: (timesteps, features)
        "metadata": {}  // Optional
    }
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        
        # Validate required fields
        if 'stope_id' not in data:
            return error_response("Missing required field: stope_id")
        
        if 'sensor_data' not in data:
            return error_response("Missing required field: sensor_data")
        
        # Validate sensor data format
        try:
            sensor_data = np.array(data['sensor_data'], dtype=np.float32)
            if sensor_data.ndim != 2:
                return error_response("sensor_data must be a 2D array")
            
        except (ValueError, TypeError) as e:
            return error_response(f"Invalid sensor_data format: {str(e)}")
        
        # Create prediction request
        prediction_request = PredictionRequest(
            stope_id=data['stope_id'],
            sensor_data=sensor_data,
            timestamp=datetime.now(),
            metadata=data.get('metadata', {})
        )
        
        # Get inference engine and make prediction
        engine = get_inference_engine()
        result = engine.predict_single(prediction_request)
        
        # Convert result to dict for JSON response
        response_data = {
            'stope_id': result.stope_id,
            'prediction': {
                'stability_class': result.prediction,
                'class_label': get_stability_label(result.prediction),
                'confidence_score': result.confidence_score,
                'uncertainty_score': result.uncertainty_score,
                'class_probabilities': {
                    f'class_{i}': float(prob) 
                    for i, prob in enumerate(result.class_probabilities)
                }
            },
            'model_info': {
                'version': result.model_version,
                'processing_time_ms': result.processing_time_ms
            },
            'metadata': result.metadata
        }
        
        return success_response(response_data, "Prediction completed successfully")
        
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return error_response(f"Prediction failed: {str(e)}", status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_batch(request):
    """
    Batch stope stability prediction endpoint
    
    Expected JSON payload:
    {
        "batch_id": "optional_batch_id",
        "requests": [
            {
                "stope_id": "STOPE_001",
                "sensor_data": [[...], [...], ...],
                "metadata": {}
            },
            ...
        ]
    }
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        
        # Validate required fields
        if 'requests' not in data:
            return error_response("Missing required field: requests")
        
        if not isinstance(data['requests'], list):
            return error_response("requests must be an array")
        
        if len(data['requests']) == 0:
            return error_response("requests array cannot be empty")
        
        # Create prediction requests
        prediction_requests = []
        for i, req_data in enumerate(data['requests']):
            try:
                # Validate individual request
                if 'stope_id' not in req_data:
                    return error_response(f"Missing stope_id in request {i}")
                
                if 'sensor_data' not in req_data:
                    return error_response(f"Missing sensor_data in request {i}")
                
                # Convert sensor data
                sensor_data = np.array(req_data['sensor_data'], dtype=np.float32)
                if sensor_data.ndim != 2:
                    return error_response(f"Invalid sensor_data format in request {i}")
                
                # Create request object
                prediction_request = PredictionRequest(
                    stope_id=req_data['stope_id'],
                    sensor_data=sensor_data,
                    timestamp=datetime.now(),
                    metadata=req_data.get('metadata', {})
                )
                prediction_requests.append(prediction_request)
                
            except (ValueError, TypeError) as e:
                return error_response(f"Invalid data format in request {i}: {str(e)}")
        
        # Get inference engine and make batch prediction
        engine = get_inference_engine()
        batch_result = engine.predict_batch(
            prediction_requests, 
            batch_id=data.get('batch_id')
        )
        
        # Convert results to dict for JSON response
        predictions_data = []
        for result in batch_result.predictions:
            prediction_data = {
                'stope_id': result.stope_id,
                'prediction': {
                    'stability_class': result.prediction,
                    'class_label': get_stability_label(result.prediction),
                    'confidence_score': result.confidence_score,
                    'uncertainty_score': result.uncertainty_score,
                    'class_probabilities': {
                        f'class_{i}': float(prob) 
                        for i, prob in enumerate(result.class_probabilities)
                    }
                },
                'model_info': {
                    'version': result.model_version,
                    'processing_time_ms': result.processing_time_ms
                },
                'metadata': result.metadata
            }
            predictions_data.append(prediction_data)
        
        response_data = {
            'batch_id': batch_result.batch_id,
            'batch_summary': {
                'total_requests': len(data['requests']),
                'successful_predictions': batch_result.success_count,
                'failed_predictions': batch_result.failure_count,
                'success_rate': batch_result.success_count / len(data['requests']),
                'total_processing_time_ms': batch_result.total_processing_time_ms,
                'average_processing_time_ms': batch_result.metadata['average_processing_time_ms']
            },
            'predictions': predictions_data,
            'batch_timestamp': batch_result.batch_timestamp.isoformat()
        }
        
        return success_response(response_data, "Batch prediction completed successfully")
        
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return error_response(f"Batch prediction failed: {str(e)}", status=500)


@require_http_methods(["GET"])
def model_performance(request):
    """
    Get current model performance metrics
    """
    try:
        engine = get_inference_engine()
        metrics = engine.get_performance_metrics()
        
        response_data = {
            'model_version': metrics.model_version,
            'performance_metrics': {
                'total_predictions': metrics.predictions_count,
                'average_confidence': metrics.average_confidence,
                'average_uncertainty': metrics.average_uncertainty,
                'average_processing_time_ms': metrics.average_processing_time_ms,
                'system_metrics': {
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'cpu_usage_percent': metrics.cpu_usage_percent
                }
            },
            'timestamp': metrics.timestamp.isoformat()
        }
        
        return success_response(response_data, "Performance metrics retrieved successfully")
        
    except Exception as e:
        logger.error(f"Performance metrics error: {str(e)}")
        return error_response(f"Failed to get performance metrics: {str(e)}", status=500)


@require_http_methods(["GET"])
def prediction_summary(request):
    """
    Get prediction summary for specified time period
    
    Query parameters:
    - hours: Number of hours to look back (default: 24)
    """
    try:
        # Get query parameters
        hours = int(request.GET.get('hours', 24))
        
        if hours <= 0 or hours > 168:  # Max 1 week
            return error_response("hours parameter must be between 1 and 168")
        
        engine = get_inference_engine()
        summary = engine.get_prediction_summary(hours=hours)
        
        return success_response(summary, f"Prediction summary for last {hours} hours")
        
    except ValueError:
        return error_response("Invalid hours parameter - must be an integer")
    except Exception as e:
        logger.error(f"Prediction summary error: {str(e)}")
        return error_response(f"Failed to get prediction summary: {str(e)}", status=500)


@require_http_methods(["GET"])
def model_info(request):
    """
    Get information about available models
    """
    try:
        engine = get_inference_engine()
        
        # Get list of available models
        available_models = engine.model_loader.list_available_models()
        
        # Get current model info
        current_model_version = engine.current_model_version
        
        response_data = {
            'current_model': current_model_version,
            'available_models': available_models,
            'model_directory': str(engine.model_loader.model_directory),
            'cached_models': len(engine.model_loader.loaded_models)
        }
        
        return success_response(response_data, "Model information retrieved successfully")
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return error_response(f"Failed to get model information: {str(e)}", status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_demo(request):
    """
    Demo endpoint for testing predictions with sample data
    
    Optional JSON payload:
    {
        "stope_id": "optional_stope_id"
    }
    """
    try:
        # Parse request data (optional)
        stope_id = "DEMO_STOPE"
        if request.body:
            data = json.loads(request.body)
            stope_id = data.get('stope_id', stope_id)
        
        # Create sample prediction request
        prediction_request = create_sample_prediction_request(stope_id)
        
        # Get inference engine and make prediction
        engine = get_inference_engine()
        result = engine.predict_single(prediction_request)
        
        # Convert result to dict for JSON response
        response_data = {
            'demo_info': {
                'description': 'This is a demo prediction using synthetic sensor data',
                'sensor_data_shape': prediction_request.sensor_data.shape,
                'features': [
                    'temperature', 'pressure', 'humidity', 'vibration_x', 
                    'vibration_y', 'vibration_z', 'total_vibration', 'displacement'
                ]
            },
            'stope_id': result.stope_id,
            'prediction': {
                'stability_class': result.prediction,
                'class_label': get_stability_label(result.prediction),
                'confidence_score': result.confidence_score,
                'uncertainty_score': result.uncertainty_score,
                'class_probabilities': {
                    f'class_{i}': float(prob) 
                    for i, prob in enumerate(result.class_probabilities)
                }
            },
            'model_info': {
                'version': result.model_version,
                'processing_time_ms': result.processing_time_ms
            }
        }
        
        return success_response(response_data, "Demo prediction completed successfully")
        
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Demo prediction error: {str(e)}")
        return error_response(f"Demo prediction failed: {str(e)}", status=500)


@require_http_methods(["GET"])
def health_check(request):
    """
    Health check endpoint for the inference service
    """
    try:
        # Check if inference engine can be initialized
        engine = get_inference_engine()
        
        # Basic health checks
        health_status = {
            'status': 'healthy',
            'model_loaded': engine.current_model is not None,
            'model_version': engine.current_model_version,
            'total_predictions': engine.total_predictions,
            'success_rate': (
                engine.successful_predictions / engine.total_predictions 
                if engine.total_predictions > 0 else 1.0
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        return success_response(health_status, "Service is healthy")
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return error_response(f"Service unhealthy: {str(e)}", status=503)


def get_stability_label(stability_class: int) -> str:
    """Convert numeric stability class to human-readable label"""
    labels = {
        0: "Stable",
        1: "Minor Concern", 
        2: "Major Concern",
        3: "Critical"
    }
    return labels.get(stability_class, f"Unknown ({stability_class})")


# API endpoint documentation
def api_docs(request):
    """
    API documentation endpoint
    """
    docs = {
        'title': 'LSTM Inference Engine API',
        'version': '1.0.0',
        'description': 'REST API for mining stope stability predictions using LSTM models',
        'endpoints': {
            '/api/predict/single': {
                'method': 'POST',
                'description': 'Make a single stope stability prediction',
                'required_fields': ['stope_id', 'sensor_data'],
                'example_request': {
                    'stope_id': 'STOPE_001',
                    'sensor_data': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] * 20,
                    'metadata': {'source': 'api_test'}
                }
            },
            '/api/predict/batch': {
                'method': 'POST',
                'description': 'Make batch stope stability predictions',
                'required_fields': ['requests'],
                'example_request': {
                    'batch_id': 'batch_001',
                    'requests': [
                        {
                            'stope_id': 'STOPE_001',
                            'sensor_data': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] * 20
                        }
                    ]
                }
            },
            '/api/predict/demo': {
                'method': 'POST',
                'description': 'Demo prediction with synthetic data',
                'required_fields': [],
                'example_request': {'stope_id': 'DEMO_001'}
            },
            '/api/model/performance': {
                'method': 'GET',
                'description': 'Get current model performance metrics'
            },
            '/api/model/info': {
                'method': 'GET', 
                'description': 'Get information about available models'
            },
            '/api/predictions/summary': {
                'method': 'GET',
                'description': 'Get prediction summary for specified time period',
                'query_params': ['hours']
            },
            '/api/health': {
                'method': 'GET',
                'description': 'Health check for the inference service'
            }
        },
        'stability_classes': {
            0: 'Stable',
            1: 'Minor Concern',
            2: 'Major Concern', 
            3: 'Critical'
        }
    }
    
    return json_response(docs)
