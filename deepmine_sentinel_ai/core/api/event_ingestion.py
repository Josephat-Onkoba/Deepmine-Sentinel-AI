"""
Event Processing API for Deepmine Sentinel AI
Handles real-time operational event ingestion and processing
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.utils import timezone
from django.db import transaction
from django.core.exceptions import ValidationError
from django.shortcuts import get_object_or_404
from django.views import View
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..models import Stope, OperationalEvent, ImpactScore
from ..impact.impact_service import ImpactCalculationService
from .event_validator import EventValidator
from .event_queue import EventProcessor

logger = logging.getLogger(__name__)


class EventIngestionError(Exception):
    """Custom exception for event ingestion errors"""
    pass


@method_decorator(csrf_exempt, name='dispatch')
class EventIngestionAPIView(View):
    """
    API endpoint for ingesting operational events
    Supports both single events and batch processing
    """
    
    def __init__(self):
        super().__init__()
        self.impact_service = ImpactCalculationService()
        self.validator = EventValidator()
        self.processor = EventProcessor()
    
    def post(self, request):
        """
        Process incoming operational events
        
        Expected JSON format:
        {
            "events": [
                {
                    "stope_id": 1,
                    "event_type": "blasting",
                    "timestamp": "2025-07-16T10:30:00Z",
                    "severity": 0.8,
                    "proximity_to_stope": 15.0,
                    "duration_hours": 2.5,
                    "description": "Major blast in adjacent area",
                    "equipment_involved": "D11 Dozer, Excavator",
                    "operator": "John Smith"
                }
            ]
        }
        """
        try:
            # Parse request body
            data = json.loads(request.body)
            events_data = data.get('events', [])
            
            if not events_data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No events provided'
                }, status=400)
            
            # Process events
            results = self._process_events_batch(events_data)
            
            return JsonResponse({
                'status': 'success',
                'message': f'Processed {len(results["successful"])} events successfully',
                'results': results,
                'timestamp': timezone.now().isoformat()
            })
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON in event ingestion request")
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON format'
            }, status=400)
            
        except Exception as e:
            logger.error(f"Event ingestion error: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': 'Internal server error during event processing'
            }, status=500)
    
    def _process_events_batch(self, events_data: List[Dict]) -> Dict[str, List]:
        """
        Process a batch of events with transaction safety
        """
        successful = []
        failed = []
        
        for i, event_data in enumerate(events_data):
            try:
                with transaction.atomic():
                    # Validate event data
                    cleaned_data = self.validator.validate_event(event_data)
                    
                    # Create operational event
                    event = self._create_operational_event(cleaned_data)
                    
                    # Queue for background processing
                    self.processor.queue_event_processing(event)
                    
                    successful.append({
                        'index': i,
                        'event_id': event.id,
                        'stope_id': event.stope.id,
                        'event_type': event.event_type,
                        'status': 'queued_for_processing'
                    })
                    
            except ValidationError as e:
                failed.append({
                    'index': i,
                    'error': 'validation_error',
                    'details': str(e)
                })
                
            except Exception as e:
                logger.error(f"Error processing event {i}: {str(e)}")
                failed.append({
                    'index': i,
                    'error': 'processing_error',
                    'details': str(e)
                })
        
        return {
            'successful': successful,
            'failed': failed
        }
    
    def _create_operational_event(self, data: Dict) -> OperationalEvent:
        """
        Create an operational event from validated data
        """
        stope = get_object_or_404(Stope, id=data['stope_id'])
        
        event = OperationalEvent.objects.create(
            stope=stope,
            event_type=data['event_type'],
            timestamp=data['timestamp'],
            severity=data['severity'],
            proximity_to_stope=data['proximity_to_stope'],
            duration_hours=data['duration_hours'],
            description=data.get('description', ''),
            equipment_involved=data.get('equipment_involved', ''),
            operator=data.get('operator', ''),
            weather_conditions=data.get('weather_conditions', ''),
            environmental_factors=data.get('environmental_factors', '')
        )
        
        logger.info(f"Created operational event {event.id} for stope {stope.id}")
        return event


@method_decorator(csrf_exempt, name='dispatch')
class RealTimeEventProcessingView(View):
    """
    API endpoint for immediate event processing
    Processes events synchronously and returns updated impact scores
    """
    
    def __init__(self):
        super().__init__()
        self.impact_service = ImpactCalculationService()
        self.validator = EventValidator()
    
    def post(self, request):
        """
        Process event immediately and return updated impact scores
        """
        try:
            data = json.loads(request.body)
            
            # Validate single event
            cleaned_data = self.validator.validate_event(data)
            
            with transaction.atomic():
                # Create event
                stope = get_object_or_404(Stope, id=cleaned_data['stope_id'])
                event = OperationalEvent.objects.create(
                    stope=stope,
                    event_type=cleaned_data['event_type'],
                    timestamp=cleaned_data['timestamp'],
                    severity=cleaned_data['severity'],
                    proximity_to_stope=cleaned_data['proximity_to_stope'],
                    duration_hours=cleaned_data['duration_hours'],
                    description=cleaned_data.get('description', ''),
                    equipment_involved=cleaned_data.get('equipment_involved', ''),
                    operator=cleaned_data.get('operator', '')
                )
                
                # Process immediately
                updated_scores = self.impact_service.process_event_immediate(event)
                
                return JsonResponse({
                    'status': 'success',
                    'event_id': event.id,
                    'updated_scores': updated_scores,
                    'timestamp': timezone.now().isoformat()
                })
                
        except ValidationError as e:
            return JsonResponse({
                'status': 'error',
                'message': 'Validation error',
                'details': str(e)
            }, status=400)
            
        except Exception as e:
            logger.error(f"Real-time processing error: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': 'Processing error'
            }, status=500)


class EventStatusView(View):
    """
    API endpoint to check event processing status
    """
    
    def get(self, request, event_id):
        """
        Get processing status for a specific event
        """
        try:
            event = get_object_or_404(OperationalEvent, id=event_id)
            
            # Get current impact scores for the stope
            impact_score = ImpactScore.objects.filter(stope=event.stope).first()
            
            return JsonResponse({
                'event_id': event.id,
                'stope_id': event.stope.id,
                'event_type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'processing_status': 'completed',
                'current_impact_score': impact_score.total_score if impact_score else 0.0,
                'current_risk_level': impact_score.risk_level if impact_score else 'stable'
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


class StopeBatchUpdateView(View):
    """
    API endpoint for batch updating multiple stopes
    """
    
    def __init__(self):
        super().__init__()
        self.impact_service = ImpactCalculationService()
    
    @method_decorator(csrf_exempt)
    def post(self, request):
        """
        Trigger batch update for specified stopes
        """
        try:
            data = json.loads(request.body)
            stope_ids = data.get('stope_ids', [])
            
            if not stope_ids:
                # Update all active stopes
                stope_ids = list(Stope.objects.filter(is_active=True).values_list('id', flat=True))
            
            # Queue batch update
            results = self.impact_service.batch_update_impact_scores(stope_ids)
            
            return JsonResponse({
                'status': 'success',
                'updated_stopes': len(results),
                'stope_ids': stope_ids,
                'timestamp': timezone.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Batch update error: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
