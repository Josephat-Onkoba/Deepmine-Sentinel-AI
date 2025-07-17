"""
Task 6: Risk Level Classification System - API Endpoints
========================================================

This module provides REST API endpoints for the risk management system including:
- Risk level classification and status
- Risk threshold configuration
- Alert management and acknowledgment
- Risk transition history and analysis

The API integrates with the Risk Classification Service to provide real-time
risk assessment and management capabilities for mining operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.db.models import Q, Count, Avg, Max, Min
from django.core.exceptions import ValidationError
from django.views.decorators.csrf import csrf_exempt

from core.models import (
    Stope, RiskThreshold, RiskTransition, RiskAlert, 
    RiskClassificationRule, ImpactScore
)
from core.risk.risk_classification_service import risk_classification_service

logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_stope_risk_status(request, stope_id):
    """
    Get comprehensive risk status for a specific stope.
    
    Returns current risk level, recent transitions, active alerts, and trends.
    """
    try:
        stope = get_object_or_404(Stope, id=stope_id)
        
        # Get comprehensive risk status
        risk_status = risk_classification_service.get_stope_risk_status(stope)
        
        return Response({
            'status': 'success',
            'stope_id': stope_id,
            'stope_name': stope.stope_name,
            'risk_status': risk_status,
            'timestamp': timezone.now().isoformat()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error getting stope risk status: {e}")
        return Response({
            'status': 'error',
            'message': f'Error retrieving risk status: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def classify_stope_risk(request, stope_id):
    """
    Manually trigger risk classification for a stope.
    
    Optionally accepts impact_score parameter for custom classification.
    """
    try:
        stope = get_object_or_404(Stope, id=stope_id)
        
        # Get impact score from request or current score
        impact_score = request.data.get('impact_score')
        if impact_score is not None:
            try:
                impact_score = float(impact_score)
            except (ValueError, TypeError):
                return Response({
                    'status': 'error',
                    'message': 'Invalid impact_score value'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Classify risk level
        risk_level = risk_classification_service.classify_stope_risk_level(
            stope, impact_score
        )
        
        # Check for risk transition
        transition = risk_classification_service.detect_risk_transition(
            stope, risk_level, trigger_value=impact_score
        )
        
        response_data = {
            'status': 'success',
            'stope_id': stope_id,
            'stope_name': stope.stope_name,
            'risk_level': risk_level,
            'impact_score': impact_score,
            'timestamp': timezone.now().isoformat(),
            'transition_detected': transition is not None
        }
        
        if transition:
            response_data['transition'] = {
                'id': transition.id,
                'previous_level': transition.previous_risk_level,
                'new_level': transition.new_risk_level,
                'trigger_type': transition.trigger_type,
                'timestamp': transition.transition_timestamp.isoformat()
            }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error classifying stope risk: {e}")
        return Response({
            'status': 'error',
            'message': f'Error classifying risk: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_risk_thresholds(request):
    """
    Get all active risk thresholds.
    
    Supports filtering by risk_level, threshold_type, and applicability.
    """
    try:
        thresholds = RiskThreshold.objects.filter(is_active=True).order_by('priority')
        
        # Apply filters
        risk_level = request.GET.get('risk_level')
        if risk_level:
            thresholds = thresholds.filter(risk_level=risk_level)
        
        threshold_type = request.GET.get('threshold_type')
        if threshold_type:
            thresholds = thresholds.filter(threshold_type=threshold_type)
        
        # Serialize threshold data
        threshold_data = []
        for threshold in thresholds:
            threshold_data.append({
                'id': threshold.id,
                'name': threshold.name,
                'risk_level': threshold.risk_level,
                'threshold_type': threshold.threshold_type,
                'minimum_value': threshold.minimum_value,
                'maximum_value': threshold.maximum_value,
                'priority': threshold.priority,
                'minimum_duration': str(threshold.minimum_duration),
                'cooldown_period': str(threshold.cooldown_period),
                'applies_to_rock_types': threshold.applies_to_rock_types,
                'applies_to_mining_methods': threshold.applies_to_mining_methods,
                'created_at': threshold.created_at.isoformat(),
                'updated_at': threshold.updated_at.isoformat()
            })
        
        return Response({
            'status': 'success',
            'count': len(threshold_data),
            'thresholds': threshold_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error getting risk thresholds: {e}")
        return Response({
            'status': 'error',
            'message': f'Error retrieving thresholds: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_risk_threshold(request):
    """
    Create a new risk threshold configuration.
    """
    try:
        # Validate required fields
        required_fields = ['name', 'risk_level', 'threshold_type', 'minimum_value']
        for field in required_fields:
            if field not in request.data:
                return Response({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create threshold
        threshold_data = request.data.copy()
        threshold_data['created_by'] = request.user.username if hasattr(request, 'user') else 'api'
        
        # Handle duration fields
        if 'minimum_duration' in threshold_data:
            try:
                # Convert seconds to timedelta
                duration_seconds = float(threshold_data['minimum_duration'])
                threshold_data['minimum_duration'] = timedelta(seconds=duration_seconds)
            except (ValueError, TypeError):
                return Response({
                    'status': 'error',
                    'message': 'Invalid minimum_duration format'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        if 'cooldown_period' in threshold_data:
            try:
                # Convert seconds to timedelta
                cooldown_seconds = float(threshold_data['cooldown_period'])
                threshold_data['cooldown_period'] = timedelta(seconds=cooldown_seconds)
            except (ValueError, TypeError):
                return Response({
                    'status': 'error',
                    'message': 'Invalid cooldown_period format'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        threshold = RiskThreshold.objects.create(**threshold_data)
        
        return Response({
            'status': 'success',
            'message': 'Risk threshold created successfully',
            'threshold_id': threshold.id,
            'threshold': {
                'id': threshold.id,
                'name': threshold.name,
                'risk_level': threshold.risk_level,
                'threshold_type': threshold.threshold_type,
                'minimum_value': threshold.minimum_value,
                'created_at': threshold.created_at.isoformat()
            }
        }, status=status.HTTP_201_CREATED)
        
    except ValidationError as e:
        return Response({
            'status': 'error',
            'message': f'Validation error: {str(e)}'
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Error creating risk threshold: {e}")
        return Response({
            'status': 'error',
            'message': f'Error creating threshold: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_risk_alerts(request):
    """
    Get risk alerts with filtering and pagination.
    
    Supports filtering by stope, status, priority, and date range.
    """
    try:
        alerts = RiskAlert.objects.all().order_by('-created_at')
        
        # Apply filters
        stope_id = request.GET.get('stope_id')
        if stope_id:
            alerts = alerts.filter(stope_id=stope_id)
        
        alert_status = request.GET.get('status')
        if alert_status:
            alerts = alerts.filter(status=alert_status)
        
        priority = request.GET.get('priority')
        if priority:
            alerts = alerts.filter(priority=priority)
        
        # Date range filtering
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        if start_date:
            try:
                start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                alerts = alerts.filter(created_at__gte=start_date)
            except ValueError:
                return Response({
                    'status': 'error',
                    'message': 'Invalid start_date format'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        if end_date:
            try:
                end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                alerts = alerts.filter(created_at__lte=end_date)
            except ValueError:
                return Response({
                    'status': 'error',
                    'message': 'Invalid end_date format'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Pagination
        limit = min(int(request.GET.get('limit', 50)), 100)  # Max 100 alerts
        offset = int(request.GET.get('offset', 0))
        
        total_count = alerts.count()
        alerts = alerts[offset:offset + limit]
        
        # Serialize alert data
        alert_data = []
        for alert in alerts:
            alert_data.append({
                'id': alert.id,
                'stope_id': alert.stope.id,
                'stope_name': alert.stope.stope_name,
                'alert_type': alert.alert_type,
                'priority': alert.priority,
                'status': alert.status,
                'title': alert.title,
                'message': alert.message,
                'recommended_actions': alert.recommended_actions,
                'created_at': alert.created_at.isoformat(),
                'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                'acknowledged_by': alert.acknowledged_by,
                'resolved_by': alert.resolved_by,
                'escalation_level': alert.escalation_level,
                'notification_sent': alert.notification_sent,
                'age_hours': alert.age.total_seconds() / 3600
            })
        
        return Response({
            'status': 'success',
            'total_count': total_count,
            'count': len(alert_data),
            'offset': offset,
            'limit': limit,
            'alerts': alert_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        return Response({
            'status': 'error',
            'message': f'Error retrieving alerts: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def acknowledge_alert(request, alert_id):
    """
    Acknowledge a risk alert.
    """
    try:
        alert = get_object_or_404(RiskAlert, id=alert_id)
        
        if alert.status not in ['active', 'investigating']:
            return Response({
                'status': 'error',
                'message': f'Alert cannot be acknowledged (current status: {alert.status})'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Acknowledge the alert
        acknowledged_by = request.user.username if hasattr(request, 'user') else 'api'
        notes = request.data.get('notes', '')
        
        alert.acknowledge(acknowledged_by, notes)
        
        return Response({
            'status': 'success',
            'message': 'Alert acknowledged successfully',
            'alert_id': alert_id,
            'acknowledged_by': acknowledged_by,
            'acknowledged_at': alert.acknowledged_at.isoformat()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        return Response({
            'status': 'error',
            'message': f'Error acknowledging alert: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def resolve_alert(request, alert_id):
    """
    Resolve a risk alert.
    """
    try:
        alert = get_object_or_404(RiskAlert, id=alert_id)
        
        if alert.status not in ['active', 'acknowledged', 'investigating']:
            return Response({
                'status': 'error',
                'message': f'Alert cannot be resolved (current status: {alert.status})'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get resolution notes
        resolution_notes = request.data.get('resolution_notes', '')
        if not resolution_notes:
            return Response({
                'status': 'error',
                'message': 'Resolution notes are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Resolve the alert
        resolved_by = request.user.username if hasattr(request, 'user') else 'api'
        alert.resolve(resolved_by, resolution_notes)
        
        return Response({
            'status': 'success',
            'message': 'Alert resolved successfully',
            'alert_id': alert_id,
            'resolved_by': resolved_by,
            'resolved_at': alert.resolved_at.isoformat(),
            'resolution_notes': resolution_notes
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        return Response({
            'status': 'error',
            'message': f'Error resolving alert: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_risk_transitions(request):
    """
    Get risk level transitions with filtering.
    
    Supports filtering by stope, date range, and transition type.
    """
    try:
        transitions = RiskTransition.objects.all().order_by('-transition_timestamp')
        
        # Apply filters
        stope_id = request.GET.get('stope_id')
        if stope_id:
            transitions = transitions.filter(stope_id=stope_id)
        
        trigger_type = request.GET.get('trigger_type')
        if trigger_type:
            transitions = transitions.filter(trigger_type=trigger_type)
        
        # Date range filtering
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        if start_date:
            try:
                start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                transitions = transitions.filter(transition_timestamp__gte=start_date)
            except ValueError:
                return Response({
                    'status': 'error',
                    'message': 'Invalid start_date format'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        if end_date:
            try:
                end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                transitions = transitions.filter(transition_timestamp__lte=end_date)
            except ValueError:
                return Response({
                    'status': 'error',
                    'message': 'Invalid end_date format'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Pagination
        limit = min(int(request.GET.get('limit', 50)), 100)  # Max 100 transitions
        offset = int(request.GET.get('offset', 0))
        
        total_count = transitions.count()
        transitions = transitions[offset:offset + limit]
        
        # Serialize transition data
        transition_data = []
        for transition in transitions:
            transition_data.append({
                'id': transition.id,
                'stope_id': transition.stope.id,
                'stope_name': transition.stope.stope_name,
                'previous_risk_level': transition.previous_risk_level,
                'new_risk_level': transition.new_risk_level,
                'trigger_type': transition.trigger_type,
                'trigger_value': transition.trigger_value,
                'transition_timestamp': transition.transition_timestamp.isoformat(),
                'is_escalation': transition.is_escalation,
                'is_de_escalation': transition.is_de_escalation,
                'risk_level_delta': transition.risk_level_delta,
                'duration_in_previous_level': str(transition.duration_in_previous_level) if transition.duration_in_previous_level else None,
                'is_confirmed': transition.is_confirmed,
                'confidence_score': transition.confidence_score,
                'impact_assessment': transition.impact_assessment
            })
        
        return Response({
            'status': 'success',
            'total_count': total_count,
            'count': len(transition_data),
            'offset': offset,
            'limit': limit,
            'transitions': transition_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error getting risk transitions: {e}")
        return Response({
            'status': 'error',
            'message': f'Error retrieving transitions: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_risk_analytics(request):
    """
    Get risk analytics and statistics.
    
    Provides comprehensive analytics on risk patterns and trends.
    """
    try:
        # Time range for analysis (default: last 30 days)
        days_back = int(request.GET.get('days_back', 30))
        start_date = timezone.now() - timedelta(days=days_back)
        
        # Risk level distribution
        current_risk_levels = {}
        all_stopes = Stope.objects.filter(is_active=True)
        for stope in all_stopes:
            risk_level = risk_classification_service.classify_stope_risk_level(stope)
            current_risk_levels[risk_level] = current_risk_levels.get(risk_level, 0) + 1
        
        # Transition statistics
        transitions = RiskTransition.objects.filter(
            transition_timestamp__gte=start_date
        )
        
        transition_stats = {
            'total_transitions': transitions.count(),
            'escalations': transitions.filter(
                new_risk_level__in=['elevated', 'high_risk', 'critical', 'emergency']
            ).count(),
            'de_escalations': sum(1 for t in transitions if t.is_de_escalation),
            'by_trigger_type': {}
        }
        
        # Group by trigger type
        for trigger_type, _ in RiskTransition.TRANSITION_TRIGGER_CHOICES:
            count = transitions.filter(trigger_type=trigger_type).count()
            if count > 0:
                transition_stats['by_trigger_type'][trigger_type] = count
        
        # Alert statistics
        alerts = RiskAlert.objects.filter(created_at__gte=start_date)
        alert_stats = {
            'total_alerts': alerts.count(),
            'active_alerts': alerts.filter(status='active').count(),
            'resolved_alerts': alerts.filter(status='resolved').count(),
            'by_priority': {}
        }
        
        for priority, _ in RiskAlert.PRIORITY_CHOICES:
            count = alerts.filter(priority=priority).count()
            if count > 0:
                alert_stats['by_priority'][priority] = count
        
        # Most active stopes (by transitions)
        stope_activity = (
            transitions.values('stope__stope_name', 'stope__id')
            .annotate(transition_count=Count('id'))
            .order_by('-transition_count')[:10]
        )
        
        return Response({
            'status': 'success',
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': timezone.now().isoformat(),
                'days_analyzed': days_back
            },
            'current_risk_distribution': current_risk_levels,
            'transition_statistics': transition_stats,
            'alert_statistics': alert_stats,
            'most_active_stopes': list(stope_activity),
            'timestamp': timezone.now().isoformat()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error getting risk analytics: {e}")
        return Response({
            'status': 'error',
            'message': f'Error retrieving analytics: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
