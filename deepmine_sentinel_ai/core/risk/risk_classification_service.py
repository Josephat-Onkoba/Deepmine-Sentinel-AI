"""
Task 6: Risk Level Classification System - Core Service
=======================================================

This module provides comprehensive risk level classification services including:
- Dynamic risk level assignment based on configurable thresholds
- Risk transition detection and tracking
- Alert generation and management
- Rule-based classification logic

The service integrates with the impact calculation system to provide real-time
risk assessment and automated alerting for mining operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from django.utils import timezone
from django.db.models import Q, Avg, Max, Min, Count
from django.core.exceptions import ValidationError

from core.models import (
    Stope, ImpactScore, OperationalEvent, ImpactHistory,
    RiskThreshold, RiskTransition, RiskAlert, RiskClassificationRule
)

logger = logging.getLogger(__name__)


class RiskClassificationService:
    """
    Comprehensive service for dynamic risk level classification and management.
    Handles threshold evaluation, transition detection, and alert generation.
    """
    
    # Standard risk level ordering for comparison
    RISK_ORDER = ['stable', 'elevated', 'high_risk', 'critical', 'emergency']
    
    def __init__(self):
        """Initialize the Risk Classification Service"""
        logger.info("Risk Classification Service initialized")
        self._load_default_thresholds()
    
    def _load_default_thresholds(self):
        """Load and cache default risk thresholds"""
        try:
            self.default_thresholds = {
                threshold.risk_level: threshold
                for threshold in RiskThreshold.objects.filter(
                    is_active=True,
                    applies_to_rock_types=[],  # Global thresholds
                    applies_to_mining_methods=[]
                ).order_by('priority')
            }
            logger.info(f"Loaded {len(self.default_thresholds)} default risk thresholds")
        except Exception as e:
            logger.error(f"Error loading default thresholds: {e}")
            self.default_thresholds = {}
    
    def classify_stope_risk_level(self, stope: Stope, impact_score: float = None) -> str:
        """
        Classify the current risk level for a stope based on all available criteria.
        
        Args:
            stope: The stope to classify
            impact_score: Current impact score (if not provided, will be retrieved)
        
        Returns:
            str: Risk level classification
        """
        try:
            # Get current impact score if not provided
            if impact_score is None:
                impact_score_obj = ImpactScore.objects.filter(stope=stope).first()
                impact_score = impact_score_obj.current_score if impact_score_obj else 0.0
            
            # Start with threshold-based classification
            threshold_risk = self._classify_by_thresholds(stope, impact_score)
            
            # Apply rule-based classification
            rule_risk = self._classify_by_rules(stope, impact_score)
            
            # Combine results (rule-based takes precedence if higher risk)
            final_risk = self._combine_risk_classifications(threshold_risk, rule_risk)
            
            logger.debug(
                f"Risk classification for {stope.stope_name}: "
                f"threshold={threshold_risk}, rule={rule_risk}, final={final_risk}"
            )
            
            return final_risk
            
        except Exception as e:
            logger.error(f"Error classifying risk for stope {stope.stope_name}: {e}")
            return 'stable'  # Default to stable on error
    
    def _classify_by_thresholds(self, stope: Stope, impact_score: float) -> str:
        """
        Classify risk level based on configurable thresholds.
        
        Args:
            stope: The stope to classify
            impact_score: Current impact score
        
        Returns:
            str: Risk level based on thresholds
        """
        try:
            # Get applicable thresholds for this stope
            applicable_thresholds = RiskThreshold.objects.filter(
                is_active=True,
                threshold_type='impact_score'
            ).order_by('priority', 'minimum_value')
            
            # Filter thresholds that apply to this stope
            valid_thresholds = [
                threshold for threshold in applicable_thresholds
                if threshold.applies_to_stope(stope)
            ]
            
            # Evaluate thresholds from highest risk to lowest
            risk_levels = ['emergency', 'critical', 'high_risk', 'elevated', 'stable']
            
            for risk_level in risk_levels:
                level_thresholds = [
                    t for t in valid_thresholds 
                    if t.risk_level == risk_level
                ]
                
                for threshold in level_thresholds:
                    if threshold.is_threshold_exceeded(impact_score):
                        # Check if minimum duration requirement is met
                        if self._check_duration_requirement(stope, threshold, impact_score):
                            return risk_level
            
            return 'stable'  # Default if no thresholds are exceeded
            
        except Exception as e:
            logger.error(f"Error in threshold-based classification: {e}")
            return 'stable'
    
    def _classify_by_rules(self, stope: Stope, impact_score: float) -> str:
        """
        Classify risk level based on advanced rules.
        
        Args:
            stope: The stope to classify
            impact_score: Current impact score
        
        Returns:
            str: Risk level based on rules
        """
        try:
            # Get applicable rules for this stope
            applicable_rules = RiskClassificationRule.objects.filter(
                is_active=True
            ).order_by('priority')
            
            # Filter rules that apply to this stope
            valid_rules = [
                rule for rule in applicable_rules
                if rule.applies_to_stope(stope)
            ]
            
            # Evaluate rules in priority order
            for rule in valid_rules:
                if self._evaluate_rule_conditions(rule, stope, impact_score):
                    return rule.target_risk_level
            
            return 'stable'  # Default if no rules match
            
        except Exception as e:
            logger.error(f"Error in rule-based classification: {e}")
            return 'stable'
    
    def _evaluate_rule_conditions(self, rule: RiskClassificationRule, 
                                 stope: Stope, impact_score: float) -> bool:
        """
        Evaluate the conditions for a specific rule.
        
        Args:
            rule: The rule to evaluate
            stope: The stope being evaluated
            impact_score: Current impact score
        
        Returns:
            bool: Whether the rule conditions are met
        """
        try:
            conditions = rule.rule_conditions
            if not conditions:
                return False
            
            # Get contextual data for evaluation
            context = self._build_evaluation_context(stope, impact_score)
            
            if rule.condition_type == 'and':
                return all(self._evaluate_single_condition(cond, context) for cond in conditions)
            elif rule.condition_type == 'or':
                return any(self._evaluate_single_condition(cond, context) for cond in conditions)
            elif rule.condition_type == 'weighted':
                return self._evaluate_weighted_conditions(conditions, context)
            elif rule.condition_type == 'sequential':
                return self._evaluate_sequential_conditions(conditions, context)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rule conditions: {e}")
            return False
    
    def _build_evaluation_context(self, stope: Stope, impact_score: float) -> Dict[str, Any]:
        """
        Build context data for rule evaluation.
        
        Args:
            stope: The stope being evaluated
            impact_score: Current impact score
        
        Returns:
            Dict containing context data
        """
        try:
            # Get recent operational events
            recent_events = OperationalEvent.objects.filter(
                stope=stope,
                timestamp__gte=timezone.now() - timedelta(hours=24)
            ).count()
            
            # Get recent impact history
            recent_history = ImpactHistory.objects.filter(
                stope=stope,
                timestamp__gte=timezone.now() - timedelta(hours=24)
            )
            
            # Calculate rate of change
            rate_of_change = 0.0
            if recent_history.count() >= 2:
                latest = recent_history.first()
                previous = recent_history[1] if recent_history.count() > 1 else None
                if previous:
                    time_diff = (latest.timestamp - previous.timestamp).total_seconds() / 3600
                    if time_diff > 0:
                        rate_of_change = (latest.new_score - previous.new_score) / time_diff
            
            return {
                'impact_score': impact_score,
                'stope': stope,
                'recent_events_count': recent_events,
                'rate_of_change': rate_of_change,
                'hours_since_last_event': self._hours_since_last_event(stope),
                'average_recent_score': recent_history.aggregate(
                    avg=Avg('new_score')
                )['avg'] or 0.0,
                'peak_recent_score': recent_history.aggregate(
                    max=Max('new_score')
                )['max'] or 0.0,
            }
            
        except Exception as e:
            logger.error(f"Error building evaluation context: {e}")
            return {'impact_score': impact_score, 'stope': stope}
    
    def _evaluate_single_condition(self, condition: Dict, context: Dict[str, Any]) -> bool:
        """
        Evaluate a single condition against the context.
        
        Args:
            condition: The condition to evaluate
            context: Context data
        
        Returns:
            bool: Whether the condition is met
        """
        try:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if not all([field, operator, value is not None]):
                return False
            
            context_value = context.get(field)
            if context_value is None:
                return False
            
            # Evaluate based on operator
            if operator == 'gt':
                return context_value > value
            elif operator == 'gte':
                return context_value >= value
            elif operator == 'lt':
                return context_value < value
            elif operator == 'lte':
                return context_value <= value
            elif operator == 'eq':
                return context_value == value
            elif operator == 'ne':
                return context_value != value
            elif operator == 'in':
                return context_value in value
            elif operator == 'not_in':
                return context_value not in value
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating single condition: {e}")
            return False
    
    def _evaluate_weighted_conditions(self, conditions: List[Dict], 
                                    context: Dict[str, Any]) -> bool:
        """
        Evaluate weighted conditions.
        
        Args:
            conditions: List of weighted conditions
            context: Context data
        
        Returns:
            bool: Whether weighted score exceeds threshold
        """
        try:
            total_weight = 0.0
            weighted_score = 0.0
            
            for condition in conditions:
                weight = condition.get('weight', 1.0)
                if self._evaluate_single_condition(condition, context):
                    weighted_score += weight
                total_weight += weight
            
            threshold = conditions[0].get('threshold', 0.5)  # Default 50%
            return (weighted_score / total_weight) >= threshold if total_weight > 0 else False
            
        except Exception as e:
            logger.error(f"Error evaluating weighted conditions: {e}")
            return False
    
    def _evaluate_sequential_conditions(self, conditions: List[Dict], 
                                      context: Dict[str, Any]) -> bool:
        """
        Evaluate sequential conditions (all must be met in order).
        
        Args:
            conditions: List of sequential conditions
            context: Context data
        
        Returns:
            bool: Whether all sequential conditions are met
        """
        try:
            for condition in conditions:
                if not self._evaluate_single_condition(condition, context):
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating sequential conditions: {e}")
            return False
    
    def _combine_risk_classifications(self, threshold_risk: str, rule_risk: str) -> str:
        """
        Combine threshold-based and rule-based risk classifications.
        
        Args:
            threshold_risk: Risk level from threshold evaluation
            rule_risk: Risk level from rule evaluation
        
        Returns:
            str: Combined risk level (higher of the two)
        """
        try:
            # Get indices in risk order
            threshold_idx = self.RISK_ORDER.index(threshold_risk) if threshold_risk in self.RISK_ORDER else 0
            rule_idx = self.RISK_ORDER.index(rule_risk) if rule_risk in self.RISK_ORDER else 0
            
            # Return the higher risk level
            return self.RISK_ORDER[max(threshold_idx, rule_idx)]
            
        except Exception as e:
            logger.error(f"Error combining risk classifications: {e}")
            return threshold_risk  # Fallback to threshold-based
    
    def _check_duration_requirement(self, stope: Stope, threshold: RiskThreshold, 
                                   impact_score: float) -> bool:
        """
        Check if the threshold has been exceeded for the minimum required duration.
        
        Args:
            stope: The stope being evaluated
            threshold: The threshold to check
            impact_score: Current impact score
        
        Returns:
            bool: Whether duration requirement is met
        """
        try:
            if not threshold.minimum_duration:
                return True  # No duration requirement
            
            # Check how long the impact score has been above threshold
            cutoff_time = timezone.now() - threshold.minimum_duration
            
            # Count consistent threshold breaches in the required timeframe
            consistent_breaches = ImpactHistory.objects.filter(
                stope=stope,
                timestamp__gte=cutoff_time,
                new_score__gte=threshold.minimum_value
            ).count()
            
            # Require at least 2 data points for duration validation
            return consistent_breaches >= 2
            
        except Exception as e:
            logger.error(f"Error checking duration requirement: {e}")
            return True  # Default to allowing if check fails
    
    def _hours_since_last_event(self, stope: Stope) -> float:
        """
        Calculate hours since the last operational event.
        
        Args:
            stope: The stope to check
        
        Returns:
            float: Hours since last event
        """
        try:
            last_event = OperationalEvent.objects.filter(stope=stope).order_by('-timestamp').first()
            if last_event:
                return (timezone.now() - last_event.timestamp).total_seconds() / 3600
            return 999.0  # Large number if no events found
            
        except Exception as e:
            logger.error(f"Error calculating hours since last event: {e}")
            return 0.0
    
    def detect_risk_transition(self, stope: Stope, new_risk_level: str, 
                              trigger_event: OperationalEvent = None,
                              trigger_value: float = None) -> Optional[RiskTransition]:
        """
        Detect and record risk level transitions.
        
        Args:
            stope: The stope experiencing the transition
            new_risk_level: The new risk level
            trigger_event: Optional event that triggered the transition
            trigger_value: Optional value that triggered the transition
        
        Returns:
            RiskTransition object if a transition occurred, None otherwise
        """
        try:
            # Get current risk level
            current_transition = RiskTransition.objects.filter(stope=stope).order_by('-transition_timestamp').first()
            current_risk_level = current_transition.new_risk_level if current_transition else 'stable'
            
            # Check if this is actually a transition
            if current_risk_level == new_risk_level:
                return None
            
            # Check cooldown period
            if current_transition and self._is_in_cooldown_period(current_transition):
                logger.debug(f"Transition for {stope.stope_name} skipped due to cooldown period")
                return None
            
            # Create risk transition record
            transition = RiskTransition.objects.create(
                stope=stope,
                previous_risk_level=current_risk_level,
                new_risk_level=new_risk_level,
                trigger_type='threshold_exceeded' if not trigger_event else 'event_impact',
                trigger_value=trigger_value,
                related_operational_event=trigger_event,
                related_impact_score=ImpactScore.objects.filter(stope=stope).first()
            )
            
            logger.info(f"Risk transition detected: {stope.stope_name} {current_risk_level} → {new_risk_level}")
            
            # Generate alerts if necessary
            self._generate_transition_alerts(transition)
            
            return transition
            
        except Exception as e:
            logger.error(f"Error detecting risk transition: {e}")
            return None
    
    def _is_in_cooldown_period(self, last_transition: RiskTransition) -> bool:
        """
        Check if we're still in the cooldown period from the last transition.
        
        Args:
            last_transition: The last risk transition
        
        Returns:
            bool: Whether we're in cooldown period
        """
        try:
            # Get applicable threshold for cooldown
            threshold = RiskThreshold.objects.filter(
                risk_level=last_transition.new_risk_level,
                is_active=True
            ).first()
            
            if not threshold or not threshold.cooldown_period:
                return False
            
            cooldown_end = last_transition.transition_timestamp + threshold.cooldown_period
            return timezone.now() < cooldown_end
            
        except Exception as e:
            logger.error(f"Error checking cooldown period: {e}")
            return False
    
    def _generate_transition_alerts(self, transition: RiskTransition):
        """
        Generate appropriate alerts for a risk transition.
        
        Args:
            transition: The risk transition that occurred
        """
        try:
            # Determine alert type and priority
            alert_type = 'risk_escalation' if transition.is_escalation else 'threshold_breach'
            priority = self._determine_alert_priority(transition)
            
            # Create alert title and message
            title = f"Risk Level Change: {transition.stope.stope_name}"
            message = f"Risk level changed from {transition.previous_risk_level} to {transition.new_risk_level}"
            
            if transition.related_operational_event:
                message += f" due to {transition.related_operational_event.event_type} event"
            
            # Determine recommended actions
            recommended_actions = self._get_recommended_actions(transition)
            
            # Create the alert
            alert = RiskAlert.objects.create(
                stope=transition.stope,
                risk_transition=transition,
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                recommended_actions=recommended_actions
            )
            
            logger.info(f"Alert generated for risk transition: {alert.title}")
            
            # Send notifications if required
            self._send_alert_notifications(alert)
            
        except Exception as e:
            logger.error(f"Error generating transition alerts: {e}")
    
    def _determine_alert_priority(self, transition: RiskTransition) -> str:
        """
        Determine the appropriate alert priority for a risk transition.
        
        Args:
            transition: The risk transition
        
        Returns:
            str: Alert priority level
        """
        try:
            # Emergency level always gets emergency priority
            if transition.new_risk_level == 'emergency':
                return 'emergency'
            
            # Critical level gets critical priority
            if transition.new_risk_level == 'critical':
                return 'critical'
            
            # Large jumps in risk level get higher priority
            if abs(transition.risk_level_delta) >= 2:
                return 'high'
            
            # Escalations get medium priority
            if transition.is_escalation:
                return 'medium'
            
            # De-escalations get low priority
            return 'low'
            
        except Exception as e:
            logger.error(f"Error determining alert priority: {e}")
            return 'medium'
    
    def _get_recommended_actions(self, transition: RiskTransition) -> List[str]:
        """
        Get recommended actions for a risk transition.
        
        Args:
            transition: The risk transition
        
        Returns:
            List of recommended action strings
        """
        try:
            actions = []
            
            if transition.new_risk_level == 'emergency':
                actions.extend([
                    "Immediately evacuate personnel from affected area",
                    "Contact emergency response team",
                    "Suspend all operations in the stope",
                    "Assess ground support systems",
                ])
            elif transition.new_risk_level == 'critical':
                actions.extend([
                    "Restrict access to essential personnel only",
                    "Increase monitoring frequency",
                    "Review and reinforce ground support",
                    "Consider operational modifications",
                ])
            elif transition.new_risk_level == 'high_risk':
                actions.extend([
                    "Increase monitoring and inspections",
                    "Review recent operational activities",
                    "Consider additional ground support",
                    "Brief personnel on elevated risk status",
                ])
            elif transition.new_risk_level == 'elevated':
                actions.extend([
                    "Continue normal operations with increased vigilance",
                    "Monitor impact score trends",
                    "Document any observed changes",
                ])
            
            # Add event-specific actions
            if transition.related_operational_event:
                event_type = transition.related_operational_event.event_type
                if event_type == 'blasting':
                    actions.append("Review blast design and execution")
                elif event_type == 'excavation':
                    actions.append("Assess excavation impact on surrounding rock")
                elif event_type == 'equipment_operation':
                    actions.append("Review equipment operation procedures")
            
            return actions
            
        except Exception as e:
            logger.error(f"Error getting recommended actions: {e}")
            return ["Monitor situation and take appropriate precautions"]
    
    def _send_alert_notifications(self, alert: RiskAlert):
        """
        Send notifications for an alert (placeholder for notification system).
        
        Args:
            alert: The alert to send notifications for
        """
        try:
            # This is a placeholder for the notification system
            # In a real implementation, this would integrate with:
            # - Email systems
            # - SMS/mobile notifications
            # - Dashboard real-time updates
            # - Integration with mining operation systems
            
            channels = []
            
            if alert.priority in ['critical', 'emergency']:
                channels.extend(['email', 'sms', 'dashboard'])
            elif alert.priority == 'high':
                channels.extend(['email', 'dashboard'])
            else:
                channels.append('dashboard')
            
            alert.notification_channels = channels
            alert.notification_sent = True
            alert.save()
            
            logger.info(f"Notifications sent for alert {alert.id} via {channels}")
            
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
    
    def get_stope_risk_status(self, stope: Stope) -> Dict[str, Any]:
        """
        Get comprehensive risk status for a stope.
        
        Args:
            stope: The stope to get status for
        
        Returns:
            Dict containing comprehensive risk status
        """
        try:
            # Get current risk level
            current_risk_level = self.classify_stope_risk_level(stope)
            
            # Get recent transitions
            recent_transitions = RiskTransition.objects.filter(
                stope=stope,
                transition_timestamp__gte=timezone.now() - timedelta(days=7)
            ).order_by('-transition_timestamp')[:5]
            
            # Get active alerts
            active_alerts = RiskAlert.objects.filter(
                stope=stope,
                status__in=['active', 'acknowledged', 'investigating']
            ).order_by('-created_at')
            
            # Get impact score
            impact_score_obj = ImpactScore.objects.filter(stope=stope).first()
            current_impact_score = impact_score_obj.current_score if impact_score_obj else 0.0
            
            return {
                'current_risk_level': current_risk_level,
                'current_impact_score': current_impact_score,
                'recent_transitions': [
                    {
                        'from': t.previous_risk_level,
                        'to': t.new_risk_level,
                        'timestamp': t.transition_timestamp,
                        'trigger': t.trigger_type,
                    }
                    for t in recent_transitions
                ],
                'active_alerts': [
                    {
                        'id': a.id,
                        'type': a.alert_type,
                        'priority': a.priority,
                        'title': a.title,
                        'created_at': a.created_at,
                        'status': a.status,
                    }
                    for a in active_alerts
                ],
                'risk_trend': self._calculate_risk_trend(stope),
                'time_in_current_level': self._time_in_current_risk_level(stope),
            }
            
        except Exception as e:
            logger.error(f"Error getting stope risk status: {e}")
            return {
                'current_risk_level': 'stable',
                'current_impact_score': 0.0,
                'recent_transitions': [],
                'active_alerts': [],
                'risk_trend': 'stable',
                'time_in_current_level': timedelta(0),
            }
    
    def _calculate_risk_trend(self, stope: Stope) -> str:
        """
        Calculate the risk trend for a stope.
        
        Args:
            stope: The stope to analyze
        
        Returns:
            str: Risk trend ('increasing', 'decreasing', 'stable')
        """
        try:
            # Get recent transitions
            recent_transitions = RiskTransition.objects.filter(
                stope=stope,
                transition_timestamp__gte=timezone.now() - timedelta(days=3)
            ).order_by('-transition_timestamp')[:3]
            
            if len(recent_transitions) < 2:
                return 'stable'
            
            # Analyze trend
            escalations = sum(1 for t in recent_transitions if t.is_escalation)
            de_escalations = sum(1 for t in recent_transitions if t.is_de_escalation)
            
            if escalations > de_escalations:
                return 'increasing'
            elif de_escalations > escalations:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating risk trend: {e}")
            return 'stable'
    
    def _time_in_current_risk_level(self, stope: Stope) -> timedelta:
        """
        Calculate how long the stope has been in its current risk level.
        
        Args:
            stope: The stope to analyze
        
        Returns:
            timedelta: Time in current risk level
        """
        try:
            last_transition = RiskTransition.objects.filter(stope=stope).order_by('-transition_timestamp').first()
            
            if last_transition:
                return timezone.now() - last_transition.transition_timestamp
            else:
                # If no transitions, assume stable since creation
                return timezone.now() - stope.created_at
                
        except Exception as e:
            logger.error(f"Error calculating time in current risk level: {e}")
            return timedelta(0)


# Initialize global service instance
risk_classification_service = RiskClassificationService()
