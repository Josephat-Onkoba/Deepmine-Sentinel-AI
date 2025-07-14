"""
Impact Calculation Service

High-level service for orchestrating impact calculations across the mining operation.
Provides real-time impact monitoring, scheduled batch updates, and alert generation.

This service integrates the mathematical calculator with the application infrastructure.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from django.utils import timezone
from django.core.cache import cache
from django.db.models import Avg
from django.db import transaction
from django.conf import settings
from concurrent.futures import ThreadPoolExecutor
import threading

from core.models import (
    Stope, OperationalEvent, ImpactScore, ImpactHistory, 
    ImpactFactor, MonitoringData
)
from .impact_calculator import MathematicalImpactCalculator
from core.utils import send_notification, log_system_event

logger = logging.getLogger(__name__)


@dataclass
class ImpactAnalysisResult:
    """Comprehensive result of impact analysis"""
    stope_id: int
    stope_name: str
    current_impact: float
    previous_impact: float
    impact_change: float
    risk_level: str
    risk_level_changed: bool
    contributing_events: List[Dict]
    analysis_timestamp: datetime
    recommendations: List[str]
    alert_triggered: bool


@dataclass
class SystemImpactSummary:
    """System-wide impact analysis summary"""
    total_stopes: int
    stable_stopes: int
    elevated_stopes: int
    high_risk_stopes: int
    critical_stopes: int
    recent_events: int
    average_impact: float
    peak_impact: float
    trending_direction: str  # 'improving', 'stable', 'deteriorating'
    analysis_timestamp: datetime


class ImpactCalculationService:
    """
    High-level service for impact calculation and monitoring.
    
    Provides:
    - Real-time impact monitoring
    - Scheduled batch updates
    - Alert generation and notifications
    - Performance optimization through caching and batching
    - Integration with monitoring systems
    """
    
    def __init__(self):
        """Initialize the impact calculation service"""
        self.calculator = MathematicalImpactCalculator(cache_calculations=True)
        self.is_running = False
        self.update_interval = 300  # 5 minutes default
        self.batch_size = 50  # Stopes per batch
        self._lock = threading.Lock()
        
        # Performance monitoring
        self.calculation_times = []
        self.error_count = 0
        self.last_update = None
        
        logger.info("Impact Calculation Service initialized")
    
    def calculate_real_time_impact(
        self,
        event: OperationalEvent,
        affected_stopes: Optional[List[Stope]] = None
    ) -> List[ImpactAnalysisResult]:
        """
        Calculate immediate impact of a new operational event.
        
        Called when new events are logged to provide immediate impact assessment.
        
        Args:
            event: The new operational event
            affected_stopes: Specific stopes to analyze (default: all within range)
            
        Returns:
            List of impact analysis results for affected stopes
        """
        start_time = timezone.now()
        logger.info(f"Calculating real-time impact for {event}")
        
        try:
            # Determine affected stopes if not provided
            if affected_stopes is None:
                affected_stopes = self._get_affected_stopes(event)
            
            if not affected_stopes:
                logger.warning(f"No stopes found in impact range of {event}")
                return []
            
            # Calculate impacts
            results = []
            for stope in affected_stopes:
                try:
                    result = self._analyze_stope_impact(stope, event, start_time)
                    results.append(result)
                    
                    # Trigger alerts if necessary
                    if result.alert_triggered:
                        self._handle_impact_alert(result, event)
                
                except Exception as e:
                    logger.error(f"Error analyzing impact on {stope}: {e}")
                    self.error_count += 1
            
            # Update performance metrics
            calculation_time = (timezone.now() - start_time).total_seconds()
            self.calculation_times.append(calculation_time)
            if len(self.calculation_times) > 100:  # Keep last 100 calculations
                self.calculation_times.pop(0)
            
            logger.info(f"Real-time impact calculation complete: {len(results)} stopes analyzed in {calculation_time:.2f}s")
            return results
        
        except Exception as e:
            logger.error(f"Critical error in real-time impact calculation: {e}")
            self.error_count += 1
            return []
    
    def run_batch_update(
        self,
        stope_ids: Optional[List[int]] = None,
        force_update: bool = False
    ) -> Dict[str, int]:
        """
        Run batch update of impact scores for multiple stopes.
        
        Optimized for periodic scheduled updates.
        
        Args:
            stope_ids: Specific stopes to update (default: all active)
            force_update: Force update even if recently calculated
            
        Returns:
            Update statistics
        """
        start_time = timezone.now()
        logger.info("Starting batch impact update")
        
        with self._lock:
            try:
                # Get stopes to update
                if stope_ids:
                    stopes = list(Stope.objects.filter(
                        id__in=stope_ids,
                        is_active=True
                    ))
                else:
                    stopes = list(Stope.objects.filter(is_active=True))
                
                # Filter out recently updated stopes unless forced
                if not force_update:
                    stopes = self._filter_stales_for_update(stopes)
                
                if not stopes:
                    logger.info("No stopes require updating")
                    return {'total_stopes': 0, 'updated_scores': 0, 'risk_level_changes': 0, 'errors': 0}
                
                # Process in batches for memory efficiency
                all_stats = {
                    'total_stopes': 0,
                    'updated_scores': 0,
                    'risk_level_changes': 0,
                    'errors': 0
                }
                
                for i in range(0, len(stopes), self.batch_size):
                    batch = stopes[i:i + self.batch_size]
                    logger.debug(f"Processing batch {i//self.batch_size + 1}: {len(batch)} stopes")
                    
                    batch_stats = self.calculator.batch_update_stope_scores(
                        stopes=batch,
                        calculation_time=start_time
                    )
                    
                    # Aggregate statistics
                    for key in all_stats:
                        all_stats[key] += batch_stats.get(key, 0)
                
                # Update service metadata
                self.last_update = start_time
                calculation_time = (timezone.now() - start_time).total_seconds()
                
                logger.info(f"Batch update complete: {all_stats['updated_scores']} stopes updated in {calculation_time:.2f}s")
                
                # Generate system-wide summary
                summary = self.generate_system_summary()
                self._cache_system_summary(summary)
                
                return all_stats
            
            except Exception as e:
                logger.error(f"Critical error in batch update: {e}")
                self.error_count += 1
                raise
    
    def generate_system_summary(self) -> SystemImpactSummary:
        """
        Generate comprehensive system-wide impact analysis.
        
        Provides executive dashboard information.
        """
        logger.debug("Generating system impact summary")
        
        try:
            # Get all active stopes with current impact scores
            stopes_with_scores = ImpactScore.objects.filter(
                stope__is_active=True
            ).select_related('stope')
            
            # Calculate risk level distribution
            risk_counts = {'stable': 0, 'elevated': 0, 'high_risk': 0, 'critical': 0}
            impact_scores = []
            
            for score in stopes_with_scores:
                risk_counts[score.risk_level] += 1
                impact_scores.append(score.current_score)
            
            # Calculate statistics
            total_stopes = len(impact_scores)
            average_impact = sum(impact_scores) / total_stopes if impact_scores else 0.0
            peak_impact = max(impact_scores) if impact_scores else 0.0
            
            # Count recent events
            recent_events = OperationalEvent.objects.filter(
                timestamp__gte=timezone.now() - timedelta(hours=24)
            ).count()
            
            # Determine trending direction
            trending_direction = self._calculate_trending_direction()
            
            summary = SystemImpactSummary(
                total_stopes=total_stopes,
                stable_stopes=risk_counts['stable'],
                elevated_stopes=risk_counts['elevated'],
                high_risk_stopes=risk_counts['high_risk'],
                critical_stopes=risk_counts['critical'],
                recent_events=recent_events,
                average_impact=average_impact,
                peak_impact=peak_impact,
                trending_direction=trending_direction,
                analysis_timestamp=timezone.now()
            )
            
            logger.debug(f"System summary: {total_stopes} stopes, avg impact {average_impact:.3f}")
            return summary
        
        except Exception as e:
            logger.error(f"Error generating system summary: {e}")
            # Return empty summary on error
            return SystemImpactSummary(
                total_stopes=0, stable_stopes=0, elevated_stopes=0,
                high_risk_stopes=0, critical_stopes=0, recent_events=0,
                average_impact=0.0, peak_impact=0.0, trending_direction='unknown',
                analysis_timestamp=timezone.now()
            )
    
    def get_stope_impact_analysis(
        self, 
        stope: Stope,
        include_contributing_events: bool = True,
        time_window_hours: int = 168
    ) -> ImpactAnalysisResult:
        """
        Get detailed impact analysis for a specific stope.
        
        Provides comprehensive analysis including contributing events and recommendations.
        """
        logger.debug(f"Generating detailed impact analysis for {stope}")
        
        try:
            # Get current impact score
            try:
                current_score = ImpactScore.objects.get(stope=stope)
                current_impact = current_score.current_score
                risk_level = current_score.risk_level
            except ImpactScore.DoesNotExist:
                # Calculate if not exists
                current_score = self.calculator.update_stope_impact_score(stope)
                current_impact = current_score.current_score
                risk_level = current_score.risk_level
            
            # Get previous impact for comparison
            previous_impact = self._get_previous_impact(stope)
            impact_change = current_impact - previous_impact
            risk_level_changed = self._check_risk_level_change(stope)
            
            # Get contributing events if requested
            contributing_events = []
            if include_contributing_events:
                contributing_events = self._get_contributing_events(stope, time_window_hours)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(stope, current_impact, risk_level)
            
            # Check if alert should be triggered
            alert_triggered = self._should_trigger_alert(stope, current_impact, risk_level_changed)
            
            result = ImpactAnalysisResult(
                stope_id=stope.id,
                stope_name=stope.stope_name,
                current_impact=current_impact,
                previous_impact=previous_impact,
                impact_change=impact_change,
                risk_level=risk_level,
                risk_level_changed=risk_level_changed,
                contributing_events=contributing_events,
                analysis_timestamp=timezone.now(),
                recommendations=recommendations,
                alert_triggered=alert_triggered
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating stope analysis for {stope}: {e}")
            # Return safe default on error
            return ImpactAnalysisResult(
                stope_id=stope.id,
                stope_name=stope.stope_name,
                current_impact=0.0,
                previous_impact=0.0,
                impact_change=0.0,
                risk_level='unknown',
                risk_level_changed=False,
                contributing_events=[],
                analysis_timestamp=timezone.now(),
                recommendations=["Error: Unable to analyze stope impact"],
                alert_triggered=False
            )
    
    def start_continuous_monitoring(self, update_interval: int = 300):
        """
        Start continuous monitoring service.
        
        Runs periodic impact updates and monitoring in background.
        """
        if self.is_running:
            logger.warning("Continuous monitoring already running")
            return
        
        self.is_running = True
        self.update_interval = update_interval
        
        logger.info(f"Starting continuous monitoring (update interval: {update_interval}s)")
        
        def monitoring_loop():
            while self.is_running:
                try:
                    # Run batch update
                    stats = self.run_batch_update()
                    
                    # Check for system-wide alerts
                    self._check_system_alerts()
                    
                    # Log performance
                    if self.calculation_times:
                        avg_time = sum(self.calculation_times) / len(self.calculation_times)
                        logger.info(f"Average calculation time: {avg_time:.2f}s, Error count: {self.error_count}")
                    
                    # Wait for next update
                    threading.Event().wait(self.update_interval)
                
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    self.error_count += 1
                    # Wait before retrying
                    threading.Event().wait(60)
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        log_system_event("impact_monitoring_started", {"update_interval": update_interval})
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring service"""
        if not self.is_running:
            logger.warning("Continuous monitoring not running")
            return
        
        self.is_running = False
        logger.info("Stopping continuous monitoring")
        log_system_event("impact_monitoring_stopped", {})
    
    # Private helper methods
    
    def _get_affected_stopes(self, event: OperationalEvent) -> List[Stope]:
        """Determine which stopes are affected by an event"""
        # Simple implementation - could be enhanced with spatial analysis
        max_distance = 500.0  # meters
        
        # For now, get all active stopes (would be filtered by distance in production)
        affected_stopes = list(Stope.objects.filter(is_active=True))
        
        # Could add distance filtering here based on actual coordinates
        return affected_stopes[:10]  # Limit to first 10 for performance
    
    def _analyze_stope_impact(
        self, 
        stope: Stope, 
        triggering_event: OperationalEvent, 
        analysis_time: datetime
    ) -> ImpactAnalysisResult:
        """Analyze impact on a single stope"""
        # Get previous state
        try:
            previous_score = ImpactScore.objects.get(stope=stope)
            previous_impact = previous_score.current_score
            previous_risk_level = previous_score.risk_level
        except ImpactScore.DoesNotExist:
            previous_impact = 0.0
            previous_risk_level = 'stable'
        
        # Update impact score
        updated_score = self.calculator.update_stope_impact_score(stope, analysis_time)
        
        # Calculate changes
        impact_change = updated_score.current_score - previous_impact
        risk_level_changed = previous_risk_level != updated_score.risk_level
        
        # Get contributing events
        contributing_events = self._get_contributing_events(stope, 24)  # Last 24 hours
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            stope, updated_score.current_score, updated_score.risk_level
        )
        
        # Check for alerts
        alert_triggered = (
            risk_level_changed and 
            updated_score.risk_level in ['high_risk', 'critical']
        )
        
        return ImpactAnalysisResult(
            stope_id=stope.id,
            stope_name=stope.stope_name,
            current_impact=updated_score.current_score,
            previous_impact=previous_impact,
            impact_change=impact_change,
            risk_level=updated_score.risk_level,
            risk_level_changed=risk_level_changed,
            contributing_events=contributing_events,
            analysis_timestamp=analysis_time,
            recommendations=recommendations,
            alert_triggered=alert_triggered
        )
    
    def _filter_stales_for_update(self, stopes: List[Stope]) -> List[Stope]:
        """Filter stopes that need updating based on last calculation time"""
        update_threshold = timezone.now() - timedelta(minutes=30)  # 30 minutes
        
        stales_to_update = []
        for stope in stopes:
            try:
                impact_score = ImpactScore.objects.get(stope=stope)
                if impact_score.last_calculated < update_threshold:
                    stales_to_update.append(stope)
            except ImpactScore.DoesNotExist:
                # No score exists, definitely needs update
                stales_to_update.append(stope)
        
        return stales_to_update
    
    def _get_previous_impact(self, stope: Stope) -> float:
        """Get previous impact score for comparison"""
        try:
            # Get most recent history record
            history = ImpactHistory.objects.filter(stope=stope).order_by('-timestamp').first()
            return history.previous_score if history else 0.0
        except Exception:
            return 0.0
    
    def _check_risk_level_change(self, stope: Stope) -> bool:
        """Check if risk level changed recently"""
        try:
            recent_history = ImpactHistory.objects.filter(
                stope=stope,
                timestamp__gte=timezone.now() - timedelta(hours=1)
            ).order_by('-timestamp').first()
            
            if recent_history:
                current_score = ImpactScore.objects.get(stope=stope)
                return recent_history.risk_level != current_score.risk_level
            
            return False
        except Exception:
            return False
    
    def _get_contributing_events(self, stope: Stope, time_window_hours: int) -> List[Dict]:
        """Get events contributing to current impact"""
        start_time = timezone.now() - timedelta(hours=time_window_hours)
        
        # Get recent events (simplified - could be enhanced with spatial filtering)
        events = OperationalEvent.objects.filter(
            timestamp__gte=start_time
        ).order_by('-timestamp')[:10]
        
        contributing = []
        for event in events:
            # Calculate this event's contribution
            impact_result = self.calculator.calculate_event_impact(
                event, stope
            )
            
            if impact_result.final_impact > 0.001:  # Significant contribution
                contributing.append({
                    'event_id': event.id,
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'timestamp': event.timestamp.isoformat(),
                    'impact_contribution': impact_result.final_impact,
                    'distance': impact_result.affected_distance
                })
        
        return contributing
    
    def _generate_recommendations(self, stope: Stope, impact_score: float, risk_level: str) -> List[str]:
        """Generate actionable recommendations based on impact analysis"""
        recommendations = []
        
        if risk_level == 'critical':
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Evacuate personnel from stope area",
                "Conduct emergency geotechnical inspection",
                "Implement additional ground support measures",
                "Review and halt high-impact operations near this stope"
            ])
        
        elif risk_level == 'high_risk':
            recommendations.extend([
                "Increase monitoring frequency for this stope",
                "Review recent operational activities in the area",
                "Consider installing additional monitoring sensors",
                "Schedule geotechnical assessment within 24 hours"
            ])
        
        elif risk_level == 'elevated':
            recommendations.extend([
                "Monitor stope closely for any changes",
                "Review operational procedures in surrounding areas",
                "Consider adjusting nearby activity schedules"
            ])
        
        else:  # stable
            recommendations.append("Continue normal monitoring protocols")
        
        return recommendations
    
    def _should_trigger_alert(self, stope: Stope, impact_score: float, risk_changed: bool) -> bool:
        """Determine if an alert should be triggered"""
        # Simple threshold-based alerting for now
        if risk_changed and impact_score > 3.0:
            return True
        if impact_score > 6.0:  # Critical threshold
            return True
        return False
    
    def _handle_impact_alert(self, result: ImpactAnalysisResult, triggering_event: OperationalEvent):
        """Handle alert generation and notification"""
        try:
            # Log the alert for now (could create Alert model records later)
            logger.warning(
                f"Impact Alert: {result.stope_name} - "
                f"Score: {result.current_impact:.3f}, Risk: {result.risk_level}"
            )
            
            # Send notifications (placeholder - could integrate with actual notification system)
            log_system_event("impact_alert", {
                "stope_name": result.stope_name,
                "stope_id": result.stope_id,
                "impact_score": result.current_impact,
                "risk_level": result.risk_level,
                "triggering_event_id": triggering_event.id
            })
            
            logger.warning(f"Alert triggered for {result.stope_name}: {result.risk_level}")
        
        except Exception as e:
            logger.error(f"Error handling impact alert: {e}")
    
    def _calculate_trending_direction(self) -> str:
        """Calculate system-wide impact trending direction"""
        try:
            # Compare current average with historical average
            current_avg = ImpactScore.objects.aggregate(
                avg_score=Avg('current_score')
            )['avg_score'] or 0.0
            
            # Get average from 24 hours ago
            day_ago = timezone.now() - timedelta(hours=24)
            historical_avg = ImpactHistory.objects.filter(
                timestamp__gte=day_ago,
                timestamp__lt=day_ago + timedelta(hours=1)
            ).aggregate(
                avg_score=Avg('impact_score')
            )['avg_score'] or current_avg
            
            change = current_avg - historical_avg
            
            if change > 0.1:
                return 'deteriorating'
            elif change < -0.1:
                return 'improving'
            else:
                return 'stable'
        
        except Exception:
            return 'unknown'
    
    def _cache_system_summary(self, summary: SystemImpactSummary):
        """Cache system summary for dashboard performance"""
        cache.set('system_impact_summary', asdict(summary), timeout=300)  # 5 minutes
    
    def _check_system_alerts(self):
        """Check for system-wide alert conditions"""
        try:
            summary = self.generate_system_summary()
            
            # Check for system-wide thresholds
            critical_percentage = (summary.critical_stopes / summary.total_stopes * 100) if summary.total_stopes > 0 else 0
            
            if critical_percentage > 10:  # More than 10% critical
                logger.warning(f"System alert: {critical_percentage:.1f}% of stopes are critical")
                # Could trigger system-wide notifications here
        
        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")


# Global service instance
impact_service = ImpactCalculationService()


# ===== IMPACT CALCULATION SERVICE COMPLETE =====
# High-level orchestration service for real-time and batch impact calculations
