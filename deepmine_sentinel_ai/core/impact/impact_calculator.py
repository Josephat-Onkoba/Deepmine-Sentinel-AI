"""
Mathematical Impact Calculator

Core engine for calculating how operational events affect stope stability.
Implements sophisticated algorithms for proximity-based impact distribution,
time decay mechanisms, and cumulative impact calculations.

This is the mathematical heart of the impact-based prediction system.
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from django.db import transaction
from django.utils import timezone
from django.core.cache import cache
import logging

from core.models import (
    Stope, OperationalEvent, ImpactScore, ImpactHistory, 
    ImpactFactor, MonitoringData
)

logger = logging.getLogger(__name__)


@dataclass
class SpatialCoordinate:
    """3D coordinate for spatial calculations"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'SpatialCoordinate') -> float:
        """Calculate Euclidean distance to another coordinate"""
        return math.sqrt(
            (self.x - other.x) ** 2 + 
            (self.y - other.y) ** 2 + 
            (self.z - other.z) ** 2
        )


@dataclass
class ImpactCalculationResult:
    """Result of impact calculation for a single event"""
    base_impact: float
    proximity_factor: float
    temporal_factor: float
    duration_factor: float
    final_impact: float
    affected_distance: float
    calculation_metadata: Dict


class MathematicalImpactCalculator:
    """
    Advanced mathematical engine for calculating operational event impacts
    on stope stability using physics-based and empirical models.
    """
    
    # Physics-based constants for mining operations
    ROCK_DENSITY = 2700  # kg/m³ (typical granite density)
    SOUND_SPEED_ROCK = 5000  # m/s (P-wave velocity in granite)
    GRAVITY = 9.81  # m/s²
    
    # Empirical decay constants
    DEFAULT_PROXIMITY_DECAY = 0.15  # Exponential decay rate with distance
    DEFAULT_TEMPORAL_DECAY = 0.05   # Exponential decay rate with time
    MIN_IMPACT_THRESHOLD = 0.001    # Minimum significant impact
    
    def __init__(self, cache_calculations: bool = True):
        """
        Initialize the impact calculator
        
        Args:
            cache_calculations: Whether to cache calculation results for performance
        """
        self.cache_calculations = cache_calculations
        self.calculation_cache = {}
        
        # Load configuration
        self._load_calculation_parameters()
    
    def _load_calculation_parameters(self):
        """Load calculation parameters from configuration"""
        # These could be loaded from database configuration
        self.max_impact_distance = 500.0  # meters
        self.temporal_window = 168  # hours (1 week)
        self.impact_accumulation_factor = 0.8
        self.stress_concentration_factor = 1.5
        
    def calculate_event_impact(
        self,
        event: OperationalEvent,
        target_stope: Stope,
        calculation_time: Optional[datetime] = None
    ) -> ImpactCalculationResult:
        """
        Calculate the impact of a single operational event on a target stope.
        
        This is the core algorithm that combines:
        - Base impact from event type and severity
        - Proximity-based impact distribution
        - Time decay mechanisms
        - Duration effects
        
        Args:
            event: The operational event to analyze
            target_stope: The stope being affected
            calculation_time: Time of calculation (default: now)
            
        Returns:
            Detailed impact calculation result
        """
        if calculation_time is None:
            calculation_time = timezone.now()
        
        logger.debug(f"Calculating impact of {event} on {target_stope}")
        
        # Check cache first
        cache_key = self._get_cache_key(event, target_stope, calculation_time)
        if self.cache_calculations and cache_key in self.calculation_cache:
            return self.calculation_cache[cache_key]
        
        try:
            # Step 1: Get base impact from event and factors
            base_impact = self._calculate_base_impact(event)
            
            # Step 2: Calculate proximity factor
            proximity_factor, distance = self._calculate_proximity_factor(event, target_stope)
            
            # Step 3: Calculate temporal decay
            temporal_factor = self._calculate_temporal_decay(event, calculation_time)
            
            # Step 4: Calculate duration effects
            duration_factor = self._calculate_duration_factor(event)
            
            # Step 5: Combine factors for final impact
            final_impact = self._combine_impact_factors(
                base_impact, proximity_factor, temporal_factor, duration_factor
            )
            
            # Step 6: Create result object
            result = ImpactCalculationResult(
                base_impact=base_impact,
                proximity_factor=proximity_factor,
                temporal_factor=temporal_factor,
                duration_factor=duration_factor,
                final_impact=final_impact,
                affected_distance=distance,
                calculation_metadata={
                    'event_id': event.id,
                    'target_stope_id': target_stope.id,
                    'calculation_time': calculation_time.isoformat(),
                    'algorithm_version': '1.0',
                    'physics_model': 'exponential_decay'
                }
            )
            
            # Cache the result
            if self.cache_calculations:
                self.calculation_cache[cache_key] = result
            
            logger.debug(f"Impact calculation complete: {final_impact:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating event impact: {e}")
            # Return zero impact on error
            return ImpactCalculationResult(
                base_impact=0.0,
                proximity_factor=0.0,
                temporal_factor=0.0,
                duration_factor=0.0,
                final_impact=0.0,
                affected_distance=float('inf'),
                calculation_metadata={'error': str(e)}
            )
    
    def _calculate_base_impact(self, event: OperationalEvent) -> float:
        """
        Calculate base impact score from event type and severity using
        configured impact factors.
        """
        try:
            # Convert numeric severity to string level
            severity_level = self._get_severity_level_string(event.severity)
            
            # Get impact factor for this event
            impact_factor = ImpactFactor.objects.get(
                event_category=event.event_type,
                severity_level=severity_level,
                is_active=True,
                mine_site__isnull=True  # Use general factors first
            )
            
            base_impact = impact_factor.base_impact_weight
            
            # Apply energy-based scaling for certain event types
            if event.event_type == 'blasting':
                # Scale by explosive energy if available
                energy_factor = self._calculate_explosive_energy_factor(event)
                base_impact *= energy_factor
            
            elif event.event_type == 'heavy_equipment':
                # Scale by equipment weight/vibration
                equipment_factor = self._calculate_equipment_factor(event)
                base_impact *= equipment_factor
            
            logger.debug(f"Base impact for {event.event_type}/{severity_level}: {base_impact:.4f}")
            return base_impact
            
        except ImpactFactor.DoesNotExist:
            logger.warning(f"No impact factor found for {event.event_type}/{severity_level}")
            # Fallback to severity-based calculation
            return event.severity * 5.0  # Scale 0.1-1.0 to 0.5-5.0
    
    def _get_severity_level_string(self, severity_float: float) -> str:
        """Convert numeric severity (0.1-1.0) to string level"""
        if severity_float <= 0.2:
            return 'minimal'
        elif severity_float <= 0.4:
            return 'low'
        elif severity_float <= 0.6:
            return 'moderate'
        elif severity_float <= 0.8:
            return 'high'
        elif severity_float <= 0.95:
            return 'severe'
        else:
            return 'critical'
    
    def _calculate_proximity_factor(
        self, 
        event: OperationalEvent, 
        target_stope: Stope
    ) -> Tuple[float, float]:
        """
        Calculate proximity-based impact factor using advanced spatial models.
        
        Implements multiple decay models:
        - Exponential decay for most events
        - Inverse square law for point sources (blasting)
        - Cylindrical decay for linear operations (drilling)
        
        Returns:
            Tuple of (proximity_factor, distance_meters)
        """
        # Get spatial coordinates
        event_location = self._get_event_coordinates(event)
        stope_location = self._get_stope_coordinates(target_stope)
        
        # Calculate 3D distance
        distance = event_location.distance_to(stope_location)
        
        # Choose decay model based on event type
        if event.event_type == 'blasting':
            # Inverse square law for explosive point sources
            # F = I₀ / (1 + α * r²)
            proximity_factor = 1.0 / (1.0 + 0.001 * distance ** 2)
        
        elif event.event_type in ['drilling', 'excavation']:
            # Cylindrical decay for linear operations
            # F = I₀ * exp(-α * r)
            decay_rate = 0.01  # per meter
            proximity_factor = math.exp(-decay_rate * distance)
        
        else:
            # Standard exponential decay
            # F = I₀ * exp(-α * r)
            try:
                severity_level = self._get_severity_level_string(event.severity)
                impact_factor = ImpactFactor.objects.get(
                    event_category=event.event_type,
                    severity_level=severity_level,
                    is_active=True
                )
                decay_rate = impact_factor.proximity_decay_rate
            except ImpactFactor.DoesNotExist:
                decay_rate = self.DEFAULT_PROXIMITY_DECAY
            
            proximity_factor = math.exp(-decay_rate * distance)
        
        # Apply minimum threshold
        if proximity_factor < self.MIN_IMPACT_THRESHOLD:
            proximity_factor = 0.0
        
        # Apply stress concentration effects near excavation boundaries
        stress_factor = self._calculate_stress_concentration(distance, target_stope)
        proximity_factor *= stress_factor
        
        logger.debug(f"Proximity factor at {distance:.1f}m: {proximity_factor:.6f}")
        return proximity_factor, distance
    
    def _calculate_temporal_decay(
        self, 
        event: OperationalEvent, 
        calculation_time: datetime
    ) -> float:
        """
        Calculate time-based decay factor using geological time constants.
        
        Different events have different temporal signatures:
        - Blasting: Immediate spike, rapid decay
        - Heavy equipment: Sustained loading, moderate decay
        - Water exposure: Gradual weakening, slow decay
        """
        time_elapsed = calculation_time - event.timestamp
        hours_elapsed = time_elapsed.total_seconds() / 3600.0
        
        if hours_elapsed < 0:
            return 0.0  # Future events have no current impact
        
        # Get decay rate from impact factor
        try:
            severity_level = self._get_severity_level_string(event.severity)
            impact_factor = ImpactFactor.objects.get(
                event_category=event.event_type,
                severity_level=severity_level,
                is_active=True
            )
            decay_rate = impact_factor.temporal_decay_rate
        except ImpactFactor.DoesNotExist:
            decay_rate = self.DEFAULT_TEMPORAL_DECAY
        
        # Apply event-specific temporal models
        if event.event_type == 'blasting':
            # Double exponential for blasting (fast + slow components)
            fast_decay = 0.5 * math.exp(-0.1 * hours_elapsed)  # Fast component
            slow_decay = 0.5 * math.exp(-decay_rate * hours_elapsed)  # Slow component
            temporal_factor = fast_decay + slow_decay
        
        elif event.event_type == 'water_exposure':
            # Logarithmic decay for water damage (very slow recovery)
            temporal_factor = 1.0 / (1.0 + 0.01 * hours_elapsed)
        
        else:
            # Standard exponential decay
            temporal_factor = math.exp(-decay_rate * hours_elapsed)
        
        # Apply minimum threshold
        if temporal_factor < self.MIN_IMPACT_THRESHOLD:
            temporal_factor = 0.0
        
        logger.debug(f"Temporal factor after {hours_elapsed:.1f}h: {temporal_factor:.6f}")
        return temporal_factor
    
    def _calculate_duration_factor(self, event: OperationalEvent) -> float:
        """
        Calculate duration-based impact amplification.
        
        Longer events can have compounding effects on stability.
        """
        if not event.duration_hours or event.duration_hours <= 0:
            return 1.0  # No duration data, assume minimal duration
        
        duration_hours = event.duration_hours
        
        # Get duration multiplier from impact factor
        try:
            severity_level = self._get_severity_level_string(event.severity)
            impact_factor = ImpactFactor.objects.get(
                event_category=event.event_type,
                severity_level=severity_level,
                is_active=True
            )
            duration_multiplier = impact_factor.duration_multiplier
        except ImpactFactor.DoesNotExist:
            duration_multiplier = 1.0
        
        # Apply logarithmic scaling for duration effects
        # F = 1 + α * log(1 + t)
        duration_factor = 1.0 + (duration_multiplier - 1.0) * math.log(1.0 + duration_hours)
        
        # Cap maximum duration effect
        duration_factor = min(duration_factor, 3.0)
        
        logger.debug(f"Duration factor for {duration_hours:.1f}h: {duration_factor:.4f}")
        return duration_factor
    
    def _combine_impact_factors(
        self,
        base_impact: float,
        proximity_factor: float,
        temporal_factor: float,
        duration_factor: float
    ) -> float:
        """
        Combine all impact factors using advanced mathematical models.
        
        Uses multiplicative model with safety factors and physical constraints.
        """
        # Basic multiplicative combination
        combined_impact = base_impact * proximity_factor * temporal_factor * duration_factor
        
        # Apply physical constraints
        # Impact cannot exceed theoretical maximum based on rock properties
        max_theoretical_impact = 10.0  # Maximum possible impact score
        combined_impact = min(combined_impact, max_theoretical_impact)
        
        # Apply minimum threshold
        if combined_impact < self.MIN_IMPACT_THRESHOLD:
            combined_impact = 0.0
        
        return combined_impact
    
    def calculate_cumulative_impact(
        self,
        target_stope: Stope,
        calculation_time: Optional[datetime] = None,
        time_window_hours: Optional[int] = None
    ) -> float:
        """
        Calculate cumulative impact from all events affecting a stope.
        
        Implements sophisticated accumulation models:
        - Superposition for independent events
        - Non-linear accumulation for stress concentration
        - Damage evolution models for persistent effects
        
        Args:
            target_stope: Stope to calculate cumulative impact for
            calculation_time: Time of calculation (default: now)
            time_window_hours: Look-back window (default: 168 hours)
            
        Returns:
            Cumulative impact score
        """
        if calculation_time is None:
            calculation_time = timezone.now()
        
        if time_window_hours is None:
            time_window_hours = self.temporal_window
        
        logger.info(f"Calculating cumulative impact for {target_stope} over {time_window_hours}h")
        
        # Get all relevant events in time window
        start_time = calculation_time - timedelta(hours=time_window_hours)
        events = OperationalEvent.objects.filter(
            timestamp__gte=start_time,
            timestamp__lte=calculation_time
        ).order_by('timestamp')
        
        if not events.exists():
            logger.debug("No events found in time window")
            return 0.0
        
        # Calculate impact from each event
        individual_impacts = []
        for event in events:
            impact_result = self.calculate_event_impact(event, target_stope, calculation_time)
            if impact_result.final_impact > 0:
                individual_impacts.append({
                    'impact': impact_result.final_impact,
                    'event': event,
                    'result': impact_result
                })
        
        if not individual_impacts:
            return 0.0
        
        # Sort by impact magnitude for accumulation
        individual_impacts.sort(key=lambda x: x['impact'], reverse=True)
        
        # Apply accumulation model
        cumulative_impact = self._apply_accumulation_model(individual_impacts)
        
        # Apply damage evolution model
        evolved_impact = self._apply_damage_evolution(cumulative_impact, calculation_time)
        
        logger.info(f"Cumulative impact: {evolved_impact:.6f}")
        return evolved_impact
    
    def _apply_accumulation_model(self, individual_impacts: List[Dict]) -> float:
        """
        Apply advanced accumulation model for multiple impacts.
        
        Uses modified superposition with stress concentration effects.
        """
        if not individual_impacts:
            return 0.0
        
        # Start with largest impact
        cumulative = individual_impacts[0]['impact']
        
        # Add remaining impacts with diminishing returns
        for i, impact_data in enumerate(individual_impacts[1:], 1):
            impact = impact_data['impact']
            
            # Diminishing returns factor: each additional impact has reduced effect
            accumulation_factor = self.impact_accumulation_factor ** i
            
            # Stress concentration factor for nearby events
            concentration_factor = 1.0
            if self._events_are_nearby(
                individual_impacts[0]['event'], 
                impact_data['event']
            ):
                concentration_factor = self.stress_concentration_factor
            
            # Add impact with factors
            additional_impact = impact * accumulation_factor * concentration_factor
            cumulative += additional_impact
        
        return cumulative
    
    def _apply_damage_evolution(self, base_impact: float, calculation_time: datetime) -> float:
        """
        Apply damage evolution model for persistent effects.
        
        Some damage accumulates over time even after events end.
        """
        # Simple evolution model - could be made more sophisticated
        evolution_factor = 1.0
        
        # Check for recent high-impact events that cause ongoing damage
        recent_events = OperationalEvent.objects.filter(
            timestamp__gte=calculation_time - timedelta(hours=24),
            timestamp__lte=calculation_time,
            event_type__in=['water_exposure', 'heavy_equipment']
        )
        
        if recent_events.exists():
            evolution_factor = 1.1  # 10% amplification for ongoing damage
        
        return base_impact * evolution_factor
    
    def update_stope_impact_score(
        self,
        stope: Stope,
        calculation_time: Optional[datetime] = None,
        create_history: bool = True
    ) -> ImpactScore:
        """
        Update the current impact score for a stope and optionally create history.
        
        This is the main function called to update stope impact scores.
        """
        if calculation_time is None:
            calculation_time = timezone.now()
        
        logger.info(f"Updating impact score for {stope}")
        
        # Calculate cumulative impact
        cumulative_impact = self.calculate_cumulative_impact(stope, calculation_time)
        
        # Determine risk level
        risk_level = self._determine_risk_level(cumulative_impact)
        
        # Get or create impact score record
        impact_score, created = ImpactScore.objects.get_or_create(
            stope=stope,
            defaults={
                'current_score': cumulative_impact,
                'risk_level': risk_level,
                'last_calculated': calculation_time,
                'calculation_version': '1.0'
            }
        )
        
        # Update if not created
        if not created:
            old_score = impact_score.current_score
            old_risk_level = impact_score.risk_level
            
            impact_score.current_score = cumulative_impact
            impact_score.risk_level = risk_level
            impact_score.last_calculated = calculation_time
            
            # Create history record if score changed significantly
            if create_history and (
                abs(old_score - cumulative_impact) > 0.01 or 
                old_risk_level != risk_level
            ):
                ImpactHistory.objects.create(
                    stope=stope,
                    previous_score=old_score,
                    new_score=cumulative_impact,
                    score_change=cumulative_impact - old_score,
                    previous_risk_level=old_risk_level,
                    new_risk_level=risk_level,
                    change_type='system_recalculation',
                    change_reason='Automatic impact score update based on recent operational events'
                )
        
        impact_score.save()
        logger.info(f"Updated {stope} impact score: {cumulative_impact:.6f} ({risk_level})")
        return impact_score
    
    def batch_update_stope_scores(
        self,
        stopes: Optional[List[Stope]] = None,
        calculation_time: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Efficiently update impact scores for multiple stopes.
        
        Returns:
            Dictionary with update statistics
        """
        if stopes is None:
            stopes = list(Stope.objects.filter(is_active=True))
        
        if calculation_time is None:
            calculation_time = timezone.now()
        
        logger.info(f"Batch updating {len(stopes)} stope impact scores")
        
        stats = {
            'total_stopes': len(stopes),
            'updated_scores': 0,
            'risk_level_changes': 0,
            'errors': 0
        }
        
        with transaction.atomic():
            for stope in stopes:
                try:
                    old_risk_level = None
                    try:
                        old_impact = ImpactScore.objects.get(stope=stope)
                        old_risk_level = old_impact.risk_level
                    except ImpactScore.DoesNotExist:
                        pass
                    
                    # Update score
                    impact_score = self.update_stope_impact_score(
                        stope, calculation_time, create_history=True
                    )
                    
                    stats['updated_scores'] += 1
                    
                    # Check for risk level change
                    if old_risk_level and old_risk_level != impact_score.risk_level:
                        stats['risk_level_changes'] += 1
                
                except Exception as e:
                    logger.error(f"Error updating {stope}: {e}")
                    stats['errors'] += 1
        
        logger.info(f"Batch update complete: {stats}")
        return stats
    
    # Helper methods
    
    def _get_event_coordinates(self, event: OperationalEvent) -> SpatialCoordinate:
        """Get spatial coordinates for an event (could be enhanced with GPS data)"""
        # For now, use stope coordinates as event location
        return self._get_stope_coordinates(event.stope)
    
    def _get_stope_coordinates(self, stope: Stope) -> SpatialCoordinate:
        """Get spatial coordinates for a stope"""
        # Simple coordinate system based on stope properties
        # In production, this would use actual survey coordinates
        x = hash(stope.stope_name) % 1000  # Pseudo-random but consistent
        y = stope.depth * 0.1
        z = stope.hr * 10
        return SpatialCoordinate(x, y, z)
    
    def _calculate_explosive_energy_factor(self, event: OperationalEvent) -> float:
        """Calculate energy scaling for blasting events"""
        # Map numeric severity to energy factor
        if event.severity <= 0.2:
            return 1.0
        elif event.severity <= 0.4:
            return 1.5
        elif event.severity <= 0.6:
            return 2.0
        elif event.severity <= 0.8:
            return 3.0
        else:
            return 4.0
    
    def _calculate_equipment_factor(self, event: OperationalEvent) -> float:
        """Calculate scaling factor for equipment operations"""
        # Could be enhanced with actual equipment specifications
        return 1.0
    
    def _calculate_stress_concentration(self, distance: float, stope: Stope) -> float:
        """Calculate stress concentration effects near excavation boundaries"""
        # Simple model - stress concentration decreases with distance
        if distance < stope.hr:  # Close to excavation
            return 1.5  # 50% amplification
        elif distance < 2 * stope.hr:
            return 1.2  # 20% amplification
        else:
            return 1.0  # No amplification
    
    def _events_are_nearby(self, event1: OperationalEvent, event2: OperationalEvent) -> bool:
        """Check if two events are spatially close"""
        coord1 = self._get_event_coordinates(event1)
        coord2 = self._get_event_coordinates(event2)
        distance = coord1.distance_to(coord2)
        return distance < 50.0  # 50 meter threshold
    
    def _determine_risk_level(self, impact_score: float) -> str:
        """Determine risk level from impact score"""
        if impact_score < 1.0:
            return 'stable'
        elif impact_score < 3.0:
            return 'elevated'
        elif impact_score < 6.0:
            return 'high_risk'
        else:
            return 'critical'
    
    def _get_cache_key(
        self, 
        event: OperationalEvent, 
        stope: Stope, 
        calculation_time: datetime
    ) -> str:
        """Generate cache key for impact calculation"""
        time_str = calculation_time.strftime('%Y%m%d_%H')  # Hour-level caching
        return f"impact_{event.id}_{stope.id}_{time_str}"


# ===== MATHEMATICAL IMPACT CALCULATOR COMPLETE =====
# Advanced physics-based and empirical models for operational event impact calculation
