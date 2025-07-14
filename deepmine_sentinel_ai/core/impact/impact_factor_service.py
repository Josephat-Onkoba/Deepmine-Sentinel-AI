"""
Impact Factor Management Service

Provides utilities for managing and applying impact factors in stability calculations.
Includes validation, calibration, and optimization functions.
"""

from django.db import transaction
from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError
from core.models import ImpactFactor, OperationalEvent
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ImpactFactorService:
    """Service for managing impact factors and their application"""
    
    @staticmethod
    def get_active_factors(mine_site: str = '', event_category: str = None) -> Dict[str, Dict]:
        """
        Get active impact factors organized by category and severity
        
        Args:
            mine_site: Specific mine site (empty for general factors)
            event_category: Filter by specific event category
            
        Returns:
            Dict: Nested dictionary of impact factors
        """
        filters = {'is_active': True}
        
        if mine_site:
            filters['mine_site'] = mine_site
        
        if event_category:
            filters['event_category'] = event_category
        
        factors = ImpactFactor.objects.filter(**filters)
        
        # Organize by category and severity
        organized_factors = {}
        for factor in factors:
            if factor.event_category not in organized_factors:
                organized_factors[factor.event_category] = {}
            
            organized_factors[factor.event_category][factor.severity_level] = {
                'id': factor.id,
                'base_weight': factor.base_impact_weight,
                'duration_multiplier': factor.duration_multiplier,
                'proximity_decay': factor.proximity_decay_rate,
                'temporal_decay': factor.temporal_decay_rate,
                'factor_object': factor
            }
        
        return organized_factors
    
    @staticmethod
    def calculate_event_impact(event: OperationalEvent, mine_site: str = '') -> Tuple[float, Dict]:
        """
        Calculate impact for an operational event using configured factors
        
        Args:
            event: OperationalEvent instance
            mine_site: Mine site for site-specific factors
            
        Returns:
            Tuple of (impact_value, calculation_details)
        """
        try:
            # Get appropriate impact factor
            factor_query = ImpactFactor.objects.filter(
                event_category=event.event_type,
                is_active=True
            )
            
            # Prefer site-specific factors
            if mine_site:
                site_factor = factor_query.filter(mine_site=mine_site).first()
                if site_factor:
                    factor = site_factor
                else:
                    factor = factor_query.filter(mine_site='').first()
            else:
                factor = factor_query.filter(mine_site='').first()
            
            if not factor:
                logger.warning(f"No impact factor found for {event.event_type}")
                return 1.0, {'error': 'No matching impact factor'}
            
            # Map severity float to severity level
            severity_mapping = {
                0.1: 'minimal', 0.3: 'low', 0.5: 'moderate',
                0.7: 'high', 0.9: 'severe', 1.0: 'critical'
            }
            
            # Find closest severity level
            closest_severity = min(severity_mapping.keys(), 
                                 key=lambda x: abs(x - event.severity))
            severity_level = severity_mapping[closest_severity]
            
            # Get factor for this severity level
            severity_factor = ImpactFactor.objects.filter(
                event_category=event.event_type,
                severity_level=severity_level,
                is_active=True
            ).first()
            
            if severity_factor:
                factor = severity_factor
            
            # Calculate impact using factor parameters
            days_elapsed = (timezone.now() - event.timestamp).total_seconds() / (24 * 3600)
            
            impact = factor.calculate_adjusted_impact(
                base_value=1.0,  # Normalized base value
                duration_hours=float(event.duration_hours),
                distance_meters=float(event.proximity_to_stope),
                days_elapsed=days_elapsed
            )
            
            calculation_details = {
                'factor_id': factor.id,
                'factor_category': factor.event_category,
                'factor_severity': factor.severity_level,
                'base_weight': factor.base_impact_weight,
                'duration_factor': factor.duration_multiplier,
                'proximity_decay': factor.proximity_decay_rate,
                'temporal_decay': factor.temporal_decay_rate,
                'days_elapsed': days_elapsed,
                'final_impact': impact
            }
            
            return impact, calculation_details
            
        except Exception as e:
            logger.error(f"Error calculating impact for event {event.id}: {e}")
            return 1.0, {'error': str(e)}
    
    @staticmethod
    @transaction.atomic
    def bulk_update_factors(categories: List[str], severity_levels: List[str], 
                          action: str, value: float = None, 
                          mine_site: str = '') -> Dict[str, int]:
        """
        Bulk update impact factors
        
        Args:
            categories: List of event categories to update
            severity_levels: List of severity levels to update
            action: Type of update action
            value: Value for numeric operations
            mine_site: Mine site filter
            
        Returns:
            Dict with update statistics
        """
        filters = {'is_active': True}
        
        if categories:
            filters['event_category__in'] = categories
        
        if severity_levels:
            filters['severity_level__in'] = severity_levels
        
        if mine_site:
            filters['mine_site'] = mine_site
        
        factors = ImpactFactor.objects.filter(**filters)
        updated_count = 0
        error_count = 0
        
        for factor in factors:
            try:
                if action == 'multiply_base':
                    factor.base_impact_weight *= value
                    factor.base_impact_weight = min(10.0, max(0.0, factor.base_impact_weight))
                
                elif action == 'add_base':
                    factor.base_impact_weight += value
                    factor.base_impact_weight = min(10.0, max(0.0, factor.base_impact_weight))
                
                elif action == 'set_duration':
                    factor.duration_multiplier = value
                    factor.duration_multiplier = min(5.0, max(0.1, factor.duration_multiplier))
                
                elif action == 'calibration_reset':
                    factor.last_calibrated = timezone.now()
                
                factor.full_clean()
                factor.save()
                updated_count += 1
                
            except ValidationError as e:
                logger.error(f"Validation error updating factor {factor.id}: {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Error updating factor {factor.id}: {e}")
                error_count += 1
        
        return {
            'updated': updated_count,
            'errors': error_count,
            'total_processed': updated_count + error_count
        }
    
    @staticmethod
    def validate_factor_consistency() -> List[Dict]:
        """
        Validate impact factor consistency and identify potential issues
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for missing factors
        categories = [choice[0] for choice in ImpactFactor.CATEGORY_CHOICES]
        severity_levels = [choice[0] for choice in ImpactFactor.SEVERITY_LEVEL_CHOICES]
        
        for category in categories:
            for severity in severity_levels:
                if not ImpactFactor.objects.filter(
                    event_category=category,
                    severity_level=severity,
                    is_active=True,
                    mine_site=''
                ).exists():
                    issues.append({
                        'type': 'missing_factor',
                        'category': category,
                        'severity': severity,
                        'message': f'Missing default factor for {category} - {severity}'
                    })
        
        # Check for unreasonable parameter combinations
        factors = ImpactFactor.objects.filter(is_active=True)
        
        for factor in factors:
            if factor.base_impact_weight > 8.0 and factor.duration_multiplier > 4.0:
                issues.append({
                    'type': 'extreme_parameters',
                    'factor_id': factor.id,
                    'category': factor.event_category,
                    'severity': factor.severity_level,
                    'message': 'Extremely high impact parameters may cause unrealistic calculations'
                })
            
            if factor.temporal_decay_rate > 0.3:
                issues.append({
                    'type': 'high_decay',
                    'factor_id': factor.id,
                    'category': factor.event_category,
                    'severity': factor.severity_level,
                    'message': 'Very high temporal decay rate - impacts may disappear too quickly'
                })
        
        # Check for inconsistent severity progression
        for category in categories:
            category_factors = factors.filter(event_category=category).order_by('severity_level')
            
            if category_factors.count() > 1:
                prev_weight = 0
                for factor in category_factors:
                    if factor.base_impact_weight < prev_weight:
                        issues.append({
                            'type': 'severity_inconsistency',
                            'factor_id': factor.id,
                            'category': factor.event_category,
                            'severity': factor.severity_level,
                            'message': 'Impact weight decreases with higher severity - check severity progression'
                        })
                    prev_weight = factor.base_impact_weight
        
        return issues
    
    @staticmethod
    def get_factor_statistics() -> Dict:
        """
        Get statistics about current impact factor configuration
        
        Returns:
            Dict with factor statistics
        """
        factors = ImpactFactor.objects.filter(is_active=True)
        
        stats = {
            'total_active_factors': factors.count(),
            'categories_covered': factors.values('event_category').distinct().count(),
            'site_specific_factors': factors.filter(site_specific=True).count(),
            'recently_calibrated': factors.filter(
                last_calibrated__gte=timezone.now() - timezone.timedelta(days=30)
            ).count(),
            'avg_base_weight': factors.aggregate(
                avg_weight=models.Avg('base_impact_weight')
            )['avg_weight'] or 0,
            'weight_distribution': {
                'low_impact': factors.filter(base_impact_weight__lt=2.0).count(),
                'medium_impact': factors.filter(
                    base_impact_weight__gte=2.0, 
                    base_impact_weight__lt=5.0
                ).count(),
                'high_impact': factors.filter(base_impact_weight__gte=5.0).count(),
            }
        }
        
        return stats
