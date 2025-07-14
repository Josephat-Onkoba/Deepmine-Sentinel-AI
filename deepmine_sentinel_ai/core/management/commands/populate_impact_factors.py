"""
Management command to populate default impact factors for mining operations.
Based on industry standards and mining engineering best practices.
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from core.models import ImpactFactor
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Populate default impact factors for mining operations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--mine-site',
            type=str,
            help='Specific mine site name (optional)',
        )
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite existing impact factors',
        )

    def handle(self, *args, **options):
        mine_site = options.get('mine_site', '')
        overwrite = options.get('overwrite', False)
        
        # Default impact factors based on mining engineering standards
        default_factors = [
            # Blasting Operations
            {
                'event_category': 'blasting',
                'severity_level': 'minimal',
                'base_impact_weight': 0.5,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.05,
                'temporal_decay_rate': 0.1,
                'description': 'Small controlled blasts with minimal rock displacement'
            },
            {
                'event_category': 'blasting',
                'severity_level': 'low',
                'base_impact_weight': 1.2,
                'duration_multiplier': 1.1,
                'proximity_decay_rate': 0.08,
                'temporal_decay_rate': 0.08,
                'description': 'Standard production blasts with moderate impact'
            },
            {
                'event_category': 'blasting',
                'severity_level': 'moderate',
                'base_impact_weight': 2.5,
                'duration_multiplier': 1.3,
                'proximity_decay_rate': 0.1,
                'temporal_decay_rate': 0.06,
                'description': 'Large production blasts affecting significant volume'
            },
            {
                'event_category': 'blasting',
                'severity_level': 'high',
                'base_impact_weight': 4.0,
                'duration_multiplier': 1.5,
                'proximity_decay_rate': 0.12,
                'temporal_decay_rate': 0.04,
                'description': 'Major blasting operations with substantial ground disturbance'
            },
            {
                'event_category': 'blasting',
                'severity_level': 'severe',
                'base_impact_weight': 6.5,
                'duration_multiplier': 2.0,
                'proximity_decay_rate': 0.15,
                'temporal_decay_rate': 0.03,
                'description': 'Massive blasting with potential for damage to adjacent areas'
            },
            {
                'event_category': 'blasting',
                'severity_level': 'critical',
                'base_impact_weight': 9.0,
                'duration_multiplier': 2.5,
                'proximity_decay_rate': 0.2,
                'temporal_decay_rate': 0.02,
                'description': 'Emergency or uncontrolled blasting events'
            },
            
            # Heavy Equipment Operations
            {
                'event_category': 'equipment',
                'severity_level': 'minimal',
                'base_impact_weight': 0.3,
                'duration_multiplier': 1.5,
                'proximity_decay_rate': 0.1,
                'temporal_decay_rate': 0.15,
                'description': 'Light equipment operations with minimal vibration'
            },
            {
                'event_category': 'equipment',
                'severity_level': 'low',
                'base_impact_weight': 0.8,
                'duration_multiplier': 1.8,
                'proximity_decay_rate': 0.12,
                'temporal_decay_rate': 0.12,
                'description': 'Standard heavy equipment operations'
            },
            {
                'event_category': 'equipment',
                'severity_level': 'moderate',
                'base_impact_weight': 1.5,
                'duration_multiplier': 2.2,
                'proximity_decay_rate': 0.15,
                'temporal_decay_rate': 0.1,
                'description': 'Multiple heavy machines operating simultaneously'
            },
            {
                'event_category': 'equipment',
                'severity_level': 'high',
                'base_impact_weight': 2.8,
                'duration_multiplier': 2.8,
                'proximity_decay_rate': 0.18,
                'temporal_decay_rate': 0.08,
                'description': 'Intensive equipment operations causing significant vibration'
            },
            {
                'event_category': 'equipment',
                'severity_level': 'severe',
                'base_impact_weight': 4.5,
                'duration_multiplier': 3.5,
                'proximity_decay_rate': 0.2,
                'temporal_decay_rate': 0.06,
                'description': 'Heavy construction equipment causing ground disturbance'
            },
            {
                'event_category': 'equipment',
                'severity_level': 'critical',
                'base_impact_weight': 7.0,
                'duration_multiplier': 4.0,
                'proximity_decay_rate': 0.25,
                'temporal_decay_rate': 0.04,
                'description': 'Equipment collisions or mechanical failures'
            },
            
            # Water-Related Events
            {
                'event_category': 'water',
                'severity_level': 'minimal',
                'base_impact_weight': 0.2,
                'duration_multiplier': 3.0,
                'proximity_decay_rate': 0.05,
                'temporal_decay_rate': 0.02,
                'description': 'Minor water seepage or controlled drainage'
            },
            {
                'event_category': 'water',
                'severity_level': 'low',
                'base_impact_weight': 0.8,
                'duration_multiplier': 3.5,
                'proximity_decay_rate': 0.08,
                'temporal_decay_rate': 0.015,
                'description': 'Moderate water ingress affecting stability'
            },
            {
                'event_category': 'water',
                'severity_level': 'moderate',
                'base_impact_weight': 2.0,
                'duration_multiplier': 4.0,
                'proximity_decay_rate': 0.1,
                'temporal_decay_rate': 0.01,
                'description': 'Significant water flow affecting rock mass'
            },
            {
                'event_category': 'water',
                'severity_level': 'high',
                'base_impact_weight': 4.0,
                'duration_multiplier': 4.5,
                'proximity_decay_rate': 0.12,
                'temporal_decay_rate': 0.008,
                'description': 'Major water inflow causing structural concerns'
            },
            {
                'event_category': 'water',
                'severity_level': 'severe',
                'base_impact_weight': 6.5,
                'duration_multiplier': 5.0,
                'proximity_decay_rate': 0.15,
                'temporal_decay_rate': 0.005,
                'description': 'Flooding conditions affecting multiple areas'
            },
            {
                'event_category': 'water',
                'severity_level': 'critical',
                'base_impact_weight': 9.5,
                'duration_multiplier': 5.0,
                'proximity_decay_rate': 0.2,
                'temporal_decay_rate': 0.003,
                'description': 'Catastrophic water events requiring immediate evacuation'
            },
            
            # Support System Operations (Positive Impact)
            {
                'event_category': 'support',
                'severity_level': 'minimal',
                'base_impact_weight': -0.1,  # Negative = improvement
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.2,
                'temporal_decay_rate': 0.001,  # Long-lasting benefit
                'description': 'Minor support installation or maintenance'
            },
            {
                'event_category': 'support',
                'severity_level': 'low',
                'base_impact_weight': -0.3,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.15,
                'temporal_decay_rate': 0.001,
                'description': 'Standard ground support installation'
            },
            {
                'event_category': 'support',
                'severity_level': 'moderate',
                'base_impact_weight': -0.8,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.1,
                'temporal_decay_rate': 0.0005,
                'description': 'Comprehensive support system installation'
            },
            {
                'event_category': 'support',
                'severity_level': 'high',
                'base_impact_weight': -1.5,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.08,
                'temporal_decay_rate': 0.0003,
                'description': 'Major structural reinforcement work'
            },
            {
                'event_category': 'support',
                'severity_level': 'severe',
                'base_impact_weight': -2.5,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.05,
                'temporal_decay_rate': 0.0001,
                'description': 'Emergency stabilization measures'
            },
            {
                'event_category': 'support',
                'severity_level': 'critical',
                'base_impact_weight': -4.0,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.03,
                'temporal_decay_rate': 0.00005,
                'description': 'Comprehensive structural rehabilitation'
            },
            
            # Geological Events
            {
                'event_category': 'geological',
                'severity_level': 'minimal',
                'base_impact_weight': 1.0,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.1,
                'temporal_decay_rate': 0.05,
                'description': 'Minor rock falls or small geological adjustments'
            },
            {
                'event_category': 'geological',
                'severity_level': 'low',
                'base_impact_weight': 2.2,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.12,
                'temporal_decay_rate': 0.04,
                'description': 'Moderate geological instability events'
            },
            {
                'event_category': 'geological',
                'severity_level': 'moderate',
                'base_impact_weight': 4.5,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.15,
                'temporal_decay_rate': 0.03,
                'description': 'Significant geological events affecting stability'
            },
            {
                'event_category': 'geological',
                'severity_level': 'high',
                'base_impact_weight': 7.0,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.18,
                'temporal_decay_rate': 0.02,
                'description': 'Major geological instability requiring intervention'
            },
            {
                'event_category': 'geological',
                'severity_level': 'severe',
                'base_impact_weight': 9.0,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.2,
                'temporal_decay_rate': 0.015,
                'description': 'Severe geological events threatening structural integrity'
            },
            {
                'event_category': 'geological',
                'severity_level': 'critical',
                'base_impact_weight': 10.0,
                'duration_multiplier': 1.0,
                'proximity_decay_rate': 0.25,
                'temporal_decay_rate': 0.01,
                'description': 'Catastrophic geological events requiring immediate evacuation'
            },
        ]
        
        created_count = 0
        updated_count = 0
        
        for factor_data in default_factors:
            # Add metadata
            factor_data.update({
                'mine_site': mine_site,
                'site_specific': bool(mine_site),
                'is_active': True,
                'validation_source': 'Mining Engineering Standards and Best Practices',
                'last_calibrated': timezone.now(),
                'calibrated_by': 'System Administrator',
                'created_by': 'Default Population Command'
            })
            
            # Check if factor already exists
            existing_factor = ImpactFactor.objects.filter(
                event_category=factor_data['event_category'],
                severity_level=factor_data['severity_level'],
                mine_site=mine_site
            ).first()
            
            if existing_factor:
                if overwrite:
                    for key, value in factor_data.items():
                        setattr(existing_factor, key, value)
                    existing_factor.save()
                    updated_count += 1
                    self.stdout.write(
                        self.style.WARNING(f'Updated: {existing_factor}')
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(f'Skipped existing: {existing_factor}')
                    )
            else:
                ImpactFactor.objects.create(**factor_data)
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created: {factor_data["event_category"]} - {factor_data["severity_level"]}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully populated {created_count} new impact factors '
                f'and updated {updated_count} existing factors'
            )
        )
