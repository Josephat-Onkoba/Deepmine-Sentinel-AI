"""
Task 6: Risk Level Classification System - Setup Default Thresholds
===================================================================

Management command to set up default risk thresholds for the classification system.
This creates a baseline configuration for risk level assignment based on impact scores
and other operational criteria.
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from core.models import RiskThreshold, Stope


class Command(BaseCommand):
    help = 'Set up default risk thresholds for the classification system'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Reset existing thresholds before creating new ones'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without actually creating'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🎯 Setting up default risk thresholds...\n')
        )
        
        if options['reset']:
            if options['dry_run']:
                count = RiskThreshold.objects.count()
                self.stdout.write(f"Would delete {count} existing thresholds")
            else:
                deleted_count = RiskThreshold.objects.all().delete()[0]
                self.stdout.write(f"Deleted {deleted_count} existing thresholds")
        
        # Default threshold configurations
        default_thresholds = [
            # Impact Score Thresholds
            {
                'name': 'Elevated Risk - Impact Score',
                'risk_level': 'elevated',
                'threshold_type': 'impact_score',
                'minimum_value': 0.3,
                'maximum_value': 0.6,
                'priority': 1,
                'minimum_duration': timedelta(minutes=10),
                'cooldown_period': timedelta(minutes=30),
                'applies_to_rock_types': [],
                'applies_to_mining_methods': [],
                'notes': 'Standard elevated risk threshold for impact scores'
            },
            {
                'name': 'High Risk - Impact Score',
                'risk_level': 'high_risk',
                'threshold_type': 'impact_score',
                'minimum_value': 0.6,
                'maximum_value': 0.8,
                'priority': 1,
                'minimum_duration': timedelta(minutes=5),
                'cooldown_period': timedelta(minutes=20),
                'applies_to_rock_types': [],
                'applies_to_mining_methods': [],
                'notes': 'Standard high risk threshold for impact scores'
            },
            {
                'name': 'Critical Risk - Impact Score',
                'risk_level': 'critical',
                'threshold_type': 'impact_score',
                'minimum_value': 0.8,
                'maximum_value': 0.95,
                'priority': 1,
                'minimum_duration': timedelta(minutes=2),
                'cooldown_period': timedelta(minutes=10),
                'applies_to_rock_types': [],
                'applies_to_mining_methods': [],
                'notes': 'Standard critical risk threshold for impact scores'
            },
            {
                'name': 'Emergency - Impact Score',
                'risk_level': 'emergency',
                'threshold_type': 'impact_score',
                'minimum_value': 0.95,
                'maximum_value': None,
                'priority': 1,
                'minimum_duration': timedelta(seconds=30),
                'cooldown_period': timedelta(minutes=5),
                'applies_to_rock_types': [],
                'applies_to_mining_methods': [],
                'notes': 'Emergency threshold for immediate response'
            },
            
            # Rate of Change Thresholds
            {
                'name': 'Rapid Risk Increase',
                'risk_level': 'high_risk',
                'threshold_type': 'rate_of_change',
                'minimum_value': 0.2,  # 0.2 points per hour
                'maximum_value': None,
                'priority': 2,
                'minimum_duration': timedelta(minutes=15),
                'cooldown_period': timedelta(hours=1),
                'applies_to_rock_types': [],
                'applies_to_mining_methods': [],
                'notes': 'Threshold for rapid increases in impact score'
            },
            
            # Rock Type Specific Thresholds
            {
                'name': 'Shale - Lower Elevated Threshold',
                'risk_level': 'elevated',
                'threshold_type': 'impact_score',
                'minimum_value': 0.2,
                'maximum_value': 0.5,
                'priority': 1,
                'minimum_duration': timedelta(minutes=10),
                'cooldown_period': timedelta(minutes=30),
                'applies_to_rock_types': ['shale'],
                'applies_to_mining_methods': [],
                'notes': 'Lower threshold for unstable rock types like shale'
            },
            {
                'name': 'Granite - Higher Critical Threshold',
                'risk_level': 'critical',
                'threshold_type': 'impact_score',
                'minimum_value': 0.85,
                'maximum_value': None,
                'priority': 1,
                'minimum_duration': timedelta(minutes=5),
                'cooldown_period': timedelta(minutes=15),
                'applies_to_rock_types': ['granite'],
                'applies_to_mining_methods': [],
                'notes': 'Higher threshold for stable rock types like granite'
            },
            
            # Mining Method Specific Thresholds
            {
                'name': 'Block Caving - Elevated Risk',
                'risk_level': 'elevated',
                'threshold_type': 'impact_score',
                'minimum_value': 0.25,
                'maximum_value': 0.55,
                'priority': 1,
                'minimum_duration': timedelta(minutes=15),
                'cooldown_period': timedelta(minutes=45),
                'applies_to_rock_types': [],
                'applies_to_mining_methods': ['block_caving'],
                'notes': 'Lower threshold for high-risk mining methods'
            },
            
            # Cumulative Events Thresholds
            {
                'name': 'High Event Frequency',
                'risk_level': 'elevated',
                'threshold_type': 'cumulative_events',
                'minimum_value': 5.0,  # 5 events in time window
                'maximum_value': None,
                'priority': 3,
                'minimum_duration': timedelta(hours=1),
                'cooldown_period': timedelta(hours=2),
                'applies_to_rock_types': [],
                'applies_to_mining_methods': [],
                'notes': 'Threshold for high frequency of operational events'
            },
            
            # Proximity Risk Thresholds
            {
                'name': 'Proximity Risk - High Activity',
                'risk_level': 'elevated',
                'threshold_type': 'proximity_risk',
                'minimum_value': 0.4,
                'maximum_value': None,
                'priority': 4,
                'minimum_duration': timedelta(minutes=20),
                'cooldown_period': timedelta(hours=1),
                'applies_to_rock_types': [],
                'applies_to_mining_methods': [],
                'notes': 'Threshold for risks from nearby stope activities'
            }
        ]
        
        # Create thresholds
        created_count = 0
        for threshold_config in default_thresholds:
            if options['dry_run']:
                self.stdout.write(f"Would create: {threshold_config['name']}")
                created_count += 1
            else:
                threshold_config['created_by'] = 'setup_command'
                threshold, created = RiskThreshold.objects.get_or_create(
                    name=threshold_config['name'],
                    defaults=threshold_config
                )
                
                if created:
                    self.stdout.write(
                        self.style.SUCCESS(f"✅ Created: {threshold.name}")
                    )
                    created_count += 1
                else:
                    self.stdout.write(
                        self.style.WARNING(f"⚠️  Already exists: {threshold.name}")
                    )
        
        self.stdout.write('\n' + '='*60)
        
        if options['dry_run']:
            self.stdout.write(
                self.style.SUCCESS(
                    f'🎯 Dry run complete: Would create {created_count} thresholds'
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f'🎯 Setup complete: Created {created_count} new thresholds'
                )
            )
            
            # Show summary of all active thresholds
            total_thresholds = RiskThreshold.objects.filter(is_active=True).count()
            self.stdout.write(
                f'📊 Total active thresholds: {total_thresholds}'
            )
            
            # Show breakdown by risk level
            for risk_level, _ in RiskThreshold.RISK_LEVEL_CHOICES:
                count = RiskThreshold.objects.filter(
                    is_active=True, 
                    risk_level=risk_level
                ).count()
                if count > 0:
                    self.stdout.write(f'   - {risk_level}: {count} thresholds')
        
        self.stdout.write(
            self.style.SUCCESS('\n✨ Risk threshold setup completed!')
        )
