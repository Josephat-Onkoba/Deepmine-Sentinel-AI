"""
Update Impact Scores Management Command

Django management command for batch updating stope impact scores.
Supports full updates, targeted updates, and performance monitoring.

Usage:
python manage.py update_impact_scores
python manage.py update_impact_scores --stope-ids 1,2,3
python manage.py update_impact_scores --force --verbose
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db import transaction
import logging
import sys
from datetime import datetime

from core.models import Stope, ImpactScore
from core.impact import impact_service


class Command(BaseCommand):
    """Management command for updating stope impact scores"""
    
    help = 'Update impact scores for stopes based on operational events'
    
    def add_arguments(self, parser):
        """Add command line arguments"""
        parser.add_argument(
            '--stope-ids',
            type=str,
            help='Comma-separated list of stope IDs to update (default: all active stopes)',
        )
        
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update even if recently calculated',
        )
        
        parser.add_argument(
            '--time-window',
            type=int,
            default=168,
            help='Time window in hours for impact calculations (default: 168)',
        )
        
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Number of stopes to process per batch (default: 50)',
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be updated without making changes',
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output',
        )
    
    def handle(self, *args, **options):
        """Main command handler"""
        # Configure logging level
        if options['verbose']:
            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
        
        logger = logging.getLogger(__name__)
        
        start_time = timezone.now()
        self.stdout.write(
            self.style.SUCCESS(f'Starting impact score update at {start_time}')
        )
        
        try:
            # Parse stope IDs if provided
            stope_ids = None
            if options['stope_ids']:
                try:
                    stope_ids = [int(id.strip()) for id in options['stope_ids'].split(',')]
                    self.stdout.write(f'Targeting specific stopes: {stope_ids}')
                except ValueError:
                    raise CommandError('Invalid stope IDs format. Use comma-separated integers.')
            
            # Configure service
            impact_service.batch_size = options['batch_size']
            
            # Dry run check
            if options['dry_run']:
                self._run_dry_run(stope_ids, options)
                return
            
            # Run the actual update
            stats = self._run_update(stope_ids, options)
            
            # Report results
            self._report_results(stats, start_time)
        
        except Exception as e:
            logger.error(f"Critical error in impact update command: {e}")
            raise CommandError(f'Update failed: {e}')
    
    def _run_dry_run(self, stope_ids, options):
        """Run a dry run to show what would be updated"""
        self.stdout.write(self.style.WARNING('DRY RUN - No changes will be made'))
        
        # Get stopes that would be updated
        if stope_ids:
            stopes = Stope.objects.filter(id__in=stope_ids, is_active=True)
        else:
            stopes = Stope.objects.filter(is_active=True)
        
        if not options['force']:
            # Filter for stales that need updates
            update_threshold = timezone.now() - timezone.timedelta(minutes=30)
            stopes_needing_update = []
            
            for stope in stopes:
                try:
                    impact_score = ImpactScore.objects.get(stope=stope)
                    if impact_score.last_calculated < update_threshold:
                        stopes_needing_update.append(stope)
                except ImpactScore.DoesNotExist:
                    stopes_needing_update.append(stope)
            
            stopes = stopes_needing_update
        
        self.stdout.write(f'Would update {len(stopes)} stopes:')
        for stope in stopes[:10]:  # Show first 10
            try:
                current_score = ImpactScore.objects.get(stope=stope)
                self.stdout.write(f'  - {stope.stope_name}: Current score {current_score.current_score:.3f}')
            except ImpactScore.DoesNotExist:
                self.stdout.write(f'  - {stope.stope_name}: No current score (new calculation)')
        
        if len(stopes) > 10:
            self.stdout.write(f'  ... and {len(stopes) - 10} more')
    
    def _run_update(self, stope_ids, options):
        """Run the actual impact score update"""
        self.stdout.write('Running impact score update...')
        
        try:
            with transaction.atomic():
                stats = impact_service.run_batch_update(
                    stope_ids=stope_ids,
                    force_update=options['force']
                )
                
                if options['verbose']:
                    self.stdout.write(f'Batch update statistics: {stats}')
                
                return stats
        
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during batch update: {e}')
            )
            raise
    
    def _report_results(self, stats, start_time):
        """Report update results"""
        end_time = timezone.now()
        duration = (end_time - start_time).total_seconds()
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('IMPACT SCORE UPDATE COMPLETE'))
        self.stdout.write('='*60)
        
        self.stdout.write(f'Total stopes processed: {stats["total_stopes"]}')
        self.stdout.write(f'Scores updated: {stats["updated_scores"]}')
        self.stdout.write(f'Risk level changes: {stats["risk_level_changes"]}')
        self.stdout.write(f'Errors encountered: {stats["errors"]}')
        self.stdout.write(f'Duration: {duration:.2f} seconds')
        
        if stats["total_stopes"] > 0:
            rate = stats["total_stopes"] / duration
            self.stdout.write(f'Processing rate: {rate:.2f} stopes/second')
        
        # Show any errors
        if stats["errors"] > 0:
            self.stdout.write(
                self.style.WARNING(f'Warning: {stats["errors"]} errors encountered during update')
            )
        
        # Get system summary
        try:
            summary = impact_service.generate_system_summary()
            self.stdout.write('\nSystem Status:')
            self.stdout.write(f'  Stable stopes: {summary.stable_stopes}')
            self.stdout.write(f'  Elevated risk: {summary.elevated_stopes}')
            self.stdout.write(f'  High risk: {summary.high_risk_stopes}')
            
            if summary.critical_stopes > 0:
                self.stdout.write(
                    self.style.WARNING(f'  CRITICAL: {summary.critical_stopes}')
                )
            else:
                self.stdout.write(f'  Critical: {summary.critical_stopes}')
            
            self.stdout.write(f'  Average impact: {summary.average_impact:.3f}')
            self.stdout.write(f'  Peak impact: {summary.peak_impact:.3f}')
            self.stdout.write(f'  Trend: {summary.trending_direction}')
        
        except Exception as e:
            self.stdout.write(
                self.style.WARNING(f'Could not generate system summary: {e}')
            )
        
        self.stdout.write('='*60)


# ===== UPDATE IMPACT SCORES COMMAND COMPLETE =====
