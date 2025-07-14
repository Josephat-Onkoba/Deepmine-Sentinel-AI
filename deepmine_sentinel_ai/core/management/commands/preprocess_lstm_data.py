"""
Management command to preprocess monitoring data for LSTM training.
Creates time series sequences with features and labels.
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
from core.models import Stope, TimeSeriesData, FeatureEngineeringConfig
from core.data_preprocessing import TimeSeriesPreprocessor, DataValidator
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Preprocess monitoring data for LSTM training'

    def add_arguments(self, parser):
        parser.add_argument(
            '--stope',
            type=str,
            help='Specific stope name to process (default: all active stopes)',
        )
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Number of days of data to process (default: 30)',
        )
        parser.add_argument(
            '--sequence-length',
            type=int,
            default=24,
            help='Length of each time series sequence in hours (default: 24)',
        )
        parser.add_argument(
            '--config',
            type=str,
            default='default',
            help='Feature engineering configuration name (default: default)',
        )
        parser.add_argument(
            '--overlap',
            type=float,
            default=0.5,
            help='Overlap between sequences (0.0-1.0, default: 0.5)',
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing time series data before processing',
        )
        parser.add_argument(
            '--validate-only',
            action='store_true',
            help='Only validate existing data, do not create new sequences',
        )

    def handle(self, *args, **options):
        stope_name = options['stope']
        days = options['days']
        sequence_length = options['sequence_length']
        config_name = options['config']
        overlap = options['overlap']
        clear_existing = options['clear_existing']
        validate_only = options['validate_only']
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Starting LSTM data preprocessing...\n'
                f'Configuration: {config_name}\n'
                f'Sequence length: {sequence_length} hours\n'
                f'Data period: {days} days\n'
                f'Overlap: {overlap:.1%}'
            )
        )
        
        if validate_only:
            self._validate_existing_data()
            return
        
        # Get stopes to process
        if stope_name:
            try:
                stopes = [Stope.objects.get(stope_name=stope_name)]
            except Stope.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Stope "{stope_name}" not found')
                )
                return
        else:
            stopes = Stope.objects.filter(is_active=True)
        
        if not stopes:
            self.stdout.write(
                self.style.WARNING('No stopes found to process')
            )
            return
        
        # Clear existing data if requested
        if clear_existing:
            self._clear_existing_data(stopes)
        
        # Initialize preprocessor
        try:
            preprocessor = TimeSeriesPreprocessor(config_name)
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to initialize preprocessor: {e}')
            )
            return
        
        # Process each stope
        total_sequences = 0
        for stope in stopes:
            self.stdout.write(f'\nProcessing {stope.stope_name}...')
            
            # Calculate time range
            end_time = timezone.now()
            start_time = end_time - timedelta(days=days)
            
            try:
                # Create sequences
                result = preprocessor.process_stope_data(
                    stope=stope,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Report results
                train_count = len(result['train'])
                val_count = len(result['validation'])
                test_count = len(result['test'])
                stope_total = train_count + val_count + test_count
                
                self.stdout.write(
                    f'  Created {stope_total} sequences:\n'
                    f'    Training: {train_count}\n'
                    f'    Validation: {val_count}\n'
                    f'    Test: {test_count}'
                )
                
                total_sequences += stope_total
                
                # Validate created sequences
                self._validate_stope_sequences(stope, result)
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error processing {stope.stope_name}: {e}')
                )
                logger.error(f"Error processing stope {stope.stope_name}: {e}")
        
        self.stdout.write(
            self.style.SUCCESS(
                f'\n✅ Preprocessing complete!\n'
                f'Total sequences created: {total_sequences}\n'
                f'Configuration used: {config_name}'
            )
        )
    
    def _clear_existing_data(self, stopes):
        """Clear existing time series data for specified stopes"""
        total_deleted = 0
        for stope in stopes:
            deleted_count = TimeSeriesData.objects.filter(stope=stope).count()
            TimeSeriesData.objects.filter(stope=stope).delete()
            total_deleted += deleted_count
        
        if total_deleted > 0:
            self.stdout.write(
                self.style.WARNING(f'Cleared {total_deleted} existing sequences')
            )
    
    def _validate_stope_sequences(self, stope, sequences_dict):
        """Validate sequences for a specific stope"""
        all_sequences = (
            sequences_dict['train'] + 
            sequences_dict['validation'] + 
            sequences_dict['test']
        )
        
        validator = DataValidator()
        valid_count = 0
        error_count = 0
        
        for sequence in all_sequences:
            result = validator.validate_time_series_data(sequence)
            if result['is_valid']:
                valid_count += 1
            else:
                error_count += 1
                self.stdout.write(
                    self.style.WARNING(
                        f'    Validation errors in {sequence.sequence_id}: '
                        f'{", ".join(result["errors"])}'
                    )
                )
        
        if error_count == 0:
            self.stdout.write(
                self.style.SUCCESS(f'  ✅ All {valid_count} sequences validated successfully')
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f'  ⚠️  {valid_count} valid, {error_count} with errors'
                )
            )
    
    def _validate_existing_data(self):
        """Validate all existing time series data"""
        self.stdout.write('Validating existing time series data...')
        
        all_sequences = TimeSeriesData.objects.all()
        if not all_sequences.exists():
            self.stdout.write(
                self.style.WARNING('No time series data found to validate')
            )
            return
        
        validator = DataValidator()
        valid_count = 0
        error_count = 0
        total_count = all_sequences.count()
        
        for i, sequence in enumerate(all_sequences):
            if i % 100 == 0:  # Progress indicator
                self.stdout.write(f'  Processed {i}/{total_count} sequences...')
            
            result = validator.validate_time_series_data(sequence)
            if result['is_valid']:
                valid_count += 1
            else:
                error_count += 1
                if error_count <= 10:  # Show first 10 errors
                    self.stdout.write(
                        self.style.WARNING(
                            f'Errors in {sequence.sequence_id}: '
                            f'{", ".join(result["errors"])}'
                        )
                    )
        
        # Summary
        error_percentage = (error_count / total_count) * 100
        self.stdout.write(
            self.style.SUCCESS(
                f'\n📊 Validation Summary:\n'
                f'Total sequences: {total_count}\n'
                f'Valid sequences: {valid_count}\n'
                f'Invalid sequences: {error_count} ({error_percentage:.1f}%)'
            )
        )
        
        if error_count > 0:
            self.stdout.write(
                self.style.WARNING(
                    '⚠️  Consider reprocessing sequences with errors'
                )
            )
