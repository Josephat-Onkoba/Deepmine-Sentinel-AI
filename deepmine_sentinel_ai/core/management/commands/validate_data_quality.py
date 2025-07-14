"""
Management command to validate data quality for LSTM training.
Analyzes time series data and generates quality reports.
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models import TimeSeriesData, DataQualityMetrics, Stope
from core.data_preprocessing import DataValidator
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Validate data quality for LSTM training sequences'

    def add_arguments(self, parser):
        parser.add_argument(
            '--stope',
            type=str,
            help='Specific stope name to validate (default: all)',
        )
        parser.add_argument(
            '--sequence-type',
            type=str,
            choices=['training', 'validation', 'test', 'prediction'],
            help='Filter by sequence type',
        )
        parser.add_argument(
            '--min-quality',
            type=float,
            default=0.0,
            help='Minimum quality score to include (0.0-1.0)',
        )
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed quality analysis for each sequence',
        )
        parser.add_argument(
            '--export-report',
            type=str,
            help='Export quality report to file (JSON format)',
        )

    def handle(self, *args, **options):
        stope_name = options['stope']
        sequence_type = options['sequence_type']
        min_quality = options['min_quality']
        detailed = options['detailed']
        export_file = options['export_report']
        
        self.stdout.write(
            self.style.SUCCESS(
                f'🔍 Data Quality Validation Report\n'
                f'{"=" * 50}'
            )
        )
        
        # Build query filters
        filters = {}
        if stope_name:
            try:
                stope = Stope.objects.get(stope_name=stope_name)
                filters['stope'] = stope
            except Stope.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Stope "{stope_name}" not found')
                )
                return
        
        if sequence_type:
            filters['sequence_type'] = sequence_type
        
        if min_quality > 0:
            filters['data_quality_score__gte'] = min_quality
        
        # Get time series data
        time_series_qs = TimeSeriesData.objects.filter(**filters)
        
        if not time_series_qs.exists():
            self.stdout.write(
                self.style.WARNING('No time series data found matching criteria')
            )
            return
        
        total_count = time_series_qs.count()
        self.stdout.write(f'📊 Analyzing {total_count} sequences...\n')
        
        # Initialize validator
        validator = DataValidator()
        
        # Collect quality statistics
        quality_stats = {
            'total_sequences': total_count,
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'quality_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
            'avg_quality_score': 0.0,
            'sequences_by_type': {},
            'issues_found': [],
            'detailed_results': []
        }
        
        # Process each sequence
        total_quality = 0.0
        for i, ts in enumerate(time_series_qs):
            if i % 50 == 0:  # Progress indicator
                self.stdout.write(f'  Processed {i}/{total_count} sequences...')
            
            # Validate sequence
            validation_result = validator.validate_time_series_data(ts)
            
            # Update statistics
            if validation_result['is_valid']:
                quality_stats['valid_sequences'] += 1
            else:
                quality_stats['invalid_sequences'] += 1
                quality_stats['issues_found'].extend(validation_result['errors'])
            
            total_quality += ts.data_quality_score
            
            # Count by sequence type
            seq_type = ts.sequence_type
            if seq_type not in quality_stats['sequences_by_type']:
                quality_stats['sequences_by_type'][seq_type] = {
                    'count': 0, 'avg_quality': 0.0, 'total_quality': 0.0
                }
            quality_stats['sequences_by_type'][seq_type]['count'] += 1
            quality_stats['sequences_by_type'][seq_type]['total_quality'] += ts.data_quality_score
            
            # Get quality grade
            try:
                quality_metrics = ts.quality_metrics
                grade = quality_metrics.quality_grade
                quality_stats['quality_distribution'][grade] += 1
            except DataQualityMetrics.DoesNotExist:
                # Create quality metrics if missing
                self._create_missing_quality_metrics(ts)
                quality_stats['quality_distribution']['F'] += 1
            
            # Store detailed results if requested
            if detailed:
                quality_stats['detailed_results'].append({
                    'sequence_id': ts.sequence_id,
                    'stope': ts.stope.stope_name,
                    'type': ts.sequence_type,
                    'quality_score': ts.data_quality_score,
                    'is_valid': validation_result['is_valid'],
                    'errors': validation_result['errors'],
                    'warnings': validation_result['warnings']
                })
        
        # Calculate averages
        quality_stats['avg_quality_score'] = total_quality / total_count
        
        for seq_type, stats in quality_stats['sequences_by_type'].items():
            stats['avg_quality'] = stats['total_quality'] / stats['count']
        
        # Display results
        self._display_quality_report(quality_stats, detailed)
        
        # Export report if requested
        if export_file:
            self._export_quality_report(quality_stats, export_file)
    
    def _display_quality_report(self, stats, detailed=False):
        """Display the quality validation report"""
        
        # Summary statistics
        self.stdout.write(
            self.style.SUCCESS(
                f'\n📈 Quality Summary:\n'
                f'  Total sequences: {stats["total_sequences"]}\n'
                f'  Valid sequences: {stats["valid_sequences"]} '
                f'({stats["valid_sequences"]/stats["total_sequences"]*100:.1f}%)\n'
                f'  Invalid sequences: {stats["invalid_sequences"]} '
                f'({stats["invalid_sequences"]/stats["total_sequences"]*100:.1f}%)\n'
                f'  Average quality: {stats["avg_quality_score"]:.3f}'
            )
        )
        
        # Quality grade distribution
        self.stdout.write('\n🏆 Quality Grade Distribution:')
        for grade, count in stats['quality_distribution'].items():
            if count > 0:
                percentage = (count / stats['total_sequences']) * 100
                self.stdout.write(f'  Grade {grade}: {count} sequences ({percentage:.1f}%)')
        
        # By sequence type
        if stats['sequences_by_type']:
            self.stdout.write('\n📊 Quality by Sequence Type:')
            for seq_type, type_stats in stats['sequences_by_type'].items():
                self.stdout.write(
                    f'  {seq_type.capitalize()}: {type_stats["count"]} sequences, '
                    f'avg quality: {type_stats["avg_quality"]:.3f}'
                )
        
        # Common issues
        if stats['issues_found']:
            issue_counts = {}
            for issue in stats['issues_found']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            self.stdout.write('\n⚠️  Common Issues Found:')
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                self.stdout.write(f'  • {issue}: {count} occurrences')
        
        # Detailed results
        if detailed and stats['detailed_results']:
            self.stdout.write('\n🔍 Detailed Results:')
            for result in stats['detailed_results'][:10]:  # Show first 10
                status = '✅' if result['is_valid'] else '❌'
                self.stdout.write(
                    f'  {status} {result["sequence_id"]} ({result["stope"]}) - '
                    f'Quality: {result["quality_score"]:.3f}'
                )
                if result['errors']:
                    for error in result['errors']:
                        self.stdout.write(f'    Error: {error}')
        
        # Recommendations
        self._provide_recommendations(stats)
    
    def _provide_recommendations(self, stats):
        """Provide recommendations based on quality analysis"""
        self.stdout.write('\n💡 Recommendations:')
        
        validity_rate = stats['valid_sequences'] / stats['total_sequences']
        avg_quality = stats['avg_quality_score']
        
        if validity_rate < 0.8:
            self.stdout.write(
                '  🔧 Low validity rate - consider reprocessing data with updated validation rules'
            )
        
        if avg_quality < 0.7:
            self.stdout.write(
                '  📊 Low average quality - review data collection and preprocessing pipeline'
            )
        
        if stats['quality_distribution']['F'] > stats['total_sequences'] * 0.1:
            self.stdout.write(
                '  ⚠️  High number of failing sequences - investigate data sources'
            )
        
        if stats['invalid_sequences'] > 0:
            self.stdout.write(
                '  🔍 Invalid sequences found - run detailed validation and fix issues'
            )
        
        if avg_quality >= 0.8 and validity_rate >= 0.9:
            self.stdout.write(
                '  ✅ Data quality is good - proceed with LSTM training'
            )
    
    def _create_missing_quality_metrics(self, time_series):
        """Create quality metrics for sequences that don't have them"""
        quality_metrics = DataQualityMetrics(
            time_series_data=time_series,
            completeness_score=1.0 - (time_series.missing_data_percentage / 100.0),
            consistency_score=1.0 - min(1.0, time_series.anomaly_count / time_series.sequence_length),
            validity_score=1.0 if time_series.is_valid else 0.5,
            temporal_resolution_score=1.0,
            outlier_count=time_series.anomaly_count,
            outlier_percentage=(time_series.anomaly_count / time_series.sequence_length) * 100,
            invalid_readings_count=0,
            timestamp_irregularities=0,
            analysis_version='1.0'
        )
        quality_metrics.calculate_overall_quality()
        quality_metrics.save()
    
    def _export_quality_report(self, stats, filename):
        """Export quality report to JSON file"""
        import json
        
        try:
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            self.stdout.write(
                self.style.SUCCESS(f'\n📁 Quality report exported to: {filename}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to export report: {e}')
            )
