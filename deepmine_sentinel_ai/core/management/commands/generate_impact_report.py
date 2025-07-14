"""
Impact Analysis Report Management Command

Django management command for generating comprehensive impact analysis reports.
Provides detailed analysis of stope impacts, trends, and system health.

Usage:
python manage.py generate_impact_report
python manage.py generate_impact_report --stope-id 123
python manage.py generate_impact_report --export-csv
python manage.py generate_impact_report --time-range 48
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db.models import Q, Avg, Max, Min, Count
import csv
import json
from datetime import datetime, timedelta

from core.models import Stope, ImpactScore, ImpactHistory, OperationalEvent
from core.impact import impact_service


class Command(BaseCommand):
    """Management command for generating impact analysis reports"""
    
    help = 'Generate comprehensive impact analysis reports'
    
    def add_arguments(self, parser):
        """Add command line arguments"""
        parser.add_argument(
            '--stope-id',
            type=int,
            help='Generate report for specific stope ID',
        )
        
        parser.add_argument(
            '--stope-name',
            type=str,
            help='Generate report for specific stope name',
        )
        
        parser.add_argument(
            '--risk-level',
            type=str,
            choices=['stable', 'elevated', 'high_risk', 'critical'],
            help='Filter stopes by risk level',
        )
        
        parser.add_argument(
            '--time-range',
            type=int,
            default=168,
            help='Time range in hours for analysis (default: 168)',
        )
        
        parser.add_argument(
            '--export-csv',
            type=str,
            help='Export results to CSV file',
        )
        
        parser.add_argument(
            '--export-json',
            type=str,
            help='Export results to JSON file',
        )
        
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Include detailed contributing events analysis',
        )
        
        parser.add_argument(
            '--summary-only',
            action='store_true',
            help='Show only system summary',
        )
    
    def handle(self, *args, **options):
        """Main command handler"""
        start_time = timezone.now()
        
        self.stdout.write(
            self.style.SUCCESS(f'Generating impact analysis report at {start_time}')
        )
        
        try:
            # Generate appropriate report based on options
            if options['summary_only']:
                self._generate_system_summary()
            elif options['stope_id'] or options['stope_name']:
                self._generate_stope_report(options)
            else:
                self._generate_comprehensive_report(options)
        
        except Exception as e:
            raise CommandError(f'Report generation failed: {e}')
        
        duration = (timezone.now() - start_time).total_seconds()
        self.stdout.write(
            self.style.SUCCESS(f'Report generated in {duration:.2f} seconds')
        )
    
    def _generate_system_summary(self):
        """Generate system-wide summary report"""
        self.stdout.write(self.style.SUCCESS('\n📊 SYSTEM IMPACT SUMMARY'))
        self.stdout.write('='*60)
        
        # Get system summary
        summary = impact_service.generate_system_summary()
        
        # Display summary
        self.stdout.write(f'Analysis timestamp: {summary.analysis_timestamp}')
        self.stdout.write(f'Total active stopes: {summary.total_stopes}')
        self.stdout.write('')
        
        # Risk distribution
        self.stdout.write('Risk Level Distribution:')
        self.stdout.write(f'  🟢 Stable: {summary.stable_stopes} ({self._percentage(summary.stable_stopes, summary.total_stopes):.1f}%)')
        self.stdout.write(f'  🟡 Elevated: {summary.elevated_stopes} ({self._percentage(summary.elevated_stopes, summary.total_stopes):.1f}%)')
        self.stdout.write(f'  🟠 High Risk: {summary.high_risk_stopes} ({self._percentage(summary.high_risk_stopes, summary.total_stopes):.1f}%)')
        
        if summary.critical_stopes > 0:
            self.stdout.write(f'  🔴 CRITICAL: {summary.critical_stopes} ({self._percentage(summary.critical_stopes, summary.total_stopes):.1f}%)')
        else:
            self.stdout.write(f'  🔴 Critical: {summary.critical_stopes}')
        
        self.stdout.write('')
        
        # Impact metrics
        self.stdout.write('Impact Metrics:')
        self.stdout.write(f'  Average impact score: {summary.average_impact:.3f}')
        self.stdout.write(f'  Peak impact score: {summary.peak_impact:.3f}')
        self.stdout.write(f'  System trend: {summary.trending_direction}')
        self.stdout.write('')
        
        # Activity metrics
        self.stdout.write('Recent Activity:')
        self.stdout.write(f'  Events (24h): {summary.recent_events}')
        
        # Alerts and recommendations
        if summary.critical_stopes > 0:
            self.stdout.write('\n🚨 IMMEDIATE ATTENTION REQUIRED:')
            critical_stopes = ImpactScore.objects.filter(
                risk_level='critical',
                stope__is_active=True
            ).select_related('stope')
            
            for score in critical_stopes:
                self.stdout.write(f'  - {score.stope.stope_name}: {score.current_score:.3f}')
        
        # Trend analysis
        if summary.trending_direction == 'deteriorating':
            self.stdout.write('\n⚠️  System trend is deteriorating - increase monitoring')
        elif summary.trending_direction == 'improving':
            self.stdout.write('\n✅ System trend is improving')
    
    def _generate_stope_report(self, options):
        """Generate detailed report for specific stope"""
        # Find the stope
        stope = self._get_target_stope(options)
        
        self.stdout.write(self.style.SUCCESS(f'\n📋 DETAILED STOPE ANALYSIS: {stope.stope_name}'))
        self.stdout.write('='*80)
        
        # Get comprehensive analysis
        analysis = impact_service.get_stope_impact_analysis(
            stope,
            include_contributing_events=options['detailed'],
            time_window_hours=options['time_range']
        )
        
        # Basic stope information
        self.stdout.write('Stope Information:')
        self.stdout.write(f'  Name: {stope.stope_name}')
        self.stdout.write(f'  Depth: {stope.depth}m')
        self.stdout.write(f'  HR: {stope.hr}')
        self.stdout.write(f'  Is Active: {stope.is_active}')
        self.stdout.write('')
        
        # Current impact status
        self.stdout.write('Current Impact Status:')
        self.stdout.write(f'  Impact Score: {analysis.current_impact:.6f}')
        self.stdout.write(f'  Risk Level: {self._format_risk_level(analysis.risk_level)}')
        self.stdout.write(f'  Previous Score: {analysis.previous_impact:.6f}')
        self.stdout.write(f'  Change: {self._format_change(analysis.impact_change)}')
        
        if analysis.risk_level_changed:
            self.stdout.write('  🔄 Risk level has changed recently!')
        
        self.stdout.write('')
        
        # Contributing events
        if options['detailed'] and analysis.contributing_events:
            self.stdout.write('Contributing Events:')
            for event in analysis.contributing_events[:10]:  # Top 10
                self.stdout.write(f'  - {event["event_type"]} ({event["severity"]})')
                self.stdout.write(f'    Impact: {event["impact_contribution"]:.6f}')
                self.stdout.write(f'    Distance: {event["distance"]:.1f}m')
                self.stdout.write(f'    Time: {event["timestamp"]}')
                self.stdout.write('')
        
        # Recommendations
        if analysis.recommendations:
            self.stdout.write('Recommendations:')
            for i, rec in enumerate(analysis.recommendations, 1):
                self.stdout.write(f'  {i}. {rec}')
            self.stdout.write('')
        
        # Historical trend
        self._show_stope_trend(stope, options['time_range'])
    
    def _generate_comprehensive_report(self, options):
        """Generate comprehensive system report"""
        self.stdout.write(self.style.SUCCESS('\n📊 COMPREHENSIVE IMPACT ANALYSIS REPORT'))
        self.stdout.write('='*80)
        
        # System summary first
        self._generate_system_summary()
        
        # Filter stopes based on options
        stopes = self._get_filtered_stopes(options)
        
        self.stdout.write(f'\n📋 DETAILED STOPE ANALYSIS ({len(stopes)} stopes)')
        self.stdout.write('-'*80)
        
        # Prepare export data
        export_data = []
        
        # Analyze each stope
        for stope in stopes:
            try:
                analysis = impact_service.get_stope_impact_analysis(
                    stope,
                    include_contributing_events=False,  # Skip for performance
                    time_window_hours=options['time_range']
                )
                
                # Display brief analysis
                risk_icon = self._get_risk_icon(analysis.risk_level)
                change_indicator = self._get_change_indicator(analysis.impact_change)
                
                self.stdout.write(
                    f'{risk_icon} {stope.stope_name:20} '
                    f'Score: {analysis.current_impact:8.3f} '
                    f'{change_indicator} '
                    f'Risk: {analysis.risk_level:10}'
                )
                
                # Prepare export data
                if options['export_csv'] or options['export_json']:
                    export_data.append({
                        'stope_name': stope.stope_name,
                        'stope_id': stope.id,
                        'current_impact': analysis.current_impact,
                        'previous_impact': analysis.previous_impact,
                        'impact_change': analysis.impact_change,
                        'risk_level': analysis.risk_level,
                        'risk_level_changed': analysis.risk_level_changed,
                        'depth': stope.depth,
                        'hr': stope.hr,
                        'is_active': stope.is_active,
                        'analysis_timestamp': analysis.analysis_timestamp.isoformat()
                    })
            
            except Exception as e:
                self.stdout.write(f'❌ Error analyzing {stope.stope_name}: {e}')
        
        # Export data if requested
        if options['export_csv']:
            self._export_csv(export_data, options['export_csv'])
        
        if options['export_json']:
            self._export_json(export_data, options['export_json'])
        
        # Show recent activity
        self._show_recent_activity(options['time_range'])
    
    def _get_target_stope(self, options):
        """Get target stope from options"""
        if options['stope_id']:
            try:
                return Stope.objects.get(id=options['stope_id'])
            except Stope.DoesNotExist:
                raise CommandError(f'Stope with ID {options["stope_id"]} not found')
        
        elif options['stope_name']:
            try:
                return Stope.objects.get(stope_name=options['stope_name'])
            except Stope.DoesNotExist:
                raise CommandError(f'Stope with name "{options["stope_name"]}" not found')
            except Stope.MultipleObjectsReturned:
                raise CommandError(f'Multiple stopes found with name "{options["stope_name"]}"')
        
        else:
            raise CommandError('Either --stope-id or --stope-name must be specified')
    
    def _get_filtered_stopes(self, options):
        """Get filtered list of stopes based on options"""
        queryset = Stope.objects.filter(is_active=True)
        
        if options['risk_level']:
            # Filter by risk level
            impact_scores = ImpactScore.objects.filter(
                risk_level=options['risk_level']
            ).values_list('stope_id', flat=True)
            queryset = queryset.filter(id__in=impact_scores)
        
        return queryset.order_by('stope_name')
    
    def _show_stope_trend(self, stope, hours):
        """Show historical trend for a stope"""
        self.stdout.write('Historical Trend:')
        
        # Get recent history
        start_time = timezone.now() - timedelta(hours=hours)
        history = ImpactHistory.objects.filter(
            stope=stope,
            timestamp__gte=start_time
        ).order_by('timestamp')
        
        if not history.exists():
            self.stdout.write('  No historical data available')
            return
        
        # Show trend data
        scores = [h.impact_score for h in history]
        if len(scores) > 1:
            first_score = scores[0]
            last_score = scores[-1]
            trend = last_score - first_score
            
            self.stdout.write(f'  Data points: {len(scores)}')
            self.stdout.write(f'  First score: {first_score:.3f}')
            self.stdout.write(f'  Last score: {last_score:.3f}')
            self.stdout.write(f'  Overall trend: {self._format_change(trend)}')
            
            # Show recent changes
            recent_changes = [h for h in history if h.change_magnitude != 0][-5:]
            if recent_changes:
                self.stdout.write('  Recent significant changes:')
                for change in recent_changes:
                    self.stdout.write(f'    {change.timestamp.strftime("%m/%d %H:%M")}: {self._format_change(change.change_magnitude)}')
    
    def _show_recent_activity(self, hours):
        """Show recent operational activity"""
        self.stdout.write(f'\n🔄 RECENT ACTIVITY ({hours}h)')
        self.stdout.write('-'*40)
        
        start_time = timezone.now() - timedelta(hours=hours)
        events = OperationalEvent.objects.filter(
            timestamp__gte=start_time
        ).order_by('-timestamp')[:20]
        
        if not events.exists():
            self.stdout.write('No recent events')
            return
        
        # Group by event type
        event_counts = {}
        for event in events:
            key = f"{event.event_type} ({event.severity})"
            event_counts[key] = event_counts.get(key, 0) + 1
        
        self.stdout.write('Event Summary:')
        for event_type, count in sorted(event_counts.items()):
            self.stdout.write(f'  {event_type}: {count}')
        
        self.stdout.write(f'\nTotal events: {events.count()}')
    
    def _export_csv(self, data, filename):
        """Export data to CSV file"""
        try:
            with open(filename, 'w', newline='') as csvfile:
                if not data:
                    return
                
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
            
            self.stdout.write(f'\n💾 Data exported to CSV: {filename}')
        except Exception as e:
            self.stdout.write(f'❌ Error exporting CSV: {e}')
    
    def _export_json(self, data, filename):
        """Export data to JSON file"""
        try:
            with open(filename, 'w') as jsonfile:
                json.dump(data, jsonfile, indent=2, default=str)
            
            self.stdout.write(f'\n💾 Data exported to JSON: {filename}')
        except Exception as e:
            self.stdout.write(f'❌ Error exporting JSON: {e}')
    
    # Helper methods for formatting
    
    def _percentage(self, part, total):
        """Calculate percentage safely"""
        return (part / total * 100) if total > 0 else 0
    
    def _format_risk_level(self, risk_level):
        """Format risk level with appropriate styling"""
        icons = {
            'stable': '🟢 Stable',
            'elevated': '🟡 Elevated',
            'high_risk': '🟠 High Risk',
            'critical': '🔴 CRITICAL'
        }
        return icons.get(risk_level, risk_level)
    
    def _format_change(self, change):
        """Format change value with direction indicator"""
        if change > 0.001:
            return f'↗️ +{change:.3f}'
        elif change < -0.001:
            return f'↘️ {change:.3f}'
        else:
            return '→ No change'
    
    def _get_risk_icon(self, risk_level):
        """Get icon for risk level"""
        icons = {
            'stable': '🟢',
            'elevated': '🟡',
            'high_risk': '🟠',
            'critical': '🔴'
        }
        return icons.get(risk_level, '⚪')
    
    def _get_change_indicator(self, change):
        """Get change indicator"""
        if change > 0.001:
            return '↗️'
        elif change < -0.001:
            return '↘️'
        else:
            return '→'


# ===== GENERATE IMPACT REPORT COMMAND COMPLETE =====
