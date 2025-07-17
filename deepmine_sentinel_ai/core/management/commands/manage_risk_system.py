"""
Task 6: Risk Level Classification System - Management Command
=============================================================

Management command for comprehensive risk system operations including:
- Risk level classification for all stopes
- Alert management and cleanup
- Risk transition analysis
- System health monitoring
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count, Avg, Max, Min
from core.models import (
    Stope, RiskThreshold, RiskTransition, RiskAlert, 
    RiskClassificationRule, ImpactScore
)
from core.risk.risk_classification_service import risk_classification_service


class Command(BaseCommand):
    help = 'Manage the risk classification system'
    
    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest='action', help='Available actions')
        
        # Classify all stopes
        classify_parser = subparsers.add_parser('classify', help='Classify risk levels for all stopes')
        classify_parser.add_argument(
            '--stope-id',
            type=int,
            help='Classify specific stope (default: all active stopes)'
        )
        classify_parser.add_argument(
            '--force-transition',
            action='store_true',
            help='Force risk transition detection even if already at same level'
        )
        
        # Manage alerts
        alert_parser = subparsers.add_parser('alerts', help='Manage risk alerts')
        alert_parser.add_argument(
            '--cleanup-resolved',
            action='store_true',
            help='Archive old resolved alerts (older than 30 days)'
        )
        alert_parser.add_argument(
            '--escalate-overdue',
            action='store_true',
            help='Escalate alerts that have been active too long'
        )
        alert_parser.add_argument(
            '--summary',
            action='store_true',
            help='Show alert summary statistics'
        )
        
        # Analyze transitions
        analyze_parser = subparsers.add_parser('analyze', help='Analyze risk patterns')
        analyze_parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Days of history to analyze (default: 7)'
        )
        analyze_parser.add_argument(
            '--export',
            help='Export analysis to file'
        )
        
        # System health
        health_parser = subparsers.add_parser('health', help='Check system health')
        health_parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed health information'
        )
        
        # Monitor system
        monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
        monitor_parser.add_argument(
            '--interval',
            type=int,
            default=60,
            help='Monitoring interval in seconds (default: 60)'
        )
        monitor_parser.add_argument(
            '--duration',
            type=int,
            default=300,
            help='Monitoring duration in seconds (default: 300)'
        )
    
    def handle(self, *args, **options):
        action = options.get('action')
        
        if action == 'classify':
            self.handle_classify(options)
        elif action == 'alerts':
            self.handle_alerts(options)
        elif action == 'analyze':
            self.handle_analyze(options)
        elif action == 'health':
            self.handle_health(options)
        elif action == 'monitor':
            self.handle_monitor(options)
        else:
            self.print_usage()
    
    def print_usage(self):
        """Print usage information"""
        self.stdout.write(
            self.style.SUCCESS('🎯 Risk Classification System Management\n')
        )
        self.stdout.write('Available commands:')
        self.stdout.write('  classify    - Classify risk levels for stopes')
        self.stdout.write('  alerts      - Manage risk alerts')
        self.stdout.write('  analyze     - Analyze risk patterns and trends')
        self.stdout.write('  health      - Check system health')
        self.stdout.write('  monitor     - Real-time system monitoring')
        self.stdout.write('\nUse --help with any command for detailed options.')
    
    def handle_classify(self, options):
        """Handle risk classification"""
        self.stdout.write(
            self.style.SUCCESS('🔍 Risk Level Classification\n')
        )
        
        # Get stopes to classify
        if options.get('stope_id'):
            try:
                stopes = [Stope.objects.get(id=options['stope_id'])]
                self.stdout.write(f"Classifying stope: {stopes[0].stope_name}")
            except Stope.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f"Stope with ID {options['stope_id']} not found")
                )
                return
        else:
            stopes = Stope.objects.filter(is_active=True)
            self.stdout.write(f"Classifying {stopes.count()} active stopes")
        
        # Classification results
        results = {
            'stable': 0,
            'elevated': 0,
            'high_risk': 0,
            'critical': 0,
            'emergency': 0,
            'transitions': 0,
            'alerts_generated': 0
        }
        
        for stope in stopes:
            try:
                # Get current risk level
                current_risk = risk_classification_service.classify_stope_risk_level(stope)
                results[current_risk] = results.get(current_risk, 0) + 1
                
                # Detect transitions
                transition = risk_classification_service.detect_risk_transition(
                    stope, current_risk
                )
                
                if transition:
                    results['transitions'] += 1
                    self.stdout.write(
                        f"  ⚠️  Transition: {stope.stope_name} → {current_risk}"
                    )
                    
                    # Count alerts generated for this transition
                    alert_count = RiskAlert.objects.filter(
                        risk_transition=transition
                    ).count()
                    results['alerts_generated'] += alert_count
                else:
                    self.stdout.write(
                        f"  ✅ {stope.stope_name}: {current_risk}"
                    )
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  ❌ Error classifying {stope.stope_name}: {e}")
                )
        
        # Print summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write('📊 Classification Summary:')
        for risk_level, count in results.items():
            if risk_level not in ['transitions', 'alerts_generated'] and count > 0:
                self.stdout.write(f"  {risk_level}: {count} stopes")
        
        self.stdout.write(f"\n🔄 Transitions detected: {results['transitions']}")
        self.stdout.write(f"🚨 Alerts generated: {results['alerts_generated']}")
    
    def handle_alerts(self, options):
        """Handle alert management"""
        self.stdout.write(
            self.style.SUCCESS('🚨 Risk Alert Management\n')
        )
        
        if options.get('summary'):
            self.show_alert_summary()
        
        if options.get('cleanup_resolved'):
            self.cleanup_resolved_alerts()
        
        if options.get('escalate_overdue'):
            self.escalate_overdue_alerts()
    
    def show_alert_summary(self):
        """Show alert summary statistics"""
        total_alerts = RiskAlert.objects.count()
        active_alerts = RiskAlert.objects.filter(
            status__in=['active', 'acknowledged', 'investigating']
        ).count()
        
        self.stdout.write(f"📊 Alert Statistics:")
        self.stdout.write(f"  Total alerts: {total_alerts}")
        self.stdout.write(f"  Active alerts: {active_alerts}")
        
        # Breakdown by status
        status_breakdown = RiskAlert.objects.values('status').annotate(
            count=Count('id')
        ).order_by('-count')
        
        self.stdout.write("\n  By Status:")
        for item in status_breakdown:
            self.stdout.write(f"    {item['status']}: {item['count']}")
        
        # Breakdown by priority
        priority_breakdown = RiskAlert.objects.values('priority').annotate(
            count=Count('id')
        ).order_by('-count')
        
        self.stdout.write("\n  By Priority:")
        for item in priority_breakdown:
            self.stdout.write(f"    {item['priority']}: {item['count']}")
        
        # Recent alerts (last 24 hours)
        recent_alerts = RiskAlert.objects.filter(
            created_at__gte=timezone.now() - timedelta(hours=24)
        ).count()
        self.stdout.write(f"\n  Recent (24h): {recent_alerts}")
    
    def cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_date = timezone.now() - timedelta(days=30)
        old_alerts = RiskAlert.objects.filter(
            status='resolved',
            resolved_at__lt=cutoff_date
        )
        
        count = old_alerts.count()
        if count > 0:
            # In a real system, you might archive rather than delete
            self.stdout.write(f"🧹 Found {count} old resolved alerts to clean up")
            
            # For now, just mark them for cleanup
            old_alerts.update(
                alert_data={'archived': True, 'archived_at': timezone.now().isoformat()}
            )
            self.stdout.write(f"✅ Marked {count} alerts for archival")
        else:
            self.stdout.write("✅ No old alerts to clean up")
    
    def escalate_overdue_alerts(self):
        """Escalate alerts that have been active too long"""
        # Define escalation timeframes
        escalation_rules = [
            (timedelta(hours=2), 'medium'),
            (timedelta(hours=6), 'high'),
            (timedelta(hours=12), 'critical'),
            (timedelta(hours=24), 'emergency'),
        ]
        
        escalated_count = 0
        
        for time_threshold, min_priority in escalation_rules:
            cutoff_time = timezone.now() - time_threshold
            
            overdue_alerts = RiskAlert.objects.filter(
                status__in=['active', 'acknowledged'],
                created_at__lt=cutoff_time,
                escalation_level__lt=2  # Don't over-escalate
            )
            
            for alert in overdue_alerts:
                # Check if should escalate based on priority
                priority_order = ['low', 'medium', 'high', 'critical', 'emergency']
                current_idx = priority_order.index(alert.priority)
                min_idx = priority_order.index(min_priority)
                
                if current_idx < min_idx:
                    alert.escalate('system_auto')
                    escalated_count += 1
                    self.stdout.write(
                        f"⬆️  Escalated: {alert.title} (age: {alert.age})"
                    )
        
        if escalated_count == 0:
            self.stdout.write("✅ No alerts require escalation")
        else:
            self.stdout.write(f"⬆️  Escalated {escalated_count} overdue alerts")
    
    def handle_analyze(self, options):
        """Handle risk pattern analysis"""
        days = options.get('days', 7)
        self.stdout.write(
            self.style.SUCCESS(f'📈 Risk Pattern Analysis ({days} days)\n')
        )
        
        start_date = timezone.now() - timedelta(days=days)
        
        # Transition analysis
        transitions = RiskTransition.objects.filter(
            transition_timestamp__gte=start_date
        )
        
        self.stdout.write(f"📊 Transition Analysis:")
        self.stdout.write(f"  Total transitions: {transitions.count()}")
        
        # Escalations vs de-escalations
        escalations = sum(1 for t in transitions if t.is_escalation)
        de_escalations = sum(1 for t in transitions if t.is_de_escalation)
        
        self.stdout.write(f"  Escalations: {escalations}")
        self.stdout.write(f"  De-escalations: {de_escalations}")
        
        # Most common triggers
        trigger_stats = transitions.values('trigger_type').annotate(
            count=Count('id')
        ).order_by('-count')[:5]
        
        self.stdout.write("\n  Top Triggers:")
        for trigger in trigger_stats:
            self.stdout.write(f"    {trigger['trigger_type']}: {trigger['count']}")
        
        # Most active stopes
        stope_stats = transitions.values(
            'stope__stope_name', 'stope__id'
        ).annotate(
            transition_count=Count('id')
        ).order_by('-transition_count')[:5]
        
        self.stdout.write("\n  Most Active Stopes:")
        for stope in stope_stats:
            self.stdout.write(
                f"    {stope['stope__stope_name']}: {stope['transition_count']} transitions"
            )
        
        # Risk level distribution over time
        self.analyze_risk_trends(start_date)
        
        # Export if requested
        if options.get('export'):
            self.export_analysis(options['export'], start_date, days)
    
    def analyze_risk_trends(self, start_date):
        """Analyze risk level trends"""
        self.stdout.write("\n📈 Risk Level Trends:")
        
        # Current distribution
        active_stopes = Stope.objects.filter(is_active=True)
        current_distribution = {}
        
        for stope in active_stopes:
            risk_level = risk_classification_service.classify_stope_risk_level(stope)
            current_distribution[risk_level] = current_distribution.get(risk_level, 0) + 1
        
        self.stdout.write("  Current Distribution:")
        for risk_level, count in current_distribution.items():
            percentage = (count / active_stopes.count()) * 100 if active_stopes.count() > 0 else 0
            self.stdout.write(f"    {risk_level}: {count} ({percentage:.1f}%)")
    
    def export_analysis(self, filename, start_date, days):
        """Export analysis to file"""
        try:
            import json
            
            # Gather analysis data
            analysis_data = {
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': timezone.now().isoformat(),
                    'days_analyzed': days
                },
                'transitions': [],
                'alerts': [],
                'current_risk_levels': {}
            }
            
            # Export transitions
            transitions = RiskTransition.objects.filter(
                transition_timestamp__gte=start_date
            )
            
            for transition in transitions:
                analysis_data['transitions'].append({
                    'stope_name': transition.stope.stope_name,
                    'previous_risk': transition.previous_risk_level,
                    'new_risk': transition.new_risk_level,
                    'trigger_type': transition.trigger_type,
                    'timestamp': transition.transition_timestamp.isoformat(),
                    'is_escalation': transition.is_escalation
                })
            
            # Export alerts
            alerts = RiskAlert.objects.filter(
                created_at__gte=start_date
            )
            
            for alert in alerts:
                analysis_data['alerts'].append({
                    'stope_name': alert.stope.stope_name,
                    'alert_type': alert.alert_type,
                    'priority': alert.priority,
                    'status': alert.status,
                    'title': alert.title,
                    'created_at': alert.created_at.isoformat()
                })
            
            # Current risk levels
            active_stopes = Stope.objects.filter(is_active=True)
            for stope in active_stopes:
                risk_level = risk_classification_service.classify_stope_risk_level(stope)
                analysis_data['current_risk_levels'][stope.stope_name] = risk_level
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            self.stdout.write(f"✅ Analysis exported to {filename}")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Error exporting analysis: {e}")
            )
    
    def handle_health(self, options):
        """Handle system health check"""
        self.stdout.write(
            self.style.SUCCESS('🏥 Risk System Health Check\n')
        )
        
        # Check thresholds
        active_thresholds = RiskThreshold.objects.filter(is_active=True).count()
        total_thresholds = RiskThreshold.objects.count()
        
        self.stdout.write(f"📏 Thresholds: {active_thresholds}/{total_thresholds} active")
        
        if active_thresholds == 0:
            self.stdout.write(
                self.style.WARNING("⚠️  No active thresholds found!")
            )
        
        # Check for stopes without impact scores
        stopes_without_scores = Stope.objects.filter(
            is_active=True,
            impact_score__isnull=True
        ).count()
        
        if stopes_without_scores > 0:
            self.stdout.write(
                self.style.WARNING(
                    f"⚠️  {stopes_without_scores} active stopes have no impact scores"
                )
            )
        
        # Check alert response times
        recent_alerts = RiskAlert.objects.filter(
            created_at__gte=timezone.now() - timedelta(hours=24)
        )
        
        unacknowledged = recent_alerts.filter(status='active').count()
        if unacknowledged > 0:
            self.stdout.write(
                self.style.WARNING(f"⚠️  {unacknowledged} unacknowledged alerts")
            )
        
        # Check for stuck transitions
        old_transitions = RiskTransition.objects.filter(
            is_confirmed=False,
            transition_timestamp__lt=timezone.now() - timedelta(hours=1)
        ).count()
        
        if old_transitions > 0:
            self.stdout.write(
                self.style.WARNING(f"⚠️  {old_transitions} unconfirmed transitions")
            )
        
        # Service health
        try:
            test_stope = Stope.objects.filter(is_active=True).first()
            if test_stope:
                risk_level = risk_classification_service.classify_stope_risk_level(test_stope)
                self.stdout.write("✅ Risk classification service: OK")
            else:
                self.stdout.write("⚠️  No active stopes to test classification service")
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Risk classification service error: {e}")
            )
        
        self.stdout.write("\n✅ Health check completed")
    
    def handle_monitor(self, options):
        """Handle real-time monitoring"""
        import time
        
        interval = options.get('interval', 60)
        duration = options.get('duration', 300)
        
        self.stdout.write(
            self.style.SUCCESS(
                f'👁️  Real-time Monitoring (interval: {interval}s, duration: {duration}s)\n'
            )
        )
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < duration:
            iterations += 1
            self.stdout.write(f"\n📊 Monitoring Cycle {iterations}:")
            self.stdout.write(f"Time: {timezone.now().strftime('%H:%M:%S')}")
            
            # Quick health check
            active_alerts = RiskAlert.objects.filter(
                status__in=['active', 'acknowledged']
            ).count()
            
            recent_transitions = RiskTransition.objects.filter(
                transition_timestamp__gte=timezone.now() - timedelta(minutes=interval/60)
            ).count()
            
            self.stdout.write(f"Active alerts: {active_alerts}")
            self.stdout.write(f"Recent transitions: {recent_transitions}")
            
            if recent_transitions > 0:
                self.stdout.write("⚠️  New transitions detected!")
            
            # Sleep until next interval
            time.sleep(interval)
        
        self.stdout.write(f"\n✅ Monitoring completed ({iterations} cycles)")
