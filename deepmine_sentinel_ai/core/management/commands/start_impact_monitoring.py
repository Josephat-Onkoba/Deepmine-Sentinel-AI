"""
Start Impact Monitoring Management Command

Django management command for starting continuous impact monitoring service.
Runs real-time impact calculation monitoring in the background.

Usage:
python manage.py start_impact_monitoring
python manage.py start_impact_monitoring --interval 600
python manage.py start_impact_monitoring --background
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
import logging
import signal
import sys
import time

from core.impact import impact_service


class Command(BaseCommand):
    """Management command for starting continuous impact monitoring"""
    
    help = 'Start continuous impact monitoring service'
    
    def add_arguments(self, parser):
        """Add command line arguments"""
        parser.add_argument(
            '--interval',
            type=int,
            default=300,
            help='Update interval in seconds (default: 300)',
        )
        
        parser.add_argument(
            '--background',
            action='store_true',
            help='Run in background mode',
        )
        
        parser.add_argument(
            '--stop-after',
            type=int,
            help='Stop monitoring after specified number of seconds',
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output',
        )
    
    def handle(self, *args, **options):
        """Main command handler"""
        # Configure logging
        if options['verbose']:
            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
        
        logger = logging.getLogger(__name__)
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting impact monitoring at {timezone.now()}')
        )
        self.stdout.write(f'Update interval: {options["interval"]} seconds')
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            self.stdout.write('\nReceived shutdown signal, stopping monitoring...')
            impact_service.stop_continuous_monitoring()
            self.stdout.write(self.style.SUCCESS('Monitoring stopped gracefully'))
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start the monitoring service
            impact_service.start_continuous_monitoring(
                update_interval=options['interval']
            )
            
            # Monitor service status
            self._monitor_service(options)
        
        except KeyboardInterrupt:
            self.stdout.write('\nMonitoring interrupted by user')
        except Exception as e:
            logger.error(f"Critical error in monitoring command: {e}")
            raise CommandError(f'Monitoring failed: {e}')
        finally:
            # Ensure service is stopped
            impact_service.stop_continuous_monitoring()
            self.stdout.write(self.style.SUCCESS('Impact monitoring stopped'))
    
    def _monitor_service(self, options):
        """Monitor the service and provide status updates"""
        start_time = timezone.now()
        check_interval = 30  # Check status every 30 seconds
        next_status_report = time.time() + check_interval
        
        self.stdout.write(self.style.SUCCESS('Impact monitoring service started'))
        self.stdout.write('Press Ctrl+C to stop monitoring gracefully')
        
        try:
            while impact_service.is_running:
                current_time = time.time()
                
                # Periodic status reports
                if current_time >= next_status_report:
                    self._print_status_report()
                    next_status_report = current_time + check_interval
                
                # Check stop condition
                if options['stop_after']:
                    elapsed = (timezone.now() - start_time).total_seconds()
                    if elapsed >= options['stop_after']:
                        self.stdout.write(f'Stopping after {elapsed:.0f} seconds as requested')
                        break
                
                # Sleep briefly to avoid busy waiting
                time.sleep(1)
        
        except KeyboardInterrupt:
            # Re-raise to be handled by main handler
            raise
    
    def _print_status_report(self):
        """Print current service status"""
        try:
            # Get service performance metrics
            if impact_service.calculation_times:
                avg_calc_time = sum(impact_service.calculation_times) / len(impact_service.calculation_times)
                max_calc_time = max(impact_service.calculation_times)
            else:
                avg_calc_time = 0
                max_calc_time = 0
            
            # Get system summary
            summary = impact_service.generate_system_summary()
            
            # Print status
            self.stdout.write(f'\n[{timezone.now().strftime("%H:%M:%S")}] Service Status:')
            self.stdout.write(f'  Running: {impact_service.is_running}')
            self.stdout.write(f'  Last update: {impact_service.last_update or "Never"}')
            self.stdout.write(f'  Error count: {impact_service.error_count}')
            self.stdout.write(f'  Avg calc time: {avg_calc_time:.2f}s')
            self.stdout.write(f'  Max calc time: {max_calc_time:.2f}s')
            
            # Print system metrics
            self.stdout.write(f'  Total stopes: {summary.total_stopes}')
            self.stdout.write(f'  Critical stopes: {summary.critical_stopes}')
            self.stdout.write(f'  High risk stopes: {summary.high_risk_stopes}')
            self.stdout.write(f'  Average impact: {summary.average_impact:.3f}')
            self.stdout.write(f'  System trend: {summary.trending_direction}')
            
            # Highlight critical conditions
            if summary.critical_stopes > 0:
                self.stdout.write(
                    self.style.WARNING(f'  ⚠️  WARNING: {summary.critical_stopes} critical stopes detected!')
                )
            
            if impact_service.error_count > 10:
                self.stdout.write(
                    self.style.ERROR(f'  ❌ HIGH ERROR COUNT: {impact_service.error_count} errors')
                )
        
        except Exception as e:
            self.stdout.write(f'Error getting status: {e}')


# ===== START IMPACT MONITORING COMMAND COMPLETE =====
