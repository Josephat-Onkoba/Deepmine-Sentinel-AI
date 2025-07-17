"""
Management command to start/manage the event processing system
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
import time
import signal
import sys
import logging

from core.api.event_queue import event_processor

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Start and manage the operational event processing system'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--action',
            type=str,
            choices=['start', 'stop', 'status', 'restart'],
            default='start',
            help='Action to perform on the event processing system'
        )
        
        parser.add_argument(
            '--monitor',
            action='store_true',
            help='Monitor processing statistics in real-time'
        )
        
        parser.add_argument(
            '--interval',
            type=int,
            default=10,
            help='Status monitoring interval in seconds (default: 10)'
        )
    
    def handle(self, *args, **options):
        action = options['action']
        
        try:
            if action == 'start':
                self._start_processing(options)
            elif action == 'stop':
                self._stop_processing()
            elif action == 'status':
                self._show_status(options)
            elif action == 'restart':
                self._restart_processing(options)
                
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nOperation interrupted by user'))
            if event_processor.is_running():
                self.stdout.write("Stopping event processor...")
                event_processor.stop_processing()
        except Exception as e:
            raise CommandError(f"Command failed: {str(e)}")
    
    def _start_processing(self, options):
        """Start the event processing system"""
        if event_processor.is_running():
            self.stdout.write(
                self.style.WARNING('Event processing system is already running')
            )
            return
        
        self.stdout.write("Starting event processing system...")
        event_processor.start_processing()
        
        self.stdout.write(
            self.style.SUCCESS('Event processing system started successfully')
        )
        
        if options['monitor']:
            self._monitor_status(options['interval'])
    
    def _stop_processing(self):
        """Stop the event processing system"""
        if not event_processor.is_running():
            self.stdout.write(
                self.style.WARNING('Event processing system is not running')
            )
            return
        
        self.stdout.write("Stopping event processing system...")
        event_processor.stop_processing()
        
        self.stdout.write(
            self.style.SUCCESS('Event processing system stopped successfully')
        )
    
    def _restart_processing(self, options):
        """Restart the event processing system"""
        self.stdout.write("Restarting event processing system...")
        
        if event_processor.is_running():
            self.stdout.write("Stopping current instance...")
            event_processor.stop_processing()
            time.sleep(2)  # Give it time to stop
        
        self.stdout.write("Starting new instance...")
        event_processor.start_processing()
        
        self.stdout.write(
            self.style.SUCCESS('Event processing system restarted successfully')
        )
        
        if options['monitor']:
            self._monitor_status(options['interval'])
    
    def _show_status(self, options):
        """Show current status of the event processing system"""
        stats = event_processor.get_processing_stats()
        
        self.stdout.write("\n" + "="*50)
        self.stdout.write("EVENT PROCESSING SYSTEM STATUS")
        self.stdout.write("="*50)
        
        # Basic status
        status_color = self.style.SUCCESS if stats['running'] else self.style.ERROR
        self.stdout.write(f"Status: {status_color('RUNNING' if stats['running'] else 'STOPPED')}")
        
        if stats['running']:
            # Detailed statistics
            self.stdout.write(f"Queue Size: {stats['queue_size']}")
            self.stdout.write(f"Processed Events: {stats['processed_count']}")
            self.stdout.write(f"Error Count: {stats['error_count']}")
            self.stdout.write(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
            self.stdout.write(f"Processing Rate: {stats['processing_rate']:.2f} events/second")
            self.stdout.write(f"Error Rate: {stats['error_rate']:.2%}")
            
            # Performance indicators
            if stats['error_rate'] > 0.1:  # More than 10% errors
                self.stdout.write(
                    self.style.WARNING("⚠️  High error rate detected")
                )
            
            if stats['queue_size'] > 100:  # Large queue
                self.stdout.write(
                    self.style.WARNING("⚠️  Large processing queue detected")
                )
            
            if stats['processing_rate'] > 0:
                self.stdout.write(
                    self.style.SUCCESS("✓ System is actively processing events")
                )
        
        self.stdout.write("="*50 + "\n")
        
        if options['monitor']:
            self._monitor_status(options['interval'])
    
    def _monitor_status(self, interval):
        """Monitor processing statistics in real-time"""
        self.stdout.write(f"\nMonitoring event processing (update every {interval}s)")
        self.stdout.write("Press Ctrl+C to stop monitoring\n")
        
        # Set up signal handler for graceful exit
        def signal_handler(sig, frame):
            self.stdout.write(self.style.WARNING('\nMonitoring stopped'))
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while True:
                stats = event_processor.get_processing_stats()
                
                # Clear previous line and show current stats
                timestamp = timezone.now().strftime('%H:%M:%S')
                status_line = (
                    f"[{timestamp}] "
                    f"Queue: {stats['queue_size']:3d} | "
                    f"Processed: {stats['processed_count']:4d} | "
                    f"Errors: {stats['error_count']:3d} | "
                    f"Rate: {stats['processing_rate']:5.2f}/s | "
                    f"Error Rate: {stats['error_rate']:5.1%}"
                )
                
                # Use carriage return to overwrite the line
                self.stdout.write(f"\r{status_line}", ending='')
                self.stdout.flush()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nMonitoring stopped'))
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def shutdown_handler(sig, frame):
            self.stdout.write(self.style.WARNING('\nReceived shutdown signal'))
            if event_processor.is_running():
                self.stdout.write("Stopping event processor...")
                event_processor.stop_processing()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)
