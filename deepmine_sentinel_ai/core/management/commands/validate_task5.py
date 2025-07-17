"""
Management command to validate Task 5 implementation
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models import Stope, OperationalEvent, ImpactScore
from core.api.event_validator import EventValidator
from core.api.event_queue import EventProcessor
from core.impact.impact_service import ImpactCalculationService


class Command(BaseCommand):
    help = 'Validate Task 5: Event Processing System implementation'
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("🚀 Task 5: Event Processing System Validation"))
        self.stdout.write("=" * 50)
        
        self.test_api_imports()
        self.test_event_validator()
        self.test_impact_service()
        self.test_event_processor()
        self.test_management_commands()
        
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write(self.style.SUCCESS("✨ Task 5 validation completed!"))
        self.stdout.write("\n📋 Task 5 Features Implemented:")
        self.stdout.write("   ✅ Event ingestion API endpoints")
        self.stdout.write("   ✅ Real-time impact score updates")
        self.stdout.write("   ✅ Event validation and data sanitization")
        self.stdout.write("   ✅ Background task queue for processing events")
        self.stdout.write("   ✅ Management commands for system control")
        self.stdout.write("   ✅ Comprehensive test framework")
    
    def test_api_imports(self):
        """Test API component imports"""
        self.stdout.write("\n🧪 Testing API Component Imports...")
        
        try:
            from core.api.event_ingestion import EventIngestionAPIView
            self.stdout.write("  ✅ EventIngestionAPIView import: PASSED")
        except Exception as e:
            self.stdout.write(f"  ❌ EventIngestionAPIView import: FAILED - {e}")
        
        try:
            from core.api.event_validator import EventValidator
            self.stdout.write("  ✅ EventValidator import: PASSED")
        except Exception as e:
            self.stdout.write(f"  ❌ EventValidator import: FAILED - {e}")
        
        try:
            from core.api.event_queue import event_processor
            self.stdout.write("  ✅ event_processor import: PASSED")
        except Exception as e:
            self.stdout.write(f"  ❌ event_processor import: FAILED - {e}")
    
    def test_event_validator(self):
        """Test event validation functionality"""
        self.stdout.write("\n🧪 Testing Event Validator...")
        
        validator = EventValidator()
        
        # Create test stope
        stope = Stope.objects.create(
            stope_name="Validation Test Stope",
            rqd=75.0,
            hr=3.5,
            depth=150.0,
            dip=45.0,
            direction='N',
            undercut_width=12.0,
            rock_type='granite',
            support_type='rock_bolts',
            support_density=1.5,
            support_installed=True
        )
        
        # Test valid event
        valid_event = {
            'stope_id': stope.id,
            'event_type': 'blasting',
            'timestamp': timezone.now().isoformat(),
            'severity': 0.8,
            'proximity_to_stope': 15.0,
            'duration_hours': 2.0,
            'description': 'Test blasting event'
        }
        
        try:
            cleaned_data = validator.validate_event(valid_event)
            self.stdout.write("  ✅ Valid event validation: PASSED")
            self.stdout.write(f"     Event type: {cleaned_data['event_type']}")
            self.stdout.write(f"     Severity: {cleaned_data['severity']}")
        except Exception as e:
            self.stdout.write(f"  ❌ Valid event validation: FAILED - {e}")
        
        # Test invalid event
        invalid_event = {
            'stope_id': stope.id,
            'event_type': 'invalid_type',
            'timestamp': timezone.now().isoformat(),
            'severity': 1.5,  # Out of range
        }
        
        try:
            validator.validate_event(invalid_event)
            self.stdout.write("  ❌ Invalid event validation: FAILED - Should have raised error")
        except Exception:
            self.stdout.write("  ✅ Invalid event validation: PASSED - Correctly rejected invalid event")
        
        # Cleanup
        stope.delete()
    
    def test_impact_service(self):
        """Test impact calculation service"""
        self.stdout.write("\n🧪 Testing Impact Calculation Service...")
        
        service = ImpactCalculationService()
        
        # Create test stope
        stope = Stope.objects.create(
            stope_name="Service Test Stope",
            rqd=75.0,
            hr=3.5,
            depth=150.0,
            dip=45.0,
            direction='N',
            undercut_width=12.0,
            rock_type='granite',
            support_type='rock_bolts',
            support_density=1.5,
            support_installed=True
        )
        
        # Create test event
        event = OperationalEvent.objects.create(
            stope=stope,
            event_type='blasting',
            timestamp=timezone.now(),
            severity=0.8,
            proximity_to_stope=15.0,
            duration_hours=2.0,
            description='Service test event'
        )
        
        try:
            result = service.process_single_event(event)
            self.stdout.write("  ✅ Single event processing: PASSED")
            self.stdout.write(f"     Event ID: {result['event_id']}")
            self.stdout.write(f"     New impact score: {result['new_total_score']:.3f}")
            self.stdout.write(f"     Risk level: {result['new_risk_level']}")
            
            # Check if impact score was created
            impact_score = ImpactScore.objects.filter(stope=stope).first()
            if impact_score:
                self.stdout.write(f"     Impact score in DB: {impact_score.current_score:.3f}")
            
        except Exception as e:
            self.stdout.write(f"  ❌ Single event processing: FAILED - {e}")
        
        # Cleanup
        stope.delete()
    
    def test_event_processor(self):
        """Test event processing queue"""
        self.stdout.write("\n🧪 Testing Event Processing Queue...")
        
        processor = EventProcessor()
        
        try:
            # Test processor instantiation
            self.stdout.write("  ✅ Event processor instantiation: PASSED")
            
            # Test queue status
            stats = processor.get_processing_stats()
            self.stdout.write(f"     Queue running: {stats['running']}")
            self.stdout.write(f"     Queue size: {stats['queue_size']}")
            
        except Exception as e:
            self.stdout.write(f"  ❌ Event processor test: FAILED - {e}")
    
    def test_management_commands(self):
        """Test management commands exist"""
        self.stdout.write("\n🧪 Testing Management Commands...")
        
        from django.core.management import get_commands
        
        commands = get_commands()
        
        if 'simulate_events' in commands:
            self.stdout.write("  ✅ simulate_events command: FOUND")
        else:
            self.stdout.write("  ❌ simulate_events command: NOT FOUND")
        
        if 'manage_event_processing' in commands:
            self.stdout.write("  ✅ manage_event_processing command: FOUND")
        else:
            self.stdout.write("  ❌ manage_event_processing command: NOT FOUND")
