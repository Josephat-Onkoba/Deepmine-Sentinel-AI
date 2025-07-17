"""
Simplified Test for Event Processing System - Task 5
"""

from django.test import TestCase
from django.utils import timezone
from core.models import Stope
from core.api.event_validator import EventValidator


class SimpleEventProcessingTest(TestCase):
    """Simple test to validate core functionality"""
    
    def setUp(self):
        self.stope = Stope.objects.create(
            stope_name="Test Stope",
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
        
    def test_stope_creation(self):
        """Test that we can create a stope for testing"""
        self.assertEqual(self.stope.stope_name, "Test Stope")
        self.assertEqual(self.stope.rqd, 75.0)
        self.assertTrue(self.stope.support_installed)
    
    def test_event_validator_import(self):
        """Test that we can import and instantiate the validator"""
        validator = EventValidator()
        self.assertIsNotNone(validator)
        self.assertIn('blasting', validator.VALID_EVENT_TYPES)
    
    def test_basic_event_validation(self):
        """Test basic event validation functionality"""
        validator = EventValidator()
        
        valid_event = {
            'stope_id': self.stope.id,
            'event_type': 'blasting',
            'timestamp': timezone.now().isoformat(),
            'severity': 0.8,
            'proximity_to_stope': 15.0,
            'duration_hours': 2.0,
            'description': 'Test blasting event'
        }
        
        # This should not raise an exception
        try:
            cleaned_data = validator.validate_event(valid_event)
            self.assertEqual(cleaned_data['stope_id'], self.stope.id)
            self.assertEqual(cleaned_data['event_type'], 'blasting')
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Validation error: {e}")
        
        self.assertTrue(test_passed, "Event validation should succeed for valid event")
    
    def test_api_endpoints_exist(self):
        """Test that API endpoints are accessible"""
        from django.urls import reverse
        
        # Test that URL patterns exist
        try:
            ingest_url = reverse('api_event_ingest')
            process_url = reverse('api_event_process')
            batch_url = reverse('api_batch_update')
            
            self.assertTrue(ingest_url.startswith('/api/v1/'))
            self.assertTrue(process_url.startswith('/api/v1/'))
            self.assertTrue(batch_url.startswith('/api/v1/'))
            
        except Exception as e:
            self.fail(f"URL reverse failed: {e}")
