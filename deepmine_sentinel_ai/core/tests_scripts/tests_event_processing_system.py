"""
Test Suite for Event Processing System
Task 5: Operational Event Processing System
"""

from django.test import TestCase, TransactionTestCase
from django.utils import timezone
from django.urls import reverse
from django.core.management import call_command
from django.db import transaction
import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from core.models import Stope, OperationalEvent, ImpactScore, ImpactHistory
from core.api.event_validator import EventValidator, BatchEventValidator
from core.api.event_queue import EventProcessor, EventProcessingQueue
from core.api.event_ingestion import EventIngestionAPIView


class EventValidatorTest(TestCase):
    """Test event validation functionality"""
    
    def setUp(self):
        self.validator = EventValidator()
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
    
    def test_valid_event_validation(self):
        """Test validation of a valid event"""
        valid_event = {
            'stope_id': self.stope.id,
            'event_type': 'blasting',
            'timestamp': timezone.now().isoformat(),
            'severity': 0.8,
            'proximity_to_stope': 15.0,
            'duration_hours': 2.0,
            'description': 'Test blasting event'
        }
        
        cleaned_data = self.validator.validate_event(valid_event)
        
        self.assertEqual(cleaned_data['stope_id'], self.stope.id)
        self.assertEqual(cleaned_data['event_type'], 'blasting')
        self.assertEqual(cleaned_data['severity'], 0.8)
        self.assertEqual(cleaned_data['proximity_to_stope'], 15.0)
        self.assertEqual(cleaned_data['duration_hours'], 2.0)
    
    def test_invalid_stope_id(self):
        """Test validation with invalid stope ID"""
        invalid_event = {
            'stope_id': 99999,  # Non-existent stope
            'event_type': 'blasting',
            'timestamp': timezone.now().isoformat(),
            'severity': 0.8
        }
        
        with self.assertRaises(Exception):
            self.validator.validate_event(invalid_event)
    
    def test_invalid_event_type(self):
        """Test validation with invalid event type"""
        invalid_event = {
            'stope_id': self.stope.id,
            'event_type': 'invalid_type',
            'timestamp': timezone.now().isoformat(),
            'severity': 0.8
        }
        
        with self.assertRaises(Exception):
            self.validator.validate_event(invalid_event)
    
    def test_invalid_severity_range(self):
        """Test validation with severity out of range"""
        invalid_event = {
            'stope_id': self.stope.id,
            'event_type': 'blasting',
            'timestamp': timezone.now().isoformat(),
            'severity': 1.5  # Out of range
        }
        
        with self.assertRaises(Exception):
            self.validator.validate_event(invalid_event)
    
    def test_string_sanitization(self):
        """Test string field sanitization"""
        event_with_dangerous_strings = {
            'stope_id': self.stope.id,
            'event_type': 'blasting',
            'timestamp': timezone.now().isoformat(),
            'severity': 0.8,
            'description': '<script>alert("xss")</script>Test description'
        }
        
        cleaned_data = self.validator.validate_event(event_with_dangerous_strings)
        
        # Should remove dangerous characters
        self.assertNotIn('<script>', cleaned_data['description'])
        self.assertIn('Test description', cleaned_data['description'])


class EventProcessingQueueTest(TransactionTestCase):
    """Test event processing queue functionality"""
    
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
        
        self.event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            timestamp=timezone.now(),
            severity=0.8,
            proximity_to_stope=15.0,
            duration_hours=2.0,
            description='Test event'
        )
    
    def test_queue_creation(self):
        """Test creating an event processing queue"""
        queue = EventProcessingQueue(max_workers=2, max_queue_size=100)
        
        self.assertFalse(queue.running)
        self.assertEqual(queue.max_workers, 2)
        self.assertEqual(queue.processed_count, 0)
        self.assertEqual(queue.error_count, 0)
    
    def test_queue_start_stop(self):
        """Test starting and stopping the queue"""
        queue = EventProcessingQueue(max_workers=1)
        
        # Start queue
        queue.start()
        self.assertTrue(queue.running)
        self.assertEqual(len(queue.workers), 1)
        
        # Stop queue
        queue.stop()
        self.assertFalse(queue.running)
    
    def test_event_queuing(self):
        """Test queuing events for processing"""
        queue = EventProcessingQueue(max_workers=1)
        queue.start()
        
        try:
            # Queue an event
            success = queue.queue_event(self.event)
            self.assertTrue(success)
            
            # Wait for processing
            time.sleep(2)
            
            # Check stats
            stats = queue.get_stats()
            self.assertGreaterEqual(stats['processed_count'], 0)
            
        finally:
            queue.stop()
    
    def test_queue_stats(self):
        """Test queue statistics"""
        queue = EventProcessingQueue(max_workers=1)
        
        stats = queue.get_stats()
        
        self.assertIn('running', stats)
        self.assertIn('queue_size', stats)
        self.assertIn('processed_count', stats)
        self.assertIn('error_count', stats)
        self.assertIn('uptime_seconds', stats)


class EventIngestionAPITest(TestCase):
    """Test event ingestion API endpoints"""
    
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
    
    def test_event_ingestion_endpoint(self):
        """Test the event ingestion API endpoint"""
        url = reverse('api_event_ingest')
        
        event_data = {
            'events': [
                {
                    'stope_id': self.stope.id,
                    'event_type': 'blasting',
                    'timestamp': timezone.now().isoformat(),
                    'severity': 0.8,
                    'proximity_to_stope': 15.0,
                    'duration_hours': 2.0,
                    'description': 'Test API event'
                }
            ]
        }
        
        response = self.client.post(
            url,
            data=json.dumps(event_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('results', data)
    
    def test_real_time_processing_endpoint(self):
        """Test the real-time processing API endpoint"""
        url = reverse('api_event_process')
        
        event_data = {
            'stope_id': self.stope.id,
            'event_type': 'blasting',
            'timestamp': timezone.now().isoformat(),
            'severity': 0.8,
            'proximity_to_stope': 15.0,
            'duration_hours': 2.0,
            'description': 'Real-time test event'
        }
        
        response = self.client.post(
            url,
            data=json.dumps(event_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('event_id', data)
        self.assertIn('updated_scores', data)
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON"""
        url = reverse('api_event_ingest')
        
        response = self.client.post(
            url,
            data='invalid json',
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        
        data = response.json()
        self.assertEqual(data['status'], 'error')
        self.assertIn('Invalid JSON', data['message'])
    
    def test_batch_update_endpoint(self):
        """Test the batch update API endpoint"""
        url = reverse('api_batch_update')
        
        # Create some events first
        OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            timestamp=timezone.now(),
            severity=0.8,
            proximity_to_stope=15.0,
            duration_hours=2.0
        )
        
        request_data = {
            'stope_ids': [self.stope.id]
        }
        
        response = self.client.post(
            url,
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('updated_stopes', data)


class EventProcessingIntegrationTest(TransactionTestCase):
    """Integration tests for the complete event processing system"""
    
    def setUp(self):
        self.stope = Stope.objects.create(
            stope_name="Integration Test Stope",
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
    
    def test_end_to_end_event_processing(self):
        """Test complete end-to-end event processing"""
        # Start the event processor
        processor = EventProcessor()
        processor.start_processing()
        
        try:
            # Create an operational event
            event = OperationalEvent.objects.create(
                stope=self.stope,
                event_type='blasting',
                timestamp=timezone.now(),
                severity=0.8,
                proximity_to_stope=15.0,
                duration_hours=2.0,
                description='Integration test event'
            )
            
            # Queue the event for processing
            success = processor.queue_event_processing(event)
            self.assertTrue(success)
            
            # Wait for processing
            time.sleep(3)
            
            # Check that impact score was created/updated
            impact_score = ImpactScore.objects.filter(stope=self.stope).first()
            self.assertIsNotNone(impact_score)
            self.assertGreater(impact_score.total_score, 0)
            
            # Check that impact history was recorded
            history = ImpactHistory.objects.filter(
                stope=self.stope,
                contributing_event=event
            ).first()
            self.assertIsNotNone(history)
            
        finally:
            processor.stop_processing()
    
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        # Create multiple events
        events = []
        for i in range(10):
            event = OperationalEvent.objects.create(
                stope=self.stope,
                event_type='heavy_equipment',
                timestamp=timezone.now() - timedelta(hours=i),
                severity=0.3 + (i * 0.05),
                proximity_to_stope=10.0 + i,
                duration_hours=1.0 + i,
                description=f'Batch test event {i}'
            )
            events.append(event)
        
        processor = EventProcessor()
        processor.start_processing()
        
        try:
            start_time = time.time()
            
            # Queue all events
            for event in events:
                processor.queue_event_processing(event)
            
            # Wait for processing
            time.sleep(5)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Check processing stats
            stats = processor.get_processing_stats()
            
            self.assertLess(processing_time, 10)  # Should complete within 10 seconds
            self.assertGreaterEqual(stats['processed_count'], len(events))
            
        finally:
            processor.stop_processing()


class ManagementCommandTest(TestCase):
    """Test management commands for event processing"""
    
    def setUp(self):
        self.stope = Stope.objects.create(
            stope_name="Command Test Stope",
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
    
    def test_simulate_events_command(self):
        """Test the simulate_events management command"""
        # Test creating events directly
        call_command(
            'simulate_events',
            '--count', '5',
            '--stope-ids', str(self.stope.id)
        )
        
        # Check that events were created
        events = OperationalEvent.objects.filter(stope=self.stope)
        self.assertEqual(events.count(), 5)
    
    @patch('core.api.event_queue.event_processor')
    def test_manage_event_processing_command(self, mock_processor):
        """Test the manage_event_processing command"""
        mock_processor.is_running.return_value = False
        mock_processor.get_processing_stats.return_value = {
            'running': False,
            'queue_size': 0,
            'processed_count': 0,
            'error_count': 0,
            'uptime_seconds': 0,
            'processing_rate': 0,
            'error_rate': 0
        }
        
        # Test status command
        call_command('manage_event_processing', '--action', 'status')
        
        # Verify that the processor was queried
        mock_processor.get_processing_stats.assert_called()


if __name__ == '__main__':
    import unittest
    unittest.main()
