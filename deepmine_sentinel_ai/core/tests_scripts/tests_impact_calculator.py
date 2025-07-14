"""
Tests for Mathematical Impact Calculator

Comprehensive tests for the impact calculation engine including:
- Base impact calculations
- Proximity factor calculations  
- Temporal decay mechanisms
- Cumulative impact calculations
- Edge cases and error handling
"""

from django.test import TestCase
from django.utils import timezone
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import math

from core.models import (
    Stope, OperationalEvent, ImpactFactor, ImpactScore, 
    ImpactHistory, MonitoringData
)
from core.impact.impact_calculator import (
    MathematicalImpactCalculator, SpatialCoordinate, 
    ImpactCalculationResult
)


class SpatialCoordinateTest(TestCase):
    """Test spatial coordinate calculations"""
    
    def test_distance_calculation(self):
        """Test 3D distance calculation"""
        coord1 = SpatialCoordinate(0, 0, 0)
        coord2 = SpatialCoordinate(3, 4, 0)
        
        distance = coord1.distance_to(coord2)
        self.assertAlmostEqual(distance, 5.0, places=6)
    
    def test_distance_3d(self):
        """Test 3D distance with z-component"""
        coord1 = SpatialCoordinate(0, 0, 0)
        coord2 = SpatialCoordinate(1, 1, 1)
        
        distance = coord1.distance_to(coord2)
        expected = math.sqrt(3)
        self.assertAlmostEqual(distance, expected, places=6)


class MathematicalImpactCalculatorTest(TestCase):
    """Test the mathematical impact calculator"""
    
    def setUp(self):
        """Set up test data"""
        self.calculator = MathematicalImpactCalculator(cache_calculations=False)
        
        # Create test stope
        self.stope = Stope.objects.create(
            stope_name="TEST_STOPE_001",
            depth=150.0,
            hr=25.0,
            rqd=75,  # Rock Quality Designation
            rock_type='granite',
            dip=45.0,  # Dip angle
            direction='N',  # Direction
            undercut_width=8.0,  # Undercut width
            mining_method='cut_fill',  # Mining method
            is_active=True
        )
        
        # Create test impact factors
        self.impact_factor = ImpactFactor.objects.create(
            event_category='blasting',
            severity_level='high',  # Use string instead of 'high'
            base_impact_weight=5.0,
            proximity_decay_rate=0.1,
            temporal_decay_rate=0.05,
            duration_multiplier=1.5,
            is_active=True
        )
        
        # Create test operational event
        self.event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity=0.7,  # Use float value instead of string
            timestamp=timezone.now() - timedelta(hours=2),
            duration_hours=0.5,  # Use duration_hours instead of duration
            description="Test blasting event"
        )
    
    def test_base_impact_calculation(self):
        """Test base impact calculation from event factors"""
        base_impact = self.calculator._calculate_base_impact(self.event)
        self.assertEqual(base_impact, 5.0)  # Should match impact factor
    
    def test_base_impact_fallback(self):
        """Test base impact fallback when no factor exists"""
        # Create event with no matching impact factor
        unknown_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='other',  # Use valid event type
            severity=0.5,  # Use float
            timestamp=timezone.now(),
            description="Unknown event test"
        )
        
        base_impact = self.calculator._calculate_base_impact(unknown_event)
        self.assertEqual(base_impact, 2.5)  # 0.5 * 5.0 = 2.5
    
    def test_proximity_factor_calculation(self):
        """Test proximity-based impact calculation"""
        proximity_factor, distance = self.calculator._calculate_proximity_factor(
            self.event, self.stope
        )
        
        # Should have some proximity factor
        self.assertGreaterEqual(proximity_factor, 0.0)
        self.assertLessEqual(proximity_factor, 1.0)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_temporal_decay_calculation(self):
        """Test time-based decay calculation"""
        calculation_time = timezone.now()
        
        # Test immediate impact (event just happened)
        recent_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity=0.7,  # Use float
            timestamp=calculation_time - timedelta(minutes=1),
            description="Recent event test"
        )
        
        temporal_factor = self.calculator._calculate_temporal_decay(
            recent_event, calculation_time
        )
        
        # Should be close to 1.0 for recent events
        self.assertGreater(temporal_factor, 0.9)
        
        # Test old event
        old_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity=0.7,  # Use float
            timestamp=calculation_time - timedelta(days=7)
        )
        
        old_temporal_factor = self.calculator._calculate_temporal_decay(
            old_event, calculation_time
        )
        
        # Should be much lower for old events
        self.assertLess(old_temporal_factor, temporal_factor)
    
    def test_future_event_handling(self):
        """Test handling of future events"""
        future_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity=0.7,  # Use float
            timestamp=timezone.now() + timedelta(hours=1)
        )
        
        temporal_factor = self.calculator._calculate_temporal_decay(
            future_event, timezone.now()
        )
        
        # Future events should have zero impact
        self.assertEqual(temporal_factor, 0.0)
    
    def test_duration_factor_calculation(self):
        """Test duration-based impact amplification"""
        # Test short event
        short_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity=0.7,  # Use float
            timestamp=timezone.now(),
            duration_hours=0.08  # 5 minutes = 0.08 hours
        )
        
        short_factor = self.calculator._calculate_duration_factor(short_event)
        
        # Test long event
        long_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='heavy_equipment',
            severity=0.7,  # Use float
            timestamp=timezone.now(),
            duration_hours=8.0  # 8 hours
        )
        
        long_factor = self.calculator._calculate_duration_factor(long_event)
        
        # Longer events should have higher factor
        self.assertGreater(long_factor, short_factor)
    
    def test_event_impact_calculation(self):
        """Test complete event impact calculation"""
        result = self.calculator.calculate_event_impact(
            self.event, self.stope
        )
        
        self.assertIsInstance(result, ImpactCalculationResult)
        self.assertGreaterEqual(result.base_impact, 0)
        self.assertGreaterEqual(result.proximity_factor, 0)
        self.assertGreaterEqual(result.temporal_factor, 0)
        self.assertGreaterEqual(result.duration_factor, 0)
        self.assertGreaterEqual(result.final_impact, 0)
        
        # Check metadata
        self.assertIn('event_id', result.calculation_metadata)
        self.assertIn('target_stope_id', result.calculation_metadata)
        self.assertEqual(result.calculation_metadata['event_id'], self.event.id)
    
    def test_cumulative_impact_calculation(self):
        """Test cumulative impact from multiple events"""
        # Create multiple events
        events = []
        for i in range(3):
            event = OperationalEvent.objects.create(
                stope=self.stope,
                event_type='blasting',
                severity='medium',
                timestamp=timezone.now() - timedelta(hours=i*2)
            )
            events.append(event)
        
        cumulative_impact = self.calculator.calculate_cumulative_impact(
            self.stope
        )
        
        self.assertGreaterEqual(cumulative_impact, 0)
    
    def test_empty_cumulative_impact(self):
        """Test cumulative impact with no events"""
        # Delete all events
        OperationalEvent.objects.all().delete()
        
        cumulative_impact = self.calculator.calculate_cumulative_impact(
            self.stope
        )
        
        self.assertEqual(cumulative_impact, 0.0)
    
    def test_impact_score_update(self):
        """Test stope impact score update"""
        impact_score = self.calculator.update_stope_impact_score(self.stope)
        
        self.assertIsInstance(impact_score, ImpactScore)
        self.assertEqual(impact_score.stope, self.stope)
        self.assertGreaterEqual(impact_score.current_score, 0)
        self.assertIn(impact_score.risk_level, ['stable', 'elevated', 'high_risk', 'critical'])
    
    def test_batch_update_stope_scores(self):
        """Test batch updating multiple stope scores"""
        # Create additional stopes
        stope2 = Stope.objects.create(
            stope_name="TEST_STOPE_002",
            depth=200.0,
            hr=30.0,
            rqd=80,  # Rock Quality Designation
            rock_type='granite',
            dip=50.0,  # Dip angle
            direction='E',  # Direction
            undercut_width=10.0,  # Undercut width
            mining_method='sublevel_stoping',  # Mining method
            is_active=True
        )
        
        stopes = [self.stope, stope2]
        stats = self.calculator.batch_update_stope_scores(stopes)
        
        self.assertIn('total_stopes', stats)
        self.assertIn('updated_scores', stats)
        self.assertIn('errors', stats)
        self.assertEqual(stats['total_stopes'], 2)
    
    def test_risk_level_determination(self):
        """Test risk level determination from impact scores"""
        test_cases = [
            (0.5, 'stable'),
            (2.0, 'elevated'),
            (4.0, 'high_risk'),
            (8.0, 'critical')
        ]
        
        for score, expected_risk in test_cases:
            risk_level = self.calculator._determine_risk_level(score)
            self.assertEqual(risk_level, expected_risk)
    
    def test_cache_functionality(self):
        """Test calculation caching"""
        calculator_with_cache = MathematicalImpactCalculator(cache_calculations=True)
        
        # First calculation
        result1 = calculator_with_cache.calculate_event_impact(
            self.event, self.stope
        )
        
        # Second calculation (should use cache)
        result2 = calculator_with_cache.calculate_event_impact(
            self.event, self.stope
        )
        
        # Results should be identical
        self.assertEqual(result1.final_impact, result2.final_impact)
    
    def test_error_handling(self):
        """Test error handling in calculations"""
        # Test with invalid event
        invalid_event = MagicMock()
        invalid_event.id = 999999
        invalid_event.event_type = 'invalid'
        invalid_event.severity = 'invalid'
        invalid_event.timestamp = timezone.now()
        
        # Should not raise exception, should return zero impact
        result = self.calculator.calculate_event_impact(
            invalid_event, self.stope
        )
        
        self.assertEqual(result.final_impact, 0.0)
        self.assertIn('error', result.calculation_metadata)
    
    def test_stress_concentration_calculation(self):
        """Test stress concentration effects"""
        # Test close distance
        close_factor = self.calculator._calculate_stress_concentration(10.0, self.stope)
        
        # Test far distance
        far_factor = self.calculator._calculate_stress_concentration(100.0, self.stope)
        
        # Close distances should have higher stress concentration
        self.assertGreaterEqual(close_factor, far_factor)
    
    def test_blasting_specific_models(self):
        """Test blasting-specific calculation models"""
        blasting_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity='high',
            timestamp=timezone.now()
        )
        
        # Test proximity calculation for blasting (should use inverse square law)
        proximity_factor, distance = self.calculator._calculate_proximity_factor(
            blasting_event, self.stope
        )
        
        self.assertGreaterEqual(proximity_factor, 0)
        self.assertLessEqual(proximity_factor, 1)
        
        # Test temporal decay for blasting (should use double exponential)
        temporal_factor = self.calculator._calculate_temporal_decay(
            blasting_event, timezone.now() + timedelta(hours=1)
        )
        
        self.assertGreaterEqual(temporal_factor, 0)
        self.assertLessEqual(temporal_factor, 1)
    
    def test_water_exposure_models(self):
        """Test water exposure specific models"""
        water_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='water_exposure',
            severity='medium',
            timestamp=timezone.now() - timedelta(hours=48)
        )
        
        # Water damage should have slow decay
        temporal_factor = self.calculator._calculate_temporal_decay(
            water_event, timezone.now()
        )
        
        # Should still have significant impact after 48 hours
        self.assertGreater(temporal_factor, 0.1)
    
    def test_accumulation_model(self):
        """Test impact accumulation with diminishing returns"""
        # Create mock impact data
        individual_impacts = [
            {'impact': 5.0, 'event': self.event, 'result': None},
            {'impact': 3.0, 'event': self.event, 'result': None},
            {'impact': 2.0, 'event': self.event, 'result': None}
        ]
        
        cumulative = self.calculator._apply_accumulation_model(individual_impacts)
        
        # Should be less than simple sum due to diminishing returns
        simple_sum = sum(item['impact'] for item in individual_impacts)
        self.assertLess(cumulative, simple_sum)
        self.assertGreater(cumulative, individual_impacts[0]['impact'])  # But more than largest single impact


class ImpactCalculatorIntegrationTest(TestCase):
    """Integration tests for the impact calculator with full system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.calculator = MathematicalImpactCalculator()
        
        # Create multiple stopes
        self.stopes = []
        for i in range(3):
            stope = Stope.objects.create(
                stope_name=f"INTEGRATION_STOPE_{i:03d}",
                depth=100 + i*50,
                hr=20 + i*5,
                rqd=70 + i*5,  # Rock Quality Designation
                rock_type='granite',
                dip=40 + i*5,  # Dip angle
                direction='N',  # Direction
                undercut_width=8 + i*2,  # Undercut width
                mining_method='cut_fill',  # Mining method
                is_active=True
            )
            self.stopes.append(stope)
        
        # Create impact factors for different event types
        impact_factors = [
            ('blasting', 'high', 5.0, 0.1, 0.05, 2.0),
            ('heavy_equipment', 'medium', 2.0, 0.05, 0.03, 1.5),
            ('water_exposure', 'low', 1.0, 0.02, 0.01, 1.2),
        ]
        
        for event_type, severity, base_weight, prox_decay, temp_decay, duration_mult in impact_factors:
            ImpactFactor.objects.create(
                event_category=event_type,
                severity_level=severity,
                base_impact_weight=base_weight,
                proximity_decay_rate=prox_decay,
                temporal_decay_rate=temp_decay,
                duration_multiplier=duration_mult,
                is_active=True
            )
    
    def test_realistic_mining_scenario(self):
        """Test realistic mining scenario with multiple events and stopes"""
        # Create sequence of realistic events
        base_time = timezone.now() - timedelta(days=1)
        
        events = [
            # Morning shift - blasting
            ('blasting', 'high', 0, 30),
            ('blasting', 'medium', 2, 20),
            
            # Day shift - heavy equipment
            ('heavy_equipment', 'medium', 8, 240),
            ('heavy_equipment', 'low', 10, 180),
            
            # Evening - water exposure incident
            ('water_exposure', 'high', 16, 60),
            
            # Night shift - more blasting
            ('blasting', 'medium', 20, 25),
        ]
        
        created_events = []
        for event_type, severity, hour_offset, duration_mins in events:
            event = OperationalEvent.objects.create(
                stope=self.stopes[0],  # All events affect first stope
                event_type=event_type,
                severity=severity,
                timestamp=base_time + timedelta(hours=hour_offset),
                duration=timedelta(minutes=duration_mins),
                description=f"Integration test {event_type}"
            )
            created_events.append(event)
        
        # Calculate impacts for all stopes
        results = []
        for stope in self.stopes:
            cumulative_impact = self.calculator.calculate_cumulative_impact(stope)
            impact_score = self.calculator.update_stope_impact_score(stope)
            results.append((stope, cumulative_impact, impact_score))
        
        # Verify results
        self.assertEqual(len(results), 3)
        
        # First stope (with events) should have higher impact
        first_stope_impact = results[0][1]
        other_stope_impacts = [result[1] for result in results[1:]]
        
        for other_impact in other_stope_impacts:
            self.assertGreaterEqual(first_stope_impact, other_impact)
        
        # All impact scores should be valid
        for stope, cumulative, score in results:
            self.assertGreaterEqual(cumulative, 0)
            self.assertIsInstance(score, ImpactScore)
            self.assertIn(score.risk_level, ['stable', 'elevated', 'high_risk', 'critical'])
    
    def test_performance_with_many_events(self):
        """Test calculator performance with many events"""
        import time
        
        # Create many events
        base_time = timezone.now() - timedelta(hours=168)  # 1 week ago
        
        for i in range(100):  # 100 events
            OperationalEvent.objects.create(
                stope=self.stopes[i % len(self.stopes)],
                event_type='blasting',
                severity='medium',
                timestamp=base_time + timedelta(hours=i),
                description=f"Performance test event {i}"
            )
        
        # Measure calculation time
        start_time = time.time()
        
        for stope in self.stopes:
            self.calculator.calculate_cumulative_impact(stope)
        
        calculation_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 10 seconds for 3 stopes with 100 events)
        self.assertLess(calculation_time, 10.0)
        
        print(f"Performance test: {calculation_time:.2f}s for {len(self.stopes)} stopes with 100 events")


# ===== MATHEMATICAL IMPACT CALCULATOR TESTS COMPLETE =====
