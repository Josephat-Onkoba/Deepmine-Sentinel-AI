"""
Simplified tests for Mathematical Impact Calculator

Basic tests to verify core functionality without complex setup.
"""

from django.test import TestCase
from django.utils import timezone
from datetime import datetime, timedelta
import math

from core.models import Stope, OperationalEvent, ImpactFactor
from core.impact import MathematicalImpactCalculator, SpatialCoordinate


class SimplifiedImpactCalculatorTest(TestCase):
    """Simplified tests for the impact calculator"""
    
    def setUp(self):
        """Set up basic test data"""
        self.calculator = MathematicalImpactCalculator(cache_calculations=False)
        
        # Create test stope with all required fields
        self.stope = Stope.objects.create(
            stope_name="TEST_STOPE_001",
            depth=150.0,
            hr=25.0,
            rqd=75,
            rock_type='granite',
            dip=45.0,
            direction='N',
            undercut_width=8.0,
            mining_method='cut_fill',
            is_active=True
        )
        
        # Create test impact factor
        self.impact_factor = ImpactFactor.objects.create(
            event_category='blasting',
            severity_level='high',
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
            severity=0.7,  # High severity (maps to 'high' string)
            timestamp=timezone.now() - timedelta(hours=2),
            duration_hours=0.5,
            description="Test blasting event"
        )
    
    def test_base_impact_calculation(self):
        """Test base impact calculation from event factors"""
        base_impact = self.calculator._calculate_base_impact(self.event)
        # For blasting event with severity 0.7 ('high'), base impact = 5.0 * 3.0 = 15.0
        self.assertEqual(base_impact, 15.0)
    
    def test_severity_mapping(self):
        """Test severity float to string mapping"""
        test_cases = [
            (0.1, 'minimal'),
            (0.3, 'low'),
            (0.5, 'moderate'),
            (0.7, 'high'),
            (0.9, 'severe'),
            (1.0, 'critical')
        ]
        
        for severity_float, expected_string in test_cases:
            result = self.calculator._get_severity_level_string(severity_float)
            self.assertEqual(result, expected_string)
    
    def test_fallback_impact_calculation(self):
        """Test fallback when no impact factor exists"""
        # Create event with unknown type/severity combination
        unknown_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='other',
            severity=0.4,  # Low severity
            timestamp=timezone.now(),
            description="Unknown event test"
        )
        
        base_impact = self.calculator._calculate_base_impact(unknown_event)
        self.assertEqual(base_impact, 2.0)  # 0.4 * 5.0 = 2.0
    
    def test_proximity_factor_calculation(self):
        """Test proximity-based impact calculation"""
        proximity_factor, distance = self.calculator._calculate_proximity_factor(
            self.event, self.stope
        )
        
        # Should have some proximity factor - can be > 1.0 due to duration multiplier
        self.assertGreaterEqual(proximity_factor, 0.0)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_temporal_decay_calculation(self):
        """Test time-based decay calculation"""
        calculation_time = timezone.now()
        
        # Test recent event (should have high temporal factor)
        recent_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity=0.7,
            timestamp=calculation_time - timedelta(minutes=5),
            description="Recent event test"
        )
        
        temporal_factor = self.calculator._calculate_temporal_decay(
            recent_event, calculation_time
        )
        
        # Should be close to 1.0 for recent events
        self.assertGreater(temporal_factor, 0.9)
    
    def test_duration_factor_calculation(self):
        """Test duration-based impact amplification"""
        # Test short event
        short_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity=0.7,
            timestamp=timezone.now(),
            duration_hours=0.1,  # 6 minutes
            description="Short event test"
        )
        
        short_factor = self.calculator._calculate_duration_factor(short_event)
        
        # Test longer event  
        long_event = OperationalEvent.objects.create(
            stope=self.stope,
            event_type='blasting',
            severity=0.7,
            timestamp=timezone.now(),
            duration_hours=4.0,  # 4 hours
            description="Long event test"
        )
        
        long_factor = self.calculator._calculate_duration_factor(long_event)
        
        # Longer events should have higher factor
        self.assertGreater(long_factor, short_factor)
    
    def test_complete_impact_calculation(self):
        """Test complete event impact calculation"""
        result = self.calculator.calculate_event_impact(
            self.event, self.stope
        )
        
        # Verify result structure
        self.assertGreaterEqual(result.base_impact, 0)
        self.assertGreaterEqual(result.proximity_factor, 0)
        self.assertGreaterEqual(result.temporal_factor, 0)
        self.assertGreaterEqual(result.duration_factor, 0)
        self.assertGreaterEqual(result.final_impact, 0)
        
        # Check metadata
        self.assertIn('event_id', result.calculation_metadata)
        self.assertEqual(result.calculation_metadata['event_id'], self.event.id)
    
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
    
    def test_cumulative_impact_calculation(self):
        """Test cumulative impact from multiple events"""
        # Create additional events
        for i in range(3):
            OperationalEvent.objects.create(
                stope=self.stope,
                event_type='blasting',
                severity=0.5,
                timestamp=timezone.now() - timedelta(hours=i*2),
                description=f"Test event {i}"
            )
        
        cumulative_impact = self.calculator.calculate_cumulative_impact(self.stope)
        self.assertGreaterEqual(cumulative_impact, 0)
    
    def test_impact_score_update(self):
        """Test stope impact score update"""
        impact_score = self.calculator.update_stope_impact_score(self.stope)
        
        self.assertEqual(impact_score.stope, self.stope)
        self.assertGreaterEqual(impact_score.current_score, 0)
        self.assertIn(impact_score.risk_level, ['stable', 'elevated', 'high_risk', 'critical'])


class SpatialCoordinateBasicTest(TestCase):
    """Basic tests for spatial coordinate calculations"""
    
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


# ===== SIMPLIFIED IMPACT CALCULATOR TESTS COMPLETE =====
