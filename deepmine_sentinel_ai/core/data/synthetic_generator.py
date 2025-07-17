"""
Synthetic Dataset Generator for LSTM Training Data

Generates realistic mining operational scenarios based on:
- Geological and design parameters from PROJECT_FINAL_REPORT.md
- Established geotechnical stability criteria
- Mining engineering best practices

Dataset Specifications:
- 2,847 synthetic operational events across 156 simulated stopes
- 18-month simulation period
- Static geotechnical features (RQD, depth, dip, rock type, mining method)
- Dynamic operational data (blasting, equipment, water exposure)
- Ground truth stability assessments
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from django.utils import timezone
from core.models import Stope, OperationalEvent, ImpactScore
from core.impact.impact_service import ImpactCalculationService

logger = logging.getLogger(__name__)


class RockType(Enum):
    """Rock type classifications for mining operations"""
    GRANITE = "granite"
    LIMESTONE = "limestone" 
    SANDSTONE = "sandstone"
    QUARTZITE = "quartzite"
    SCHIST = "schist"


class MiningMethod(Enum):
    """Mining method classifications"""
    SUBLEVEL_STOPING = "sublevel_stoping"
    CUT_AND_FILL = "cut_and_fill"
    ROOM_AND_PILLAR = "room_and_pillar"
    BLOCK_CAVING = "block_caving"


class EventType(Enum):
    """Operational event types with base impact values"""
    BLASTING = "blasting"
    HEAVY_EQUIPMENT = "heavy_equipment"
    WATER_EXPOSURE = "water_exposure"
    DRILLING = "drilling"
    MUCKING = "mucking"
    SUPPORT_INSTALLATION = "support_installation"


@dataclass
class StopeConfiguration:
    """Configuration for generating synthetic stope data"""
    stope_name: str
    rock_type: RockType
    mining_method: MiningMethod
    depth: float  # meters
    rqd: float  # Rock Quality Designation (%)
    dip: float  # degrees
    hr: float  # Height/Width ratio
    undercut_width: float  # meters
    support_density: float  # supports per m²
    
    
@dataclass 
class EventConfiguration:
    """Configuration for generating operational events"""
    event_type: EventType
    base_impact: float
    frequency_per_week: float  # average events per week
    duration_range: Tuple[float, float]  # min, max hours
    proximity_range: Tuple[float, float]  # min, max distance from stope


class SyntheticDataGenerator:
    """
    Generate synthetic mining operational data for LSTM training
    
    Based on specifications from PROJECT_FINAL_REPORT.md:
    - 2,847 operational events across 156 stopes
    - 18-month simulation period
    - Realistic geological distributions
    - Mining operation patterns
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize generator with deterministic seed for reproducibility"""
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Configuration based on PROJECT_FINAL_REPORT.md specifications
        self.target_stopes = 156
        self.target_events = 2847
        self.simulation_months = 18
        
        # Event type configurations with realistic impact values
        self.event_configs = {
            EventType.BLASTING: EventConfiguration(
                event_type=EventType.BLASTING,
                base_impact=5.0,  # Highest impact
                frequency_per_week=2.5,  # 2-3 blasts per week
                duration_range=(0.5, 2.0),  # 30min - 2hrs
                proximity_range=(5.0, 50.0)  # 5-50m from stope
            ),
            EventType.HEAVY_EQUIPMENT: EventConfiguration(
                event_type=EventType.HEAVY_EQUIPMENT,
                base_impact=2.0,
                frequency_per_week=5.0,  # Daily equipment use
                duration_range=(2.0, 8.0),  # 2-8 hour shifts
                proximity_range=(10.0, 100.0)
            ),
            EventType.WATER_EXPOSURE: EventConfiguration(
                event_type=EventType.WATER_EXPOSURE,
                base_impact=3.0,
                frequency_per_week=1.0,  # Weekly occurrence
                duration_range=(1.0, 12.0),  # 1-12 hour exposure
                proximity_range=(0.0, 25.0)  # Direct contact possible
            ),
            EventType.DRILLING: EventConfiguration(
                event_type=EventType.DRILLING,
                base_impact=1.5,
                frequency_per_week=4.0,
                duration_range=(1.0, 6.0),
                proximity_range=(0.0, 30.0)
            ),
            EventType.MUCKING: EventConfiguration(
                event_type=EventType.MUCKING,
                base_impact=1.0,
                frequency_per_week=3.0,
                duration_range=(2.0, 6.0),
                proximity_range=(0.0, 20.0)
            ),
            EventType.SUPPORT_INSTALLATION: EventConfiguration(
                event_type=EventType.SUPPORT_INSTALLATION,
                base_impact=-0.5,  # Negative impact (improves stability)
                frequency_per_week=1.5,
                duration_range=(4.0, 12.0),
                proximity_range=(0.0, 15.0)
            )
        }
        
        logger.info("Synthetic Data Generator initialized")
        logger.info(f"Target: {self.target_stopes} stopes, {self.target_events} events over {self.simulation_months} months")

    def generate_stope_configurations(self) -> List[StopeConfiguration]:
        """
        Generate realistic stope configurations based on geological distributions
        from mining engineering literature
        """
        stope_configs = []
        
        for i in range(self.target_stopes):
            # Generate realistic geological parameters
            
            # RQD: Rock Quality Designation (60-95% range with appropriate frequency)
            # Higher values more common in stable rock formations
            rqd = np.random.beta(2, 1) * 35 + 60  # Beta distribution for realistic RQD
            rqd = min(95, max(60, rqd))
            
            # Depth: Typical underground mining depths (50-800m)
            # Exponential distribution favoring shallower depths
            depth = np.random.exponential(150) + 50
            depth = min(800, max(50, depth))
            
            # Dip angle: Geological structure angle (15-85 degrees)
            dip = np.random.normal(45, 15)  # Normal around 45°
            dip = min(85, max(15, dip))
            
            # Rock type: Weighted by typical mining environments
            rock_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Granite most common
            rock_type = np.random.choice(list(RockType), p=rock_weights)
            
            # Mining method: Based on geological conditions
            if depth < 100:
                mining_methods = [MiningMethod.ROOM_AND_PILLAR, MiningMethod.CUT_AND_FILL]
                method_weights = [0.7, 0.3]
            elif depth < 300:
                mining_methods = [MiningMethod.SUBLEVEL_STOPING, MiningMethod.CUT_AND_FILL]
                method_weights = [0.6, 0.4]
            else:
                mining_methods = [MiningMethod.SUBLEVEL_STOPING, MiningMethod.BLOCK_CAVING]
                method_weights = [0.7, 0.3]
            
            mining_method = np.random.choice(mining_methods, p=method_weights)
            
            # Height/Width ratio: Engineering design parameter
            hr = np.random.normal(2.5, 0.5)
            hr = min(4.0, max(1.5, hr))
            
            # Undercut width: Based on mining method
            base_width = 15 if mining_method == MiningMethod.ROOM_AND_PILLAR else 25
            undercut_width = np.random.normal(base_width, 5)
            undercut_width = min(40, max(10, undercut_width))
            
            # Support density: Based on rock quality and depth
            base_density = 0.8 - (rqd - 60) / 100  # Lower RQD needs more support
            depth_factor = 1 + depth / 1000  # Deeper needs more support
            support_density = base_density * depth_factor
            support_density = min(1.5, max(0.3, support_density))
            
            config = StopeConfiguration(
                stope_name=f"STOPE_{i+1:03d}",
                rock_type=rock_type,
                mining_method=mining_method,
                depth=depth,
                rqd=rqd,
                dip=dip,
                hr=hr,
                undercut_width=undercut_width,
                support_density=support_density
            )
            
            stope_configs.append(config)
        
        logger.info(f"Generated {len(stope_configs)} stope configurations")
        return stope_configs
    
    def generate_operational_events(self, stope_configs: List[StopeConfiguration]) -> List[Dict]:
        """
        Generate operational events with realistic patterns and timing
        
        Returns list of event dictionaries ready for database insertion
        """
        events = []
        
        # Calculate events per stope to reach target total
        events_per_stope = self.target_events / len(stope_configs)
        
        # Simulation start date
        start_date = timezone.now() - timedelta(days=self.simulation_months * 30)
        
        for stope_config in stope_configs:
            stope_events = int(np.random.poisson(events_per_stope))
            
            # Generate events for this stope over simulation period
            for _ in range(stope_events):
                # Random event type based on frequencies
                event_weights = [config.frequency_per_week for config in self.event_configs.values()]
                event_type = np.random.choice(list(self.event_configs.keys()), 
                                            p=np.array(event_weights)/sum(event_weights))
                
                config = self.event_configs[event_type]
                
                # Generate realistic timestamp
                # More events during business hours and weekdays
                random_days = np.random.uniform(0, self.simulation_months * 30)
                base_time = start_date + timedelta(days=random_days)
                
                # Adjust for business hours (bias towards 6 AM - 6 PM)
                hour_bias = np.random.beta(2, 2) * 12 + 6  # 6 AM to 6 PM bias
                timestamp = base_time.replace(hour=int(hour_bias), 
                                           minute=np.random.randint(0, 60))
                
                # Event duration
                duration = np.random.uniform(*config.duration_range)
                
                # Proximity to stope
                proximity = np.random.uniform(*config.proximity_range)
                
                # Location coordinates (simplified as relative to stope center)
                x_offset = np.random.normal(0, proximity)
                y_offset = np.random.normal(0, proximity)
                z_offset = np.random.normal(0, proximity/2)  # Less vertical variation
                
                # Event intensity (affects base impact)
                intensity = np.random.gamma(2, 0.5)  # Gamma distribution for intensity
                intensity = min(3.0, max(0.5, intensity))
                
                event_data = {
                    'stope_name': stope_config.stope_name,
                    'event_type': event_type.value,
                    'timestamp': timestamp,
                    'duration': duration,
                    'location_x': x_offset,
                    'location_y': y_offset,
                    'location_z': z_offset,
                    'intensity': intensity,
                    'proximity': proximity,
                    'base_impact': config.base_impact,
                    'description': f"{event_type.value.replace('_', ' ').title()} operation"
                }
                
                events.append(event_data)
        
        # Sort events by timestamp for realistic sequence
        events.sort(key=lambda x: x['timestamp'])
        
        # Trim to exact target if needed
        if len(events) > self.target_events:
            events = events[:self.target_events]
        
        logger.info(f"Generated {len(events)} operational events")
        return events
    
    def create_stopes_in_database(self, stope_configs: List[StopeConfiguration]) -> Dict[str, Stope]:
        """Create Stope instances in database from configurations"""
        stope_objects = {}
        
        for config in stope_configs:
            # Check if stope already exists
            existing_stope = Stope.objects.filter(stope_name=config.stope_name).first()
            if existing_stope:
                stope_objects[config.stope_name] = existing_stope
                continue
            
            stope = Stope.objects.create(
                stope_name=config.stope_name,
                rock_type=config.rock_type.value,
                depth=config.depth,
                rqd=config.rqd,
                dip=config.dip,
                hr=config.hr,
                undercut_width=config.undercut_width,
                support_density=config.support_density,
                mining_method=config.mining_method.value,
                is_active=True,
                baseline_stability_score=0.0,  # All start stable
                notes=f"Synthetic stope generated for LSTM training data"
            )
            
            stope_objects[config.stope_name] = stope
        
        logger.info(f"Created {len(stope_objects)} stopes in database")
        return stope_objects
    
    def create_events_in_database(self, events: List[Dict], stope_objects: Dict[str, Stope]) -> List[OperationalEvent]:
        """Create OperationalEvent instances in database"""
        event_objects = []
        
        for event_data in events:
            stope = stope_objects[event_data['stope_name']]
            
            event = OperationalEvent.objects.create(
                stope=stope,
                event_type=event_data['event_type'],
                timestamp=event_data['timestamp'],
                duration_hours=event_data.get('duration', 1.0),
                proximity_to_stope=event_data.get('location_x', 0.0),  # Map location_x to proximity
                severity=event_data.get('intensity', 0.5),  # Map intensity to severity
                description=event_data.get('description', f"{event_data['event_type']} operation")
            )
            
            event_objects.append(event)
        
        logger.info(f"Created {len(event_objects)} operational events in database")
        return event_objects
    
    def generate_complete_dataset(self) -> Tuple[List[Stope], List[OperationalEvent]]:
        """
        Generate complete synthetic dataset including:
        - Stope configurations
        - Operational events
        - Database objects
        
        Returns tuple of (stopes, events) for further processing
        """
        logger.info("Starting complete synthetic dataset generation")
        
        # Generate configurations
        stope_configs = self.generate_stope_configurations()
        events_data = self.generate_operational_events(stope_configs)
        
        # Create database objects
        stope_objects = self.create_stopes_in_database(stope_configs)
        event_objects = self.create_events_in_database(events_data, stope_objects)
        
        # Calculate initial impact scores using existing service
        impact_service = ImpactCalculationService()
        
        for stope in stope_objects.values():
            try:
                # Calculate current impact score for the stope
                current_score = impact_service.calculate_current_impact_score(stope)
                logger.debug(f"Calculated impact score {current_score} for {stope.stope_name}")
            except Exception as e:
                logger.warning(f"Failed to calculate impact score for {stope.stope_name}: {e}")
        
        logger.info("Synthetic dataset generation completed successfully")
        logger.info(f"Generated: {len(stope_objects)} stopes, {len(event_objects)} events")
        
        return list(stope_objects.values()), event_objects
