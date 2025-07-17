"""
Management command to simulate operational events for testing
Task 5: Event Processing System Testing
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
import random
import time
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict

from core.models import Stope, OperationalEvent


class Command(BaseCommand):
    help = 'Simulate operational events for testing the event processing system'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=10,
            help='Number of events to simulate (default: 10)'
        )
        
        parser.add_argument(
            '--rate',
            type=float,
            default=1.0,
            help='Events per second (default: 1.0)'
        )
        
        parser.add_argument(
            '--duration',
            type=int,
            default=60,
            help='Duration to run simulation in seconds (default: 60)'
        )
        
        parser.add_argument(
            '--api-endpoint',
            type=str,
            default='http://localhost:8000/api/v1/events/ingest/',
            help='API endpoint for event ingestion'
        )
        
        parser.add_argument(
            '--use-api',
            action='store_true',
            help='Send events via API instead of direct database insertion'
        )
        
        parser.add_argument(
            '--realistic',
            action='store_true',
            help='Use realistic event patterns (mining shift schedules, etc.)'
        )
        
        parser.add_argument(
            '--stope-ids',
            type=str,
            help='Comma-separated list of stope IDs to use (default: all active stopes)'
        )
        
        parser.add_argument(
            '--export',
            type=str,
            help='Export generated events to JSON file'
        )
    
    def handle(self, *args, **options):
        try:
            # Get target stopes
            stopes = self._get_target_stopes(options['stope_ids'])
            if not stopes:
                raise CommandError("No active stopes found for simulation")
            
            self.stdout.write(f"Starting event simulation with {len(stopes)} stopes")
            
            # Generate events
            if options['realistic']:
                events = self._generate_realistic_events(stopes, options)
            else:
                events = self._generate_random_events(stopes, options)
            
            # Export events if requested
            if options['export']:
                self._export_events(events, options['export'])
            
            # Process events
            if options['use_api']:
                self._send_events_via_api(events, options['api_endpoint'], options['rate'])
            else:
                self._create_events_directly(events, options['rate'])
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully simulated {len(events)} events')
            )
            
        except Exception as e:
            raise CommandError(f"Simulation failed: {str(e)}")
    
    def _get_target_stopes(self, stope_ids_str: str) -> List[Stope]:
        """Get the stopes to use for simulation"""
        if stope_ids_str:
            stope_ids = [int(id.strip()) for id in stope_ids_str.split(',')]
            stopes = list(Stope.objects.filter(id__in=stope_ids))
        else:
            stopes = list(Stope.objects.filter(is_active=True))
        
        return stopes
    
    def _generate_random_events(self, stopes: List[Stope], options: Dict) -> List[Dict]:
        """Generate random operational events"""
        events = []
        event_types = [
            'blasting', 'heavy_equipment', 'excavation', 'drilling',
            'loading', 'transport', 'water_exposure', 'vibration_external',
            'support_installation', 'inspection'
        ]
        
        for i in range(options['count']):
            stope = random.choice(stopes)
            event_type = random.choice(event_types)
            
            # Generate realistic parameters based on event type
            severity, proximity, duration = self._get_event_parameters(event_type)
            
            event = {
                'stope_id': stope.id,
                'event_type': event_type,
                'timestamp': timezone.now().isoformat(),
                'severity': severity,
                'proximity_to_stope': proximity,
                'duration_hours': duration,
                'description': f'Simulated {event_type} event #{i+1}',
                'operator': f'Operator_{random.randint(1, 10)}',
                'equipment_involved': self._get_equipment_for_type(event_type)
            }
            
            events.append(event)
        
        return events
    
    def _generate_realistic_events(self, stopes: List[Stope], options: Dict) -> List[Dict]:
        """Generate realistic mining operational events"""
        events = []
        
        # Simulate a typical mining shift pattern
        current_time = timezone.now()
        simulation_duration = timedelta(seconds=options['duration'])
        end_time = current_time + simulation_duration
        
        # Define shift patterns
        shift_patterns = {
            'day_shift': {'start': 6, 'end': 14, 'activity_level': 0.8},
            'swing_shift': {'start': 14, 'end': 22, 'activity_level': 0.6},
            'night_shift': {'start': 22, 'end': 6, 'activity_level': 0.3}
        }
        
        event_count = 0
        while current_time < end_time and event_count < options['count']:
            # Determine current shift
            hour = current_time.hour
            current_shift = None
            for shift_name, shift_info in shift_patterns.items():
                if shift_info['start'] <= hour < shift_info['end'] or \
                   (shift_name == 'night_shift' and (hour >= 22 or hour < 6)):
                    current_shift = shift_info
                    break
            
            if current_shift is None:
                current_shift = shift_patterns['day_shift']  # Default
            
            # Generate events based on shift activity level
            if random.random() < current_shift['activity_level']:
                stope = random.choice(stopes)
                event_type = self._get_shift_appropriate_event_type(hour)
                
                severity, proximity, duration = self._get_event_parameters(event_type)
                
                event = {
                    'stope_id': stope.id,
                    'event_type': event_type,
                    'timestamp': current_time.isoformat(),
                    'severity': severity,
                    'proximity_to_stope': proximity,
                    'duration_hours': duration,
                    'description': f'Realistic {event_type} during shift operation',
                    'operator': f'Shift_Operator_{random.randint(1, 5)}',
                    'equipment_involved': self._get_equipment_for_type(event_type),
                    'weather_conditions': self._get_weather_conditions(),
                    'environmental_factors': self._get_environmental_factors()
                }
                
                events.append(event)
                event_count += 1
            
            # Advance time
            current_time += timedelta(seconds=random.uniform(10, 60))
        
        return events
    
    def _get_event_parameters(self, event_type: str) -> tuple:
        """Get realistic parameters for event type"""
        params = {
            'blasting': (random.uniform(0.7, 1.0), random.uniform(5, 50), random.uniform(0.5, 4.0)),
            'heavy_equipment': (random.uniform(0.3, 0.7), random.uniform(0, 20), random.uniform(1.0, 8.0)),
            'excavation': (random.uniform(0.4, 0.8), random.uniform(0, 15), random.uniform(2.0, 12.0)),
            'drilling': (random.uniform(0.2, 0.6), random.uniform(0, 25), random.uniform(1.0, 6.0)),
            'loading': (random.uniform(0.2, 0.5), random.uniform(0, 10), random.uniform(0.5, 3.0)),
            'transport': (random.uniform(0.1, 0.4), random.uniform(5, 30), random.uniform(0.5, 2.0)),
            'water_exposure': (random.uniform(0.3, 0.9), random.uniform(0, 5), random.uniform(1.0, 24.0)),
            'vibration_external': (random.uniform(0.2, 0.8), random.uniform(10, 100), random.uniform(0.1, 2.0)),
            'support_installation': (random.uniform(0.1, 0.3), random.uniform(0, 5), random.uniform(2.0, 8.0)),
            'inspection': (random.uniform(0.1, 0.2), random.uniform(0, 5), random.uniform(0.5, 2.0))
        }
        
        return params.get(event_type, (0.3, 10.0, 1.0))
    
    def _get_shift_appropriate_event_type(self, hour: int) -> str:
        """Get event types appropriate for the time of day"""
        if 6 <= hour < 14:  # Day shift - high activity
            return random.choice([
                'blasting', 'heavy_equipment', 'excavation', 'drilling',
                'loading', 'transport', 'support_installation'
            ])
        elif 14 <= hour < 22:  # Swing shift - moderate activity
            return random.choice([
                'heavy_equipment', 'excavation', 'loading', 'transport',
                'support_maintenance', 'inspection'
            ])
        else:  # Night shift - low activity
            return random.choice([
                'inspection', 'support_maintenance', 'water_exposure',
                'vibration_external'
            ])
    
    def _get_equipment_for_type(self, event_type: str) -> str:
        """Get appropriate equipment for event type"""
        equipment_map = {
            'blasting': 'Explosive charges, Detonators',
            'heavy_equipment': 'D11 Dozer, CAT 793 Truck',
            'excavation': 'Hydraulic Excavator, Backhoe',
            'drilling': 'Drill Rig, Pneumatic Drill',
            'loading': 'Front-end Loader, Conveyor',
            'transport': 'Haul Truck, Rail Car',
            'water_exposure': 'Pumps, Drainage system',
            'vibration_external': 'External machinery',
            'support_installation': 'Rock bolts, Mesh, Shotcrete gun',
            'inspection': 'Measurement tools, Safety equipment'
        }
        
        return equipment_map.get(event_type, 'General equipment')
    
    def _get_weather_conditions(self) -> str:
        """Get random weather conditions"""
        conditions = ['Clear', 'Cloudy', 'Light rain', 'Heavy rain', 'Windy', 'Humid']
        return random.choice(conditions)
    
    def _get_environmental_factors(self) -> str:
        """Get random environmental factors"""
        factors = [
            'Normal conditions', 'High humidity', 'Low temperature',
            'Dust present', 'Good ventilation', 'Poor visibility'
        ]
        return random.choice(factors)
    
    def _export_events(self, events: List[Dict], filename: str):
        """Export events to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'generated_at': timezone.now().isoformat(),
                    'event_count': len(events),
                    'simulation_type': 'operational_events'
                },
                'events': events
            }, f, indent=2)
        
        self.stdout.write(f"Events exported to {filename}")
    
    def _send_events_via_api(self, events: List[Dict], endpoint: str, rate: float):
        """Send events via API endpoint"""
        self.stdout.write(f"Sending {len(events)} events to {endpoint}")
        
        # Send in batches
        batch_size = 10
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            
            try:
                response = requests.post(
                    endpoint,
                    json={'events': batch},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    successful = len(result.get('results', {}).get('successful', []))
                    failed = len(result.get('results', {}).get('failed', []))
                    self.stdout.write(
                        f"Batch {i//batch_size + 1}: {successful} successful, {failed} failed"
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"Batch {i//batch_size + 1} failed: HTTP {response.status_code}"
                        )
                    )
                
            except requests.RequestException as e:
                self.stdout.write(
                    self.style.ERROR(f"API request failed: {str(e)}")
                )
            
            # Rate limiting
            time.sleep(1.0 / rate)
    
    def _create_events_directly(self, events: List[Dict], rate: float):
        """Create events directly in the database"""
        self.stdout.write(f"Creating {len(events)} events directly in database")
        
        created_count = 0
        for event_data in events:
            try:
                stope = Stope.objects.get(id=event_data['stope_id'])
                
                OperationalEvent.objects.create(
                    stope=stope,
                    event_type=event_data['event_type'],
                    timestamp=timezone.now(),
                    severity=event_data['severity'],
                    proximity_to_stope=event_data['proximity_to_stope'],
                    duration_hours=event_data['duration_hours'],
                    description=event_data['description'],
                    operator=event_data.get('operator', ''),
                    equipment_involved=event_data.get('equipment_involved', ''),
                    weather_conditions=event_data.get('weather_conditions', ''),
                    environmental_factors=event_data.get('environmental_factors', '')
                )
                
                created_count += 1
                
                if created_count % 10 == 0:
                    self.stdout.write(f"Created {created_count} events...")
                
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"Failed to create event: {str(e)}")
                )
            
            # Rate limiting
            time.sleep(1.0 / rate)
