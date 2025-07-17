"""
Event Validation System for Operational Event Processing
Handles data sanitization and validation for incoming events
"""

from django.core.exceptions import ValidationError
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class EventValidator:
    """
    Comprehensive validation system for operational events
    """
    
    # Valid event types from the model
    VALID_EVENT_TYPES = {
        'blasting', 'heavy_equipment', 'excavation', 'drilling', 
        'loading', 'transport', 'water_exposure', 'vibration_external',
        'support_installation', 'support_maintenance', 'inspection',
        'emergency', 'geological_event', 'other'
    }
    
    # Severity ranges
    MIN_SEVERITY = 0.1
    MAX_SEVERITY = 1.0
    
    # Reasonable limits
    MAX_PROXIMITY_METERS = 1000.0  # 1km max distance
    MAX_DURATION_HOURS = 168.0     # 1 week max duration
    MIN_DURATION_HOURS = 0.1       # 6 minutes minimum
    
    def validate_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize a single operational event
        
        Args:
            event_data: Raw event data dictionary
            
        Returns:
            Cleaned and validated event data
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(event_data, dict):
            raise ValidationError("Event data must be a dictionary")
        
        cleaned_data = {}
        errors = []
        
        # Validate stope_id
        try:
            cleaned_data['stope_id'] = self._validate_stope_id(event_data.get('stope_id'))
        except ValidationError as e:
            errors.append(f"stope_id: {str(e)}")
        
        # Validate event_type
        try:
            cleaned_data['event_type'] = self._validate_event_type(event_data.get('event_type'))
        except ValidationError as e:
            errors.append(f"event_type: {str(e)}")
        
        # Validate timestamp
        try:
            cleaned_data['timestamp'] = self._validate_timestamp(event_data.get('timestamp'))
        except ValidationError as e:
            errors.append(f"timestamp: {str(e)}")
        
        # Validate severity
        try:
            cleaned_data['severity'] = self._validate_severity(event_data.get('severity'))
        except ValidationError as e:
            errors.append(f"severity: {str(e)}")
        
        # Validate proximity_to_stope
        try:
            cleaned_data['proximity_to_stope'] = self._validate_proximity(
                event_data.get('proximity_to_stope', 0.0)
            )
        except ValidationError as e:
            errors.append(f"proximity_to_stope: {str(e)}")
        
        # Validate duration_hours
        try:
            cleaned_data['duration_hours'] = self._validate_duration(
                event_data.get('duration_hours', 1.0)
            )
        except ValidationError as e:
            errors.append(f"duration_hours: {str(e)}")
        
        # Validate optional string fields
        cleaned_data['description'] = self._sanitize_string(
            event_data.get('description', ''), max_length=500
        )
        cleaned_data['equipment_involved'] = self._sanitize_string(
            event_data.get('equipment_involved', ''), max_length=200
        )
        cleaned_data['operator'] = self._sanitize_string(
            event_data.get('operator', ''), max_length=100
        )
        cleaned_data['weather_conditions'] = self._sanitize_string(
            event_data.get('weather_conditions', ''), max_length=100
        )
        cleaned_data['environmental_factors'] = self._sanitize_string(
            event_data.get('environmental_factors', ''), max_length=200
        )
        
        # Check for validation errors
        if errors:
            raise ValidationError('; '.join(errors))
        
        # Cross-field validation
        self._validate_event_consistency(cleaned_data)
        
        return cleaned_data
    
    def _validate_stope_id(self, stope_id: Any) -> int:
        """Validate stope ID"""
        if stope_id is None:
            raise ValidationError("stope_id is required")
        
        try:
            stope_id = int(stope_id)
            if stope_id <= 0:
                raise ValidationError("stope_id must be positive")
            return stope_id
        except (ValueError, TypeError):
            raise ValidationError("stope_id must be a valid integer")
    
    def _validate_event_type(self, event_type: Any) -> str:
        """Validate event type"""
        if not event_type:
            raise ValidationError("event_type is required")
        
        event_type = str(event_type).strip().lower()
        if event_type not in self.VALID_EVENT_TYPES:
            valid_types = ', '.join(sorted(self.VALID_EVENT_TYPES))
            raise ValidationError(f"Invalid event_type. Must be one of: {valid_types}")
        
        return event_type
    
    def _validate_timestamp(self, timestamp: Any) -> datetime:
        """Validate timestamp"""
        if not timestamp:
            raise ValidationError("timestamp is required")
        
        # If it's already a datetime object
        if isinstance(timestamp, datetime):
            parsed_time = timestamp
        else:
            # Try to parse string timestamp
            try:
                parsed_time = parse_datetime(str(timestamp))
                if parsed_time is None:
                    # Try alternative formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M']:
                        try:
                            parsed_time = datetime.strptime(str(timestamp), fmt)
                            break
                        except ValueError:
                            continue
                    
                    if parsed_time is None:
                        raise ValidationError("Invalid timestamp format")
            except Exception:
                raise ValidationError("Invalid timestamp format")
        
        # Ensure timezone awareness
        if timezone.is_naive(parsed_time):
            parsed_time = timezone.make_aware(parsed_time)
        
        # Validate timestamp is not too far in the future or past
        now = timezone.now()
        if parsed_time > now + timedelta(hours=24):
            raise ValidationError("Timestamp cannot be more than 24 hours in the future")
        if parsed_time < now - timedelta(days=365):
            raise ValidationError("Timestamp cannot be more than 1 year in the past")
        
        return parsed_time
    
    def _validate_severity(self, severity: Any) -> float:
        """Validate severity level"""
        if severity is None:
            raise ValidationError("severity is required")
        
        try:
            severity = float(severity)
            if not (self.MIN_SEVERITY <= severity <= self.MAX_SEVERITY):
                raise ValidationError(
                    f"severity must be between {self.MIN_SEVERITY} and {self.MAX_SEVERITY}"
                )
            return severity
        except (ValueError, TypeError):
            raise ValidationError("severity must be a valid number")
    
    def _validate_proximity(self, proximity: Any) -> float:
        """Validate proximity to stope"""
        try:
            proximity = float(proximity)
            if proximity < 0:
                raise ValidationError("proximity_to_stope cannot be negative")
            if proximity > self.MAX_PROXIMITY_METERS:
                raise ValidationError(
                    f"proximity_to_stope cannot exceed {self.MAX_PROXIMITY_METERS} meters"
                )
            return proximity
        except (ValueError, TypeError):
            raise ValidationError("proximity_to_stope must be a valid number")
    
    def _validate_duration(self, duration: Any) -> float:
        """Validate event duration"""
        try:
            duration = float(duration)
            if not (self.MIN_DURATION_HOURS <= duration <= self.MAX_DURATION_HOURS):
                raise ValidationError(
                    f"duration_hours must be between {self.MIN_DURATION_HOURS} and {self.MAX_DURATION_HOURS}"
                )
            return duration
        except (ValueError, TypeError):
            raise ValidationError("duration_hours must be a valid number")
    
    def _sanitize_string(self, value: Any, max_length: int = 500) -> str:
        """Sanitize string input"""
        if value is None:
            return ''
        
        # Convert to string and strip whitespace
        sanitized = str(value).strip()
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>\"\'&]', '', sanitized)
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def _validate_event_consistency(self, cleaned_data: Dict[str, Any]) -> None:
        """
        Perform cross-field validation for event consistency
        """
        # High-impact events should have reasonable proximity
        if cleaned_data['severity'] > 0.8 and cleaned_data['proximity_to_stope'] > 200:
            logger.warning(
                f"High severity event ({cleaned_data['severity']}) at large distance "
                f"({cleaned_data['proximity_to_stope']}m) - unusual but allowed"
            )
        
        # Certain event types should have reasonable durations
        event_type = cleaned_data['event_type']
        duration = cleaned_data['duration_hours']
        
        if event_type == 'blasting' and duration > 8:
            raise ValidationError("Blasting events should not exceed 8 hours duration")
        
        if event_type == 'inspection' and duration > 24:
            raise ValidationError("Inspection events should not exceed 24 hours duration")
        
        # Emergency events should have high severity
        if event_type == 'emergency' and cleaned_data['severity'] < 0.7:
            logger.warning("Emergency event with low severity - unusual but allowed")


class BatchEventValidator:
    """
    Validator for batch event processing
    """
    
    def __init__(self):
        self.event_validator = EventValidator()
    
    def validate_batch(self, events_data: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Validate a batch of events
        
        Returns:
            Dictionary with 'valid' and 'invalid' lists
        """
        valid_events = []
        invalid_events = []
        
        for i, event_data in enumerate(events_data):
            try:
                cleaned_event = self.event_validator.validate_event(event_data)
                valid_events.append({
                    'index': i,
                    'data': cleaned_event
                })
            except ValidationError as e:
                invalid_events.append({
                    'index': i,
                    'original_data': event_data,
                    'error': str(e)
                })
        
        return {
            'valid': valid_events,
            'invalid': invalid_events
        }
    
    def get_validation_summary(self, validation_result: Dict[str, List]) -> Dict[str, Any]:
        """
        Generate validation summary statistics
        """
        total_events = len(validation_result['valid']) + len(validation_result['invalid'])
        valid_count = len(validation_result['valid'])
        invalid_count = len(validation_result['invalid'])
        
        return {
            'total_events': total_events,
            'valid_events': valid_count,
            'invalid_events': invalid_count,
            'validation_rate': (valid_count / total_events * 100) if total_events > 0 else 0,
            'has_errors': invalid_count > 0
        }
