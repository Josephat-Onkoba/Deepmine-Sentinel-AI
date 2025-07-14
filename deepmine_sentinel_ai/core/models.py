from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
from datetime import timedelta


class Stope(models.Model):
    """
    Core model representing a mining stope.
    Enhanced for impact-based stability prediction.
    """
    ROCK_TYPE_CHOICES = [
        ('granite', 'Granite'),
        ('basalt', 'Basalt'),
        ('limestone', 'Limestone'),
        ('sandstone', 'Sandstone'),
        ('shale', 'Shale'),
        ('quartzite', 'Quartzite'),
        ('gneiss', 'Gneiss'),
        ('schist', 'Schist'),
        ('slate', 'Slate'),
        ('marble', 'Marble'),
        ('other', 'Other'),
    ]
    
    DIRECTION_CHOICES = [
        ('N', 'North'),
        ('S', 'South'), 
        ('E', 'East'),
        ('W', 'West'),
        ('NE', 'Northeast'),
        ('NW', 'Northwest'),
        ('SE', 'Southeast'),
        ('SW', 'Southwest'),
    ]
    
    SUPPORT_TYPE_CHOICES = [
        ('none', 'No Support'),
        ('rock_bolts', 'Rock Bolts'),
        ('mesh', 'Wire Mesh'),
        ('shotcrete', 'Shotcrete'),
        ('timber', 'Timber Support'),
        ('cable_bolts', 'Cable Bolts'),
        ('steel_sets', 'Steel Sets'),
        ('combined', 'Combined Support'),
    ]

    MINING_METHOD_CHOICES = [
        ('open_stope', 'Open Stoping'),
        ('cut_fill', 'Cut and Fill'),
        ('sublevel_stoping', 'Sublevel Stoping'),
        ('block_caving', 'Block Caving'),
        ('room_pillar', 'Room and Pillar'),
        ('longwall', 'Longwall'),
        ('other', 'Other Method'),
    ]

    # Basic identification
    stope_name = models.CharField(
        max_length=100, 
        unique=True, 
        help_text="Unique identifier for the stope"
    )
    
    # Geological parameters (critical for baseline stability)
    rqd = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Rock Quality Designation (0-100%)"
    )
    rock_type = models.CharField(
        max_length=50, 
        choices=ROCK_TYPE_CHOICES,
        help_text="Primary rock type"
    )
    
    # Geometric parameters (affect stability calculations)
    depth = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text="Depth below surface (meters)"
    )
    hr = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text="Hydraulic Radius (meters)"
    )
    dip = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(90)],
        help_text="Dip angle (degrees)"
    )
    direction = models.CharField(
        max_length=2, 
        choices=DIRECTION_CHOICES,
        help_text="Primary orientation"
    )
    undercut_width = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text="Undercut width (meters)"
    )
    
    # Mining operation details
    mining_method = models.CharField(
        max_length=50,
        choices=MINING_METHOD_CHOICES,
        default='open_stope',
        help_text="Mining method used"
    )
    
    # Support system (affects baseline stability)
    support_type = models.CharField(
        max_length=50, 
        choices=SUPPORT_TYPE_CHOICES,
        default='none',
        help_text="Type of ground support installed"
    )
    support_density = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0)],
        help_text="Support density (bolts/m² or equivalent)"
    )
    support_installed = models.BooleanField(
        default=False,
        help_text="Whether ground support is currently installed"
    )
    
    # Operational status
    is_active = models.BooleanField(
        default=True,
        help_text="Whether stope is currently being mined"
    )
    excavation_started = models.DateTimeField(
        null=True, blank=True,
        help_text="When excavation began"
    )
    excavation_completed = models.DateTimeField(
        null=True, blank=True,
        help_text="When excavation was completed"
    )
    
    # Baseline stability factors
    baseline_stability_score = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Baseline stability score based on geological factors (0.0-1.0)"
    )
    
    # Metadata
    notes = models.TextField(
        blank=True,
        help_text="Additional notes about this stope"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Stope {self.stope_name}"
    
    def calculate_baseline_stability(self):
        """
        Calculate baseline stability score based on geological and design factors.
        Lower scores indicate more stable conditions.
        """
        # RQD factor (higher RQD = more stable)
        rqd_factor = (100 - self.rqd) / 100  # Invert so higher RQD gives lower risk
        
        # Depth factor (deeper = less stable)
        depth_factor = min(self.depth / 1000, 0.5)  # Cap at 0.5 for very deep mines
        
        # Support factor (better support = more stable)
        support_factors = {
            'none': 0.3,
            'rock_bolts': 0.2,
            'mesh': 0.25,
            'shotcrete': 0.15,
            'timber': 0.2,
            'cable_bolts': 0.1,
            'steel_sets': 0.1,
            'combined': 0.05,
        }
        support_factor = support_factors.get(self.support_type, 0.3)
        if self.support_installed:
            support_factor *= max(0.5, 1 - self.support_density / 10)  # More density = better
        
        # Rock type factor
        rock_stability = {
            'granite': 0.1, 'quartzite': 0.1,
            'basalt': 0.15, 'gneiss': 0.15,
            'limestone': 0.2, 'sandstone': 0.25,
            'slate': 0.2, 'marble': 0.2,
            'schist': 0.3, 'shale': 0.35,
            'other': 0.25,
        }
        rock_factor = rock_stability.get(self.rock_type, 0.25)
        
        # Combine factors
        baseline = (rqd_factor * 0.3 + depth_factor * 0.2 + 
                   support_factor * 0.3 + rock_factor * 0.2)
        
        return min(baseline, 1.0)
    
    def save(self, *args, **kwargs):
        """Override save to calculate baseline stability"""
        self.baseline_stability_score = self.calculate_baseline_stability()
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Mining Stope"
        verbose_name_plural = "Mining Stopes"
        indexes = [
            models.Index(fields=['stope_name']),
            models.Index(fields=['is_active']),
            models.Index(fields=['mining_method']),
        ]


class MonitoringData(models.Model):
    """
    Simplified model for storing sensor monitoring data.
    Used to feed into LSTM predictions and detect anomalies.
    """
    SENSOR_TYPE_CHOICES = [
        ('vibration', 'Vibration Sensor'),
        ('deformation', 'Deformation Sensor'),
        ('stress', 'Stress Sensor'),
        ('temperature', 'Temperature Sensor'),
        ('humidity', 'Humidity Sensor'),
        ('acoustic', 'Acoustic Emission'),
        ('strain', 'Strain Gauge'),
        ('displacement', 'Displacement Sensor'),
    ]
    
    stope = models.ForeignKey(
        Stope, 
        on_delete=models.CASCADE, 
        related_name='monitoring_data'
    )
    sensor_type = models.CharField(
        max_length=20,
        choices=SENSOR_TYPE_CHOICES,
        help_text="Type of sensor/measurement"
    )
    timestamp = models.DateTimeField(
        help_text="When the measurement was taken"
    )
    value = models.FloatField(
        help_text="Measured value"
    )
    unit = models.CharField(
        max_length=20,
        help_text="Unit of measurement (mm/s, MPa, °C, etc.)"
    )
    sensor_id = models.CharField(
        max_length=50,
        blank=True,
        help_text="Physical sensor identifier"
    )
    
    # Quality indicators
    is_anomaly = models.BooleanField(
        default=False,
        help_text="Flagged as anomalous reading"
    )
    confidence = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Confidence in this measurement (0.0-1.0)"
    )
    
    def __str__(self):
        return f"{self.get_sensor_type_display()} for {self.stope.stope_name}: {self.value} {self.unit}"
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Monitoring Data Point"
        verbose_name_plural = "Monitoring Data Points"
        unique_together = ['stope', 'sensor_type', 'timestamp', 'sensor_id']
        indexes = [
            models.Index(fields=['stope', 'timestamp']),
            models.Index(fields=['sensor_type', 'timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['is_anomaly']),
        ]


class OperationalEvent(models.Model):
    """
    Enhanced model to store operational events that affect stope stability.
    Core of the impact-based prediction system.
    """
    EVENT_TYPE_CHOICES = [
        ('blasting', 'Blasting Operations'),
        ('heavy_equipment', 'Heavy Equipment Operations'),
        ('excavation', 'Excavation Activities'),
        ('drilling', 'Drilling Operations'),
        ('loading', 'Material Loading/Unloading'),
        ('transport', 'Heavy Transport'),
        ('water_exposure', 'Water Exposure/Flooding'),
        ('vibration_external', 'External Vibration Events'),
        ('support_installation', 'Ground Support Installation'),
        ('support_maintenance', 'Support System Maintenance'),
        ('inspection', 'Routine Inspection'),
        ('emergency', 'Emergency Response'),
        ('geological_event', 'Geological Event (rockfall, etc.)'),
        ('other', 'Other Operations'),
    ]
    
    SEVERITY_CHOICES = [
        (0.1, 'Minimal Impact'),
        (0.3, 'Low Impact'),
        (0.5, 'Moderate Impact'),
        (0.7, 'High Impact'),
        (0.9, 'Severe Impact'),
        (1.0, 'Critical Impact'),
    ]
    
    # Core event information
    stope = models.ForeignKey(
        Stope, 
        on_delete=models.CASCADE, 
        related_name='operational_events',
        help_text="Stope affected by this operational event"
    )
    event_type = models.CharField(
        max_length=30, 
        choices=EVENT_TYPE_CHOICES,
        help_text="Type of operational event"
    )
    timestamp = models.DateTimeField(
        help_text="Date and time when the event occurred"
    )
    
    # Impact calculation parameters
    severity = models.FloatField(
        choices=SEVERITY_CHOICES,
        default=0.5,
        validators=[MinValueValidator(0.1), MaxValueValidator(1.0)],
        help_text="Severity level of the event (0.1-1.0)"
    )
    proximity_to_stope = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0)],
        help_text="Distance from stope center (meters)"
    )
    duration_hours = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.1)],
        help_text="Duration of the event (hours)"
    )
    affected_area = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0)],
        help_text="Area affected by the event (square meters)"
    )
    
    # Event details
    description = models.TextField(
        help_text="Detailed description of the operational event"
    )
    operator_crew = models.CharField(
        max_length=100,
        blank=True,
        help_text="Crew or operator responsible"
    )
    equipment_involved = models.CharField(
        max_length=200,
        blank=True,
        help_text="Equipment involved in the operation"
    )
    safety_measures = models.TextField(
        blank=True,
        help_text="Safety measures taken during the event"
    )
    
    # Impact assessment
    immediate_impact_score = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Calculated immediate impact on stability (0.0-1.0)"
    )
    decay_rate = models.FloatField(
        default=0.1,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Rate at which impact decreases over time (per day)"
    )
    
    # Related monitoring data
    triggered_monitoring = models.BooleanField(
        default=False,
        help_text="Whether this event triggered additional monitoring"
    )
    
    # Metadata
    event_metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional event-specific data (coordinates, weather, etc.)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    recorded_by = models.CharField(
        max_length=100,
        help_text="Person who recorded this event"
    )
    verified = models.BooleanField(
        default=False,
        help_text="Whether this event has been verified by supervisor"
    )
    verified_by = models.CharField(
        max_length=100,
        blank=True,
        help_text="Supervisor who verified this event"
    )
    
    def __str__(self):
        return f"{self.get_event_type_display()} - {self.stope.stope_name} at {self.timestamp}"
    
    def calculate_impact_score(self):
        """
        Calculate the immediate impact score for this event.
        Considers event type, severity, proximity, and duration.
        """
        # Base impact factors by event type
        base_impact_factors = {
            'blasting': 0.4,
            'heavy_equipment': 0.2,
            'excavation': 0.3,
            'drilling': 0.15,
            'loading': 0.1,
            'transport': 0.05,
            'water_exposure': 0.25,
            'vibration_external': 0.2,
            'support_installation': -0.15,  # Positive event - reduces risk
            'support_maintenance': -0.1,   # Positive event
            'inspection': 0.0,
            'emergency': 0.5,
            'geological_event': 0.6,
            'other': 0.1,
        }
        
        base_factor = base_impact_factors.get(self.event_type, 0.1)
        
        # Proximity factor (closer = more impact)
        proximity_factor = 1.0 if self.proximity_to_stope <= 10 else max(0.1, 100 / (self.proximity_to_stope + 10))
        
        # Duration factor (longer = more impact, but with diminishing returns)
        duration_factor = min(2.0, 1 + (self.duration_hours - 1) * 0.1)
        
        # Calculate final impact
        impact = abs(base_factor) * self.severity * proximity_factor * duration_factor
        
        # Apply sign (negative for beneficial events)
        if base_factor < 0:
            impact = -impact
            
        return max(-0.5, min(1.0, impact))  # Clamp between -0.5 and 1.0
    
    def get_current_impact(self, as_of_date=None):
        """
        Get the current impact of this event, accounting for time decay.
        """
        if as_of_date is None:
            as_of_date = timezone.now()
            
        if self.immediate_impact_score is None:
            self.immediate_impact_score = self.calculate_impact_score()
            
        # Calculate time decay
        days_elapsed = (as_of_date - self.timestamp).total_seconds() / (24 * 3600)
        decay_factor = max(0, 1 - (self.decay_rate * days_elapsed))
        
        return self.immediate_impact_score * decay_factor
    
    def save(self, *args, **kwargs):
        """Override save to calculate immediate impact score"""
        if self.immediate_impact_score is None:
            self.immediate_impact_score = self.calculate_impact_score()
        super().save(*args, **kwargs)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Operational Event"
        verbose_name_plural = "Operational Events"
        indexes = [
            models.Index(fields=['stope', 'timestamp']),
            models.Index(fields=['event_type', 'timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['verified']),
            models.Index(fields=['stope', 'event_type']),
        ]


class ImpactScore(models.Model):
    """
    Enhanced model to store current impact scores for each stope.
    Central hub for real-time stability risk assessment.
    """
    RISK_LEVEL_CHOICES = [
        ('stable', 'STABLE - Normal Operations'),
        ('elevated', 'ELEVATED - Increased Monitoring'),
        ('high_risk', 'HIGH RISK - Restricted Access'),
        ('critical', 'CRITICAL - Immediate Action Required'),
        ('emergency', 'EMERGENCY - Evacuation Protocol'),
    ]
    
    # Core relationship
    stope = models.OneToOneField(
        Stope,
        on_delete=models.CASCADE,
        related_name='impact_score',
        help_text="Stope for which this impact score applies"
    )
    
    # Current impact assessment
    current_score = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Current impact score (0.0 = stable, 1.0 = critical)"
    )
    risk_level = models.CharField(
        max_length=20,
        choices=RISK_LEVEL_CHOICES,
        default='stable',
        help_text="Current risk level classification"
    )
    
    # Score composition breakdown
    baseline_component = models.FloatField(
        default=0.0,
        help_text="Component from geological baseline factors"
    )
    operational_component = models.FloatField(
        default=0.0,
        help_text="Component from recent operational events"
    )
    temporal_component = models.FloatField(
        default=0.0,
        help_text="Component from time-based factors and trends"
    )
    monitoring_component = models.FloatField(
        default=0.0,
        help_text="Component from sensor anomalies and monitoring data"
    )
    
    # Contributing factors detail
    contributing_factors = models.JSONField(
        default=dict,
        help_text="Detailed breakdown of contributing factors"
    )
    
    # Temporal tracking
    last_updated = models.DateTimeField(
        auto_now=True,
        help_text="Last time the impact score was recalculated"
    )
    last_significant_change = models.DateTimeField(
        null=True, blank=True,
        help_text="Last time the risk level changed significantly"
    )
    last_calculation_method = models.CharField(
        max_length=50,
        default='manual',
        help_text="Method used for last calculation (manual, automatic, lstm)"
    )
    
    # LSTM Predictions
    predicted_24h = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="LSTM predicted score for 24 hours ahead"
    )
    predicted_48h = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="LSTM predicted score for 48 hours ahead"
    )
    predicted_7d = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="LSTM predicted score for 7 days ahead"
    )
    predicted_30d = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="LSTM predicted score for 30 days ahead"
    )
    prediction_confidence = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Confidence level of LSTM predictions (0.0-1.0)"
    )
    last_prediction_update = models.DateTimeField(
        null=True, blank=True,
        help_text="Last time LSTM predictions were updated"
    )
    
    # Alerts and thresholds
    alert_threshold_elevated = models.FloatField(
        default=0.25,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Score threshold for elevated risk alert"
    )
    alert_threshold_high = models.FloatField(
        default=0.50,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Score threshold for high risk alert"
    )
    alert_threshold_critical = models.FloatField(
        default=0.75,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Score threshold for critical risk alert"
    )
    
    def __str__(self):
        return f"{self.stope.stope_name} - {self.get_risk_level_display()} ({self.current_score:.3f})"
    
    def get_risk_level_from_score(self, score=None):
        """Determine risk level based on impact score and custom thresholds"""
        if score is None:
            score = self.current_score
            
        if score < self.alert_threshold_elevated:
            return 'stable'
        elif score < self.alert_threshold_high:
            return 'elevated'
        elif score < self.alert_threshold_critical:
            return 'high_risk'
        elif score < 0.9:
            return 'critical'
        else:
            return 'emergency'
    
    def update_risk_level(self):
        """Update risk level based on current score"""
        old_risk_level = self.risk_level
        new_risk_level = self.get_risk_level_from_score()
        
        if new_risk_level != old_risk_level:
            self.last_significant_change = timezone.now()
            self.risk_level = new_risk_level
            return True
        return False
    
    def calculate_comprehensive_score(self):
        """
        Calculate comprehensive impact score from all components.
        This is the main scoring algorithm.
        """
        # Get baseline from stope geological factors
        self.baseline_component = self.stope.baseline_stability_score
        
        # Calculate operational component from recent events
        recent_events = self.stope.operational_events.filter(
            timestamp__gte=timezone.now() - timedelta(days=30)
        )
        operational_impact = 0
        for event in recent_events:
            operational_impact += event.get_current_impact()
        self.operational_component = min(0.5, max(0, operational_impact))
        
        # Calculate monitoring component from anomalies
        recent_anomalies = self.stope.monitoring_data.filter(
            timestamp__gte=timezone.now() - timedelta(days=7),
            is_anomaly=True
        ).count()
        self.monitoring_component = min(0.2, recent_anomalies * 0.05)
        
        # Temporal component (time since last major event, mining duration, etc.)
        if self.stope.excavation_started:
            days_mining = (timezone.now() - self.stope.excavation_started).days
            self.temporal_component = min(0.1, days_mining * 0.001)  # Gradual increase over time
        else:
            self.temporal_component = 0.0
            
        # Combine components with weights
        self.current_score = (
            self.baseline_component * 0.4 +      # Geological baseline
            self.operational_component * 0.4 +   # Recent operations
            self.monitoring_component * 0.15 +   # Sensor anomalies
            self.temporal_component * 0.05       # Time factors
        )
        
        # Update contributing factors
        self.contributing_factors = {
            'baseline': float(self.baseline_component),
            'operational': float(self.operational_component),
            'monitoring': float(self.monitoring_component),
            'temporal': float(self.temporal_component),
            'total': float(self.current_score),
            'last_updated': timezone.now().isoformat(),
        }
        
        return self.current_score
    
    def needs_prediction_update(self):
        """Check if LSTM predictions need updating"""
        if not self.last_prediction_update:
            return True
        hours_since_update = (timezone.now() - self.last_prediction_update).total_seconds() / 3600
        return hours_since_update >= 6  # Update every 6 hours
    
    def save(self, *args, **kwargs):
        """Override save to automatically update risk level and calculations"""
        # Recalculate score if needed
        if not kwargs.pop('skip_calculation', False):
            self.calculate_comprehensive_score()
            
        # Update risk level
        self.update_risk_level()
        
        super().save(*args, **kwargs)
    
    class Meta:
        verbose_name = "Impact Score"
        verbose_name_plural = "Impact Scores"
        indexes = [
            models.Index(fields=['risk_level']),
            models.Index(fields=['current_score']),
            models.Index(fields=['last_updated']),
        ]


class ImpactHistory(models.Model):
    """
    Enhanced model to track historical changes in impact scores.
    Provides comprehensive audit trail and supports trend analysis.
    """
    CHANGE_TYPE_CHOICES = [
        ('event_impact', 'Operational Event Impact'),
        ('time_decay', 'Natural Time Decay'),
        ('manual_adjustment', 'Manual Score Adjustment'),
        ('support_improvement', 'Ground Support Enhancement'),
        ('maintenance_activity', 'Maintenance Activity'),
        ('monitoring_anomaly', 'Monitoring Anomaly Detected'),
        ('system_recalculation', 'Automated System Recalculation'),
        ('lstm_prediction', 'LSTM Model Update'),
        ('threshold_adjustment', 'Alert Threshold Change'),
        ('emergency_override', 'Emergency Manual Override'),
    ]
    
    # Core references
    stope = models.ForeignKey(
        Stope,
        on_delete=models.CASCADE,
        related_name='impact_history',
        help_text="Stope for which this history record applies"
    )
    
    # Score change information
    previous_score = models.FloatField(
        help_text="Impact score before this change"
    )
    new_score = models.FloatField(
        help_text="Impact score after this change"
    )
    score_change = models.FloatField(
        help_text="Change in impact score (positive = increase, negative = decrease)"
    )
    
    # Risk level change information
    previous_risk_level = models.CharField(
        max_length=20,
        help_text="Risk level before this change"
    )
    new_risk_level = models.CharField(
        max_length=20,
        help_text="Risk level after this change"
    )
    
    # Change classification
    change_type = models.CharField(
        max_length=30,
        choices=CHANGE_TYPE_CHOICES,
        help_text="Type of change that caused this impact score update"
    )
    change_magnitude = models.CharField(
        max_length=20,
        choices=[
            ('minimal', 'Minimal Change (<0.05)'),
            ('minor', 'Minor Change (0.05-0.15)'),
            ('moderate', 'Moderate Change (0.15-0.30)'),
            ('significant', 'Significant Change (0.30-0.50)'),
            ('major', 'Major Change (>0.50)'),
        ],
        default='minimal',
        help_text="Magnitude classification of the change"
    )
    
    # Detailed information
    change_reason = models.TextField(
        help_text="Detailed explanation of what caused this change"
    )
    component_changes = models.JSONField(
        default=dict,
        help_text="Breakdown of how each score component changed"
    )
    
    # Related events and data
    related_operational_event = models.ForeignKey(
        OperationalEvent,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='score_impacts',
        help_text="Operational event that triggered this change (if applicable)"
    )
    related_monitoring_data = models.ManyToManyField(
        MonitoringData,
        blank=True,
        related_name='score_impacts',
        help_text="Monitoring data points that contributed to this change"
    )
    
    # Metadata and tracking
    timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="When this change occurred"
    )
    calculation_method = models.CharField(
        max_length=50,
        choices=[
            ('automatic', 'Automatic System Calculation'),
            ('manual', 'Manual Calculation'),
            ('lstm', 'LSTM Model Prediction'),
            ('hybrid', 'Hybrid Automatic/Manual'),
        ],
        default='automatic',
        help_text="Method used to calculate this change"
    )
    calculated_by = models.CharField(
        max_length=100,
        default='system',
        help_text="What/who calculated this change"
    )
    confidence_level = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Confidence in this calculation (0.0-1.0)"
    )
    
    # Additional context
    external_factors = models.JSONField(
        default=dict,
        blank=True,
        help_text="External factors that may have influenced this change"
    )
    notes = models.TextField(
        blank=True,
        help_text="Additional notes about this change"
    )
    
    def __str__(self):
        direction = "↑" if self.score_change > 0 else "↓" if self.score_change < 0 else "="
        return f"{self.stope.stope_name} {direction} {abs(self.score_change):.3f} - {self.get_change_type_display()}"
    
    @property
    def risk_level_changed(self):
        """Check if this change resulted in a risk level transition"""
        return self.previous_risk_level != self.new_risk_level
    
    @property
    def is_significant_change(self):
        """Check if this is considered a significant change"""
        return abs(self.score_change) >= 0.15
    
    def classify_change_magnitude(self):
        """Automatically classify the magnitude of this change"""
        abs_change = abs(self.score_change)
        if abs_change < 0.05:
            return 'minimal'
        elif abs_change < 0.15:
            return 'minor'
        elif abs_change < 0.30:
            return 'moderate'
        elif abs_change < 0.50:
            return 'significant'
        else:
            return 'major'
    
    def save(self, *args, **kwargs):
        """Override save to automatically classify change magnitude"""
        self.score_change = self.new_score - self.previous_score
        self.change_magnitude = self.classify_change_magnitude()
        super().save(*args, **kwargs)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Impact History Record"
        verbose_name_plural = "Impact History Records"
        indexes = [
            models.Index(fields=['stope', 'timestamp']),
            models.Index(fields=['change_type', 'timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['stope', 'change_type']),
            models.Index(fields=['change_magnitude']),
        ]


class ImpactFactor(models.Model):
    """
    Configurable impact factors for different operational events.
    Allows mining engineers to adjust impact weights based on site-specific conditions.
    """
    CATEGORY_CHOICES = [
        ('blasting', 'Blasting Operations'),
        ('equipment', 'Heavy Equipment'),
        ('excavation', 'Excavation Activities'),
        ('drilling', 'Drilling Operations'),
        ('loading', 'Material Loading'),
        ('transport', 'Heavy Transport'),
        ('water', 'Water-Related Events'),
        ('vibration', 'External Vibrations'),
        ('support', 'Support Installation/Maintenance'),
        ('geological', 'Geological Events'),
        ('environmental', 'Environmental Factors'),
        ('other', 'Other Operations'),
    ]
    
    SEVERITY_LEVEL_CHOICES = [
        ('minimal', 'Minimal Impact'),
        ('low', 'Low Impact'),
        ('moderate', 'Moderate Impact'),
        ('high', 'High Impact'),
        ('severe', 'Severe Impact'),
        ('critical', 'Critical Impact'),
    ]
    
    # Factor identification
    event_category = models.CharField(
        max_length=20,
        choices=CATEGORY_CHOICES,
        help_text="Category of operational event"
    )
    severity_level = models.CharField(
        max_length=10,
        choices=SEVERITY_LEVEL_CHOICES,
        help_text="Severity level for this impact factor"
    )
    
    # Impact calculation parameters
    base_impact_weight = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(10.0)],
        help_text="Base impact weight (0.0-10.0 scale)"
    )
    duration_multiplier = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(5.0)],
        help_text="Multiplier for event duration (0.1-5.0)"
    )
    proximity_decay_rate = models.FloatField(
        default=0.1,
        validators=[MinValueValidator(0.01), MaxValueValidator(1.0)],
        help_text="Rate of impact decay with distance (0.01-1.0 per meter)"
    )
    temporal_decay_rate = models.FloatField(
        default=0.05,
        validators=[MinValueValidator(0.001), MaxValueValidator(0.5)],
        help_text="Rate of impact decay over time (0.001-0.5 per day)"
    )
    
    # Configuration metadata
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this factor is currently active"
    )
    site_specific = models.BooleanField(
        default=False,
        help_text="Site-specific override of default factors"
    )
    mine_site = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Specific mine site (if site-specific)"
    )
    
    # Documentation and validation
    description = models.TextField(
        help_text="Detailed description of this impact factor"
    )
    validation_source = models.CharField(
        max_length=200,
        blank=True,
        help_text="Source of validation (research, field data, expert opinion)"
    )
    last_calibrated = models.DateTimeField(
        null=True, blank=True,
        help_text="When this factor was last calibrated/validated"
    )
    calibrated_by = models.CharField(
        max_length=100,
        blank=True,
        help_text="Engineer who last calibrated this factor"
    )
    
    # Audit trail
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.CharField(
        max_length=100,
        help_text="User who created this impact factor"
    )
    
    def __str__(self):
        return f"{self.get_event_category_display()} - {self.get_severity_level_display()}"
    
    def calculate_adjusted_impact(self, base_value, duration_hours=1.0, distance_meters=0.0, days_elapsed=0.0):
        """
        Calculate adjusted impact based on this factor's parameters
        
        Args:
            base_value: Base impact value
            duration_hours: Duration of the event in hours
            distance_meters: Distance from stope center
            days_elapsed: Days since the event occurred
            
        Returns:
            float: Adjusted impact value
        """
        # Apply base weight
        impact = base_value * self.base_impact_weight
        
        # Apply duration multiplier
        if duration_hours > 1.0:
            duration_factor = 1.0 + (duration_hours - 1.0) * (self.duration_multiplier - 1.0)
            impact *= duration_factor
        
        # Apply proximity decay
        if distance_meters > 0:
            proximity_factor = max(0.1, 1.0 - (distance_meters * self.proximity_decay_rate))
            impact *= proximity_factor
        
        # Apply temporal decay
        if days_elapsed > 0:
            temporal_factor = max(0.01, 1.0 - (days_elapsed * self.temporal_decay_rate))
            impact *= temporal_factor
        
        return max(0.0, impact)
    
    def clean(self):
        """Validate impact factor parameters"""
        from django.core.exceptions import ValidationError
        
        # Ensure reasonable parameter combinations
        if self.base_impact_weight > 5.0 and self.duration_multiplier > 3.0:
            raise ValidationError(
                "High base impact weight combined with high duration multiplier may cause excessive impacts"
            )
        
        # Validate temporal decay isn't too aggressive
        if self.temporal_decay_rate > 0.2 and self.base_impact_weight < 1.0:
            raise ValidationError(
                "High temporal decay rate with low base impact may cause impacts to disappear too quickly"
            )
    
    class Meta:
        unique_together = ['event_category', 'severity_level', 'mine_site']
        ordering = ['event_category', 'severity_level']
        verbose_name = "Impact Factor"
        verbose_name_plural = "Impact Factors"
        indexes = [
            models.Index(fields=['event_category', 'severity_level']),
            models.Index(fields=['is_active']),
            models.Index(fields=['site_specific', 'mine_site']),
            models.Index(fields=['last_calibrated']),
        ]


# ===== OPTIMIZED IMPACT-BASED DATABASE SCHEMA COMPLETE =====
# All models specifically designed for revolutionary impact-based prediction system
# with LSTM integration and comprehensive monitoring capabilities

# ===== LSTM TIME SERIES DATA STRUCTURES =====
# Enhanced models for machine learning training and prediction

class TimeSeriesData(models.Model):
    """
    Enhanced time series data model specifically designed for LSTM training.
    Stores preprocessed features, labels, and metadata for machine learning.
    """
    
    SEQUENCE_TYPE_CHOICES = [
        ('training', 'Training Sequence'),
        ('validation', 'Validation Sequence'),
        ('test', 'Test Sequence'),
        ('prediction', 'Live Prediction Input'),
    ]
    
    FEATURE_SET_CHOICES = [
        ('basic', 'Basic Features (sensor data only)'),
        ('enhanced', 'Enhanced Features (sensor + operational)'),
        ('engineered', 'Engineered Features (with derived metrics)'),
        ('full', 'Full Feature Set (all available features)'),
    ]
    
    # Core identification
    stope = models.ForeignKey(
        Stope,
        on_delete=models.CASCADE,
        related_name='time_series_data',
        help_text="Associated stope"
    )
    
    sequence_id = models.CharField(
        max_length=100,
        help_text="Unique identifier for this time series sequence"
    )
    
    sequence_type = models.CharField(
        max_length=20,
        choices=SEQUENCE_TYPE_CHOICES,
        default='training',
        help_text="Type of sequence for ML pipeline"
    )
    
    # Time series metadata
    start_timestamp = models.DateTimeField(
        help_text="Start time of the sequence"
    )
    end_timestamp = models.DateTimeField(
        help_text="End time of the sequence"
    )
    sequence_length = models.IntegerField(
        validators=[MinValueValidator(1)],
        help_text="Number of time steps in sequence"
    )
    sampling_interval = models.DurationField(
        default=timedelta(hours=1),
        help_text="Time interval between samples"
    )
    
    # Feature configuration
    feature_set = models.CharField(
        max_length=20,
        choices=FEATURE_SET_CHOICES,
        default='enhanced',
        help_text="Set of features included in this sequence"
    )
    
    feature_count = models.IntegerField(
        validators=[MinValueValidator(1)],
        help_text="Number of features per time step"
    )
    
    # Preprocessed data storage (JSON fields for efficiency)
    raw_features = models.JSONField(
        help_text="Raw feature values [timesteps, features]"
    )
    
    normalized_features = models.JSONField(
        help_text="Normalized feature values for ML training"
    )
    
    feature_names = models.JSONField(
        help_text="List of feature names in order"
    )
    
    # Labels and targets
    impact_score_sequence = models.JSONField(
        help_text="Sequence of impact scores (target values)"
    )
    
    risk_level_sequence = models.JSONField(
        help_text="Sequence of risk levels (categorical targets)"
    )
    
    future_impact_scores = models.JSONField(
        null=True,
        blank=True,
        help_text="Future impact scores for prediction horizons"
    )
    
    # Quality and validation metrics
    data_quality_score = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Overall data quality score (0.0-1.0)"
    )
    
    missing_data_percentage = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Percentage of missing data points"
    )
    
    anomaly_count = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Number of anomalous data points detected"
    )
    
    # Processing metadata
    preprocessing_version = models.CharField(
        max_length=20,
        default='1.0',
        help_text="Version of preprocessing pipeline used"
    )
    
    processing_timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="When this sequence was processed"
    )
    
    is_valid = models.BooleanField(
        default=True,
        help_text="Whether sequence passes validation checks"
    )
    validation_errors = models.JSONField(
        default=list,
        blank=True,
        help_text="List of validation errors if any"
    )
    
    # LSTM specific metadata
    sequence_overlap = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Overlap with adjacent sequences (0.0-1.0)"
    )
    
    is_augmented = models.BooleanField(
        default=False,
        help_text="Whether this sequence is from data augmentation"
    )
    
    augmentation_type = models.CharField(
        max_length=50,
        blank=True,
        help_text="Type of augmentation applied"
    )
    
    def get_feature_matrix(self):
        """Return normalized features as numpy array for ML"""
        import numpy as np
        return np.array(self.normalized_features)
    
    def get_target_vector(self):
        """Return impact score sequence as numpy array"""
        import numpy as np
        return np.array(self.impact_score_sequence)
    
    def validate_sequence(self):
        """Validate the time series sequence for ML training"""
        errors = []
        
        # Check sequence length consistency
        if len(self.normalized_features) != self.sequence_length:
            errors.append(f"Feature length {len(self.normalized_features)} != sequence_length {self.sequence_length}")
        
        # Check feature count consistency
        if self.normalized_features and len(self.normalized_features[0]) != self.feature_count:
            errors.append(f"Feature count mismatch: expected {self.feature_count}, got {len(self.normalized_features[0])}")
        
        # Check target sequence length
        if len(self.impact_score_sequence) != self.sequence_length:
            errors.append(f"Target length {len(self.impact_score_sequence)} != sequence_length {self.sequence_length}")
        
        # Check for NaN or infinite values
        import numpy as np
        features_array = np.array(self.normalized_features)
        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            errors.append("Features contain NaN or infinite values")
        
        # Update validation status
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        
        return self.is_valid
    
    def __str__(self):
        return f"TimeSeriesData {self.sequence_id} ({self.stope.stope_name}): {self.sequence_length} steps"
    
    class Meta:
        ordering = ['-processing_timestamp']
        verbose_name = "Time Series Data"
        verbose_name_plural = "Time Series Data"
        unique_together = ['stope', 'sequence_id']
        indexes = [
            models.Index(fields=['stope', 'start_timestamp']),
            models.Index(fields=['sequence_type', 'is_valid']),
            models.Index(fields=['feature_set', 'preprocessing_version']),
            models.Index(fields=['data_quality_score']),
            models.Index(fields=['processing_timestamp']),
        ]


class FeatureEngineeringConfig(models.Model):
    """
    Configuration for feature engineering pipeline used in LSTM training.
    Stores parameters for feature extraction and transformation.
    """
    
    FEATURE_TYPE_CHOICES = [
        ('raw', 'Raw Sensor Values'),
        ('statistical', 'Statistical Aggregations'),
        ('temporal', 'Temporal Patterns'),
        ('frequency', 'Frequency Domain Features'),
        ('operational', 'Operational Event Features'),
        ('derived', 'Derived/Calculated Features'),
    ]
    
    AGGREGATION_CHOICES = [
        ('mean', 'Mean'),
        ('median', 'Median'),
        ('std', 'Standard Deviation'),
        ('min', 'Minimum'),
        ('max', 'Maximum'),
        ('range', 'Range (Max-Min)'),
        ('percentile_25', '25th Percentile'),
        ('percentile_75', '75th Percentile'),
        ('skewness', 'Skewness'),
        ('kurtosis', 'Kurtosis'),
    ]
    
    config_name = models.CharField(
        max_length=100,
        unique=True,
        help_text="Unique name for this feature configuration"
    )
    
    version = models.CharField(
        max_length=20,
        default='1.0',
        help_text="Version of this configuration"
    )
    
    description = models.TextField(
        help_text="Description of feature engineering approach"
    )
    
    # Feature selection
    enabled_sensor_types = models.JSONField(
        help_text="List of sensor types to include"
    )
    
    enabled_feature_types = models.JSONField(
        help_text="List of feature types to generate"
    )
    
    # Temporal aggregation settings
    window_sizes = models.JSONField(
        default=list,
        help_text="List of time window sizes for aggregation (in hours)"
    )
    
    aggregation_functions = models.JSONField(
        default=list,
        help_text="List of aggregation functions to apply"
    )
    
    # Frequency domain settings
    include_fft_features = models.BooleanField(
        default=False,
        help_text="Include FFT-based frequency features"
    )
    
    fft_frequency_bands = models.JSONField(
        default=list,
        blank=True,
        help_text="Frequency bands for FFT analysis"
    )
    
    # Operational event integration
    include_event_features = models.BooleanField(
        default=True,
        help_text="Include operational event-based features"
    )
    
    event_decay_factor = models.FloatField(
        default=0.95,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Decay factor for event impact over time"
    )
    
    # Normalization settings
    normalization_method = models.CharField(
        max_length=20,
        choices=[
            ('minmax', 'Min-Max Scaling'),
            ('zscore', 'Z-Score Normalization'),
            ('robust', 'Robust Scaling'),
            ('quantile', 'Quantile Normalization'),
        ],
        default='zscore',
        help_text="Method for feature normalization"
    )
    
    # Quality control
    outlier_detection_method = models.CharField(
        max_length=20,
        choices=[
            ('iqr', 'Interquartile Range'),
            ('zscore', 'Z-Score Method'),
            ('isolation_forest', 'Isolation Forest'),
            ('none', 'No Outlier Detection'),
        ],
        default='iqr',
        help_text="Method for outlier detection"
    )
    
    outlier_threshold = models.FloatField(
        default=3.0,
        validators=[MinValueValidator(0)],
        help_text="Threshold for outlier detection"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this configuration is active"
    )
    
    def __str__(self):
        return f"Feature Config: {self.config_name} v{self.version}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Feature Engineering Configuration"
        verbose_name_plural = "Feature Engineering Configurations"
        unique_together = ['config_name', 'version']


class DataQualityMetrics(models.Model):
    """
    Track data quality metrics for time series data and LSTM training.
    """
    
    time_series_data = models.OneToOneField(
        TimeSeriesData,
        on_delete=models.CASCADE,
        related_name='quality_metrics'
    )
    
    # Completeness metrics
    completeness_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Data completeness score (0.0-1.0)"
    )
    
    missing_sensor_types = models.JSONField(
        default=list,
        help_text="List of sensor types with missing data"
    )
    
    missing_data_gaps = models.JSONField(
        default=list,
        help_text="List of time gaps with missing data"
    )
    
    # Consistency metrics
    consistency_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Data consistency score (0.0-1.0)"
    )
    
    outlier_count = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Number of outlier data points"
    )
    
    outlier_percentage = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Percentage of outlier data points"
    )
    
    # Validity metrics
    validity_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Data validity score (0.0-1.0)"
    )
    
    invalid_readings_count = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Number of invalid sensor readings"
    )
    
    sensor_failure_events = models.JSONField(
        default=list,
        help_text="List of detected sensor failure events"
    )
    
    # Temporal quality
    temporal_resolution_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Quality of temporal resolution (0.0-1.0)"
    )
    
    timestamp_irregularities = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Number of timestamp irregularities"
    )
    
    # Overall quality
    overall_quality_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Overall data quality score (0.0-1.0)"
    )
    
    quality_grade = models.CharField(
        max_length=2,
        choices=[
            ('A', 'Excellent (>0.9)'),
            ('B', 'Good (0.8-0.9)'),
            ('C', 'Fair (0.7-0.8)'),
            ('D', 'Poor (0.6-0.7)'),
            ('F', 'Failing (<0.6)'),
        ],
        help_text="Letter grade for data quality"
    )
    
    # Analysis metadata
    analysis_timestamp = models.DateTimeField(auto_now_add=True)
    analysis_version = models.CharField(
        max_length=20,
        default='1.0',
        help_text="Version of quality analysis algorithm"
    )
    
    def calculate_overall_quality(self):
        """Calculate overall quality score from component metrics"""
        weights = {
            'completeness': 0.3,
            'consistency': 0.3,
            'validity': 0.25,
            'temporal': 0.15,
        }
        
        self.overall_quality_score = (
            weights['completeness'] * self.completeness_score +
            weights['consistency'] * self.consistency_score +
            weights['validity'] * self.validity_score +
            weights['temporal'] * self.temporal_resolution_score
        )
        
        # Assign letter grade
        if self.overall_quality_score >= 0.9:
            self.quality_grade = 'A'
        elif self.overall_quality_score >= 0.8:
            self.quality_grade = 'B'
        elif self.overall_quality_score >= 0.7:
            self.quality_grade = 'C'
        elif self.overall_quality_score >= 0.6:
            self.quality_grade = 'D'
        else:
            self.quality_grade = 'F'
    
    def __str__(self):
        return f"Quality Metrics for {self.time_series_data.sequence_id}: {self.quality_grade} ({self.overall_quality_score:.2f})"
    
    class Meta:
        verbose_name = "Data Quality Metrics"
        verbose_name_plural = "Data Quality Metrics"


# ===== LSTM TIME SERIES DATA STRUCTURES COMPLETE =====
# Comprehensive models for machine learning training, validation, and quality control
