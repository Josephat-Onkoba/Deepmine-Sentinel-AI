from django.db import models
from django.utils import timezone
from datetime import timedelta

class Stope(models.Model):
    # Choice definitions
    ROCK_TYPE_CHOICES = [
        ('Granite', 'Granite'),
        ('Basalt', 'Basalt'),
        ('Obsidian', 'Obsidian'),
        ('Shale', 'Shale'),
        ('Marble', 'Marble'),
        ('Slate', 'Slate'),
        ('Gneiss', 'Gneiss'),
        ('Schist', 'Schist'),
        ('Quartzite', 'Quartzite'),
        ('Limestone', 'Limestone'),
        ('Sandstone', 'Sandstone'),
    ]
    
    DIRECTION_CHOICES = [
        ('North', 'North'),
        ('South', 'South'),
        ('East', 'East'),
        ('West', 'West'),
        ('Northeast', 'Northeast'),
        ('Northwest', 'Northwest'),
        ('Southeast', 'Southeast'),
        ('Southwest', 'Southwest'),
    ]
    
    SUPPORT_TYPE_CHOICES = [
        ('None', 'None'),
        ('Rock Bolts', 'Rock Bolts'),
        ('Mesh', 'Mesh'),
        ('Shotcrete', 'Shotcrete'),
        ('Timber', 'Timber'),
        ('Cable Bolts', 'Cable Bolts'),
        ('Steel Sets', 'Steel Sets'),
    ]

    stope_name = models.CharField(max_length=100, unique=True, help_text="Unique name for the stope (e.g., 'North Wing Level 5', 'Main Stope A')")
    rqd = models.FloatField(help_text="Rock Quality Designation (%)")
    hr = models.FloatField(help_text="Hydraulic Radius")
    depth = models.FloatField(help_text="Depth below surface (m)")
    dip = models.FloatField(help_text="Dip angle (degrees)")
    direction = models.CharField(max_length=50, choices=DIRECTION_CHOICES, help_text="Direction (e.g., North, South)")
    undercut_wdt = models.FloatField(help_text="Undercut width (m)")
    rock_type = models.CharField(max_length=50, choices=ROCK_TYPE_CHOICES, help_text="Rock type")
    support_type = models.CharField(max_length=50, choices=SUPPORT_TYPE_CHOICES, help_text="Support type")
    support_density = models.FloatField(help_text="Support density")
    support_installed = models.BooleanField(default=False, help_text="Is support installed?")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Stope {self.stope_name}"

    class Meta:
        ordering = ['-created_at']

class StopeProfile(models.Model):
    stope = models.OneToOneField(Stope, on_delete=models.CASCADE, related_name='profile')
    summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Profile for {self.stope.stope_name}"

class TimeSeriesUpload(models.Model):
    stope = models.ForeignKey(Stope, on_delete=models.CASCADE, related_name='timeseries')
    csv_file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Timeseries for {self.stope.stope_name} - {self.uploaded_at}"

class TimeSeriesData(models.Model):
    """
    Model to store individual time series data points for monitoring
    """
    stope = models.ForeignKey(Stope, on_delete=models.CASCADE, related_name='timeseries_data')
    timestamp = models.DateTimeField(help_text="Date and time of measurement")
    
    # Common mining monitoring parameters
    vibration_velocity = models.FloatField(
        null=True, blank=True,
        help_text="Peak particle velocity (mm/s)"
    )
    deformation_rate = models.FloatField(
        null=True, blank=True,
        help_text="Deformation rate measurement (mm/day)"
    )
    stress = models.FloatField(
        null=True, blank=True,
        help_text="Stress measurement (MPa)"
    )
    temperature = models.FloatField(
        null=True, blank=True,
        help_text="Temperature (°C)"
    )
    humidity = models.FloatField(
        null=True, blank=True,
        help_text="Humidity (%)"
    )
    notes = models.TextField(
        blank=True,
        help_text="Additional notes or observations"
    )
    
    # Metadata
    upload_source = models.ForeignKey(
        TimeSeriesUpload, 
        on_delete=models.SET_NULL, 
        null=True, blank=True,
        related_name='data_points',
        help_text="Reference to upload batch if data came from file"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Data for {self.stope.stope_name} at {self.timestamp}"
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Time Series Data Point"
        verbose_name_plural = "Time Series Data Points"
        unique_together = ['stope', 'timestamp']  # Prevent duplicate timestamps for same stope

class Prediction(models.Model):
    stope = models.ForeignKey(Stope, on_delete=models.CASCADE, related_name='predictions')
    risk_level = models.CharField(max_length=20)
    impact_score = models.FloatField()
    explanation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.stope.stope_name} - {self.risk_level}"

    class Meta:
        ordering = ['-created_at']

class PredictionFeedback(models.Model):
    """
    Model to collect user feedback on predictions for continuous improvement
    """
    prediction = models.ForeignKey(
        Prediction, 
        on_delete=models.CASCADE, 
        related_name='feedbacks',
        help_text="The prediction this feedback relates to"
    )
    stope = models.ForeignKey(
        Stope, 
        on_delete=models.CASCADE, 
        related_name='prediction_feedbacks',
        help_text="The stope this feedback relates to"
    )
    
    user_feedback = models.BooleanField(
        help_text="True = helpful/accurate, False = not helpful/inaccurate"
    )
    corrected_text = models.TextField(
        blank=True, 
        null=True,
        help_text="Optional user-provided correction or additional information"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        feedback_type = "Positive" if self.user_feedback else "Negative"
        return f"{feedback_type} feedback for {self.prediction} on {self.stope.stope_name}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Prediction Feedback"
        verbose_name_plural = "Prediction Feedbacks"

# Add new models for future predictions
class FuturePrediction(models.Model):
    """
    Model to store future risk predictions with temporal forecasting
    """
    PREDICTION_TYPE_CHOICES = [
        ('current', 'Current Status'),
        ('short_term', 'Short Term (1-3 days)'),
        ('medium_term', 'Medium Term (1-2 weeks)'),
        ('long_term', 'Long Term (1+ months)'),
    ]
    
    RISK_LEVEL_CHOICES = [
        ('stable', 'Stable'),
        ('slight_elevated', 'Slightly Elevated'),
        ('elevated', 'Elevated'),
        ('high', 'High Risk'),
        ('critical', 'Critical'),
        ('unstable', 'Unstable'),
    ]
    
    stope = models.ForeignKey(Stope, on_delete=models.CASCADE, related_name='future_predictions')
    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPE_CHOICES, default='current')
    
    # Temporal information
    prediction_for_date = models.DateTimeField(
        help_text="Date/time this prediction is for"
    )
    days_ahead = models.IntegerField(
        default=0,
        help_text="How many days ahead this prediction is for"
    )
    
    # Risk assessment
    risk_level = models.CharField(max_length=20, choices=RISK_LEVEL_CHOICES)
    confidence_score = models.FloatField(
        help_text="Model confidence in this prediction (0.0-1.0)"
    )
    risk_probability = models.FloatField(
        help_text="Probability of risk occurrence (0.0-1.0)"
    )
    
    # Risk factors that contributed to this prediction
    contributing_factors = models.JSONField(
        default=dict,
        help_text="Factors that led to this prediction"
    )
    
    # Detailed explanation
    explanation = models.TextField(help_text="Detailed explanation of the prediction")
    recommended_actions = models.TextField(
        blank=True,
        help_text="Recommended preventive or corrective actions"
    )
    
    # Model metadata
    model_version = models.CharField(max_length=50, default="v1.0")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Future Prediction for {self.stope.stope_name} - {self.risk_level} ({self.days_ahead} days ahead)"
    
    def is_current_prediction(self):
        return self.prediction_type == 'current'
    
    def is_future_prediction(self):
        return self.prediction_type != 'current'
    
    @property
    def risk_trend(self):
        """Calculate risk trend compared to current status"""
        current_pred = FuturePrediction.objects.filter(
            stope=self.stope,
            prediction_type='current'
        ).first()
        
        if not current_pred:
            return "unknown"
        
        risk_levels_order = ['stable', 'slight_elevated', 'elevated', 'high', 'critical', 'unstable']
        current_idx = risk_levels_order.index(current_pred.risk_level)
        future_idx = risk_levels_order.index(self.risk_level)
        
        if future_idx > current_idx:
            return "increasing"
        elif future_idx < current_idx:
            return "decreasing"
        else:
            return "stable"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Future Prediction"
        verbose_name_plural = "Future Predictions"
        indexes = [
            models.Index(fields=['stope', 'prediction_for_date']),
            models.Index(fields=['prediction_type', 'created_at']),
        ]

class PredictionAlert(models.Model):
    """
    Model to store alerts generated from predictions
    """
    ALERT_TYPE_CHOICES = [
        ('risk_increase', 'Risk Increase'),
        ('threshold_exceeded', 'Threshold Exceeded'),
        ('pattern_anomaly', 'Pattern Anomaly'),
        ('maintenance_due', 'Maintenance Due'),
    ]
    
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    stope = models.ForeignKey(Stope, on_delete=models.CASCADE, related_name='alerts')
    future_prediction = models.ForeignKey(
        FuturePrediction, 
        on_delete=models.CASCADE, 
        related_name='alerts'
    )
    
    alert_type = models.CharField(max_length=30, choices=ALERT_TYPE_CHOICES)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    
    title = models.CharField(max_length=200)
    message = models.TextField()
    
    # Alert status
    is_active = models.BooleanField(default=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.severity.upper()} Alert: {self.title} - {self.stope.stope_name}"
    
    def acknowledge(self):
        self.acknowledged_at = timezone.now()
        self.save()
    
    def resolve(self):
        self.resolved_at = timezone.now()
        self.is_active = False
        self.save()
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Prediction Alert"
        verbose_name_plural = "Prediction Alerts"

class ModelPerformanceMetrics(models.Model):
    """
    Track model performance over time for continuous improvement
    """
    model_version = models.CharField(max_length=50)
    
    # Performance metrics
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    
    # Temporal performance
    prediction_horizon_days = models.IntegerField(
        help_text="How many days ahead the model was predicting"
    )
    
    # Training information
    training_data_count = models.IntegerField()
    validation_data_count = models.IntegerField()
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"Model {self.model_version} Performance - Accuracy: {self.accuracy:.3f}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Model Performance Metrics"
        verbose_name_plural = "Model Performance Metrics"
