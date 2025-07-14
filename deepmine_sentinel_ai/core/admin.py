from django.contrib import admin
from django.utils import timezone
from django.db import transaction
from .models import (
    Stope, MonitoringData, OperationalEvent, ImpactScore, ImpactHistory, ImpactFactor,
    TimeSeriesData, FeatureEngineeringConfig, DataQualityMetrics
)

# ===== OPTIMIZED ADMIN FOR IMPACT-BASED SYSTEM =====

@admin.register(Stope)
class StopeAdmin(admin.ModelAdmin):
    list_display = [
        'stope_name', 'rock_type', 'mining_method', 'baseline_stability_score', 
        'is_active', 'support_installed', 'created_at'
    ]
    list_filter = [
        'rock_type', 'mining_method', 'support_type', 'support_installed', 
        'is_active', 'direction'
    ]
    search_fields = ['stope_name', 'rock_type', 'notes']
    readonly_fields = ['baseline_stability_score', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('stope_name', 'notes')
        }),
        ('Geological Parameters', {
            'fields': ('rqd', 'rock_type', 'depth', 'hr', 'dip', 'direction')
        }),
        ('Mining Operations', {
            'fields': ('mining_method', 'undercut_width', 'is_active', 
                      'excavation_started', 'excavation_completed')
        }),
        ('Support System', {
            'fields': ('support_type', 'support_density', 'support_installed')
        }),
        ('Stability Assessment', {
            'fields': ('baseline_stability_score',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(MonitoringData)
class MonitoringDataAdmin(admin.ModelAdmin):
    list_display = [
        'stope', 'sensor_type', 'value', 'unit', 'timestamp', 
        'is_anomaly', 'confidence'
    ]
    list_filter = ['sensor_type', 'is_anomaly', 'stope', 'timestamp']
    search_fields = ['stope__stope_name', 'sensor_id']
    ordering = ['-timestamp']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Measurement Details', {
            'fields': ('stope', 'sensor_type', 'value', 'unit', 'timestamp')
        }),
        ('Sensor Information', {
            'fields': ('sensor_id',)
        }),
        ('Quality Assessment', {
            'fields': ('is_anomaly', 'confidence')
        }),
    )


@admin.register(OperationalEvent)
class OperationalEventAdmin(admin.ModelAdmin):
    list_display = [
        'stope', 'event_type', 'severity', 'immediate_impact_score', 
        'timestamp', 'verified', 'recorded_by'
    ]
    list_filter = [
        'event_type', 'severity', 'verified', 'triggered_monitoring', 'stope'
    ]
    search_fields = ['stope__stope_name', 'description', 'operator_crew']
    readonly_fields = ['immediate_impact_score', 'created_at']
    ordering = ['-timestamp']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Event Details', {
            'fields': ('stope', 'event_type', 'timestamp', 'description')
        }),
        ('Impact Parameters', {
            'fields': ('severity', 'proximity_to_stope', 'duration_hours', 
                      'affected_area', 'decay_rate')
        }),
        ('Operational Details', {
            'fields': ('operator_crew', 'equipment_involved', 'safety_measures')
        }),
        ('Impact Assessment', {
            'fields': ('immediate_impact_score', 'triggered_monitoring'),
            'classes': ('collapse',)
        }),
        ('Verification', {
            'fields': ('verified', 'verified_by', 'recorded_by')
        }),
        ('Metadata', {
            'fields': ('event_metadata', 'created_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ImpactScore)
class ImpactScoreAdmin(admin.ModelAdmin):
    list_display = [
        'stope', 'current_score', 'risk_level', 'last_updated', 
        'last_calculation_method'
    ]
    list_filter = ['risk_level', 'last_calculation_method']
    search_fields = ['stope__stope_name']
    readonly_fields = [
        'current_score', 'baseline_component', 'operational_component', 
        'temporal_component', 'monitoring_component', 'contributing_factors',
        'last_updated', 'last_significant_change'
    ]
    ordering = ['-current_score', '-last_updated']
    
    fieldsets = (
        ('Current Assessment', {
            'fields': ('stope', 'current_score', 'risk_level')
        }),
        ('Score Components', {
            'fields': ('baseline_component', 'operational_component', 
                      'temporal_component', 'monitoring_component'),
            'classes': ('collapse',)
        }),
        ('LSTM Predictions', {
            'fields': ('predicted_24h', 'predicted_48h', 'predicted_7d', 
                      'predicted_30d', 'prediction_confidence', 'last_prediction_update'),
            'classes': ('collapse',)
        }),
        ('Alert Thresholds', {
            'fields': ('alert_threshold_elevated', 'alert_threshold_high', 
                      'alert_threshold_critical'),
            'classes': ('collapse',)
        }),
        ('Calculation Details', {
            'fields': ('contributing_factors', 'last_calculation_method', 
                      'last_updated', 'last_significant_change'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ImpactHistory)
class ImpactHistoryAdmin(admin.ModelAdmin):
    list_display = [
        'stope', 'change_type', 'score_change', 'change_magnitude', 
        'new_risk_level', 'timestamp', 'calculated_by'
    ]
    list_filter = [
        'change_type', 'change_magnitude', 'calculation_method', 
        'new_risk_level', 'stope'
    ]
    search_fields = ['stope__stope_name', 'change_reason', 'notes']
    readonly_fields = ['score_change', 'change_magnitude', 'timestamp']
    ordering = ['-timestamp']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Change Summary', {
            'fields': ('stope', 'change_type', 'change_magnitude', 'timestamp')
        }),
        ('Score Changes', {
            'fields': ('previous_score', 'new_score', 'score_change')
        }),
        ('Risk Level Changes', {
            'fields': ('previous_risk_level', 'new_risk_level')
        }),
        ('Change Details', {
            'fields': ('change_reason', 'component_changes')
        }),
        ('Related Data', {
            'fields': ('related_operational_event',),
            'classes': ('collapse',)
        }),
        ('Calculation Info', {
            'fields': ('calculation_method', 'calculated_by', 'confidence_level'),
            'classes': ('collapse',)
        }),
        ('Additional Context', {
            'fields': ('external_factors', 'notes'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ImpactFactor)
class ImpactFactorAdmin(admin.ModelAdmin):
    """Enhanced admin interface for configurable impact factors"""
    
    list_display = [
        'event_category', 'severity_level', 'base_impact_weight', 
        'duration_multiplier', 'is_active', 'mine_site', 'last_calibrated'
    ]
    list_filter = [
        'event_category', 'severity_level', 'is_active', 
        'site_specific', 'mine_site', 'last_calibrated'
    ]
    search_fields = ['event_category', 'severity_level', 'description', 'mine_site']
    ordering = ['event_category', 'severity_level']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('event_category', 'severity_level', 'description')
        }),
        ('Impact Parameters', {
            'fields': (
                'base_impact_weight', 'duration_multiplier', 
                'proximity_decay_rate', 'temporal_decay_rate'
            ),
            'description': 'Configure how this factor affects stability calculations'
        }),
        ('Site Configuration', {
            'fields': ('is_active', 'site_specific', 'mine_site'),
            'classes': ('collapse',)
        }),
        ('Validation & Calibration', {
            'fields': (
                'validation_source', 'last_calibrated', 
                'calibrated_by', 'created_by'
            ),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ['created_at', 'updated_at']
    
    def get_queryset(self, request):
        """Optimize queryset for admin list view"""
        return super().get_queryset(request).select_related()
    
    def save_model(self, request, obj, form, change):
        """Auto-populate created_by and calibrated_by fields"""
        if not change:  # Creating new object
            obj.created_by = request.user.get_full_name() or request.user.username
        
        if 'base_impact_weight' in form.changed_data or 'duration_multiplier' in form.changed_data:
            obj.last_calibrated = timezone.now()
            obj.calibrated_by = request.user.get_full_name() or request.user.username
        
        super().save_model(request, obj, form, change)
    
    actions = ['activate_factors', 'deactivate_factors', 'reset_calibration_date']
    
    def activate_factors(self, request, queryset):
        """Bulk activate impact factors"""
        with transaction.atomic():
            updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} impact factors were activated.')
    activate_factors.short_description = "Activate selected impact factors"
    
    def deactivate_factors(self, request, queryset):
        """Bulk deactivate impact factors"""
        with transaction.atomic():
            updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} impact factors were deactivated.')
    deactivate_factors.short_description = "Deactivate selected impact factors"
    
    def reset_calibration_date(self, request, queryset):
        """Reset calibration date to now"""
        from django.utils import timezone
        with transaction.atomic():
            updated = queryset.update(
                last_calibrated=timezone.now(),
                calibrated_by=request.user.get_full_name() or request.user.username
            )
        self.message_user(request, f'Calibration date reset for {updated} impact factors.')
    reset_calibration_date.short_description = "Reset calibration date to now"


# ===== OPTIMIZED ADMIN INTERFACE COMPLETE =====
# Enhanced admin views for revolutionary impact-based prediction system

# ===== LSTM TIME SERIES ADMIN INTERFACES =====

@admin.register(TimeSeriesData)
class TimeSeriesDataAdmin(admin.ModelAdmin):
    list_display = [
        'sequence_id', 'stope', 'sequence_type', 'sequence_length', 
        'feature_count', 'data_quality_score', 'is_valid', 'processing_timestamp'
    ]
    list_filter = [
        'sequence_type', 'feature_set', 'is_valid', 'is_augmented',
        'preprocessing_version', 'stope'
    ]
    search_fields = ['sequence_id', 'stope__stope_name']
    readonly_fields = [
        'sequence_id', 'processing_timestamp', 'validation_errors',
        'feature_names', 'sequence_length'
    ]
    ordering = ['-processing_timestamp']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('sequence_id', 'stope', 'sequence_type', 'processing_timestamp')
        }),
        ('Time Series Configuration', {
            'fields': (
                'start_timestamp', 'end_timestamp', 'sequence_length',
                'sampling_interval', 'feature_set', 'feature_count'
            )
        }),
        ('Data Quality', {
            'fields': (
                'data_quality_score', 'missing_data_percentage', 'anomaly_count',
                'is_valid', 'validation_errors'
            ),
            'classes': ('collapse',)
        }),
        ('LSTM Configuration', {
            'fields': (
                'sequence_overlap', 'is_augmented', 'augmentation_type',
                'preprocessing_version'
            ),
            'classes': ('collapse',)
        }),
        ('Feature Data', {
            'fields': ('feature_names',),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('stope')
    
    actions = ['validate_sequences', 'mark_as_invalid', 'recalculate_quality']
    
    def validate_sequences(self, request, queryset):
        """Validate selected time series sequences"""
        from core.data_preprocessing import DataValidator
        validator = DataValidator()
        
        validated = 0
        errors = 0
        
        for ts in queryset:
            result = validator.validate_time_series_data(ts)
            ts.is_valid = result['is_valid']
            ts.validation_errors = result['errors']
            ts.save()
            
            if result['is_valid']:
                validated += 1
            else:
                errors += 1
        
        self.message_user(
            request, 
            f'Validated {validated} sequences, {errors} with errors.'
        )
    validate_sequences.short_description = "Validate selected sequences"
    
    def mark_as_invalid(self, request, queryset):
        """Mark sequences as invalid"""
        updated = queryset.update(is_valid=False)
        self.message_user(request, f'{updated} sequences marked as invalid.')
    mark_as_invalid.short_description = "Mark as invalid"
    
    def recalculate_quality(self, request, queryset):
        """Recalculate data quality scores"""
        updated = 0
        for ts in queryset:
            # Simple quality recalculation
            quality_score = 1.0 - (ts.missing_data_percentage / 100.0)
            quality_score -= min(0.5, ts.anomaly_count / ts.sequence_length)
            ts.data_quality_score = max(0.0, quality_score)
            ts.save()
            updated += 1
        
        self.message_user(request, f'Recalculated quality for {updated} sequences.')
    recalculate_quality.short_description = "Recalculate quality scores"


@admin.register(FeatureEngineeringConfig)
class FeatureEngineeringConfigAdmin(admin.ModelAdmin):
    list_display = [
        'config_name', 'version', 'normalization_method', 
        'include_event_features', 'is_active', 'created_at'
    ]
    list_filter = [
        'normalization_method', 'include_event_features', 'include_fft_features',
        'is_active', 'outlier_detection_method'
    ]
    search_fields = ['config_name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Configuration', {
            'fields': ('config_name', 'version', 'description', 'is_active')
        }),
        ('Feature Selection', {
            'fields': (
                'enabled_sensor_types', 'enabled_feature_types',
                'include_event_features', 'include_fft_features'
            )
        }),
        ('Temporal Aggregation', {
            'fields': ('window_sizes', 'aggregation_functions'),
            'description': 'Configure time window aggregations'
        }),
        ('Frequency Analysis', {
            'fields': ('fft_frequency_bands',),
            'classes': ('collapse',)
        }),
        ('Event Integration', {
            'fields': ('event_decay_factor',),
            'classes': ('collapse',)
        }),
        ('Data Processing', {
            'fields': (
                'normalization_method', 'outlier_detection_method', 
                'outlier_threshold'
            ),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['duplicate_config', 'activate_configs', 'deactivate_configs']
    
    def duplicate_config(self, request, queryset):
        """Create duplicates of selected configurations"""
        duplicated = 0
        for config in queryset:
            new_config = config
            new_config.pk = None
            new_config.config_name = f"{config.config_name}_copy"
            new_config.is_active = False
            new_config.save()
            duplicated += 1
        
        self.message_user(request, f'Duplicated {duplicated} configurations.')
    duplicate_config.short_description = "Duplicate configurations"
    
    def activate_configs(self, request, queryset):
        """Activate selected configurations"""
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} configurations activated.')
    activate_configs.short_description = "Activate configurations"
    
    def deactivate_configs(self, request, queryset):
        """Deactivate selected configurations"""
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} configurations deactivated.')
    deactivate_configs.short_description = "Deactivate configurations"


@admin.register(DataQualityMetrics)
class DataQualityMetricsAdmin(admin.ModelAdmin):
    list_display = [
        'time_series_sequence_id', 'quality_grade', 'overall_quality_score',
        'completeness_score', 'consistency_score', 'validity_score',
        'analysis_timestamp'
    ]
    list_filter = [
        'quality_grade', 'analysis_version', 'analysis_timestamp'
    ]
    search_fields = ['time_series_data__sequence_id', 'time_series_data__stope__stope_name']
    readonly_fields = [
        'overall_quality_score', 'quality_grade', 'analysis_timestamp',
        'time_series_sequence_id'
    ]
    ordering = ['-analysis_timestamp']
    
    fieldsets = (
        ('Associated Data', {
            'fields': ('time_series_sequence_id', 'analysis_timestamp', 'analysis_version')
        }),
        ('Quality Scores', {
            'fields': (
                'overall_quality_score', 'quality_grade',
                'completeness_score', 'consistency_score', 
                'validity_score', 'temporal_resolution_score'
            )
        }),
        ('Detailed Metrics', {
            'fields': (
                'outlier_count', 'outlier_percentage',
                'invalid_readings_count', 'timestamp_irregularities'
            ),
            'classes': ('collapse',)
        }),
        ('Quality Issues', {
            'fields': (
                'missing_sensor_types', 'missing_data_gaps',
                'sensor_failure_events'
            ),
            'classes': ('collapse',)
        }),
    )
    
    def time_series_sequence_id(self, obj):
        return obj.time_series_data.sequence_id
    time_series_sequence_id.short_description = 'Sequence ID'
    time_series_sequence_id.admin_order_field = 'time_series_data__sequence_id'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('time_series_data__stope')
    
    actions = ['recalculate_quality_grades']
    
    def recalculate_quality_grades(self, request, queryset):
        """Recalculate quality grades for selected metrics"""
        updated = 0
        for metrics in queryset:
            metrics.calculate_overall_quality()
            metrics.save()
            updated += 1
        
        self.message_user(request, f'Recalculated quality grades for {updated} metrics.')
    recalculate_quality_grades.short_description = "Recalculate quality grades"


# ===== LSTM TIME SERIES ADMIN INTERFACES COMPLETE =====
