from django.contrib import admin
from .models import (
    Stope, StopeProfile, TimeSeriesUpload, TimeSeriesData, 
    Prediction, PredictionFeedback, FuturePrediction, 
    PredictionAlert, ModelPerformanceMetrics
)

# Enhanced admin registrations
@admin.register(Stope)
class StopeAdmin(admin.ModelAdmin):
    list_display = ['stope_name', 'rqd', 'depth', 'rock_type', 'support_type', 'support_installed', 'created_at']
    list_filter = ['rock_type', 'support_type', 'support_installed', 'direction']
    search_fields = ['stope_name', 'rock_type']
    ordering = ['-created_at']

@admin.register(StopeProfile)
class StopeProfileAdmin(admin.ModelAdmin):
    list_display = ['stope', 'created_at']
    search_fields = ['stope__stope_name']
    ordering = ['-created_at']

@admin.register(TimeSeriesUpload)
class TimeSeriesUploadAdmin(admin.ModelAdmin):
    list_display = ['stope', 'csv_file', 'uploaded_at']
    list_filter = ['uploaded_at']
    search_fields = ['stope__stope_name']
    ordering = ['-uploaded_at']

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['stope', 'risk_level', 'impact_score', 'created_at']
    list_filter = ['risk_level', 'created_at']
    search_fields = ['stope__stope_name', 'explanation']
    ordering = ['-created_at']

@admin.register(TimeSeriesData)
class TimeSeriesDataAdmin(admin.ModelAdmin):
    list_display = ['stope', 'timestamp', 'vibration_velocity', 'deformation_rate', 'stress', 'temperature', 'created_at']
    list_filter = ['stope', 'timestamp', 'upload_source']
    search_fields = ['stope__stope_name', 'notes']
    readonly_fields = ['created_at']
    ordering = ['-timestamp']
    date_hierarchy = 'timestamp'

@admin.register(PredictionFeedback)
class PredictionFeedbackAdmin(admin.ModelAdmin):
    list_display = ['prediction', 'stope', 'user_feedback', 'created_at']
    list_filter = ['user_feedback', 'created_at']
    search_fields = ['prediction__explanation', 'stope__stope_name', 'corrected_text']
    readonly_fields = ['created_at']
    ordering = ['-created_at']

@admin.register(FuturePrediction)
class FuturePredictionAdmin(admin.ModelAdmin):
    list_display = ['stope', 'prediction_type', 'days_ahead', 'risk_level', 'confidence_score', 'created_at']
    list_filter = ['prediction_type', 'risk_level', 'days_ahead', 'created_at']
    search_fields = ['stope__stope_name', 'explanation']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('stope', 'prediction_type', 'days_ahead', 'prediction_for_date')
        }),
        ('Risk Assessment', {
            'fields': ('risk_level', 'confidence_score', 'risk_probability')
        }),
        ('Details', {
            'fields': ('explanation', 'recommended_actions', 'contributing_factors')
        }),
        ('Metadata', {
            'fields': ('model_version',),
            'classes': ('collapse',)
        })
    )

@admin.register(PredictionAlert)
class PredictionAlertAdmin(admin.ModelAdmin):
    list_display = ['stope', 'alert_type', 'severity', 'title', 'is_active', 'acknowledged_at', 'created_at']
    list_filter = ['alert_type', 'severity', 'is_active', 'created_at']
    search_fields = ['stope__stope_name', 'title', 'message']
    ordering = ['-created_at']
    
    actions = ['acknowledge_alerts', 'resolve_alerts']
    
    def acknowledge_alerts(self, request, queryset):
        from django.utils import timezone
        updated = queryset.update(acknowledged_at=timezone.now())
        self.message_user(request, f'{updated} alerts acknowledged.')
    acknowledge_alerts.short_description = "Acknowledge selected alerts"
    
    def resolve_alerts(self, request, queryset):
        from django.utils import timezone
        updated = queryset.update(resolved_at=timezone.now(), is_active=False)
        self.message_user(request, f'{updated} alerts resolved.')
    resolve_alerts.short_description = "Resolve selected alerts"

@admin.register(ModelPerformanceMetrics)
class ModelPerformanceMetricsAdmin(admin.ModelAdmin):
    list_display = ['model_version', 'accuracy', 'precision', 'recall', 'f1_score', 'prediction_horizon_days', 'created_at']
    list_filter = ['model_version', 'prediction_horizon_days', 'created_at']
    ordering = ['-created_at']
