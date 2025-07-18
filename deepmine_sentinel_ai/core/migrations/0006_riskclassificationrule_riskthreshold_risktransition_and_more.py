# Generated by Django 5.2.4 on 2025-07-16 07:48

import datetime
import django.core.validators
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0005_auto_20250714_1232'),
    ]

    operations = [
        migrations.CreateModel(
            name='RiskClassificationRule',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='Descriptive name for this classification rule', max_length=100)),
                ('description', models.TextField(help_text='Detailed description of what this rule evaluates')),
                ('target_risk_level', models.CharField(choices=[('stable', 'Stable'), ('elevated', 'Elevated Risk'), ('high_risk', 'High Risk'), ('critical', 'Critical Risk'), ('emergency', 'Emergency')], help_text='Risk level this rule assigns when conditions are met', max_length=20)),
                ('condition_type', models.CharField(choices=[('and', 'All Conditions Must Be Met (AND)'), ('or', 'Any Condition Must Be Met (OR)'), ('weighted', 'Weighted Score Calculation'), ('sequential', 'Sequential Condition Evaluation')], default='and', help_text='How multiple conditions should be evaluated', max_length=20)),
                ('rule_conditions', models.JSONField(default=list, help_text='List of conditions that must be evaluated')),
                ('is_active', models.BooleanField(default=True, help_text='Whether this rule is currently active')),
                ('priority', models.IntegerField(default=1, help_text='Rule evaluation priority (lower = higher priority)')),
                ('applies_to_rock_types', models.JSONField(blank=True, default=list, help_text='Rock types this rule applies to (empty = all types)')),
                ('applies_to_mining_methods', models.JSONField(blank=True, default=list, help_text='Mining methods this rule applies to (empty = all methods)')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by', models.CharField(default='system', help_text='Who created this rule', max_length=100)),
            ],
            options={
                'verbose_name': 'Risk Classification Rule',
                'verbose_name_plural': 'Risk Classification Rules',
                'ordering': ['priority', 'name'],
            },
        ),
        migrations.CreateModel(
            name='RiskThreshold',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='Descriptive name for this threshold configuration', max_length=100)),
                ('risk_level', models.CharField(choices=[('stable', 'Stable'), ('elevated', 'Elevated Risk'), ('high_risk', 'High Risk'), ('critical', 'Critical Risk'), ('emergency', 'Emergency')], help_text='Risk level this threshold defines', max_length=20)),
                ('threshold_type', models.CharField(choices=[('impact_score', 'Impact Score Threshold'), ('rate_of_change', 'Rate of Change Threshold'), ('cumulative_events', 'Cumulative Events Threshold'), ('time_since_event', 'Time Since Last Event Threshold'), ('proximity_risk', 'Proximity Risk Threshold'), ('composite_risk', 'Composite Risk Threshold')], help_text='Type of threshold measurement', max_length=30)),
                ('minimum_value', models.FloatField(help_text='Minimum value to trigger this risk level')),
                ('maximum_value', models.FloatField(blank=True, help_text='Maximum value for this risk level (optional)', null=True)),
                ('is_active', models.BooleanField(default=True, help_text='Whether this threshold is currently active')),
                ('priority', models.IntegerField(default=1, help_text='Priority order for threshold evaluation (lower = higher priority)')),
                ('requires_confirmation', models.BooleanField(default=False, help_text='Whether risk level changes require manual confirmation')),
                ('minimum_duration', models.DurationField(default=datetime.timedelta(seconds=300), help_text='Minimum time threshold must be exceeded before triggering')),
                ('cooldown_period', models.DurationField(default=datetime.timedelta(seconds=900), help_text='Minimum time between risk level changes')),
                ('applies_to_rock_types', models.JSONField(blank=True, default=list, help_text='Rock types this threshold applies to (empty = all types)')),
                ('applies_to_mining_methods', models.JSONField(blank=True, default=list, help_text='Mining methods this threshold applies to (empty = all methods)')),
                ('depth_range_min', models.FloatField(blank=True, help_text='Minimum depth this threshold applies to (meters)', null=True)),
                ('depth_range_max', models.FloatField(blank=True, help_text='Maximum depth this threshold applies to (meters)', null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by', models.CharField(default='system', help_text='Who created this threshold configuration', max_length=100)),
                ('notes', models.TextField(blank=True, help_text='Additional notes about this threshold')),
            ],
            options={
                'verbose_name': 'Risk Threshold',
                'verbose_name_plural': 'Risk Thresholds',
                'ordering': ['priority', 'risk_level', 'minimum_value'],
                'indexes': [models.Index(fields=['risk_level', 'is_active'], name='core_riskth_risk_le_a1ee1c_idx'), models.Index(fields=['threshold_type', 'is_active'], name='core_riskth_thresho_46e7ca_idx'), models.Index(fields=['priority'], name='core_riskth_priorit_6687ee_idx')],
            },
        ),
        migrations.CreateModel(
            name='RiskTransition',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('previous_risk_level', models.CharField(help_text='Risk level before transition', max_length=20)),
                ('new_risk_level', models.CharField(help_text='Risk level after transition', max_length=20)),
                ('trigger_type', models.CharField(choices=[('threshold_exceeded', 'Threshold Exceeded'), ('manual_override', 'Manual Override'), ('system_calculation', 'System Recalculation'), ('event_impact', 'Operational Event Impact'), ('time_decay', 'Natural Time Decay'), ('support_improvement', 'Ground Support Enhancement'), ('emergency_declaration', 'Emergency Declaration'), ('maintenance_completion', 'Maintenance Activity Completion'), ('monitoring_anomaly', 'Monitoring Anomaly Detected'), ('lstm_prediction', 'LSTM Model Prediction')], help_text='What caused this risk level transition', max_length=30)),
                ('trigger_value', models.FloatField(blank=True, help_text='Value that triggered the transition (if applicable)', null=True)),
                ('transition_timestamp', models.DateTimeField(auto_now_add=True, help_text='When this transition occurred')),
                ('duration_in_previous_level', models.DurationField(blank=True, help_text='How long the stope was in the previous risk level', null=True)),
                ('is_confirmed', models.BooleanField(default=False, help_text='Whether this transition has been confirmed by an operator')),
                ('confirmed_by', models.CharField(blank=True, help_text='Who confirmed this transition', max_length=100)),
                ('confirmed_at', models.DateTimeField(blank=True, help_text='When this transition was confirmed', null=True)),
                ('confidence_score', models.FloatField(default=1.0, help_text='Confidence in this transition (0.0-1.0)', validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(1)])),
                ('contributing_factors', models.JSONField(default=dict, help_text='Factors that contributed to this transition')),
                ('impact_assessment', models.TextField(blank=True, help_text='Assessment of the impact and implications of this transition')),
                ('notes', models.TextField(blank=True, help_text='Additional notes about this transition')),
                ('related_impact_score', models.ForeignKey(blank=True, help_text='Impact score at the time of transition', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='risk_transitions', to='core.impactscore')),
                ('related_operational_event', models.ForeignKey(blank=True, help_text='Operational event that triggered this transition (if applicable)', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='risk_transitions', to='core.operationalevent')),
                ('stope', models.ForeignKey(help_text='Stope that experienced the risk level transition', on_delete=django.db.models.deletion.CASCADE, related_name='risk_transitions', to='core.stope')),
                ('threshold_used', models.ForeignKey(blank=True, help_text='Threshold configuration that triggered this transition', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='triggered_transitions', to='core.riskthreshold')),
            ],
            options={
                'verbose_name': 'Risk Transition',
                'verbose_name_plural': 'Risk Transitions',
                'ordering': ['-transition_timestamp'],
            },
        ),
        migrations.CreateModel(
            name='RiskAlert',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('alert_type', models.CharField(choices=[('risk_escalation', 'Risk Level Escalation'), ('threshold_breach', 'Threshold Breach'), ('rapid_change', 'Rapid Risk Change'), ('persistent_risk', 'Persistent High Risk'), ('anomaly_detected', 'Anomaly Detected'), ('prediction_warning', 'LSTM Prediction Warning'), ('system_error', 'System Error'), ('manual_alert', 'Manual Alert')], help_text='Type of alert', max_length=30)),
                ('priority', models.CharField(choices=[('low', 'Low Priority'), ('medium', 'Medium Priority'), ('high', 'High Priority'), ('critical', 'Critical Priority'), ('emergency', 'Emergency Priority')], help_text='Alert priority level', max_length=20)),
                ('status', models.CharField(choices=[('active', 'Active'), ('acknowledged', 'Acknowledged'), ('investigating', 'Under Investigation'), ('resolved', 'Resolved'), ('false_positive', 'False Positive'), ('suppressed', 'Suppressed')], default='active', help_text='Current status of this alert', max_length=20)),
                ('title', models.CharField(help_text='Brief alert title', max_length=200)),
                ('message', models.TextField(help_text='Detailed alert message')),
                ('recommended_actions', models.JSONField(default=list, help_text='List of recommended actions to address this alert')),
                ('created_at', models.DateTimeField(auto_now_add=True, help_text='When this alert was created')),
                ('acknowledged_at', models.DateTimeField(blank=True, help_text='When this alert was acknowledged', null=True)),
                ('resolved_at', models.DateTimeField(blank=True, help_text='When this alert was resolved', null=True)),
                ('acknowledged_by', models.CharField(blank=True, help_text='Who acknowledged this alert', max_length=100)),
                ('resolved_by', models.CharField(blank=True, help_text='Who resolved this alert', max_length=100)),
                ('resolution_notes', models.TextField(blank=True, help_text='Notes about how this alert was resolved')),
                ('escalation_level', models.IntegerField(default=0, help_text='Current escalation level (0 = initial, higher = more escalated)')),
                ('last_escalated_at', models.DateTimeField(blank=True, help_text='When this alert was last escalated', null=True)),
                ('escalated_to', models.CharField(blank=True, help_text='Who this alert was escalated to', max_length=100)),
                ('notification_sent', models.BooleanField(default=False, help_text='Whether notification has been sent for this alert')),
                ('notification_channels', models.JSONField(default=list, help_text='Channels where notifications were sent')),
                ('alert_data', models.JSONField(default=dict, help_text='Additional structured data about this alert')),
                ('stope', models.ForeignKey(help_text='Stope this alert relates to', on_delete=django.db.models.deletion.CASCADE, related_name='risk_alerts', to='core.stope')),
                ('risk_transition', models.ForeignKey(help_text='Risk transition that triggered this alert', on_delete=django.db.models.deletion.CASCADE, related_name='alerts', to='core.risktransition')),
            ],
            options={
                'verbose_name': 'Risk Alert',
                'verbose_name_plural': 'Risk Alerts',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='risktransition',
            index=models.Index(fields=['stope', 'transition_timestamp'], name='core_risktr_stope_i_fddea6_idx'),
        ),
        migrations.AddIndex(
            model_name='risktransition',
            index=models.Index(fields=['new_risk_level', 'transition_timestamp'], name='core_risktr_new_ris_c79346_idx'),
        ),
        migrations.AddIndex(
            model_name='risktransition',
            index=models.Index(fields=['trigger_type', 'transition_timestamp'], name='core_risktr_trigger_506e92_idx'),
        ),
        migrations.AddIndex(
            model_name='risktransition',
            index=models.Index(fields=['is_confirmed'], name='core_risktr_is_conf_9f1e15_idx'),
        ),
        migrations.AddIndex(
            model_name='riskalert',
            index=models.Index(fields=['stope', 'status', 'created_at'], name='core_riskal_stope_i_3f9fa4_idx'),
        ),
        migrations.AddIndex(
            model_name='riskalert',
            index=models.Index(fields=['priority', 'status'], name='core_riskal_priorit_5e67b8_idx'),
        ),
        migrations.AddIndex(
            model_name='riskalert',
            index=models.Index(fields=['alert_type', 'created_at'], name='core_riskal_alert_t_42f965_idx'),
        ),
        migrations.AddIndex(
            model_name='riskalert',
            index=models.Index(fields=['status', 'created_at'], name='core_riskal_status_4acc7a_idx'),
        ),
    ]
