from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator
from .models import Stope, OperationalEvent, MonitoringData, ImpactScore, ImpactFactor

class StopeForm(forms.ModelForm):
    """Form for creating and editing stopes"""
    
    class Meta:
        model = Stope
        fields = [
            'stope_name', 'rock_type', 'rqd', 'hr', 'depth', 'dip', 
            'direction', 'undercut_width', 'support_type', 'support_density', 
            'support_installed', 'mining_method', 'excavation_started',
            'excavation_completed', 'is_active'
        ]
        widgets = {
            'stope_name': forms.TextInput(attrs={'class': 'form-control'}),
            'rock_type': forms.Select(attrs={'class': 'form-control'}),
            'rqd': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'hr': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'depth': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'dip': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'direction': forms.Select(attrs={'class': 'form-control'}),
            'undercut_width': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'support_type': forms.Select(attrs={'class': 'form-control'}),
            'support_density': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'mining_method': forms.Select(attrs={'class': 'form-control'}),
            'excavation_started': forms.DateTimeInput(attrs={'class': 'form-control', 'type': 'datetime-local'}),
            'excavation_completed': forms.DateTimeInput(attrs={'class': 'form-control', 'type': 'datetime-local'}),
            'support_installed': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class OperationalEventForm(forms.ModelForm):
    """Form for recording operational events that impact stability"""
    
    class Meta:
        model = OperationalEvent
        fields = [
            'stope', 'event_type', 'severity', 'description', 
            'equipment_involved', 'operator_crew', 'duration_hours',
            'proximity_to_stope', 'affected_area', 'safety_measures'
        ]
        widgets = {
            'stope': forms.Select(attrs={'class': 'form-control'}),
            'event_type': forms.Select(attrs={'class': 'form-control'}),
            'severity': forms.Select(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'equipment_involved': forms.TextInput(attrs={'class': 'form-control'}),
            'operator_crew': forms.TextInput(attrs={'class': 'form-control'}),
            'duration_hours': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'proximity_to_stope': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'affected_area': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'safety_measures': forms.Textarea(attrs={'class': 'form-control', 'rows': 2}),
        }

class MonitoringDataForm(forms.ModelForm):
    """Form for adding monitoring sensor data"""
    
    class Meta:
        model = MonitoringData
        fields = [
            'stope', 'sensor_type', 'value', 'unit', 'sensor_id',
            'confidence'
        ]
        widgets = {
            'stope': forms.Select(attrs={'class': 'form-control'}),
            'sensor_type': forms.Select(attrs={'class': 'form-control'}),
            'value': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.001'}),
            'unit': forms.TextInput(attrs={'class': 'form-control'}),
            'sensor_id': forms.TextInput(attrs={'class': 'form-control'}),
            'confidence': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0', 'max': '1'}),
        }

class BulkMonitoringUploadForm(forms.Form):
    """Form for bulk uploading monitoring data via CSV"""
    
    csv_file = forms.FileField(
        label="CSV File",
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv'}),
        help_text="Upload CSV file with columns: stope_name, sensor_type, value, unit, timestamp"
    )
    
    def clean_csv_file(self):
        file = self.cleaned_data['csv_file']
        if not file.name.endswith('.csv'):
            raise forms.ValidationError("File must be a CSV file")
        return file

class ExcelUploadForm(forms.Form):
    """Simple form for uploading Excel files with stope data"""
    
    excel_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.xlsx,.xls',
        }),
        help_text='Upload Excel file with stope data (XLSX, XLS formats supported)'
    )

class ImpactFactorForm(forms.ModelForm):
    """Form for configuring impact factors with validation"""
    
    class Meta:
        model = ImpactFactor
        fields = [
            'event_category', 'severity_level', 'base_impact_weight',
            'duration_multiplier', 'proximity_decay_rate', 'temporal_decay_rate',
            'description', 'validation_source', 'mine_site', 'is_active'
        ]
        widgets = {
            'event_category': forms.Select(attrs={'class': 'form-control'}),
            'severity_level': forms.Select(attrs={'class': 'form-control'}),
            'base_impact_weight': forms.NumberInput(attrs={
                'class': 'form-control', 'step': '0.1', 'min': '0', 'max': '10'
            }),
            'duration_multiplier': forms.NumberInput(attrs={
                'class': 'form-control', 'step': '0.1', 'min': '0.1', 'max': '5'
            }),
            'proximity_decay_rate': forms.NumberInput(attrs={
                'class': 'form-control', 'step': '0.01', 'min': '0.01', 'max': '1'
            }),
            'temporal_decay_rate': forms.NumberInput(attrs={
                'class': 'form-control', 'step': '0.001', 'min': '0.001', 'max': '0.5'
            }),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'validation_source': forms.TextInput(attrs={'class': 'form-control'}),
            'mine_site': forms.TextInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def clean(self):
        """Validate impact factor parameter combinations"""
        cleaned_data = super().clean()
        
        base_weight = cleaned_data.get('base_impact_weight', 0)
        duration_mult = cleaned_data.get('duration_multiplier', 1)
        proximity_decay = cleaned_data.get('proximity_decay_rate', 0.1)
        temporal_decay = cleaned_data.get('temporal_decay_rate', 0.05)
        
        # Validate parameter relationships
        if base_weight > 5.0 and duration_mult > 3.0:
            raise forms.ValidationError(
                "High base impact weight combined with high duration multiplier "
                "may cause excessive impact calculations."
            )
        
        if temporal_decay > 0.2 and base_weight < 1.0:
            raise forms.ValidationError(
                "High temporal decay rate with low base impact may cause "
                "impacts to disappear too quickly."
            )
        
        if proximity_decay > 0.5:
            self.add_error('proximity_decay_rate', 
                          'Proximity decay rate seems very high - impacts may not propagate effectively.')
        
        return cleaned_data

class ImpactFactorBulkUpdateForm(forms.Form):
    """Form for bulk updating impact factors"""
    
    BULK_ACTION_CHOICES = [
        ('multiply_base', 'Multiply Base Weight'),
        ('add_base', 'Add to Base Weight'),
        ('set_duration', 'Set Duration Multiplier'),
        ('calibration_reset', 'Reset Calibration Date'),
    ]
    
    action = forms.ChoiceField(
        choices=BULK_ACTION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    value = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
        help_text="Value to apply (required for numeric operations)"
    )
    
    categories = forms.MultipleChoiceField(
        choices=ImpactFactor.CATEGORY_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=False,
        help_text="Select categories to update (leave empty for all)"
    )
    
    severity_levels = forms.MultipleChoiceField(
        choices=ImpactFactor.SEVERITY_LEVEL_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=False,
        help_text="Select severity levels to update (leave empty for all)"
    )
    
    def clean(self):
        cleaned_data = super().clean()
        action = cleaned_data.get('action')
        value = cleaned_data.get('value')
        
        # Validate that value is provided for numeric operations
        numeric_actions = ['multiply_base', 'add_base', 'set_duration']
        if action in numeric_actions and value is None:
            raise forms.ValidationError(f"Value is required for action: {action}")
        
        return cleaned_data
