from django import forms
from .models import Stope, TimeSeriesUpload

class StopeForm(forms.ModelForm):
    class Meta:
        model = Stope
        fields = [
            'stope_name', 'rqd', 'hr', 'depth', 'dip', 'direction',
            'undercut_wdt', 'rock_type', 'support_type', 
            'support_density', 'support_installed'
        ]
        widgets = {
            'stope_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., North Wing Level 5'}),
            'rqd': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '0', 'max': '100'}),
            'hr': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '0'}),
            'depth': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '0'}),
            'dip': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '0', 'max': '90'}),
            'direction': forms.Select(attrs={'class': 'form-control'}),
            'undercut_wdt': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '0'}),
            'rock_type': forms.Select(attrs={'class': 'form-control'}),
            'support_type': forms.Select(attrs={'class': 'form-control'}),
            'support_density': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '0'}),
            'support_installed': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add empty choice for required dropdowns
        self.fields['rock_type'].empty_label = "Select rock type"
        self.fields['direction'].empty_label = "Select direction"
        self.fields['support_type'].empty_label = "Select support type"
        
        # Make support_density not required by default (will be validated conditionally)
        self.fields['support_density'].required = False

    def clean(self):
        cleaned_data = super().clean()
        support_type = cleaned_data.get('support_type')
        support_density = cleaned_data.get('support_density')
        support_installed = cleaned_data.get('support_installed')
        
        # Conditional validation based on support type
        if support_type and support_type != 'None':
            # If support type is selected (not None), support density is required
            if not support_density:
                self.add_error('support_density', 'Support density is required when a support type is selected.')
            
            # Auto-set support_installed to True when support type is selected
            if not support_installed:
                cleaned_data['support_installed'] = True
                
        else:
            # If support type is None, ensure support_installed is False and clear support_density
            if support_installed:
                cleaned_data['support_installed'] = False
            
            # Clear support density if it was provided but support type is None
            if support_density:
                cleaned_data['support_density'] = 0.0
        
        return cleaned_data
    
    def clean_stope_name(self):
        stope_name = self.cleaned_data.get('stope_name')
        if stope_name:
            # Check if stope name already exists (excluding current instance for updates)
            existing_stope = Stope.objects.filter(stope_name=stope_name)
            if self.instance and self.instance.pk:
                # For updates, exclude the current instance
                existing_stope = existing_stope.exclude(pk=self.instance.pk)
            
            if existing_stope.exists():
                raise forms.ValidationError(f'A stope with the name "{stope_name}" already exists. Please choose a different name.')
        
        return stope_name


class TimeSeriesUploadForm(forms.ModelForm):
    """Form for uploading time series data via Excel/CSV file"""
    class Meta:
        model = TimeSeriesUpload
        fields = ['csv_file']
        widgets = {
            'csv_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.xlsx,.xls',
                'help_text': 'Upload CSV or Excel file with time series data'
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['csv_file'].help_text = 'Supported formats: CSV, XLSX, XLS'


class TimeSeriesEntryForm(forms.Form):
    """Form for manual time series data entry"""
    timestamp = forms.DateTimeField(
        widget=forms.DateTimeInput(attrs={
            'class': 'form-control',
            'type': 'datetime-local'
        }),
        help_text='Date and time of measurement'
    )
    
    # Common mining time series parameters
    vibration_velocity = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.001',
            'placeholder': 'mm/s'
        }),
        help_text='Peak particle velocity (mm/s)'
    )
    
    deformation_rate = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': 'mm/day'
        }),
        help_text='Deformation rate measurement (mm/day)'
    )
    
    stress = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': 'MPa'
        }),
        help_text='Stress measurement (MPa)'
    )
    
    temperature = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': '°C'
        }),
        help_text='Temperature (°C)'
    )
    
    humidity = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0',
            'max': '100',
            'placeholder': '%'
        }),
        help_text='Humidity (%)'
    )
    
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Additional notes or observations...'
        }),
        help_text='Optional notes about this measurement'
    )
