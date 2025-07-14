from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.contrib import messages
from django.core.paginator import Paginator
from django.db import IntegrityError
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import Stope, MonitoringData, OperationalEvent, ImpactScore, ImpactHistory
from .forms import StopeForm
import logging
import json

logger = logging.getLogger(__name__)

def home(request):
    """Enhanced dashboard view showing impact-based statistics"""
    total_stopes = Stope.objects.count()
    active_stopes = Stope.objects.filter(is_active=True).count()
    recent_stopes = Stope.objects.order_by('-created_at')[:5]
    
    # Get risk distribution from impact scores
    risk_distribution = {}
    if ImpactScore.objects.exists():
        risk_distribution = {
            'stable': ImpactScore.objects.filter(risk_level='stable').count(),
            'elevated': ImpactScore.objects.filter(risk_level='elevated').count(),
            'high_risk': ImpactScore.objects.filter(risk_level='high_risk').count(),
            'critical': ImpactScore.objects.filter(risk_level='critical').count(),
            'emergency': ImpactScore.objects.filter(risk_level='emergency').count(),
        }
        high_risk_count = risk_distribution.get('high_risk', 0) + risk_distribution.get('critical', 0) + risk_distribution.get('emergency', 0)
    else:
        high_risk_count = 0
    
    # Get recent operational events
    recent_events = OperationalEvent.objects.select_related('stope').order_by('-timestamp')[:5]
    
    context = {
        'total_stopes': total_stopes,
        'active_stopes': active_stopes,
        'recent_stopes': recent_stopes,
        'high_risk_count': high_risk_count,
        'risk_distribution': risk_distribution,
        'recent_events': recent_events,
    }
    return render(request, 'core/base.html', context)

# List all stopes
def stope_list(request):
    """List all stopes with pagination"""
    stopes = Stope.objects.all().order_by('-created_at')
    paginator = Paginator(stopes, 10)  # Show 10 stopes per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'stope/stope_list.html', {'page_obj': page_obj})


# 1. Create stope via manual form
def stope_create(request):
    if request.method == 'POST':
        form = StopeForm(request.POST)
        if form.is_valid():
            try:
                # Simply save the stope - the signal handler will automatically create the profile
                stope = form.save()
                messages.success(request, f'Stope {stope.stope_name} created successfully!')
                return redirect('stope_detail', pk=stope.pk)
                    
            except IntegrityError as e:
                logger.error(f"IntegrityError creating stope: {e}")
                error_message = str(e).lower()
                
                if 'stope_name' in error_message or 'unique constraint failed: core_stope.stope_name' in error_message:
                    messages.error(request, f'A stope with the name "{form.cleaned_data.get("stope_name", "")}" already exists. Please choose a different name.')
                else:
                    messages.error(request, f'Database constraint error: {str(e)}')
            except Exception as e:
                logger.error(f"Error creating stope: {e}")
                messages.error(request, f'An error occurred while creating the stope: {str(e)}')
    else:
        form = StopeForm()
    return render(request, 'stope/stope_form.html', {'form': form})


# 2. View stope + summary
def stope_detail(request, pk):
    """Enhanced stope detail with impact assessment"""
    stope = get_object_or_404(Stope, pk=pk)
    
    # Get impact score for this stope
    try:
        impact_score = stope.impact_score
    except ImpactScore.DoesNotExist:
        impact_score = None
    
    # Get recent monitoring data and operational events
    recent_monitoring = stope.monitoring_data.order_by('-timestamp')[:10]
    recent_events = stope.operational_events.order_by('-timestamp')[:10]
    impact_history = stope.impact_history.order_by('-timestamp')[:10]
    
    # Calculate basic statistics
    monitoring_count = stope.monitoring_data.count()
    events_count = stope.operational_events.count()
    
    context = {
        'stope': stope,
        'impact_score': impact_score,
        'recent_monitoring': recent_monitoring,
        'recent_events': recent_events,
        'impact_history': impact_history,
        'monitoring_count': monitoring_count,
        'events_count': events_count,
    }
    
    return render(request, 'stope/stope_detail.html', context)


def upload_excel(request):
    """Upload Excel file with stope data - Updated for impact-based system"""
    if request.method == 'POST' and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        
        try:
            import pandas as pd
            
            # Read Excel file using pandas
            df = pd.read_excel(excel_file)
            
            # Process rows
            created_count = 0
            for index, row in df.iterrows():
                if pd.isna(row.iloc[0]):  # Skip rows with empty first column
                    continue
                    
                try:
                    # Create stope from Excel row data
                    stope = Stope.objects.create(
                        stope_name=row.iloc[0],
                        rqd=row.iloc[1] if not pd.isna(row.iloc[1]) else 50,
                        hr=row.iloc[2] if not pd.isna(row.iloc[2]) else 5,
                        depth=row.iloc[3] if not pd.isna(row.iloc[3]) else 400,
                        dip=row.iloc[4] if not pd.isna(row.iloc[4]) else 45,
                        direction=row.iloc[5] if not pd.isna(row.iloc[5]) else 'North',
                        undercut_wdt=row.iloc[6] if not pd.isna(row.iloc[6]) else 3,
                        rock_type=row.iloc[7] if not pd.isna(row.iloc[7]) else 'Granite',
                        support_type=row.iloc[8] if not pd.isna(row.iloc[8]) else 'Rock Bolts',
                        support_density=row.iloc[9] if not pd.isna(row.iloc[9]) else 0.5,
                        support_installed=bool(row.iloc[10]) if not pd.isna(row.iloc[10]) else True,
                        stability_status='stable'  # All new stopes start stable
                    )
                    created_count += 1
                    
                except IntegrityError as e:
                    logger.warning(f"Stope {row.iloc[0]} already exists, skipping")
                except Exception as e:
                    logger.error(f"Error creating stope from row {index}: {e}")
            
            if created_count > 0:
                messages.success(request, f'Successfully created {created_count} stopes from Excel file.')
            else:
                messages.warning(request, 'No new stopes were created. Check if stopes already exist.')
                
            return redirect('stope_list')

        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            messages.error(request, f'Error processing Excel file: {str(e)}. Please check the format.')

    return render(request, 'stope/upload_excel.html')


# ===== CLEANED UP VIEWS - ML FUNCTIONS REMOVED =====
# Ready for new impact-based implementation
