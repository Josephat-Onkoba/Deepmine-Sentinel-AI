from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.contrib import messages
from django.core.paginator import Paginator
from django.db import IntegrityError
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import Stope, StopeProfile, TimeSeriesUpload, TimeSeriesData, Prediction, PredictionFeedback, FuturePrediction, PredictionAlert
from .forms import StopeForm, TimeSeriesUploadForm, TimeSeriesEntryForm
from .utils import profile_stope
from .ml_service import MLPredictionService
import openpyxl
from io import BytesIO
import logging
import json
import os
import numpy as np

logger = logging.getLogger(__name__)

# Home page view
def home(request):
    """Dashboard view showing summary statistics"""
    total_stopes = Stope.objects.count()
    recent_stopes = Stope.objects.order_by('-created_at')[:5]
    
    # Calculate some basic stats
    high_risk_count = 0
    if total_stopes > 0:
        # Count stopes with poor rock quality and large hydraulic radius
        high_risk_count = Stope.objects.filter(rqd__lt=50, hr__gt=9).count()
    
    context = {
        'total_stopes': total_stopes,
        'recent_stopes': recent_stopes,
        'high_risk_count': high_risk_count,
    }
    return render(request, 'core/home.html', context)

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
    stope = get_object_or_404(Stope, pk=pk)
    
    # Get stope profile
    try:
        profile = get_object_or_404(StopeProfile, stope=stope)
        profile_summary = profile.summary
    except:
        profile_summary = "Profile not available."
    
    # Get time series data for predictions
    timeseries_data = None
    has_timeseries = stope.timeseries_data.exists()
    
    if has_timeseries:
        # Convert to pandas DataFrame format for prediction
        import pandas as pd
        timeseries_queryset = stope.timeseries_data.all().order_by('timestamp')
        timeseries_data = pd.DataFrame(list(timeseries_queryset.values(
            'timestamp', 'vibration_velocity', 'deformation_rate', 
            'stress', 'temperature', 'humidity'
        )))
    
    # Enhanced ML prediction using enhanced MLPredictionService
    ml_prediction = None
    future_predictions = None
    ml_prediction_status = None
    alerts = []
    
    if has_timeseries:
        try:
            # Use the enhanced MLPredictionService for predictions
            ml_service = MLPredictionService()
            
            if ml_service.is_model_trained():
                prediction_result = ml_service.predict_stope_stability(stope, save_prediction=False)
                
                if 'error' not in prediction_result:
                    ml_prediction = {
                        'prediction': prediction_result['prediction'],
                        'risk_level': prediction_result['risk_level'],
                        'confidence_score': prediction_result['confidence'],
                        'explanation': prediction_result['explanation'],
                        'model_type': prediction_result['model_info']['model_type'],
                        'stable': prediction_result['prediction'] == 'Stable',
                        'enhanced_mode': prediction_result['model_info'].get('enhanced_mode', False),
                        'future_predictions': prediction_result.get('future_predictions', []),
                        'risk_trend': prediction_result.get('risk_trend', 'stable'),
                        'recommendations': prediction_result.get('recommendations', [])
                    }
                    ml_prediction_status = "available"
                    
                    # Store future predictions for display
                    future_predictions = prediction_result.get('future_predictions', [])
                    
                    # Create a Prediction record with enhanced data
                    try:
                        prediction_obj = Prediction.objects.create(
                            stope=stope,
                            risk_level=prediction_result['risk_level'],
                            impact_score=prediction_result['confidence'],
                            explanation=prediction_result['explanation']
                        )
                        
                        # Save future predictions if available
                        if future_predictions:
                            for future_pred in future_predictions:
                                FuturePrediction.objects.create(
                                    stope=stope,
                                    prediction_type=ml_service._map_horizon_to_type(future_pred['horizon_days']),
                                    prediction_for_date=timezone.now() + timezone.timedelta(days=future_pred['horizon_days']),
                                    days_ahead=future_pred['horizon_days'],
                                    risk_level=future_pred['predicted_risk_level'],
                                    confidence_score=future_pred['confidence'],
                                    risk_probability=max(future_pred['risk_probabilities'].values()),
                                    contributing_factors=future_pred.get('risk_probabilities', {}),
                                    explanation=f"Enhanced ML prediction for {future_pred['horizon_days']} days ahead",
                                    model_version=prediction_result['model_info'].get('version', 'v1.0')
                                )
                                
                    except Exception as e:
                        logger.error(f"Error saving enhanced prediction to database: {e}")
                        
                else:
                    ml_prediction_status = "error"
                    logger.warning(f"Enhanced ML prediction error for stope {pk}: {prediction_result['error']}")
            else:
                ml_prediction_status = "model_not_trained"
                
        except Exception as e:
            logger.error(f"Error getting enhanced ML prediction for stope {pk}: {e}")
            ml_prediction_status = "error"
    else:
        ml_prediction_status = "no_timeseries"
    
    # Get existing future predictions for display
    existing_future_predictions = stope.future_predictions.filter(
        created_at__date=timezone.now().date()
    ).order_by('days_ahead')
    
    # Get active alerts
    active_alerts = stope.alerts.filter(is_active=True).order_by('-created_at')
    
    # Get recent predictions and feedback
    recent_predictions = stope.predictions.order_by('-created_at')[:5]
    
    # Handle form submissions
    if request.method == 'POST':
        if 'add_timeseries' in request.POST:
            # Handle manual time series entry
            timeseries_form = TimeSeriesEntryForm(request.POST)
            if timeseries_form.is_valid():
                # Create time series data point
                TimeSeriesData.objects.create(
                    stope=stope,
                    timestamp=timeseries_form.cleaned_data['timestamp'],
                    vibration_velocity=timeseries_form.cleaned_data.get('vibration_velocity'),
                    deformation_rate=timeseries_form.cleaned_data.get('deformation_rate'),
                    stress=timeseries_form.cleaned_data.get('stress'),
                    temperature=timeseries_form.cleaned_data.get('temperature'),
                    humidity=timeseries_form.cleaned_data.get('humidity'),
                    notes=timeseries_form.cleaned_data.get('notes', '')
                )
                
                # 🎯 TRIGGER ENHANCED ML PREDICTION after adding time series data
                try:
                    ml_service = MLPredictionService()
                    
                    if ml_service.is_model_trained():
                        prediction_result = ml_service.predict_stope_stability(stope, save_prediction=True)
                        
                        if 'error' not in prediction_result:
                            # Enhanced success message with current prediction and future outlook
                            future_info = ""
                            if prediction_result.get('future_predictions'):
                                next_risk = prediction_result['future_predictions'][0]
                                future_info = f" | Next 24h: {next_risk['predicted_risk_level'].title()}"
                            
                            messages.success(request, 
                                f'Time series data added successfully! Current Risk: {prediction_result["risk_level"]} '
                                f'(Confidence: {prediction_result["confidence"]:.1%}){future_info}')
                                
                            # Add trend information
                            if prediction_result.get('risk_trend'):
                                trend_msg = f"Risk trend: {prediction_result['risk_trend']}"
                                if prediction_result['risk_trend'] == 'increasing':
                                    messages.warning(request, f"⚠️ {trend_msg} - Enhanced monitoring recommended")
                                else:
                                    messages.info(request, f"📊 {trend_msg}")
                        else:
                            messages.success(request, 'Time series data added successfully!')
                            messages.warning(request, f'Enhanced ML prediction error: {prediction_result["error"]}')
                    else:
                        messages.success(request, 'Time series data added successfully!')
                        messages.warning(request, 'Enhanced ML model is not trained. Predictions will be available once the model is trained.')
                        
                except Exception as e:
                    logger.error(f"Error generating enhanced ML prediction after time series addition: {e}")
                    messages.success(request, 'Time series data added successfully!')
                    messages.warning(request, 'Could not generate ML prediction at this time.')
                
                return redirect('stope_detail', pk=pk)
            else:
                messages.error(request, 'Please correct the errors in the time series form.')
                
        elif 'upload_file' in request.POST and request.FILES.get('csv_file'):
            # Handle file upload
            upload_form = TimeSeriesUploadForm(request.POST, request.FILES)
            if upload_form.is_valid():
                upload = upload_form.save(commit=False)
                upload.stope = stope
                upload.save()
                
                # TODO: Process the uploaded file and create TimeSeriesData objects
                # For now, just save the file reference
                messages.success(request, 'File uploaded successfully! Processing time series data...')
                messages.info(request, 'Once time series data is processed, ML predictions will be automatically generated.')
                return redirect('stope_detail', pk=pk)
            else:
                messages.error(request, 'Please select a valid file to upload.')
    else:
        timeseries_form = TimeSeriesEntryForm()
        upload_form = TimeSeriesUploadForm()
    
    # Get existing time series data
    timeseries_data_display = stope.timeseries_data.all()[:10]  # Latest 10 records
    timeseries_uploads = stope.timeseries.all()[:5]  # Latest 5 uploads
    
    return render(request, 'stope/stope_detail.html', {
        'stope': stope,
        'profile': profile_summary,
        'timeseries_form': timeseries_form,
        'upload_form': upload_form,
        'timeseries_data': timeseries_data_display,
        'timeseries_uploads': timeseries_uploads,
        'ml_prediction': ml_prediction,
        'future_predictions': future_predictions,  # Enhanced future predictions
        'active_alerts': active_alerts,
        'ml_prediction_status': ml_prediction_status,
        'has_timeseries': has_timeseries,
        'recent_predictions': recent_predictions,
    })


def upload_excel(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        
        # Get valid choices from the model
        from .models import Stope
        VALID_ROCK_TYPES = [choice[0] for choice in Stope.ROCK_TYPE_CHOICES]
        VALID_DIRECTIONS = [choice[0] for choice in Stope.DIRECTION_CHOICES]
        VALID_SUPPORT_TYPES = [choice[0] for choice in Stope.SUPPORT_TYPE_CHOICES]
        
        try:
            wb = openpyxl.load_workbook(BytesIO(excel_file.read()))
            sheet = wb.active

            headers = [cell.value for cell in sheet[1]]
            rows = list(sheet.iter_rows(min_row=2, values_only=True))
            
            # Check if we have the required headers
            required_headers = [
                'stope_name', 'rqd', 'hr', 'depth', 'dip', 'direction',
                'undercut_wdt', 'rock_type', 'support_type', 
                'support_density', 'support_installed'
            ]
            
            missing_headers = [h for h in required_headers if h not in headers]
            if missing_headers:
                messages.error(request, f'Missing required columns: {", ".join(missing_headers)}')
                return render(request, 'stope/upload_excel.html')

            success_count = 0
            error_count = 0

            for row_num, row in enumerate(rows, start=2):
                try:
                    data = dict(zip(headers, row))

                    # Skip empty rows
                    if not data.get('stope_name'):
                        continue

                    # Validate dropdown values
                    direction = str(data['direction']).strip()
                    rock_type = str(data['rock_type']).strip()
                    support_type = str(data['support_type']).strip()
                    
                    if direction not in VALID_DIRECTIONS:
                        logger.warning(f"Row {row_num}: Invalid direction '{direction}'. Valid options: {VALID_DIRECTIONS}")
                        error_count += 1
                        continue
                    
                    if rock_type not in VALID_ROCK_TYPES:
                        logger.warning(f"Row {row_num}: Invalid rock type '{rock_type}'. Valid options: {VALID_ROCK_TYPES}")
                        error_count += 1
                        continue
                    
                    if support_type not in VALID_SUPPORT_TYPES:
                        logger.warning(f"Row {row_num}: Invalid support type '{support_type}'. Valid options: {VALID_SUPPORT_TYPES}")
                        error_count += 1
                        continue

                    # Handle conditional support logic
                    support_density_value = data.get('support_density', '')
                    support_installed_value = data.get('support_installed', '')
                    
                    if support_type == 'None':
                        # When support type is None, set defaults
                        support_density_value = 0.0
                        support_installed_value = False
                    else:
                        # When support type is selected, validate support density
                        try:
                            support_density_value = float(support_density_value) if support_density_value else 0.0
                            if support_density_value <= 0:
                                logger.warning(f"Row {row_num}: Support density must be greater than 0 when support type is '{support_type}'")
                                error_count += 1
                                continue
                        except (ValueError, TypeError):
                            logger.warning(f"Row {row_num}: Invalid support density value when support type is '{support_type}'")
                            error_count += 1
                            continue
                        
                        # Auto-set support_installed to True when support type is not None
                        support_installed_value = True

                    # Check for duplicate stope name before creating
                    if Stope.objects.filter(stope_name=data['stope_name']).exists():
                        logger.warning(f"Row {row_num}: Stope with name '{data['stope_name']}' already exists")
                        error_count += 1
                        continue

                    stope = Stope.objects.create(
                        stope_name=data['stope_name'],
                        rqd=float(data['rqd']),
                        hr=float(data['hr']),
                        depth=float(data['depth']),
                        dip=float(data['dip']),
                        direction=direction,
                        undercut_wdt=float(data['undercut_wdt']),
                        rock_type=rock_type,
                        support_type=support_type,
                        support_density=support_density_value,
                        support_installed=support_installed_value,
                    )

                    features = {
                        'rqd': stope.rqd,
                        'depth': stope.depth,
                        'dip': stope.dip,
                        'direction': stope.direction,
                        'hr': stope.hr,
                        'Undercut_wdt': stope.undercut_wdt,
                        'rock_type': stope.rock_type,
                        'support_type': stope.support_type,
                        'support_density': stope.support_density,
                        'support_installed': stope.support_installed,
                    }

                    summary = profile_stope(features)
                    
                    # Check if profile already exists before creating
                    existing_profile = StopeProfile.objects.filter(stope=stope).first()
                    if existing_profile:
                        logger.warning(f"Profile already exists for stope {stope.stope_name}, updating instead of creating new")
                        existing_profile.summary = summary
                        existing_profile.save()
                    else:
                        StopeProfile.objects.create(stope=stope, summary=summary)
                    
                    success_count += 1

                except (ValueError, TypeError) as e:
                    error_count += 1
                    logger.error(f"Error processing row {row_num} - Data validation error: {e}")
                except IntegrityError as e:
                    error_count += 1
                    if 'stope_name' in str(e) or 'UNIQUE constraint failed' in str(e):
                        logger.error(f"Error processing row {row_num} - Duplicate stope name '{data.get('stope_name', 'Unknown')}': {e}")
                    else:
                        logger.error(f"Error processing row {row_num} - Database constraint error: {e}")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Unexpected error processing row {row_num}: {e}")

            if success_count > 0:
                messages.success(request, f"{success_count} stopes processed successfully.")
            if error_count > 0:
                messages.warning(request, f"{error_count} rows had errors and were skipped.")
                
            return redirect('stope_list')

        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            messages.error(request, 'Error processing Excel file. Please check the format.')

    return render(request, 'stope/upload_excel.html')

# ML Prediction Views

@require_http_methods(["POST"])
def predict_stability(request, pk):
    """Enhanced AJAX endpoint for getting ML stability prediction for a stope"""
    stope = get_object_or_404(Stope, pk=pk)
    
    try:
        ml_service = MLPredictionService()
        
        if not ml_service.is_model_trained():
            return JsonResponse({
                'error': 'Enhanced ML model is not trained or not available. Please ensure the model is trained first.',
                'success': False
            })
        
        prediction_result = ml_service.predict_stope_stability(stope, save_prediction=True)
        
        if 'error' in prediction_result:
            return JsonResponse({
                'error': prediction_result['error'],
                'success': False
            })
        
        # Enhanced response with future predictions
        response_data = {
            'success': True,
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'explanation': prediction_result['explanation'],
            'probabilities': prediction_result.get('probabilities', {}),
            'model_info': prediction_result.get('model_info', {}),
            'enhanced_mode': prediction_result['model_info'].get('enhanced_mode', False)
        }
        
        # Add enhanced features if available
        if 'future_predictions' in prediction_result:
            response_data['future_predictions'] = prediction_result['future_predictions']
        if 'risk_trend' in prediction_result:
            response_data['risk_trend'] = prediction_result['risk_trend']
        if 'recommendations' in prediction_result:
            response_data['recommendations'] = prediction_result['recommendations']
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error in enhanced predict_stability for stope {pk}: {e}")
        return JsonResponse({
            'error': f'An error occurred during enhanced prediction: {str(e)}',
            'success': False
        })


@require_http_methods(["POST"])
def submit_prediction_feedback(request, prediction_id):
    """Submit user feedback on a prediction"""
    prediction = get_object_or_404(Prediction, pk=prediction_id)
    
    try:
        data = json.loads(request.body)
        is_helpful = data.get('is_helpful', False)
        corrected_text = data.get('corrected_text', '').strip()
        
        # Create feedback entry
        feedback = PredictionFeedback.objects.create(
            prediction=prediction,
            stope=prediction.stope,
            user_feedback=is_helpful,
            corrected_text=corrected_text if corrected_text else None
        )
        
        return JsonResponse({
            'success': True,
            'message': 'Thank you for your feedback! This helps improve our model.',
            'feedback_id': feedback.id
        })
        
    except Exception as e:
        logger.error(f"Error submitting feedback for prediction {prediction_id}: {e}")
        return JsonResponse({
            'success': False,
            'error': 'Failed to submit feedback. Please try again.'
        })


def ml_dashboard(request):
    """Enhanced dashboard showing ML model performance and statistics"""
    try:
        ml_service = MLPredictionService()
        
        # Get enhanced model information
        if ml_service.is_model_trained():
            model_info = ml_service.get_model_performance_metrics()
            feature_importance = ml_service.get_feature_importance(top_n=10)
            is_model_trained = True
            
            # Get model health status
            health_status = ml_service.validate_model_health()
        else:
            model_info = {'error': 'Enhanced model is not trained or not available'}
            feature_importance = {}
            is_model_trained = False
            health_status = {'overall_health': 'unhealthy', 'components': {}, 'recommendations': ['Train the enhanced model']}
            
    except Exception as e:
        logger.error(f"Error accessing enhanced ML service: {e}")
        model_info = {'error': f'Error accessing enhanced model: {str(e)}'}
        feature_importance = {}
        is_model_trained = False
        health_status = {'overall_health': 'error', 'error': str(e)}
    
    # Get enhanced prediction statistics
    total_predictions = Prediction.objects.count()
    recent_predictions = Prediction.objects.order_by('-created_at')[:10]
    
    # Get future prediction statistics
    total_future_predictions = FuturePrediction.objects.count()
    recent_future_predictions = FuturePrediction.objects.order_by('-created_at')[:5]
    
    # Get active alerts
    total_alerts = PredictionAlert.objects.count()
    active_alerts = PredictionAlert.objects.filter(is_active=True).count()
    
    # Get feedback statistics
    total_feedback = PredictionFeedback.objects.count()
    positive_feedback = PredictionFeedback.objects.filter(user_feedback=True).count()
    negative_feedback = PredictionFeedback.objects.filter(user_feedback=False).count()
    
    feedback_accuracy = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
    
    # Get risk level distribution
    from django.db.models import Count
    risk_distribution = Prediction.objects.values('risk_level').annotate(count=Count('id'))
    
    context = {
        'model_info': model_info,
        'feature_importance': feature_importance,
        'health_status': health_status,
        'total_predictions': total_predictions,
        'recent_predictions': recent_predictions,
        'total_future_predictions': total_future_predictions,
        'recent_future_predictions': recent_future_predictions,
        'total_alerts': total_alerts,
        'active_alerts': active_alerts,
        'total_feedback': total_feedback,
        'positive_feedback': positive_feedback,
        'negative_feedback': negative_feedback,
        'feedback_accuracy': round(feedback_accuracy, 1),
        'risk_distribution': list(risk_distribution),
        'is_model_trained': is_model_trained,
    }
    
    return render(request, 'core/ml_dashboard.html', context)


def batch_predict(request):
    """Batch prediction for multiple stopes"""
    if request.method == 'POST':
        stope_ids = request.POST.getlist('stope_ids')
        
        if not stope_ids:
            messages.error(request, 'Please select at least one stope for prediction.')
            return redirect('batch_predict')
        
        try:
            ml_service = MLPredictionService()
            
            if not ml_service.is_model_trained():
                messages.error(request, 'Enhanced ML model is not trained or not available. Please ensure the model is trained first.')
                return redirect('batch_predict')
            
            # Convert to integers
            stope_ids = [int(sid) for sid in stope_ids]
            
            # Get enhanced predictions
            results = ml_service.predict_multiple_stopes(stope_ids)
            
            success_count = 0
            error_count = 0
            enhanced_count = 0
            
            for stope_id, result in results.items():
                if 'error' not in result:
                    success_count += 1
                    if result.get('model_info', {}).get('enhanced_mode', False):
                        enhanced_count += 1
                else:
                    error_count += 1
            
            if success_count > 0:
                enhanced_msg = f" ({enhanced_count} enhanced)" if enhanced_count > 0 else ""
                messages.success(request, f'Generated predictions for {success_count} stopes{enhanced_msg}.')
            if error_count > 0:
                messages.warning(request, f'{error_count} predictions failed.')
            
            # Store results in session for display
            request.session['batch_results'] = results
            return redirect('batch_predict_results')
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            messages.error(request, f'Error generating predictions: {str(e)}')
    
    # Get all stopes for selection
    stopes = Stope.objects.all().order_by('stope_name')
    
    return render(request, 'core/batch_predict.html', {'stopes': stopes})


def batch_predict_results(request):
    """Display results of batch prediction"""
    results = request.session.get('batch_results', {})
    
    if not results:
        messages.info(request, 'No batch prediction results to display.')
        return redirect('batch_predict')
    
    # Get stope objects for the results
    stope_ids = list(results.keys())
    stopes = {stope.id: stope for stope in Stope.objects.filter(id__in=stope_ids)}
    
    # Combine results with stope objects
    detailed_results = []
    for stope_id, result in results.items():
        stope = stopes.get(stope_id)
        if stope:
            detailed_results.append({
                'stope': stope,
                'result': result
            })
    
    # Clear results from session
    if 'batch_results' in request.session:
        del request.session['batch_results']
    
    return render(request, 'core/batch_predict_results.html', {
        'results': detailed_results
    })


def temporal_prediction_dashboard(request, pk):
    """
    Enhanced dashboard showing temporal predictions and trends
    """
    stope = get_object_or_404(Stope, pk=pk)
    
    # Get all future predictions for this stope
    future_predictions = stope.future_predictions.order_by('days_ahead', '-created_at')
    
    # Get latest predictions for each horizon
    latest_predictions = {}
    for pred in future_predictions:
        key = f"{pred.days_ahead}_days"
        if key not in latest_predictions:
            latest_predictions[key] = pred
    
    # Get active alerts
    active_alerts = stope.alerts.filter(is_active=True).order_by('-created_at')
    
    # Get historical time series data for trends
    timeseries_data = stope.timeseries_data.all().order_by('-timestamp')[:30]
    
    # Calculate risk trend
    risk_trend = None
    if len(latest_predictions) > 1:
        current_risk = latest_predictions.get('1_days')
        future_risk = latest_predictions.get('7_days')
        
        if current_risk and future_risk:
            risk_levels = ['stable', 'slight_elevated', 'elevated', 'high', 'critical', 'unstable']
            current_idx = risk_levels.index(current_risk.risk_level)
            future_idx = risk_levels.index(future_risk.risk_level)
            
            if future_idx > current_idx:
                risk_trend = "increasing"
            elif future_idx < current_idx:
                risk_trend = "decreasing"
            else:
                risk_trend = "stable"
    
    # Prepare chart data for frontend
    chart_data = {
        'dates': [pred.prediction_for_date.strftime('%Y-%m-%d') for pred in latest_predictions.values()],
        'risk_levels': [pred.risk_level for pred in latest_predictions.values()],
        'confidence_scores': [pred.confidence_score for pred in latest_predictions.values()],
        'risk_probabilities': [pred.risk_probability for pred in latest_predictions.values()]
    }
    
    context = {
        'stope': stope,
        'latest_predictions': latest_predictions,
        'active_alerts': active_alerts,
        'timeseries_data': timeseries_data,
        'risk_trend': risk_trend,
        'chart_data': json.dumps(chart_data),
    }
    
    return render(request, 'stope/temporal_dashboard.html', context)

def acknowledge_alert(request, alert_id):
    """
    Acknowledge a prediction alert
    """
    if request.method == 'POST':
        alert = get_object_or_404(PredictionAlert, id=alert_id)
        alert.acknowledge()
        messages.success(request, f'Alert "{alert.title}" has been acknowledged.')
    
    return redirect('stope_detail', pk=request.POST.get('stope_id', 1))

def resolve_alert(request, alert_id):
    """
    Resolve a prediction alert
    """
    if request.method == 'POST':
        alert = get_object_or_404(PredictionAlert, id=alert_id)
        alert.resolve()
        messages.success(request, f'Alert "{alert.title}" has been resolved.')
    
    return redirect('stope_detail', pk=request.POST.get('stope_id', 1))

@require_http_methods(["POST"])
def trigger_prediction_update(request, pk):
    """
    Manually trigger a prediction update for a stope
    """
    stope = get_object_or_404(Stope, pk=pk)
    
    try:
        # Check if time series data exists
        if not stope.timeseries_data.exists():
            return JsonResponse({
                'success': False,
                'error': 'No time series data available for prediction'
            })
        
        # Use enhanced MLPredictionService for predictions
        ml_service = MLPredictionService()
        
        if not ml_service.is_model_trained():
            return JsonResponse({
                'success': False,
                'error': 'Enhanced ML model is not trained or not available'
            })
        
        # Get enhanced prediction
        prediction_result = ml_service.predict_stope_stability(stope, save_prediction=True)
        
        if 'error' not in prediction_result:
            response_data = {
                'success': True,
                'prediction': prediction_result['prediction'],
                'risk_level': prediction_result['risk_level'],
                'confidence': prediction_result['confidence'],
                'explanation': prediction_result['explanation'],
                'model_info': prediction_result['model_info'],
                'message': 'Enhanced prediction updated successfully'
            }
            
            # Add enhanced features if available
            if 'future_predictions' in prediction_result:
                response_data['future_predictions'] = prediction_result['future_predictions']
            if 'risk_trend' in prediction_result:
                response_data['risk_trend'] = prediction_result['risk_trend']
            if 'recommendations' in prediction_result:
                response_data['recommendations'] = prediction_result['recommendations']
                
            return JsonResponse(response_data)
        else:
            return JsonResponse({
                'success': False,
                'error': prediction_result['error']
            })
            
    except Exception as e:
        logger.error(f"Error updating predictions for stope {pk}: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@require_http_methods(["POST"])
def train_enhanced_model(request):
    """
    Train the enhanced ML model with current data.
    """
    try:
        ml_service = MLPredictionService()
        
        # Check model health first
        health_status = ml_service.validate_model_health()
        
        if health_status['overall_health'] == 'error':
            return JsonResponse({
                'success': False,
                'error': f"Model health check failed: {health_status.get('error', 'Unknown error')}"
            })
        
        if health_status['overall_health'] == 'unhealthy':
            return JsonResponse({
                'success': False,
                'error': 'Model cannot be trained due to missing components',
                'recommendations': health_status.get('recommendations', [])
            })
        
        # Start training
        training_result = ml_service.train_model_with_current_data()
        
        if training_result.get('success', False):
            return JsonResponse({
                'success': True,
                'message': training_result['message'],
                'epochs': training_result.get('epochs', 'N/A'),
                'final_accuracy': training_result.get('final_accuracy', 'N/A')
            })
        else:
            return JsonResponse({
                'success': False,
                'error': training_result.get('error', 'Training failed for unknown reason')
            })
            
    except Exception as e:
        logger.error(f"Error training enhanced model: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Training error: {str(e)}'
        })


def model_health_dashboard(request):
    """
    Dashboard for monitoring enhanced model health and status.
    """
    try:
        ml_service = MLPredictionService()
        
        # Get comprehensive health status
        health_status = ml_service.validate_model_health()
        model_info = ml_service.get_model_performance_metrics() if ml_service.is_model_trained() else None
        
        # Get data file information
        data_info = {
            'static_csv_exists': os.path.exists(ml_service.static_csv_path),
            'timeseries_csv_exists': os.path.exists(ml_service.timeseries_csv_path),
            'models_dir_exists': os.path.exists(ml_service.models_dir)
        }
        
        # Get recent training/prediction activity
        recent_predictions = Prediction.objects.order_by('-created_at')[:5]
        recent_future_predictions = FuturePrediction.objects.order_by('-created_at')[:5]
        
        context = {
            'health_status': health_status,
            'model_info': model_info,
            'data_info': data_info,
            'recent_predictions': recent_predictions,
            'recent_future_predictions': recent_future_predictions,
            'csv_paths': {
                'static': ml_service.static_csv_path,
                'timeseries': ml_service.timeseries_csv_path,
                'models_dir': ml_service.models_dir
            }
        }
        
        return render(request, 'core/model_health_dashboard.html', context)
        
    except Exception as e:
        logger.error(f"Error in model health dashboard: {e}")
        return render(request, 'core/model_health_dashboard.html', {
            'error': f'Error accessing model health information: {str(e)}'
        })

# Continue with existing views...
