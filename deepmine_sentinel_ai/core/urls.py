# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Add home page
    path('create/', views.stope_create, name='stope_create'),
    path('stopes/', views.stope_list, name='stope_list'),  # Add list view
    path('stopes/<int:pk>/', views.stope_detail, name='stope_detail'),
    path('upload/', views.upload_excel, name='upload_excel'),
    
    # ML Prediction URLs
    path('api/predict/<int:pk>/', views.predict_stability, name='predict_stability'),
    path('api/feedback/<int:prediction_id>/', views.submit_prediction_feedback, name='submit_prediction_feedback'),
    path('ml/dashboard/', views.ml_dashboard, name='ml_dashboard'),
    path('ml/batch-predict/', views.batch_predict, name='batch_predict'),
    path('ml/batch-results/', views.batch_predict_results, name='batch_predict_results'),

    # Enhanced ML URLs
    path('api/train-model/', views.train_enhanced_model, name='train_enhanced_model'),
    path('ml/health/', views.model_health_dashboard, name='model_health_dashboard'),
    path('api/trigger-prediction/<int:pk>/', views.trigger_prediction_update, name='trigger_prediction_update'),
    
    # Temporal prediction and alert URLs
    path('stopes/<int:pk>/temporal/', views.temporal_prediction_dashboard, name='temporal_prediction_dashboard'),
    path('alerts/<int:alert_id>/acknowledge/', views.acknowledge_alert, name='acknowledge_alert'),
    path('alerts/<int:alert_id>/resolve/', views.resolve_alert, name='resolve_alert'),
]
