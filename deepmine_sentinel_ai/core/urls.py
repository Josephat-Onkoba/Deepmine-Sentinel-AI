from django.urls import path
from . import views, api_views

urlpatterns = [
    # Web views
    path('', views.home, name='home'),
    path('upload/', views.upload_excel, name='upload_excel'),
    path('stope/<int:pk>/', views.stope_detail, name='stope_detail'),
    path('stopes/', views.stope_list, name='stope_list'),
    path('create/', views.stope_create, name='stope_create'),
    
    # API endpoints for LSTM inference
    path('api/predict/single', api_views.predict_single, name='api_predict_single'),
    path('api/predict/batch', api_views.predict_batch, name='api_predict_batch'),
    path('api/predict/demo', api_views.predict_demo, name='api_predict_demo'),
    path('api/model/performance', api_views.model_performance, name='api_model_performance'),
    path('api/model/info', api_views.model_info, name='api_model_info'),
    path('api/predictions/summary', api_views.prediction_summary, name='api_prediction_summary'),
    path('api/health', api_views.health_check, name='api_health_check'),
    path('api/docs', api_views.api_docs, name='api_docs'),
]

