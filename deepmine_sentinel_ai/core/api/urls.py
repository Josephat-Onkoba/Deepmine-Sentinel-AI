"""
URL Configuration for Event Processing API
Task 5: Operational Event Processing System
"""

from django.urls import path, include
from .event_ingestion import (
    EventIngestionAPIView,
    RealTimeEventProcessingView,
    EventStatusView,
    StopeBatchUpdateView
)

# API endpoints for event processing
api_urlpatterns = [
    # Event ingestion endpoints
    path('events/ingest/', EventIngestionAPIView.as_view(), name='api_event_ingest'),
    path('events/process/', RealTimeEventProcessingView.as_view(), name='api_event_process'),
    path('events/<int:event_id>/status/', EventStatusView.as_view(), name='api_event_status'),
    
    # Batch processing endpoints
    path('stopes/batch-update/', StopeBatchUpdateView.as_view(), name='api_batch_update'),
]

urlpatterns = [
    path('api/v1/', include(api_urlpatterns)),
]
