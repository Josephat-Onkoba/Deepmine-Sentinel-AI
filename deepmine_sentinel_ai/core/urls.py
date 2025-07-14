# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Add home page
    path('create/', views.stope_create, name='stope_create'),
    path('stopes/', views.stope_list, name='stope_list'),  # Add list view
    path('stopes/<int:pk>/', views.stope_detail, name='stope_detail'),
    path('upload/', views.upload_excel, name='upload_excel'),
]

