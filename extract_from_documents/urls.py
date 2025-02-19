from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('extract_info/', views.extract_info),
]