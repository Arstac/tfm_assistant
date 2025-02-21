from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('extract_info/', views.extract_info),
    path('extract_mats/', views.extract_mats),
    path('question_answering/', views.question_answering),
]