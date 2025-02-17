from django.contrib import admin
from django.urls import path
from .views import evaluate_risk, evaluate_feasibility

urlpatterns = [
    path('evaluate_feasibility/', evaluate_feasibility, name='evaluate_feasibility'),
    path('evaluate_risk/', evaluate_risk, name='evaluate_risk'),
]
