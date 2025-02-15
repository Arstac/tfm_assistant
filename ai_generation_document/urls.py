from django.contrib import admin
from django.urls import path
from .views import generate_report, generate_budget, generate_report_pdf, generate_budget_csv, integrate_predictions_to_report, integrate_predictions_to_budget, project_dashboard, evaluate_feasibility

urlpatterns = [
    path('generate-report/', generate_report, name='generate_report'),
    path('generate-budget/', generate_budget, name='generate_budget'),
    path('generate-report-pdf/', generate_report_pdf, name='generate_report_pdf'),
    path('generate-budget-csv/', generate_budget_csv, name='generate_budget_csv'),
    path('integrate-predictions-report/', integrate_predictions_to_report, name='integrate_predictions_report'),
    path('integrate-predictions-budget/', integrate_predictions_to_budget, name='integrate_predictions_budget'),
    path('dashboard/', project_dashboard, name='project_dashboard'),
    path('api/evaluate_feasibility/', evaluate_feasibility, name='evaluar_viabilidad'),
]
