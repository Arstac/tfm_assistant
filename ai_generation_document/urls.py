from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('evaluate_feasibility/', views.evaluate_feasibility, name='evaluate_feasibility'),
    # path('evaluate_feasibility_pdf/', views.evaluate_feasibility_pdf, name='evaluate_feasibilit_pdf'),
    path('evaluate_risk/', views.evaluate_risk, name='evaluate_risk'),
    path('cost_variance/', views.cost_variance, name='cost_variance'),
]
