#path chat/urls.py
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import UploadPDFView
from .views import predict_view, chatbot_api, evaluate_feasibility, evaluate_risk


urlpatterns = [
    path('chat/', views.chat ),
    path('chat/delete/', views.delete_conversation),
    path('chat/get-titles', views.get_title),
    path('chat/get-data/', views.get_data),
    path('chatbot/', chatbot_api, name='chatbot_api'),
    path('predict/', predict_view, name='predict'),
    path('upload-pdf/', UploadPDFView.as_view(), name='upload-pdf'),
    path('evaluate_feasibility/', evaluate_feasibility, name='evaluate_feasibility'),
    path('evaluate_risk/', evaluate_risk, name='evaluate_risk'),
]