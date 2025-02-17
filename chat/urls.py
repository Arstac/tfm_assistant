#path chat/urls.py
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import UploadPDFView
from .views import predict_view, chatbot_api, predict_costo_final_view, predict_duracion_real_view, predict_satisfaccion_cliente_view, predict_desviacion_presupuestaria_view, evaluate_feasibility, evaluate_risk


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

    #Testing Pablo
    path("predict/costo_final/", predict_costo_final_view, name="predict_costo_final"),
    path("predict/duracion_real/", predict_duracion_real_view, name="predict_duracion_real"),
    path("predict/satisfaccion_cliente/", predict_satisfaccion_cliente_view, name="predict_satisfaccion_cliente"),
    path("predict/desviacion_presupuestaria/", predict_desviacion_presupuestaria_view, name="predict_desviacion_presupuestaria"),
]