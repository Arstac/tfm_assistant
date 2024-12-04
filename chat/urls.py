#path chat/urls.py
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import chatbot_api, predict_view


urlpatterns = [
    path('chat/', views.chat ),
    path('chat/delete/', views.delete_conversation),
    path('chat/get-titles', views.get_title),
    path('chat/get-data/', views.get_data),
    path('api/chatbot/', chatbot_api, name='chatbot_api'),
    path('predict/', predict_view, name='predict'),
]