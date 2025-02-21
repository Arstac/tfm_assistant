#path project_name/urls.py
from django.contrib import admin
from django.urls import path, include
import chat, authentication, extract_from_documents

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('chat.urls')),
    path('',include('authentication.urls')),
    path('extract_from_documents/',include('extract_from_documents.urls')),
    path('ai_generation_document/',include('ai_generation_document.urls')),
]