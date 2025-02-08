#path project_name/urls.py
from django.contrib import admin
from django.urls import path, include
import chat, authentication

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('chat.urls')),
    path('',include('authentication.urls')),
]