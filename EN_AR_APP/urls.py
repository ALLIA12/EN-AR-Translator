from . import views
from django.urls import path

urlpatterns = [
    path('', views.index, name='index'),
    path('translate/', views.translate_text, name='translate_text'),
]