# urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('agent-a/', views.agent_a, name='agent_a'),
    path('agent-b/', views.agent_b, name='agent_b'),
]