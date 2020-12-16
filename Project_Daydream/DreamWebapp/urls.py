from django.urls import path
from . import views

urlpatterns = [
    path('',views.DreamView.as_view()),



]
