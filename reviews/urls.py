from django.urls import path

from . import views

app_name = 'reviews'
urlpatterns = [
    path('', views.get_review, name='get_review'),
]
