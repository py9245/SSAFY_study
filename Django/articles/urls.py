from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.index),
    path('hello/', views.hello),
    path('dinner/', views.dinner),
    path('search/', views.search),
    path('throw/', views.throw, name='throw'),
    path('catch/', views.catch),
    path('articles/<int:num>/', views.detail),
]