from django.urls import path

from . import views

urlpatterns = [
    path('hello_world',views.hello_world),
    path('send_data', views.send_data),
]
