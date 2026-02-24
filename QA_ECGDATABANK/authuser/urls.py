from django.urls import path
from .views import login, logout,home
from . import views

urlpatterns = [
    path('', home, name='home'),
    path('login/', login, name='login'),
    path('logout/', logout, name='logout'),
    path('get_patients_data/', views.get_patients_data, name='get_patients_data'),
]

