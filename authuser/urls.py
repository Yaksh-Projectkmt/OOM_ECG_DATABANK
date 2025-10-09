from django.urls import path
from .views import register, login, logout,dashboard,profile,change_password,update_profile,home
#from report.views import index

urlpatterns = [
    path('', home, name='home'),
    path('register/', register, name='register'),
    path('login/', login, name='login'),
    path('dashboard/', dashboard, name='dashboard'),
    path('logout/', logout, name='logout'),
    path('profile/', profile, name='profile'),  # Redirect to index view
    path('change_password/',change_password,name='change_password'),
    path('update-profile/',update_profile, name='update_profile'),  # New endpoint

]

