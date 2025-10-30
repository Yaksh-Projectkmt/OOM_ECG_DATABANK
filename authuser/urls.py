from django.urls import path
from .views import register, login, logout,profile,change_password,update_profile,home,help
# from report.views import total_data,refresh_totals_cache
from . import views
urlpatterns = [
    path('', home, name='home'),
    path('login/', login, name='login'),
    path('register/', register, name='register'),
    path('logout/', logout, name='logout'),
    path('help/', help, name='help'),  # Redirect to index view
    path('profile/', profile, name='profile'),  # Redirect to index view
    path('change_password/',change_password,name='change_password'),
    path('update-profile/',update_profile, name='update_profile'),  # New endpoint
    path('save-contact/', views.save_contact, name='save_contact'),
]

