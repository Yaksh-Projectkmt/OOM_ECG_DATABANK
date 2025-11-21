from django.urls import path
from .views import register, login, logout,profile,change_password,update_profile,home,help
# from report.views import total_data,refresh_totals_cache
from . import views
from oom_ecg_data.views import fetch_random_ecg_data

urlpatterns = [
    path('', home, name='home'),
    path('login/', login, name='login'),
    path('register/', register, name='register'),
    path('logout/', logout, name='logout'),
    path('help/', help, name='help'), 
    path('profile/', profile, name='profile'),  # Redirect to index view
    path('change_password/',change_password,name='change_password'),
    path('update-profile/',update_profile, name='update_profile'),  # New endpoint
    path('save-contact/', views.save_contact, name='save_contact'),
    path('patient_list/', views.patient_list, name='patient_list'),
    path("patient-arrhythmias/", views.get_patient_arrhythmia_records, name="patient_arrhythmias"),
    path("fetch_random_ecg_data/<str:arrhythmia>/",fetch_random_ecg_data, name="fetch_random_ecg_data"),
    path('get_patients_data/', views.get_patients_data, name='get_patients_data'),
    path('create-order/', views.create_order, name='create_order'),
    path('payment-status/', views.payment_status, name='payment_status'),
    # path('admin/', views.admin_dashboard, name='admin_dashboard'),
    path('get_registrations/', views.get_registrations, name='get_registrations'),
    path('update_registration_status/', views.update_registration_status, name='update_registration_status'),
    path("get_user_status/", views.get_user_status, name="get_user_status"),
    path("wallet_add_money/",views.wallet_add_money,name="wallet_add_money"),
    path("payment-failed/",views.payment_failed,name="payment-failed"),
    path("payment_status_add_money/",views.payment_status_add_money,name="payment_status_add_money"),
    # path("subscription/",views.subscription_plan,name="subscription_plan"),


]

