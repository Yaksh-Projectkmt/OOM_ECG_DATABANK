from django.urls import path
from morphology_drow import views

urlpatterns = [
    path('', views.index, name='morphology_index'),
    path('api/', views.api_ecg_data, name='api_ecg_data'),
    path('uploads/', views.uploaded_file, name='uploaded_file'),
    path('csv_data/',views.csv_data, name='csv_data'),
    path("upload_ecg/", views.upload_ecg, name="upload_ecg"),
    path("remove_all_csvs/", views.remove_all_csvs, name="remove_all_csvs"),
    path('open_morphology_script/', views.open_morphology_script, name='open_morphology_script'),
]

