from django.urls import path
from oom_ecg_data import views
urlpatterns = [

    path('', views.index, name='oom_ecg_index'),
    path('new_insert_data/', views.new_insert_data, name="new_insert_data"),
    path('api/',views.api_ecg_data, name='api_ecg_data'),
    path('table/',views.new_table_data, name='new_table_data'),
    path('submit-form/', views.submit_form, name='submit_form'),
    path('fetch_ecg_data/',views.fetch_ecg_data, name='fetch_ecg_data'),
    path("fetch_random_ecg_data/<str:arrhythmia>/",views.fetch_random_ecg_data, name="fetch_random_ecg_data"),
    path("ecg_details/<str:arrhythmia>/",views.ecg_details,name="ecg_details"),
    path('select-arrhythmia/', views.select_arrhythmia, name='select_arrhythmia'),
    path("get_object_id/", views.get_object_id, name="get_object_id"),
    path('edit_data/',views.edit_datas, name="edit_data"),
    path('selecteddownload/', views.selecteddownload, name="selecteddownload"),
    path('delete_data/',views.delete_data, name='delete_data'), 
    path('process_ecg/',views.process_and_return_ecg, name='process_ecg'),
    path('get_pqrst_data/', views.get_pqrst_data, name='get_pqrst_data'),
    path("get_multiple_segments/", views.get_multiple_segments, name="get_multiple_segments"),
    path("insert_db_Data/",views.insert_db_Data,name="insert_db_Data"),
    path("upload_plot/", views.upload_plot, name="upload_plot"),   
    path("check_patient/",views.check_patient, name="check_patient"),
    path("patient_search_view",views.patient_search_view, name="patient_search_view"), 
    # path("fetch_more_ecg",views.fetch_more_ecg, name="fetch_more_ecg"),
    path("get_patient_hex_data",views.get_patient_hex_data, name="get_patient_hex_data"),
]
