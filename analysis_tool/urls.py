from django.urls import path
from django.conf import settings
from . import views
from django.conf.urls.static import static

urlpatterns = [
    path('',views.index, name='analysis_index'),
    path('upload/',views.uploads_file, name='upload_file'),
    path('process_image/<str:img>/',views.process_img, name='process_img'),
    # path('files/<str:category>/',views.fetch_files, name='fetch_files'),
    path('run_model_arrhythmia/<str:category>/<str:filename>/',views.run_model_arrhythmia, name='run_model_arrhythmia'),
    path('api/',views.api_ecg_data, name='api_ecg_data'),
    # path('get_image/<str:filename>/',views.get_processed_image, name='get_processed_image'),
    path('upload_tmt_pdf/', views.upload_tmt_pdf, name='upload_tmt_pdf'),
    # path('download_result/',views.download_analysis_result, name='download_result'),
    path('plot_csv_view/', views.plot_csv_view, name='plot_csv_view'),
    path('download-tmt/<str:filename>',views.download_tmt_file, name='download_tmt'),
    path("download_patient_pdf/", views.download_patient_pdf, name="download_patient_pdf")

    # path('delete-files/',views.delete_files, name='delete_files'),
    # path("run_all_arrhythmia/<str:category>/<str:filename>/", views.run_all_arrhythmia, name="run_all_arrhythmia"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    