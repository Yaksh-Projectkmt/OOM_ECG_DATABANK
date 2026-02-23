from django.urls import path
from django.conf import settings
from . import views
from django.conf.urls.static import static

urlpatterns = [
    path('',views.index, name='analysis_index'),
    path('analyzing_history/',views.analyzing_history,name='analyzing_history'),
    path('upload/',views.uploads_file, name='upload_file'),
    path('process_image/<str:img>/<str:task_id>',views.process_img, name='process_img'),
    path('run_model_arrhythmia/<str:category>/<str:filename>/',views.run_model_arrhythmia, name='run_model_arrhythmia'),
    path('api/',views.api_ecg_data, name='api_ecg_data'),
    path('upload_tmt_pdf/', views.process_tmt_pdf, name='upload_tmt_pdf'),
    path('plot_csv_view/', views.plot_csv_view, name='plot_csv_view'),

    # path("download_patient_pdf/", views.download_patient_pdf, name="download_patient_pdf"),
    path("download/<str:file_id>/", views.download_by_file_id, name="download_by_file_id"),
    path("history/", views.get_analysis_history,name="get_analysis_history"),
    path("download-all-receipt/",views.download_all_receipt, name="download_all_receipt")

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    