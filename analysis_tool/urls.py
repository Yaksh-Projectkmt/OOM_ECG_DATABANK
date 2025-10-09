from django.urls import path
from .views import index, api_ecg_data, uploads_file, fetch_files, process_arrhythmia, process_img,get_processed_image,upload_tmt_pdf,download_analysis_result,download_tmt_file,delete_files
from django.conf import settings
from . import views
from django.conf.urls.static import static

urlpatterns = [
    path('', index, name='analysis_index'),
    path('upload/', uploads_file, name='upload_file'),
    path('process_image/<str:filename>/', process_img, name='process_img'),
    path('files/<str:category>/', fetch_files, name='fetch_files'),
    path('process_arrhythmia/<str:category>/<str:filename>/', process_arrhythmia, name='process_arrhythmia'),
    path('api/', api_ecg_data, name='api_ecg_data'),
    path('get_image/<str:filename>/', get_processed_image, name='get_processed_image'),
    path('upload_tmt_pdf/', upload_tmt_pdf, name='upload_tmt_pdf'),
    path('download_result/',download_analysis_result, name='download_result'),
    path('plot_csv_view/', views.plot_csv_view, name='plot_csv_view'),
    path('download-tmt/<str:filename>', download_tmt_file, name='download_tmt'),
    path('delete-files/', delete_files, name='delete_files'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)