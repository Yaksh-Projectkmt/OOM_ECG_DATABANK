from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path("", views.index,name='St_Segment'),
    path("run/", views.run_ecg_analysis, name="run"),
    path("view-pdf/<str:file_id>/", views.view_pdf, name="view_pdf"),
    path("download-pdf/<str:file_id>/", views.download_pdf, name="download_pdf"),
]

urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT
)