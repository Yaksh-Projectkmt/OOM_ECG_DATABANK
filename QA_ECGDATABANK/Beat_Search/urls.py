from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path('', views.index, name='Beat_Search'),
    path("start-batch/", views.start_batch_search, name="start_batch_search"),
    path("stream/<str:batch_id>/", views.stream_batch_results),
    path("save-reference/",views.save_reference_pattern,name="save_reference"),
    path("download/<str:batch_id>/", views.download_lead_pdf),
]   
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT
)