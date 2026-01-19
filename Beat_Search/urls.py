from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path('', views.index, name='Beat_Search'),
    path("start-batch/", views.start_batch_search, name="start_batch_search"),
    path("save-reference/",views.save_reference_pattern,name="save_reference"),
    path("get-images/<str:batch_id>/", views.get_batch_images),
    path("image/<str:image_id>/", views.serve_image),
    path("download-pdf/<str:batch_id>/", views.download_batch_pdf),
    path("check-status/<str:batch_id>/",views.check_batch_status,name="check_batch_status"),
]
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT
)