from django.contrib import admin
from django.urls import path, include
from authuser.views import home
from django.conf import settings
from django.conf.urls.static import static
from authuser import views as auth_views 
 
urlpatterns = [
    path('', home, name='home'), 
    path('auth/', include('authuser.urls')),
    path('analysis/', include('analysis_tool.urls')),
    path('morphology/', include('morphology_drow.urls')),
    path('ommecgdata/', include('oom_ecg_data.urls')),
    path('report/', include('report.urls')),
    path('admin-a92bf71c01e24f5bb8f8/', admin.site.urls),
    path('registration-requests/', auth_views.registration_requests_admin, name='registration_requests_admin'),

    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
