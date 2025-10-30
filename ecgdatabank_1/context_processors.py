from django.shortcuts import render
from django.conf import settings

def user_session(request):
    return {
        'user_session': request.session.get('user_session', None)
    }
  
class MaintenanceModeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check maintenance flag
        if getattr(settings, 'MAINTENANCE_MODE', False):
            return render(request, 'authuser/maintenance.html', status=503)
        return self.get_response(request)