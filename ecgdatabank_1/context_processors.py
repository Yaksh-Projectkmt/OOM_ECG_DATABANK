from django.shortcuts import render
from django.conf import settings

def user_session(request):
    return {
        'user_session': request.session.get('user_session', None)
    }

#class MaintenanceModeMiddleware:
#    def __init__(self, get_response):
#        self.get_response = get_response
#
#    def __call__(self, request):
#        # Check maintenance flag
#        if getattr(settings, 'MAINTENANCE_MODE', False):
#            return render(request, 'authuser/maintenance.html', status=503)
#        return self.get_response(request)
class MaintenanceModeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Your developer/desktop IP(s)
        DEV_IPS = ['192.168.1.16']  # replace with your desktop IP

        # Get client IP
        client_ip = request.META.get('REMOTE_ADDR')

        # Allow access if from your desktop IP
        if client_ip in DEV_IPS:
            return self.get_response(request)

        # Maintenance mode ON for everyone else
        if getattr(settings, 'MAINTENANCE_MODE', False):
            return render(request, 'authuser/maintenance.html', status=503)

        # Maintenance mode OFF
        return self.get_response(request)
    
