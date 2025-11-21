
from django.http import HttpResponseForbidden
from django.contrib.auth import get_user_model

def feature_required(feature_code):
    def decorator(view_func):
        def _wrapped_view(request, *args, **kwargs):
            user = request.user
            if not user.is_authenticated:
                return HttpResponseForbidden("Please log in.")
            plan = getattr(user, "plan", None)
            if plan and plan.features.filter(code=feature_code).exists():
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden("Feature not available in your subscription plan.")
        return _wrapped_view
    return decorator

from functools import wraps
from subscription.utils import sync_subscription_from_mongo

def sync_subscription(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated:
            sync_subscription_from_mongo(request.user)
        return view_func(request, *args, **kwargs)
    return wrapper