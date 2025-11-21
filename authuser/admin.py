from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.shortcuts import redirect
from django.urls import reverse

from .models import CustomUser, RegistrationRequest


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ("username", "email", "role", "package", "plan", "is_active", "is_staff")
    fieldsets = UserAdmin.fieldsets + (
        ("User Type & Plan", {"fields": ("role", "package", "plan")}),
    )


@admin.register(RegistrationRequest)
class RegistrationRequestAdmin(admin.ModelAdmin):
    def changelist_view(self, request, extra_context=None):
        url = reverse("registration_requests_admin")
        return redirect(url)

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    class Meta:
        verbose_name = "Registration Request"
        verbose_name_plural = "Registration Requests"
