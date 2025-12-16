from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.shortcuts import redirect
from django.urls import reverse

from .models import CustomUser, Wallet
from subscription.models import Plan


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ("username", "email", "role", "package", "plan", "is_active", "is_staff")
    fieldsets = UserAdmin.fieldsets + (
        ("User Type & Plan", {"fields": ("role", "package", "plan")}),
    )
        
@admin.register(Wallet)
class WalletAdmin(admin.ModelAdmin):
    list_display = ("user", "balance", "updated_at")