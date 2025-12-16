from django.contrib import admin
from .models import Plan, Feature, UserSubscription, DownloadPrice

@admin.register(Feature)
class FeatureAdmin(admin.ModelAdmin):
    list_display = ('name', 'code')

@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ("name", "description", "price", "duration_days")

@admin.register(UserSubscription)
class UserSubscriptionAdmin(admin.ModelAdmin):
    list_display = ('user', 'plan', 'start_date', 'end_date')

@admin.register(DownloadPrice)
class DownloadPriceAdmin(admin.ModelAdmin):
    list_display = ('role', 'file_type', 'price')
    list_filter = ('role', 'file_type')
