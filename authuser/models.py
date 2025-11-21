from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db import models
from subscription.models import Plan

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('student', 'Student'),
        ('doctor', 'Doctor'),
        ('other', 'Other'),
    )
    PACKAGE_CHOICES = (
        ('free', 'Free'),
        ('basic', 'Basic'),
        ('pro', 'Pro'),
        ('premium', 'Premium'),
    )

    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='other')
    package = models.CharField(max_length=10, choices=PACKAGE_CHOICES, default='free')
    plan = models.ForeignKey(Plan, null=True, blank=True, on_delete=models.SET_NULL)  # <-- Added here

from django.db import models

class RegistrationRequest(models.Model):
    class Meta:
        managed = False
        verbose_name = "Registration Request"
        verbose_name_plural = "Registration Requests"

    def __str__(self):
        return "Registration Requests"
