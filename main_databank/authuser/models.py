from django.db import models
from django.contrib.auth.models import AbstractUser
from subscription.models import Plan

class MongoUser(models.Model):
    username = models.CharField(max_length=150)
    email = models.EmailField()
    role = models.CharField(max_length=50)
    package = models.CharField(max_length=50)
    plan = models.CharField(max_length=50, default="-")
    status = models.CharField(max_length=50)

    class Meta:
        db_table = "users"   # MongoDB collection name
        managed = False      # Important (do not create SQL table)

    def __str__(self):
        return self.username
    
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
    plan = models.ForeignKey(Plan, null=True, blank=True, on_delete=models.SET_NULL) 

    def __str__(self):
        return self.username

class Wallet(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    balance = models.FloatField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.email} - {self.balance}"