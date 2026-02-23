from django.db import models
from django.conf import settings
class Feature(models.Model):
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.name

class Plan(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(null=True, blank=True)
    price = models.FloatField(default=0)
    duration_days = models.IntegerField(default=30)
    features = models.ManyToManyField(Feature, blank=True)

    def __str__(self):
        return self.name

class UserSubscription(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    plan = models.ForeignKey(Plan, on_delete=models.SET_NULL, null=True)
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.plan.name}"
        
class DownloadPrice(models.Model):

    ROLE_CHOICES = (
        ('student', 'student'),
        ('doctor', 'doctor'),
        ('other', 'other'),
    )

    FILE_TYPE_CHOICES = (
        ('image', 'image'),
        ('pdf', 'pdf'),
        ('csv', 'csv'),
    )

    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    file_type = models.CharField(max_length=20, choices=FILE_TYPE_CHOICES)
    price = models.FloatField(default=0)

    class Meta:
        unique_together = ('role', 'file_type')

    def __str__(self):
        return f"{self.role} | {self.file_type} | ?{self.price}"

