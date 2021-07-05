from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Account(models.Model):
    objects = models.Manager()
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='users') # id
    name = models.CharField(max_length=10)
    email = models.CharField(max_length=48)
    gender = models.CharField(max_length=5)
    age = models.IntegerField()
    created_date = models.DateTimeField(auto_now = True)

    def __str__(self):
        return self.name

class Diagnosis(models.Model):
    objects = models.Manager()
    diagnosis = models.ForeignKey(User, on_delete=models.CASCADE, related_name='diagnosis')
    image = models.ImageField(blank=True, upload_to='images', null=True)
    selection = models.CharField(max_length=50, null=True)
    doing = models.CharField(max_length = 50, null= True)
    feel = models.IntegerField(null=True)
    tree_state = models.CharField(max_length=60, null =True)
    tree_forward = models.CharField(max_length = 50, null= True)
    home_mood = models.CharField(max_length = 50, null= True)
    home_door = models.IntegerField(null=True)
    last_q = models.CharField(max_length = 80, null= True)
    
    def __str__(self):
        return self.diagnosis.username

