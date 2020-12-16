from rest_framework import serializers
from .models import Dream
class Dream_serializer(serializers.ModelSerializer):
    class Meta:
        model = Dream
        fields = ['image']
