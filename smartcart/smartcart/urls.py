from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detection.urls')),
    path('detection/', include('detection.urls')),
]
