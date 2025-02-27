from django.urls import path
from .views import index, video_feed, get_detected_items

urlpatterns = [
    path("", index, name="index"),
    path("video_feed/", video_feed, name="video_feed"),
    path("get_detected_items/", get_detected_items, name="get_detected_items"),  # ✅ Add this line
]
