from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('llmapp/', include('llmapp.urls')),
    path('', include('llmapp.urls'))
]
