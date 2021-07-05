"""testproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from bloomingapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('signup', views.signup, name='signup'),
    path('login', views.login, name='login'),
    path('logout', views.logout, name='logout'),
    path('diagnosis_intro/<int:user_pk>', views.diagnosis_intro, name='diagnosis_intro'),
    path('draw/<int:user_pk>', views.draw, name='draw'),
    path('que1/<int:user_pk>', views.que1, name='que1'),
    path('que2/<int:user_pk>/<int:dia_pk>', views.que2, name='que2'),
    path('que3/<int:user_pk>/<int:dia_pk>', views.que3, name='que3'),
    path('que4/<int:user_pk>/<int:dia_pk>', views.que4, name='que4'),
    path('que5/<int:user_pk>/<int:dia_pk>', views.que5, name='que5'),
    path('result/<int:user_pk>/<int:dia_pk>', views.result, name='result'),
    path('service_intro/<int:user_pk>', views.service_intro, name='service_intro'),
    path('music/<int:user_pk>', views.music, name='music'),
    path('music_1/<int:user_pk>', views.music_1, name='music_1'),
    path('apiapi', views.apiapi, name='apiapi'),
    path('facegame', views.face, name='facegame'),
    path('face_1', views.face_1, name='face_1')
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

