
from pathlib import Path
import os
from mongoengine import connect
from dotenv import load_dotenv
load_dotenv()

connect(
    db=os.getenv("MONGO_DB"),
    host=os.getenv("MONGO_HOST"),
)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
SESSION_ENGINE = "django.contrib.sessions.backends.db"

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
# SECURITY WARNING: don't run with debug turned on in production!

DEBUG = os.getenv("DJANGO_DEBUG") == "True"
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS").split(",")
MAINTENANCE_MODE = os.getenv("MAINTENANCE_MODE") == "True"


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'authuser',
    'Beat_Search',
    'ST_Segment',
    'Scripts_Models.apps.Script_ModelsConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'searchs.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, "searchs", "templates")],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'searchs.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / os.getenv("SQLITE_NAME")

    },
    'mongodb': {
    'ENGINE': 'djongo',
    'NAME': os.getenv("DJONGO_DB_NAME"),
    'ENFORCE_SCHEMA': os.getenv("DJONGO_ENFORCE_SCHEMA") == "True",
    'CLIENT': {
        'host': os.getenv("DJONGO_HOST"),
    }
    }

}


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'searchs', 'static')
] 
# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

CUSTOM_SESSION_EXPIRY_SECONDS = int(os.getenv("CUSTOM_SESSION_EXPIRY_SECONDS", "604800"))
SESSION_ENGINE = "django.contrib.sessions.backends.cached_db"

SESSION_COOKIE_AGE = CUSTOM_SESSION_EXPIRY_SECONDS
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = False   # True in HTTPS production
SESSION_COOKIE_NAME = "django_session"
SESSION_SERIALIZER = "django.contrib.sessions.serializers.JSONSerializer"

# ===============================
# CUSTOM AUTH TOKEN COOKIE
# ===============================
CUSTOM_AUTH_TOKEN_COOKIE = "session_token"

CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SECURE = False  # True in HTTPS production