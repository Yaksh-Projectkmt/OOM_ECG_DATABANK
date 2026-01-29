from pathlib import Path
import os
from mongoengine import connect
from dotenv import load_dotenv
load_dotenv()
import os

connect(
    db=os.getenv("MONGO_DB"),
    host=os.getenv("MONGO_HOST"),
)

DATABASE_ROUTERS = ['ecgdatabank_1.db_router.ECGDBRouter']
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"

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

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
# SECRET_KEY = None

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv("DJANGO_DEBUG") == "True"
AUTH_USER_MODEL = 'authuser.CustomUser'
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS").split(",")
MAINTENANCE_MODE = os.getenv("MAINTENANCE_MODE") == "True"

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',  # keep THIS
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'subscription.apps.SubscriptionConfig',
    'authuser',
    'analysis_tool',
    'morphology_drow', 
    'oom_ecg_data',
    'report',
    'Beat_Search',
    'St_Segment',
    'Scripts_Models.apps.Script_ModelsConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # MUST be here
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'ecgdatabank_1.context_processors.MaintenanceModeMiddleware',


]

ROOT_URLCONF = 'ecgdatabank_1.urls'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, "ecgdatabank_1", "templates")],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'ecgdatabank_1.context_processors.user_session',
            ],
        },
    },
]

WSGI_APPLICATION = 'ecgdatabank_1.wsgi.application'

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ]
}



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



LANGUAGE_CODE = 'en-us'

TIME_ZONE = "Asia/Kolkata"

USE_I18N = True

USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'ecgdatabank_1', 'static')
] 

# Media files (Uploaded Files)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CUSTOM TOKEN-BASED SESSION EXPIRY
CUSTOM_SESSION_EXPIRY_SECONDS = int(os.getenv("CUSTOM_SESSION_EXPIRY_SECONDS", "604800"))

# DJANGO SESSION SETTINGS
SESSION_ENGINE = "django.contrib.sessions.backends.cached_db"

SESSION_COOKIE_AGE = CUSTOM_SESSION_EXPIRY_SECONDS
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = True  # True in HTTPS production
SESSION_COOKIE_NAME = "django_session"
SESSION_SERIALIZER = "django.contrib.sessions.serializers.JSONSerializer"

# ===============================
# CUSTOM AUTH TOKEN COOKIE
# ===============================
CUSTOM_AUTH_TOKEN_COOKIE = "session_token"

CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SECURE = True  # True in HTTPS production

# ===============================
# SECURITY HEADERS
# ===============================
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"

SECURE_SSL_REDIRECT = os.getenv("SECURE_SSL_REDIRECT") == "True"
SECURE_HSTS_SECONDS = int(os.getenv("SECURE_HSTS_SECONDS"))

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

DATA_UPLOAD_MAX_NUMBER_FILES = None
DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024