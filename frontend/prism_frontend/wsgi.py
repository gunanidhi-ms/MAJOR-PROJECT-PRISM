"""WSGI config for PRISM Frontend."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "prism_frontend.settings")

application = get_wsgi_application()
