"""Global template context for PRISM frontend."""

from django.conf import settings


def prism_settings(request):
    return {
        "PRISM_API_BASE_URL": settings.PRISM_API_BASE_URL,
        "INSTITUTION_NAME": "PRISM Radiology Centre",
    }
